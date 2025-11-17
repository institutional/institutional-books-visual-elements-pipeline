import os
import json
import base64
import io
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime, timezone, timedelta
from PIL import Image
import openai
import click
from loguru import logger
from models import PipelineBatchItem, CaptionTokenLedger, Detection
from datetime import date
from iso639 import Lang
import multiprocessing
from functools import partial
from more_itertools import chunked

client = openai.OpenAI()

from const import (
    CAPTION_BATCH_SIZE,
    CAPTION_MAX_FILE_MB,
    CAPTION_MAX_IMG_DIM,
    CAPTION_MAX_IMG_TOKENS,
    CAPTION_MODEL_NAME,
    CPUS_LIMIT,
    MAX_TOKENS_PER_DAY,
    CAPTION_JSONL_FILES_PATH,
)

CAPTION_MAX_JSONL_BYTES = CAPTION_MAX_FILE_MB * 1024 * 1024


@click.command("step04-caption")
@click.option("--id-pipeline-batch", type=int, required=True)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT,
    help="Allows for limiting the number of CPU cores this command can use.",
)
def request_captions(
    id_pipeline_batch: int,
    cpus_limit: int,
):
    """
    Parallel caption JSONL creation per volume.
    Enforces daily token limits and JSONL file size.
    """
    os.makedirs(CAPTION_JSONL_FILES_PATH, exist_ok=True)

    # Query all detections in this batch, group by volume barcode
    query = (
        Detection.select()
        .join(PipelineBatchItem)
        .where(PipelineBatchItem.pipeline_batch == id_pipeline_batch)
        .order_by(PipelineBatchItem.id_pipeline_batch_item, Detection.id_detection)
    )
    dets_by_volume = {}
    for det in query:
        vol = det.pipeline_batch_item.ib_volume
        dets_by_volume.setdefault(vol.barcode, []).append(det)

    # Use multiprocessing to process per-volume
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    shared_token = manager.Value("i", get_today_token_count())
    lock = manager.Lock()  # Ensures token increments atomic

    def process_volume(barcode, dets, id_pipeline_batch, shared_token, lock):
        item = dets[0].pipeline_batch_item
        volume = item.ib_volume
        lang = get_language(volume)
        outfiles = []
        n_captions = 0
        failed_crops = 0
        failed_captions = 0

        # Split into batches of size CAPTION_BATCH_SIZE
        for batch_idx, dets_chunk in enumerate(chunked(dets, CAPTION_BATCH_SIZE)):
            jsonl_entries = []
            jsonl_bytes = 0
            batch_captions = 0
            file_idx = batch_idx  # Will be incremented if file size hits limit

            for det in dets_chunk:
                # OCR/context
                try:
                    context_text = get_context_text(det, item)
                except Exception:
                    context_text = ""
                    failed_captions += 1
                    continue

                # Crop & encode image
                try:
                    image_bytes = item.data.images[str(det.scan_filename)]
                    scan_img = decode_image_bytes(image_bytes)
                    crop_img = det.crop(scan_img)
                    img_b64 = base64_png_bytes(crop_img)
                except Exception as e:
                    logger.warning(f"Failed to crop/encode {det.scan_filename}: {e}")
                    failed_crops += 1
                    continue

                # Compose JSONL item
                system_message, user_messages = create_prompt(lang)
                messages = (
                    [system_message]
                    + user_messages
                    + [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": context_text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                                },
                            ],
                        }
                    ]
                )
                body = {
                    "model": CAPTION_MODEL_NAME,
                    "messages": messages,
                    "max_tokens": CAPTION_MAX_IMG_TOKENS,
                    "logprobs": True,
                    "top_logprobs": 2,
                    "temperature": 0,
                }
                obj = {
                    "custom_id": f"{id_pipeline_batch}-{det.id_detection}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                json_line = json.dumps(obj, ensure_ascii=False) + "\n"
                jsonl_entries.append(json_line)
                jsonl_bytes += len(json_line.encode("utf-8"))
                batch_captions += 1
                n_captions += 1

                # If adding the next would push over file size, dump so far
                if jsonl_bytes >= CAPTION_MAX_JSONL_BYTES:
                    jsonl_fn = os.path.join(
                        CAPTION_JSONL_FILES_PATH, f"captions_vol_{barcode}_{file_idx:04d}.jsonl"
                    )
                    with open(jsonl_fn, "w", encoding="utf-8") as outfile:
                        outfile.writelines(jsonl_entries)
                    outfiles.append(jsonl_fn)
                    file_idx += 1
                    jsonl_entries = []
                    jsonl_bytes = 0

            # Write any remainder from this chunk
            if jsonl_entries:
                jsonl_fn = os.path.join(
                    CAPTION_JSONL_FILES_PATH, f"captions_vol_{barcode}_{file_idx:04d}.jsonl"
                )
                with open(jsonl_fn, "w", encoding="utf-8") as outfile:
                    outfile.writelines(jsonl_entries)
                outfiles.append(jsonl_fn)

        # ----- Enforce token daily limit safely -----
        tokens_for_captions = n_captions * CAPTION_MAX_IMG_TOKENS
        with lock:
            if shared_token.value + tokens_for_captions > MAX_TOKENS_PER_DAY:
                logger.error(
                    f"{barcode} | Would exceed MAX_TOKENS_PER_DAY. Skipping: {n_captions} captions."
                )
                failed_captions += n_captions
                for fn in outfiles:
                    try:
                        os.remove(fn)
                    except Exception:
                        pass
                outfiles = []
                n_captions = 0
            else:
                shared_token.value += tokens_for_captions
                add_token_count(tokens_for_captions)

        logger.info(
            f"{barcode} | n_captions: {n_captions} - jsonl_filename: {outfiles} - failed crops: {failed_crops} - failed_captions: {failed_captions}"
        )

    # Compose partial function for the pool
    partial_fn = partial(
        process_volume, id_pipeline_batch=id_pipeline_batch, shared_token=shared_token, lock=lock
    )

    # Kick off the jobs
    pool = ctx.Pool(cpus_limit)
    jobs = []
    for barcode, dets in dets_by_volume.items():
        jobs.append(pool.apply_async(partial_fn, args=(barcode, dets)))
    pool.close()
    pool.join()


def get_language(volume):
    try:
        lang = json.loads(volume.metadata)["language_src"]
        lang = Lang(lang)
        lang = lang.name
    except Exception:
        lang = "English"
    return lang


def get_context_text(det, item):
    ocr_filename = os.path.splitext(det.scan_filename)[0] + ".txt"
    return item.data.texts.get(ocr_filename, "")


def base64_png_bytes(ndarray_img) -> str:
    img = Image.fromarray(ndarray_img)
    img = resize_image(img, CAPTION_MAX_IMG_DIM)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def resize_image(image: Image.Image, max_dimension: int):
    width, height = image.size
    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image


def get_today_token_count():
    today = date.today()
    try:
        rec, _ = CaptionTokenLedger.get_or_create(date=today)
        return rec.tokens_used
    except Exception as e:
        logger.error(f"Could not get today's token usage: {e}")
        return 0


def add_token_count(count):
    today = date.today()
    db = CaptionTokenLedger._meta.database
    with db.atomic():
        rec, _ = CaptionTokenLedger.get_or_create(date=today)
        query = CaptionTokenLedger.update(tokens_used=CaptionTokenLedger.tokens_used + count).where(
            CaptionTokenLedger.date == today
        )
        query.execute()


def decode_image_bytes(image_bytes) -> Any:
    import numpy as np, cv2

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR_RGB)


def create_prompt(language: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a librarian that captions images in precise and concise language."
                ),
            }
        ],
    }
    user_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Create a caption in 50 words or less for the image given the context by the page text."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Reply only with the caption."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Write the caption in {language}"}],
        },
    ]
    return system_message, user_messages


# actually process in seperate script

# def process_batch(batch_file, metadata):
#     """Upload BATCH_FILE and create a batch job with optional --metadata."""
#     # Handle metadata
#     metadata_dict = {}
#     if metadata:
#         try:
#             metadata_dict = json.loads(metadata)
#         except json.JSONDecodeError:
#             try:
#                 with open(metadata, "r") as f:
#                     metadata_dict = json.load(f)
#             except Exception as e:
#                 click.echo(f"Error with metadata: {e}")
#                 return
#     # Upload the file
#     click.echo(f"Uploading {batch_file}...")
#     file = upload_batch(client, batch_file)
#     # Create the batch
#     click.echo("Creating batch...")
#     batch = create_batch(client, file, metadata=metadata_dict)
#     click.echo(f"Batch created: {batch}")


# def upload_batch(client: Any, batch_filename: str) -> Any:
#     """
#     Uploads a batch file to the API client.

#     Args:
#         client: The API client with file upload capability.
#         batch_filename: Path to the batch file to upload.

#     Returns:
#         The uploaded file object as returned by the client.
#     """
#     # Use context manager to ensure file is closed properly
#     with open(batch_filename, "rb") as file_obj:
#         batch_input_file = client.files.create(file=file_obj, purpose="batch")
#     return batch_input_file


# def create_batch(client: Any, batch_file: Any, metadata: Optional[Dict[str, str]] = None) -> Any:
#     """
#     Creates a new batch using the uploaded file's ID.

#     Args:
#         client: The API client.
#         batch_file: The uploaded batch file object (should have 'id').
#         metadata: Optional metadata dictionary.

#     Returns:
#         The created batch object.
#     """
#     if metadata is None:
#         metadata = {}
#     batch_input_file_id = batch_file.id
#     batch = client.batches.create(
#         input_file_id=batch_input_file_id,
#         endpoint="/v1/chat/completions",
#         completion_window="24h",
#         metadata=metadata,
#     )
#     print(f"Created batch: {batch}")
#     return batch
