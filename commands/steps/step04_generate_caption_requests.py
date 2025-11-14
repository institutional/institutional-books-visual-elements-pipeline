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
from more_itertools import chunked
from models import PipelineBatchItem, Detection, IBVolume
from const import CPUS_LIMIT, MAX_TOKENS_PER_DAY

client = openai.OpenAI()

from const import (
    CAPTION_BATCH_DIR,
    CAPTION_BATCH_SIZE,
    CAPTION_MAX_FILE_MB,
    CAPTION_MAX_IMG_DIM,
    CAPTION_MAX_IMG_TOKENS,
    CAPTIONS_TABLE,
    CAPTION_TOKENS_TABLE,
    CAPTION_MODEL_NAME,
)


@click.command("step04-caption")
@click.option("--id-pipeline-batch", type=int, required=True)
def request_captions(id_pipeline_batch: int):
    """
    Generate OpenAI batch files for image (crop) captions with OCR and submit as jobs.
    """
    os.makedirs(CAPTION_BATCH_DIR, exist_ok=True)

    # --- Aggregate instances to caption (Detection records for this batch with crops)
    # Each detection should have: volume_id, scan_filename, bbox (to crop), and OCR text
    query = (
        Detection.select()
        .join(PipelineBatchItem)
        .where(PipelineBatchItem.pipeline_batch == id_pipeline_batch)
        .order_by(PipelineBatchItem.id_pipeline_batch_item, Detection.id_detection)
    )

    caption_requests = []
    token_counter = 0

    logger.info(f"Generating caption jobs for {query.count()} crops (batch {id_pipeline_batch})")
    for det in query:
        # Get associated batch item and volume to retrieve text/lang
        item = det.pipeline_batch_item
        volume = item.ib_volume

        # Get language from IBVolume metadata
        try:
            lang = json.loads(volume.metadata)["language_src"]
        except Exception:
            lang = "English"  # sensible fallback

        # Get OCR text for scan/crop
        ocr_filename = os.path.splitext(det.scan_filename)[0] + ".txt"
        try:
            context_text = item.data.texts.get(ocr_filename, "")
            if not context_text:
                logger.info(
                    f"No OCR context for crop {det.scan_filename} (volume {volume.barcode}), using empty."
                )
        except Exception as e:
            context_text = ""
            logger.error(f"Error reading OCR for crop {det.scan_filename}: {e}")

        # Get and crop image bytes from cache
        try:
            image_bytes = item.data.images[str(det.scan_filename)]
            from models import Detection  # local to avoid circular import

            scan_img = decode_image_bytes(image_bytes)
            crop_img = det.crop(scan_img)
        except Exception as e:
            logger.warning(f"Failed to crop {det.scan_filename}: {e}")
            continue

        # Prepare image as base64
        try:
            img_b64 = base64_png_bytes(crop_img)
        except Exception as e:
            logger.warning(f"Failed to encode image for {det.scan_filename}: {e}")
            continue

        caption_requests.append(
            {
                "image_b64": img_b64,
                "ocr_text": context_text,
                "language": lang,
                "batch_item_id": item.id_pipeline_batch_item,
                "detection_id": det.id_detection,
                "scan_filename": det.scan_filename,
                "volume_barcode": volume.barcode,
            }
        )
        token_counter += CAPTION_MAX_IMG_TOKENS

    # --- Batch into chunked files; enforce OpenAI limits and daily token quota
    today_tok = get_today_token_count()
    if today_tok + token_counter > MAX_TOKENS_PER_DAY:
        logger.error(
            f"Token budget for today exceeded: planned {token_counter}, today {today_tok}, max {MAX_TOKENS_PER_DAY}"
        )
        raise click.ClickException("Would exceed MAX_TOKENS_PER_DAY")

    add_token_count(token_counter)
    batch_chunks = list(chunked(caption_requests, min(CAPTION_BATCH_SIZE, len(caption_requests))))
    batch_file_paths = []

    for i, batch in enumerate(batch_chunks):
        # Write batch to JSONL as per create_batch_file spec.
        batch_jsonl_fn = os.path.join(
            CAPTION_BATCH_DIR, f"captions_batch_{id_pipeline_batch}_{i:04d}.jsonl"
        )
        with open(batch_jsonl_fn, "w", encoding="utf-8") as outfile:
            for idx, req in enumerate(batch, 1):
                system_message, user_messages = create_prompt(req["language"])
                # Compose just like in create_batch_file
                messages = (
                    [system_message]
                    + user_messages
                    + [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": req["ocr_text"]},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{req['image_b64']}"
                                    },
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
                    "custom_id": f"{id_pipeline_batch}-{req['detection_id']}-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
        batch_file_paths.append(batch_jsonl_fn)
        logger.info(f"Wrote {len(batch)} requests to {batch_jsonl_fn}")

    # Submit the batches and log their openai batch IDs
    # for batch_file in batch_file_paths:
    #     batch = process_batch(batch_file, metadata={...})
    #     TODO: Save batch info to log


def base64_png_bytes(ndarray_img) -> str:
    # Convert np.array uint8 image to base64 PNG for OpenAI Batch format
    img = Image.fromarray(ndarray_img)
    # Resize for safety
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
    # TODO: Fetch sum(tokens) from caption_tokens where date is today
    # return int
    return 0  # stub


def add_token_count(count):
    # TODO: Write to DB or update your token ledger for the day
    pass


def decode_image_bytes(image_bytes) -> Any:
    import numpy as np, cv2

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR_RGB)


def create_prompt(language: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Constructs a system message and user prompt messages for image captioning in a given language.

    Args:
        language: The target language for the caption (e.g., "English", "Spanish").

    Returns:
        A tuple containing:
            - system_message: Dictionary with the system's context.
            - user_messages: List of user message dictionaries, including the caption instruction and the specified language.
    """
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
