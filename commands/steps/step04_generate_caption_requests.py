import os
import json
import base64
import io
from typing import List, Dict, Tuple, Any
from PIL import Image
import openai
import click
from loguru import logger
from models import PipelineBatchItem, Detection, IBVolume
from iso639 import Lang
from functools import partial
from more_itertools import chunked
import os
import multiprocessing

client = openai.OpenAI()

from const import (
    CAPTION_MAX_IMG_DIM,
    CAPTION_MAX_TOKENS,
    CAPTION_MODEL_NAME,
    CPUS_LIMIT_CAPTIONS,
    CAPTION_JSONL_FILES_PATH,
    MAX_REQUESTS_PER_FILE,
)


@click.command("step04-generate-caption-requests")
@click.option("--id-pipeline-batch", type=int, required=True)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT_CAPTIONS,
    help="Allows for limiting the number of CPU cores this command can use.",
)
def step04_generate_caption_requests(
    id_pipeline_batch: int,
    cpus_limit: int,
):
    """
    Parallel caption JSONL creation per volume.
    Enforces daily token limits and JSONL file size.
    """
    os.makedirs(CAPTION_JSONL_FILES_PATH, exist_ok=True)

    # TODO: save batch in subfolder named based on id-pipeline-batch

    # Query all detections in this batch, group by volume barcode

    query = (
        Detection.select()
        .join(PipelineBatchItem)
        .switch(Detection)
        .where(PipelineBatchItem.pipeline_batch == id_pipeline_batch)
        .order_by(PipelineBatchItem.id_pipeline_batch_item, Detection.id_detection)
        .prefetch(PipelineBatchItem, IBVolume)
    )

    detections = list(query)

    dets_by_volume = {}

    for det in detections:
        vol = det.pipeline_batch_item.ib_volume
        dets_by_volume.setdefault(vol.barcode, []).append(det)

    # Use multiprocessing to process per-volume
    ctx = multiprocessing.get_context("spawn")

    # Kick off the jobs
    pool = ctx.Pool(cpus_limit)

    jobs = []
    for barcode, dets in dets_by_volume.items():
        jobs.append(
            pool.apply_async(
                process_volume,
                args=(barcode, dets),
                kwds={"id_pipeline_batch": id_pipeline_batch, "cpus_limit": cpus_limit},
            )
        )
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


def process_volume(
    barcode,
    dets,
    id_pipeline_batch,
    cpus_limit: int = 1,
):
    item = dets[0].pipeline_batch_item
    volume = item.ib_volume
    lang = get_language(volume)
    outfiles = []
    n_captions = 0
    failed_crops = 0
    failed_captions = 0

    # Decode images
    image_bytes_by_filename = dict(list(item.data.images.items()))
    image_bytes_by_filename = {str(k): v for k, v in image_bytes_by_filename.items()}

    # get context texts
    texts_by_filename = dict(list(item.data.texts.items()))
    texts_by_filename = {str(k): v for k, v in texts_by_filename.items()}

    used_filenames = set(str(det.scan_filename) for det in dets)

    loaded_images = {}
    from concurrent.futures import ThreadPoolExecutor, wait

    with ThreadPoolExecutor(max_workers=cpus_limit) as decode_executor:
        futures = {}
        for fn in used_filenames:
            if fn not in image_bytes_by_filename:
                logger.warning(
                    f"Missing image bytes for scan {barcode}.{fn} - skipping this scan in captioning"
                )
                continue
            futures[decode_executor.submit(decode_image_bytes, image_bytes_by_filename[fn])] = fn
        done, _ = wait(futures)
        for future in done:
            fn = futures[future]
            try:
                loaded_images[fn] = future.result()
            except Exception:
                logger.warning(f"Could not decode scan {barcode}.{fn}")

    # Split into batches of size MAX_REQUESTS_PER_FILE
    for batch_idx, dets_chunk in enumerate(chunked(dets, MAX_REQUESTS_PER_FILE)):
        jsonl_entries = []
        batch_captions = 0
        file_idx = batch_idx  # Will be incremented if file size hits limit

        for det in dets_chunk:
            # OCR/context
            try:
                filename = det.scan_filename.split(".")[0] + ".txt"
                context_text = texts_by_filename[filename]
            except Exception:
                context_text = ""
                failed_captions += 1
                continue

            # Look up scanned image via filename
            scan_img = loaded_images.get(str(det.scan_filename))
            if scan_img is None:
                logger.warning(f"Decoded image missing for {barcode}.{det.scan_filename}")
                failed_crops += 1
                continue

            # Crop & encode image
            try:
                crop_img = det.crop(scan_img)
                img_b64 = base64_png_bytes(crop_img)
            except Exception as e:
                logger.warning(f"Failed to crop/encode {det.scan_filename}: {e}")
                failed_crops += 1
                continue

            # Compose JSONL item (unchanged)
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
                "max_tokens": CAPTION_MAX_TOKENS,
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
            batch_captions += 1
            n_captions += 1

        # Write any remainder from this chunk
        if jsonl_entries:
            jsonl_fn = os.path.join(
                CAPTION_JSONL_FILES_PATH, f"captions_vol_{barcode}_{file_idx:04d}.jsonl"
            )
            with open(jsonl_fn, "w", encoding="utf-8") as outfile:
                outfile.writelines(jsonl_entries)
            outfiles.append(jsonl_fn)

    logger.info(
        f"{barcode} | n_captions: {n_captions} - jsonl_filename: {outfiles} - failed crops: {failed_crops} - failed_captions: {failed_captions}"
    )
