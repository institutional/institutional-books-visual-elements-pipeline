import json
import io
import base64
import traceback
import time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
import click
from loguru import logger
from PIL import Image

import numpy as np
import cv2
from more_itertools import chunked
from iso639 import Lang

from utils import get_db, process_db_write_batch
from models import PipelineBatchItem, Detection, Caption

from const import (
    CAPTION_MAX_IMG_DIM,
    CAPTION_MODEL_NAME,
    CAPTION_MODEL_TEMPERATURE,
    CAPTION_MAX_TOKENS,
    CAPTION_TOP_LOGPROBS,
    CPUS_LIMIT_CAPTIONS,
    CAPTION_MAX_REQUESTS,
    OPENAI_REQUEST_TIMEOUT,
    MAX_OPENAI_CONCURRENT_REQUESTS,
)
import openai
from openai import APITimeoutError


client = openai.OpenAI()


@click.command("step04_process_caption_requests")
@click.option("--id-pipeline-batch", type=int, required=True)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT_CAPTIONS,
    help="Allows for limiting the number of CPU cores this command can use.",
)
def step04_process_caption_requests(id_pipeline_batch: int, cpus_limit: int):
    """
    Runs caption-generation on the cropped regions of each volume that contains detections.
    """

    processes_total = cpus_limit
    logger.info(f"Launching {processes_total} CPU processes ...")

    if processes_total > 1:
        per_task_cpus_limit = max(2, cpus_limit // 2)

    # Select only items with detections
    eligible_query = (
        PipelineBatchItem.select(PipelineBatchItem)
        .where(
            (PipelineBatchItem.pipeline_batch == id_pipeline_batch)
            & PipelineBatchItem.id_pipeline_batch_item.in_(
                Detection.select(Detection.pipeline_batch_item)
            )
        )
        .distinct()
    )
    eligible_items = list(eligible_query)

    # REMOVE WHEN ACTUALLY RUNNING PIPELINE - HERE FOR BUDGET REASONS
    eligible_items = eligible_items[:CAPTION_MAX_REQUESTS]

    if not eligible_items:
        logger.warning("No items with detections found. Exiting.")
        return

    item_batches = [[] for _ in range(processes_total)]
    for i, item in enumerate(eligible_items):
        item_batches[i % processes_total].append(item.id_pipeline_batch_item)

    with ProcessPoolExecutor(max_workers=processes_total, initializer=get_db) as executor:
        futures = {}
        for idx, item_ids in enumerate(item_batches):
            if not item_ids:
                continue
            future = executor.submit(
                caption_batch_of_items,
                item_ids=item_ids,
                cpus_limit=per_task_cpus_limit,
            )
            futures[future] = idx
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                logger.error("Error in a worker process:\n" + traceback.format_exc())
                executor.shutdown(wait=False, cancel_futures=True)
                return


def caption_batch_of_items(item_ids: list[int], cpus_limit: int):
    for id_pipeline_batch_item in item_ids:
        item = PipelineBatchItem.get(id_pipeline_batch_item=id_pipeline_batch_item)
        volume = item.ib_volume
        barcode = volume.barcode

        dets = (
            Detection.select()
            .where(Detection.pipeline_batch_item == id_pipeline_batch_item)
            .order_by(Detection.id_detection)
        )

        if dets.count() == 0:
            logger.info(f"{barcode}: No detections - skipping.")
            continue

        # images + text context
        image_bytes_by_filename = {str(k): v for k, v in item.data.images.items()}
        texts_by_filename = {str(k): v for k, v in item.data.texts.items()}

        # decode scans in parallel
        loaded_images = {}
        used_filenames = set(det.scan_filename for det in dets)

        start_decode = datetime.now()
        with ThreadPoolExecutor(max_workers=cpus_limit) as pool:
            futures = {
                pool.submit(decode_image_bytes, image_bytes_by_filename[fn]): fn
                for fn in used_filenames
                if fn in image_bytes_by_filename
            }
            done, _ = wait(futures)
            for fut in done:
                fn = futures[fut]
                try:
                    loaded_images[fn] = fut.result()
                except:
                    logger.warning(f"{barcode}: Failed to decode {fn}")
        time_decode = datetime.now() - start_decode

        # crops
        crop_records = []
        failed_crops = 0
        failed_captions = 0
        n_crops = 0
        for det in dets:
            img = loaded_images.get(det.scan_filename)
            if img is None:
                failed_crops += 1
                continue
            try:
                crop = det.crop(img)
                crop_records.append((det, crop))
                n_crops += 1
            except:
                failed_crops += 1

        if not crop_records:
            logger.info(f"{barcode}: All crops failed.")
            continue

        # caption batches (OpenAI API calls)
        captioned_entries = []
        max_batch = 8

        lang = get_language(volume)

        for batch in chunked(crop_records, max_batch):

            max_openai_concurrent = MAX_OPENAI_CONCURRENT_REQUESTS  # tune based on rate limits

            with ThreadPoolExecutor(max_workers=max_openai_concurrent) as api_pool:
                future_to_det = {}
                for det, crop in batch:
                    context_file = det.scan_filename.split(".")[0] + ".txt"
                    context = texts_by_filename.get(context_file, "")
                    b64 = base64_png_bytes(crop)
                    instruction = build_instruction(lang)

                    input_blocks = [
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": instruction}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": context},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{b64}",
                                },
                            ],
                        },
                    ]
                    fut = api_pool.submit(send_to_openai, input_blocks, det.id_detection)
                    future_to_det[fut] = det

                for fut in as_completed(future_to_det):
                    det = future_to_det[fut]
                    caption_text = fut.result()

                    if not caption_text:
                        failed_captions += 1

                    captioned_entries.append(
                        Caption(
                            detection_id=det.id_detection,
                            caption=caption_text,
                            lang=lang,
                            pipeline_batch_item=id_pipeline_batch_item,
                            scan_filename=det.scan_filename,
                            created=datetime.now(timezone.utc),
                        )
                    )

            # DB write
        Caption.delete().where(Caption.pipeline_batch_item == id_pipeline_batch_item).execute()
        process_db_write_batch(Caption, captioned_entries)

        logger.info(
            f"{barcode} | Captions: {n_crops} | Failed crops: {failed_crops} | Decode time: {time_decode} | Failed Captions {failed_captions}"
        )

    return True


def get_language(volume):
    try:
        meta = json.loads(volume.metadata)
        lang = Lang(meta.get("language_src")).name
    except:
        lang = "English"
    return lang


def decode_image_bytes(image_bytes):
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def resize_image(img: Image.Image, max_dim: int):
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img
    if w > h:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def base64_png_bytes(arr) -> str:
    img = Image.fromarray(arr)
    img = resize_image(img, CAPTION_MAX_IMG_DIM)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def send_to_openai(input_blocks, id):
    try:
        response = client.responses.create(
            model=CAPTION_MODEL_NAME,
            input=input_blocks,
            max_output_tokens=CAPTION_MAX_TOKENS,
            temperature=CAPTION_MODEL_TEMPERATURE,
            top_logprobs=CAPTION_TOP_LOGPROBS,
            timeout=OPENAI_REQUEST_TIMEOUT,
        )
        for item in response.output:
            for content in item.content:
                if getattr(content, "type", None) in ("output_text", "summary_text"):
                    return content.text.strip()
    except APITimeoutError:
        logger.info(f"OpenAI API request timed out for {id}.")
        return False
    except Exception as e:
        logger.info(f"An error occurred for {id}: {e}")
        return False


def build_instruction(language: str) -> str:
    return (
        "You are a librarian that captions images in precise and concise language. "
        "Create a caption in 50 words or less for the image, given the page text as context. "
        "Reply only with the caption. "
        f"Write the caption in {language}."
    )
