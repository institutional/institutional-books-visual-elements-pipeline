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
from models import PipelineBatchItem, Detection, Caption, Classification

from const import (
    CAPTION_MAX_IMG_DIM,
    CAPTION_MODEL_NAME,
    CAPTION_MODEL_TEMPERATURE,
    CAPTION_MAX_TOKENS,
    CAPTION_TOP_LOGPROBS,
    CAPTION_MAX_REQUESTS,
    OPENAI_REQUEST_TIMEOUT,
    CAPTION_REQUEST_RETRY_ATTEMPTS,
    CPUS_LIMIT,
    CAPTION_CLASSES_EXCLUDED,
    CAPTION_MAX_BATCH_SIZE,
)
import openai
from openai import APITimeoutError


client = openai.OpenAI()


@click.command("step04_process_caption_requests")
@click.option("--id-pipeline-batch", type=int, required=True)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT,
    help="Allows for limiting the number of CPU cores this command can use.",
)
def step04_process_caption_requests(id_pipeline_batch: int, cpus_limit: int):
    """
    Runs caption-generation on the cropped regions of each volume that contains detections.

    NOTE:
    - This command is intended to be run by the orchestrator. See `orchestration/execute.py` for details.
    - This is a batch-level step, which expects to process a batch for which images and text are already cached on disk.
    - Adjust `CAPTION_MAX_REQUESTS` env var based on your OpenAI API tier and usage.
    """

    processes_total = cpus_limit
    logger.info(f"Launching {processes_total} CPU processes ...")

    if processes_total > 1:
        per_task_cpus_limit = max(2, cpus_limit // 2)

    # Select only items with detections that have classifications where pred_class is not in excluded classes
    eligible_query = (
        PipelineBatchItem.select(PipelineBatchItem)
        .where(
            (PipelineBatchItem.pipeline_batch == id_pipeline_batch)
            & PipelineBatchItem.id_pipeline_batch_item.in_(
                Detection.select(Detection.pipeline_batch_item)
                .join(Classification, on=(Detection.id_detection == Classification.detection))
                .where(Classification.pred_class.not_in(CAPTION_CLASSES_EXCLUDED))
            )
        )
        .distinct()
    )
    eligible_items = list(eligible_query)

    # TODO: REMOVE WHEN ACTUALLY RUNNING PIPELINE - HERE FOR BUDGET REASONS
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

    # Fetch all items
    items = list(
        PipelineBatchItem.select().where(PipelineBatchItem.id_pipeline_batch_item.in_(item_ids))
    )

    # Fetch ALL detections for these items at once (with non-excluded classifications)
    detections_query = (
        Detection.select()
        .join(Classification, on=(Detection.id_detection == Classification.detection))
        .where(
            (Detection.pipeline_batch_item.in_(item_ids))
            & (Classification.pred_class.not_in(CAPTION_CLASSES_EXCLUDED))
        )
        .order_by(Detection.id_detection)
        .distinct()
    )

    # Group detections by pipeline_batch_item
    detections_by_item = {}
    for det in detections_query:
        item_id = det.pipeline_batch_item_id
        if item_id not in detections_by_item:
            detections_by_item[item_id] = []
        detections_by_item[item_id].append(det)

    # Collect all captions and track statistics
    all_captions = []
    total_n_crops = 0
    total_failed_crops = 0
    total_failed_captions = 0
    total_decode_time = 0

    for item in items:
        # Access pre-fetched detections (no duplicate query needed)
        dets = detections_by_item.get(item.id_pipeline_batch_item, [])
        if not dets:
            continue

        id_pipeline_batch_item = item.id_pipeline_batch_item
        volume = item.ib_volume
        barcode = volume.barcode

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
        time_decode = (datetime.now() - start_decode).total_seconds()
        total_decode_time += time_decode

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
        max_batch = CAPTION_MAX_BATCH_SIZE

        lang = get_language(volume)

        for batch in chunked(crop_records, max_batch):

            with ThreadPoolExecutor(max_workers=cpus_limit) as api_pool:
                future_to_det = {}
                for det, crop in batch:
                    context_file = det.scan_filename.split(".")[0] + ".txt"
                    context = texts_by_filename.get(context_file, "")
                    b64 = base64_jpg_bytes(crop)
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

                # Process results with logprobs
                for fut in as_completed(future_to_det):
                    det = future_to_det[fut]
                    result = fut.result()

                    if not result:
                        failed_captions += 1
                        caption_text = ""
                        logprobs_data = None
                    else:
                        caption_text = result["text"]
                        logprobs_data = result["logprobs"]

                    captioned_entries.append(
                        Caption(
                            detection=det.id_detection,
                            caption=caption_text,
                            lang=lang,
                            logprobs=logprobs_data,
                            pipeline_batch_item=id_pipeline_batch_item,
                            scan_filename=det.scan_filename,
                            created=datetime.now(timezone.utc),
                        )
                    )

        # Add to batch totals
        all_captions.extend(captioned_entries)
        total_n_crops += n_crops
        total_failed_crops += failed_crops
        total_failed_captions += failed_captions

        logger.info(
            f"{barcode} | Captions: {n_crops} | Failed crops: {failed_crops} | Decode time: {time_decode:.2f}s | Failed Captions: {failed_captions}"
        )

    #
    # Batch DB operations - delete and write all at once
    #
    Caption.delete().where(Caption.pipeline_batch_item.in_(item_ids)).execute()
    process_db_write_batch(Caption, all_captions)

    return True


def get_language(volume):
    try:
        metadata = volume.metadata
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}
        lang = metadata.get("language_src")
        lang = Lang(lang).name
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


def base64_jpg_bytes(arr) -> str:
    # Convert OpenCV (BGR) to RGB once
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    # Using JPEG to send to OpenAI for faster encoding and smaller size
    img = resize_image(img, CAPTION_MAX_IMG_DIM)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def send_to_openai(input_blocks, id):
    """Send request to OpenAI with retry logic"""
    last_exception = None

    for attempt in range(CAPTION_REQUEST_RETRY_ATTEMPTS):
        try:
            response = client.responses.create(
                model=CAPTION_MODEL_NAME,
                input=input_blocks,
                max_output_tokens=CAPTION_MAX_TOKENS,
                temperature=CAPTION_MODEL_TEMPERATURE,
                include=["message.output_text.logprobs"],
                top_logprobs=CAPTION_TOP_LOGPROBS,
                timeout=OPENAI_REQUEST_TIMEOUT,
            )
            for item in response.output:
                for content in item.content:
                    if getattr(content, "type", None) in ("output_text", "summary_text"):
                        # Extract and serialize logprobs if available
                        logprobs_data = None
                        if hasattr(content, "logprobs") and content.logprobs:
                            logprobs_data = serialize_logprobs(content.logprobs)

                        return {"text": content.text.strip(), "logprobs": logprobs_data}

            return None

        except APITimeoutError as e:
            last_exception = e
            logger.warning(
                f"OpenAI API request timed out for {id} (attempt {attempt + 1}/{CAPTION_REQUEST_RETRY_ATTEMPTS})."
            )
            if attempt < CAPTION_REQUEST_RETRY_ATTEMPTS - 1:
                time.sleep(2**attempt)  # Exponential backoff
                continue
        except Exception as e:
            last_exception = e
            logger.warning(
                f"An error occurred for {id} (attempt {attempt + 1}/{CAPTION_REQUEST_RETRY_ATTEMPTS}): {e}"
            )
            if attempt < CAPTION_REQUEST_RETRY_ATTEMPTS - 1:
                time.sleep(2**attempt)  # Exponential backoff
                continue

    # All retries failed
    logger.error(
        f"All {CAPTION_REQUEST_RETRY_ATTEMPTS} attempts failed for {id}. Last error: {last_exception}"
    )
    return None


def serialize_logprobs(logprobs_list):
    """
    Serialize logprobs from OpenAI response to a JSON-compatible format.
    """
    if not logprobs_list:
        return None

    serialized = []
    for logprob_item in logprobs_list:
        item_dict = {
            "token": logprob_item.token,
            "bytes": logprob_item.bytes if hasattr(logprob_item, "bytes") else None,
            "logprob": logprob_item.logprob,
        }

        # Serialize top_logprobs if present
        if hasattr(logprob_item, "top_logprobs") and logprob_item.top_logprobs:
            item_dict["top_logprobs"] = [
                {
                    "token": top.token,
                    "bytes": top.bytes if hasattr(top, "bytes") else None,
                    "logprob": top.logprob,
                }
                for top in logprob_item.top_logprobs
            ]

        serialized.append(item_dict)

    return serialized


def build_instruction(language: str) -> str:
    return (
        "You are a librarian who writes precise, concise captions for image crops.\n"
        "- Use the page text only as supporting context, and only when it clearly "
        "relates to what is visible in the image crop.\n"
        "- Focus on describing what is visually present in the crop. Do not summarize "
        "or restate the page text.\n"
        "- Do not add background information, interpretations, or educated guesses "
        "that are not directly shown in the image crop.\n"
        "- If you are unsure about something, describe it generically (for example, "
        '"a person," "a diagram," "a building") rather than guessing.\n'
        "- The caption must be 50 words or less.\n"
        f"- Reply only with the caption, in {language}, with no additional commentary."
    )
