import json
import io
import base64
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import click
from loguru import logger
from PIL import Image

import cv2
import numpy as np
from iso639 import Lang

from models.pipeline_batch_item import PipelineBatchItemData
from utils import (
    get_db,
    process_db_write_batch,
    get_time,
    decode_image_bytes,
)
from models import PipelineBatchItem, Detection, Caption, Classification

from const import (
    CAPTION_MAX_IMG_DIM,
    CAPTION_MODEL_NAME,
    CAPTION_MODEL_TEMPERATURE,
    CAPTION_MAX_TOKENS,
    CAPTION_TOP_LOGPROBS,
    OPENAI_REQUEST_TIMEOUT,
    CAPTION_REQUEST_RETRY_ATTEMPTS,
    CPUS_LIMIT_CAPTION,
    CAPTION_CLASSES_EXCLUDED,
    CAPTION_MAX_BATCH_SIZE,
)
import openai

# Module-level client for connection reuse within each worker process
_openai_client = None


def get_openai_client():
    """Get or create a reusable OpenAI client for this worker process."""
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI().with_options(
            max_retries=CAPTION_REQUEST_RETRY_ATTEMPTS,
            timeout=OPENAI_REQUEST_TIMEOUT,
        )
    return _openai_client


@click.command("step04-caption")
@click.option("--id-pipeline-batch", type=int, required=True)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT_CAPTION,
    help="Allows for limiting the number of CPU cores this command can use.",
)
def step04_caption(id_pipeline_batch: int, cpus_limit: int):
    """
    Runs caption-generation on the cropped regions of each volume that contains detections.

    NOTE:
    - This command is intended to be run by the orchestrator. See orchestration/execute.py for details.
    - This is a batch-level step, which expects to process a batch for which images and text are already cached on disk.
    - Adjust CAPTION_MAX_BATCH_SIZE env var based on your OpenAI API tier and usage.
    """

    # Concurrency model:
    # - We use a ProcessPoolExecutor with processes_total worker processes, each
    #   responsible for a disjoint subset of PipelineBatchItem IDs (round‑robin
    #   assignment via item_batches).
    # - Each worker process:
    #     * Initializes its own DB connection (initializer=get_db).
    #     * Calls caption_batch_of_items to handle decoding, cropping, and
    #       OpenAI API calls for its assigned items.
    # - Within each worker, we create small ThreadPoolExecutors for:
    #     * Decoding scans (CPU‑bound but easily parallelizable).
    #     * Sending caption requests to OpenAI (I/O‑bound, lots of waiting).
    # - The global cpus_limit controls how many *processes* we launch; inside
    #   each process we cap the number of threads via per_task_cpus_limit so
    #   that processes_total * per_task_cpus_limit stays roughly bounded and we
    #   avoid excessive oversubscription of CPU threads across the machine.

    processes_total = cpus_limit
    logger.info(f"Launching {processes_total} CPU processes ...")

    if processes_total > 1:
        per_task_cpus_limit = max(2, cpus_limit // 2)
    else:
        # Single-process mode: allow the worker to use the full CPU budget.
        per_task_cpus_limit = cpus_limit

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

    # Use Peewee's iterator() to avoid materializing the entire result set in memory.
    item_batches = [[] for _ in range(processes_total)]
    got_any_items = False
    for i, item in enumerate(eligible_query.iterator()):
        got_any_items = True
        item_batches[i % processes_total].append(item.id_pipeline_batch_item)

    if not got_any_items:
        logger.warning("No items with eligible detections/classifications found. Exiting.")
        return

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
                click.get_current_context().exit(1)


def caption_batch_of_items(item_ids: list[int], cpus_limit: int) -> bool:
    """
    Generate captions for all detections belonging to a batch of PipelineBatchItem IDs.
    """
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
    detections_by_item: dict[int, list[Detection]] = {}
    for det in detections_query:
        item_id = det.pipeline_batch_item_id
        detections_by_item.setdefault(item_id, []).append(det)

    # Collect all captions and track statistics
    all_captions: list[Caption] = []
    total_n_crops = 0
    total_failed_crops = 0
    total_failed_captions = 0
    total_decode_time = 0.0

    for item in items:
        # Access pre-fetched detections
        dets = detections_by_item.get(item.id_pipeline_batch_item, [])
        if not dets:
            continue

        id_pipeline_batch_item = item.id_pipeline_batch_item
        volume = item.ib_volume
        barcode = volume.barcode

        item_data: PipelineBatchItemData = item.data

        image_bytes_by_filename = {str(k): v for k, v in item_data.images.items()}
        texts_by_filename = {str(k): v for k, v in item_data.texts.items()}

        del item_data

        #
        # Group detections by scan filename for this item.
        # Decode and process one scan at a time to limit memory.
        #
        dets_by_scan: dict[str, list[Detection]] = {}
        for det in dets:
            fn = str(det.scan_filename)
            dets_by_scan.setdefault(fn, []).append(det)

        max_batch = CAPTION_MAX_BATCH_SIZE

        lang = get_language(volume)
        # lang is a human-readable language name derived via iso639 (e.g., "English").

        n_crops = 0
        failed_crops = 0
        failed_captions = 0
        decode_time_for_item = 0.0

        # Small in-memory batch of (Detection, np.ndarray crop)
        batch: list[tuple[Detection, np.ndarray]] = []

        for fn, det_list in dets_by_scan.items():
            img_bytes = image_bytes_by_filename.get(fn)
            if img_bytes is None:
                # Missing image -> all detections on this scan fail as crops
                failed_crops += len(det_list)
                continue

            # Decode this scan only once
            start_decode = datetime.now()
            try:
                scan_img = decode_image_bytes(img_bytes)
            except Exception:
                # If we can't decode this image, all detections on it fail
                failed_crops += len(det_list)
                logger.warning(f"{barcode}: Could not decode image for {fn}; skipping detections")
                continue
            decode_time_for_item += (datetime.now() - start_decode).total_seconds()

            # Build crops for detections on this scan, but stream them into small batches
            for det in det_list:
                try:
                    crop = det.crop(scan_img)
                except Exception:
                    logger.warning(f"Could not crop detection in {barcode}.{fn}; skipping")
                    failed_crops += 1
                    continue

                batch.append((det, crop))
                n_crops += 1

                if len(batch) >= max_batch:
                    failed_captions += _process_caption_batch(
                        batch=batch,
                        texts_by_filename=texts_by_filename,
                        lang=lang,
                        barcode=barcode,
                        id_pipeline_batch_item=id_pipeline_batch_item,
                        cpus_limit=cpus_limit,
                        all_captions=all_captions,
                    )
                    batch.clear()

            # Done with this scan; free decoded image
            del scan_img

        # Final partial batch (across all scans for this item)
        if batch:
            failed_captions += _process_caption_batch(
                batch=batch,
                texts_by_filename=texts_by_filename,
                lang=lang,
                barcode=barcode,
                id_pipeline_batch_item=id_pipeline_batch_item,
                cpus_limit=cpus_limit,
                all_captions=all_captions,
            )
            batch.clear()

        total_decode_time += decode_time_for_item
        total_n_crops += n_crops
        total_failed_crops += failed_crops
        total_failed_captions += failed_captions

        logger.info(
            f"{barcode} | Captions: {n_crops} | Failed crops: {failed_crops} | "
            f"Decode time: {decode_time_for_item:.2f}s | Failed Captions: {failed_captions}"
        )

    #
    # Batch DB operations - delete and write all at once
    #
    Caption.delete().where(Caption.pipeline_batch_item.in_(item_ids)).execute()
    process_db_write_batch(Caption, all_captions)

    return True


def _process_caption_batch(
    batch: list[tuple[Detection, np.ndarray]],
    texts_by_filename: dict[str, str],
    lang: str,
    barcode: str,
    id_pipeline_batch_item: int,
    cpus_limit: int,
    all_captions: list[Caption],
) -> int:
    """
    Caption a single batch of (Detection, crop) records.

    Returns:
        Number of failed captions in this batch.
    """
    failed_captions = 0

    # caption batches (OpenAI API calls)
    with ThreadPoolExecutor(max_workers=cpus_limit) as api_pool:
        future_to_det: dict = {}

        for det, crop in batch:
            context_file = det.scan_filename.split(".")[0] + ".txt"
            context = texts_by_filename.get(context_file, "")

            # base64-encode and free crop ASAP to limit memory
            b64 = base64_jpg_bytes(crop)
            del crop

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
            try:
                result = fut.result()
            except Exception as e:
                # Any OpenAI / network error is treated as a failed caption for that crop.
                failed_captions += 1
                logger.warning(
                    f"{barcode}: Caption request failed for detection {det.id_detection}: {e}"
                )
                caption_text = ""
                logprobs_data = None
            else:
                if not result:
                    failed_captions += 1
                    caption_text = ""
                    logprobs_data = None
                else:
                    caption_text = result["text"]
                    logprobs_data = result["logprobs"]

            all_captions.append(
                Caption(
                    detection=det.id_detection,
                    text=caption_text,
                    lang=lang,
                    logprobs=logprobs_data,
                    pipeline_batch_item=id_pipeline_batch_item,
                    created=get_time(),
                )
            )

    return failed_captions


def get_language(volume) -> str:
    """
    Infer the language for a volume from its metadata.

    Input:
        volume: A Volume-like object with a metadata field that may contain
                a JSON string or dict, and an optional "language_src" key.

    Output:
        A human-readable language name (e.g., "English"), derived via iso639.
        Defaults to "English" if language metadata is missing or invalid.
    """
    try:
        metadata = volume.metadata
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}
        lang = metadata.get("language_src")
        lang = Lang(lang).name
    except Exception:
        lang = "English"
    return lang


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
    """
    Send a single captioning request to OpenAI.

    - Uses the OpenAI Python SDK's built-in retry mechanism instead of manual retry loops.
    - Raises exceptions on failure; callers are responsible for handling/logging them.
    """
    client = get_openai_client()

    response = client.responses.create(
        model=CAPTION_MODEL_NAME,
        input=input_blocks,
        max_output_tokens=CAPTION_MAX_TOKENS,
        temperature=CAPTION_MODEL_TEMPERATURE,
        include=["message.output_text.logprobs"],
        top_logprobs=CAPTION_TOP_LOGPROBS,
    )

    for item in response.output:
        for content in item.content:
            if getattr(content, "type", None) in ("output_text", "summary_text"):
                # Extract and serialize logprobs if available
                logprobs_data = None
                if hasattr(content, "logprobs") and content.logprobs:
                    logprobs_data = serialize_logprobs(content.logprobs)

                return {"text": content.text.strip(), "logprobs": logprobs_data}

    # No suitable content found
    return None


@dataclass
class SerializedLogprob:
    token: str
    bytes: list[int] | None
    logprob: float
    top_logprobs: list[dict] | None = None


def serialize_logprobs(logprobs_list):
    """
    Serialize logprobs from OpenAI response to a JSON-compatible format.

    Returns:
        A list of dicts (converted from SerializedLogprob dataclass instances), or
        None if no logprobs are present.
    """
    if not logprobs_list:
        return None

    serialized: list[dict] = []
    for logprob_item in logprobs_list:
        top_logprobs_serialized = None

        # Serialize top_logprobs if present
        if hasattr(logprob_item, "top_logprobs") and logprob_item.top_logprobs:
            top_logprobs_serialized = [
                {
                    "token": top.token,
                    "bytes": top.bytes if hasattr(top, "bytes") else None,
                    "logprob": top.logprob,
                }
                for top in logprob_item.top_logprobs
            ]

        entry = SerializedLogprob(
            token=logprob_item.token,
            bytes=getattr(logprob_item, "bytes", None),
            logprob=logprob_item.logprob,
            top_logprobs=top_logprobs_serialized,
        )
        serialized.append(asdict(entry))

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
