import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
import gc
import time

import click
from loguru import logger
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
from more_itertools import chunked

from utils import get_db, process_db_write_batch
from models import PipelineBatchItem, Detection, Classification

import torch
from ultralytics import YOLO, settings
from datetime import datetime, timedelta

from const import (
    CLASSIFICATION_MODEL_REPO,
    CLASSIFICATION_MODEL_FILEPATH,
    CLASSIFICATION_MODEL_IMGSZ,
    CLASSIFICATION_MODEL_CONF,
    CLASSIFICATION_MODEL_PROCESSES_FORK_DELAY,
    CUDA_GPUS,
    CPUS_LIMIT,
)


@click.command("step02-classify")
@click.option(
    "--id-pipeline-batch",
    type=int,
)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT,
    help="Allows for limiting the number of CPU cores this command can use.",
)
@click.option(
    "--cuda-gpus",
    type=click.Choice(CUDA_GPUS),
    multiple=True,
    required=True,
    default=CUDA_GPUS if CUDA_GPUS else ["cuda:0"],
    help="Determines on which specific CUDA device(s) this command should use.",
)
def step02_classify(id_pipeline_batch: int, cpus_limit: int, cuda_gpus: list[str]):
    """
    Runs the classification model on element crops from each volume containing detections.
    """
    model_filepath: Path | None = None
    cuda_gpus_total = len(cuda_gpus)
    processes_total = cuda_gpus_total  # 1 process per GPU (can adjust if needed)
    item_id_batches: list[list[int]] = [[] for _ in range(processes_total)]

    per_task_cpus_limit = int(round(cpus_limit / cuda_gpus_total))
    if processes_total > 1:
        per_task_cpus_limit = max(2, per_task_cpus_limit // 2)

    # 1. Download model if needed
    logger.info(f"Pulling {CLASSIFICATION_MODEL_REPO} from HuggingFace or cache ...")
    model_filepath = Path(
        hf_hub_download(
            repo_id=CLASSIFICATION_MODEL_REPO,
            filename=CLASSIFICATION_MODEL_FILEPATH,
        ),
    )

    # 2. Populate batches ONLY for pipeline batch items that actually had detections:
    eligible_items_query = (
        PipelineBatchItem.select(PipelineBatchItem)
        .where(
            (PipelineBatchItem.pipeline_batch == id_pipeline_batch)
            & PipelineBatchItem.id_pipeline_batch_item.in_(
                Detection.select(
                    Detection.pipeline_batch_item
                )  # Only volumes with at least 1 detection
            )
        )
        .order_by(PipelineBatchItem.id_pipeline_batch_item)
        .distinct()
    )

    eligible_items = list(eligible_items_query)
    for i, item in enumerate(eligible_items):
        process_i = i % processes_total
        item_id_batches[process_i].append(item.id_pipeline_batch_item)

    if not any(item_id_batches):
        logger.warning("No eligible items with detections found for this batch. Exiting.")
        click.get_current_context().exit(0)

    # 3. Pool
    with ProcessPoolExecutor(max_workers=processes_total, initializer=get_db) as executor:
        futures = {}
        for i, item_ids in enumerate(item_id_batches):
            cuda_gpus_i = i % cuda_gpus_total
            future = executor.submit(
                classify_batch_of_items,
                item_ids=item_ids,
                model_filepath=model_filepath,
                cuda_device=cuda_gpus[cuda_gpus_i],
                cpus_limit=per_task_cpus_limit,
            )
            futures[future] = cuda_gpus[cuda_gpus_i]
            time.sleep(CLASSIFICATION_MODEL_PROCESSES_FORK_DELAY)
        for future in as_completed(futures):
            cuda_gpu: str = futures[future]
            try:
                future.result()
            except Exception as err:
                logger.debug(traceback.print_exc())
                logger.error(
                    f"A blocking error occured while processing batch on {cuda_gpu}. Exiting."
                )
                executor.shutdown(wait=False, cancel_futures=True)
                click.get_current_context().exit(1)
            except KeyboardInterrupt as err:
                logger.warning("Received interrupt signal")
                raise err


def classify_batch_of_items(
    item_ids: list[int],
    model_filepath: Path,
    cuda_device: str,
    cpus_limit: int,
):

    settings.update({"sync": False})

    # Load model
    model = YOLO(model_filepath, verbose=False)
    model.to(cuda_device)
    class_names = list(model.names.values())

    # For each item/volume
    for id_pipeline_batch_item in item_ids:
        item = PipelineBatchItem.get(id_pipeline_batch_item=id_pipeline_batch_item)
        volume_barcode = item.ib_volume.barcode

        # Get all Detection records for this item
        item_detections = (
            Detection.select()
            .where(Detection.pipeline_batch_item == id_pipeline_batch_item)
            .order_by(Detection.id_detection)
        )

        if item_detections.count() == 0:
            logger.info(f"{volume_barcode}: No detections - skipping classification for this item.")
            continue

        # images = dict: filename -> bytes
        image_bytes_by_filename = dict(list(item.data.images.items()))

        # Force change to string
        image_bytes_by_filename = {str(k): v for k, v in image_bytes_by_filename.items()}

        # Aggregators
        classified_entries: list[Classification] = []
        n_crops = 0
        failed_crops = 0

        time_preproc = timedelta(0)
        time_infer = timedelta(0)
        time_db = timedelta(0)
        time_clear_gpu = timedelta(0)

        # We'll group crops in batches for GPU efficiency
        crop_image_records = []  # tuples of (Detection, np.ndarray, filename)

        # Preprocessing: decode scan images, do crop according to bbox
        start = datetime.now()
        with ThreadPoolExecutor(max_workers=cpus_limit) as decode_executor:
            futures = {}
            # First, decode all scans in parallel (only scans used by this item's detections)
            used_filenames = set(det.scan_filename for det in item_detections)
            loaded_images: dict[str, np.ndarray] = {}
            for fn in used_filenames:
                futures[decode_executor.submit(decode_image_bytes, image_bytes_by_filename[fn])] = (
                    fn
                )
            done, _ = wait(futures)
            for future in done:
                fn = futures[future]
                try:
                    loaded_images[fn] = future.result()
                except Exception:
                    logger.warning(f"Could not decode scan {volume_barcode}.{fn}")
        # Now, for each detection, get the crop
        for det in item_detections:
            scan_img = loaded_images.get(det.scan_filename)
            if scan_img is None:
                failed_crops += 1
                continue
            try:
                crop = det.crop(scan_img)
                crop_image_records.append((det, crop, det.scan_filename))
                n_crops += 1
            except Exception:
                logger.warning(
                    f"Could not crop detection in {volume_barcode}.{det.scan_filename}; skipping"
                )
                failed_crops += 1

        end = datetime.now()
        time_preproc += end - start

        if n_crops == 0:
            logger.info(f"{volume_barcode}: All crops failed, skipping.")
            continue

        # Augment with classification
        max_batch = 16  # safety: don't overload GPU with too many big crops
        crop_batches = list(chunked(crop_image_records, max_batch))

        for batch in crop_batches:
            detections_batch, images_batch, filenames_batch = zip(*batch)

            # Inference!
            start_inf = datetime.now()
            results = model(
                list(images_batch),
                imgsz=CLASSIFICATION_MODEL_IMGSZ,
                conf=CLASSIFICATION_MODEL_CONF,
                device=cuda_device,
                verbose=False,
            )
            end_inf = datetime.now()
            time_infer += end_inf - start_inf

            for idx, result in enumerate(results):
                probs = getattr(result, "probs", None)
                if probs is None:
                    logger.warning(
                        f"No probs for crop of {filenames_batch[idx]} in {volume_barcode}: skipping."
                    )
                    continue
                pred_idx = probs.top1
                logger.info(str(pred_idx))
                try:
                    pred_class = class_names[pred_idx]
                except Exception:
                    pred_class = str(pred_idx)
                pred_conf = None
                if hasattr(probs, "data"):
                    pred_conf = float(probs.data[pred_idx])
                elif hasattr(probs, "__getitem__"):
                    try:
                        pred_conf = float(probs[pred_idx])
                    except Exception:
                        pass
                if pred_conf is None:
                    logger.warning(
                        f"No pred_conf for crop {filenames_batch[idx]} on {volume_barcode}; skipping."
                    )
                    continue

                classified_entries.append(
                    Classification(
                        detection_id=detections_batch[idx].id_detection,
                        pred_idx=int(pred_idx),
                        pred_class=pred_class,
                        pred_probs=probs.data.tolist() if hasattr(probs, "data") else None,
                        pipeline_batch_item=id_pipeline_batch_item,
                        scan_filename=filenames_batch[idx],
                        pred_conf=pred_conf,
                    )
                )

        # Save all to the database
        start_db = datetime.now()
        # WARNING: you may want to delete previous Classification entries for this item
        Classification.delete().where(
            Classification.pipeline_batch_item == id_pipeline_batch_item
        ).execute()
        process_db_write_batch(
            model=Classification,
            entries_to_create=classified_entries,
        )
        end_db = datetime.now()
        time_db += end_db - start_db

        # Clear GPU cache and run GC
        start_gpu = datetime.now()
        with torch.cuda.device(int(cuda_device.replace("cuda:", ""))):
            torch.cuda.empty_cache()
        gc.collect()
        end_gpu = datetime.now()
        time_clear_gpu += end_gpu - start_gpu

        # Stats
        logger.info(
            f"{volume_barcode} | Crops: {n_crops} - "
            f"Failed crops: {failed_crops} - "
            f"Device: {cuda_device} - "
            f"Preproc: {time_preproc} - "
            f"Infer: {time_infer} - "
            f"DB: {time_db} - "
            f"Clear GPU: {time_clear_gpu} - "
            f"Total: {time_preproc + time_infer + time_db + time_clear_gpu}"
        )

    return True


# Use the detection/decode code as in detection pipeline
def decode_image_bytes(image_bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR_RGB)
