import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import gc
from datetime import timedelta
import multiprocessing as mp

import click
from loguru import logger
from huggingface_hub import hf_hub_download
import numpy as np
from more_itertools import chunked

from utils import (
    get_db,
    process_db_write_batch,
    get_time,
    load_scans_for_detections,
    build_detection_crops,
)
from models import PipelineBatchItem, Detection, Classification

from const import (
    CLASSIFICATION_MODEL_PROCESSES_PER_GPU,
    CLASSIFICATION_MODEL_REPO,
    CLASSIFICATION_MODEL_FILEPATH,
    CLASSIFICATION_MODEL_IMGSZ,
    CLASSIFICATION_MODEL_CONF,
    CUDA_GPUS,
    CPUS_LIMIT,
    CLASSIFICATION_MAX_BATCH,
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
    Runs the classification model on the visual elements detected for each volume.
    """

    # Concurrency model:
    # - We create a pool of worker processes and assign them across the available CUDA devices.
    # - CLASSIFICATION_MODEL_PROCESSES_PER_GPU controls how many processes share a single GPU.
    #   processes_total = cuda_gpus_total * processes_per_gpu is the total number of workers.
    # - Each worker process:
    #     * Initializes its own DB connection (via initializer=get_db).
    #     * Loads the classification model once and pins it to a specific CUDA device.
    #     * Processes only the subset of item IDs assigned to it in item_id_batches.
    # - Item IDs are distributed round‑robin across all workers for simple, reasonably even
    #   load balancing.
    # - If multiple processes share a GPU, we reduce the CPU cores per process
    #   (per_task_cpus_limit) to avoid oversubscribing the host CPU and causing contention.
    # - We use a process pool (instead of threads) because:
    #     * PyTorch/Ultralytics workloads are CPU- and GPU-heavy and benefit from process-level
    #       isolation (no GIL issues, more predictable memory usage).
    #     * CUDA + fork can be problematic; we explicitly use the "spawn" start method below.

    model_filepath: Path | None = None
    cuda_gpus_total = len(cuda_gpus)
    processes_per_gpu = CLASSIFICATION_MODEL_PROCESSES_PER_GPU
    processes_total = cuda_gpus_total * processes_per_gpu
    item_id_batches: list[list[int]] = [[] for _ in range(processes_total)]

    per_task_cpus_limit = int(round(cpus_limit / cuda_gpus_total))

    if processes_total > 1:
        per_task_cpus_limit = max(2, per_task_cpus_limit // 2)

    # Download model if needed
    logger.info(f"Pulling {CLASSIFICATION_MODEL_REPO} from HuggingFace or cache ...")
    model_filepath = Path(
        hf_hub_download(
            repo_id=CLASSIFICATION_MODEL_REPO,
            filename=CLASSIFICATION_MODEL_FILEPATH,
        ),
    )

    # Populate batches ONLY for pipeline batch items that actually had detections:
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

    # Using Peewee's iterator() to avoid materializing the entire result set
    # in memory at once. We assign items round-robin to workers.
    for i, item in enumerate(eligible_items_query.iterator()):
        process_i = i % processes_total
        item_id_batches[process_i].append(item.id_pipeline_batch_item)

    if not any(item_id_batches):
        logger.warning("No eligible items with detections found for this batch. Exiting.")
        click.get_current_context().exit(0)

    # Pool - Classifying batches
    # Use "spawn" to avoid CUDA / fork issues when starting worker processes.
    mp_ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=processes_total,
        initializer=get_db,
        mp_context=mp_ctx,
    ) as executor:
        futures = {}
        for i, item_ids in enumerate(item_id_batches):
            cuda_gpus_i = i % cuda_gpus_total
            # We map worker index i to a specific CUDA device by taking
            # i % cuda_gpus_total, so workers are evenly distributed across
            # the available GPUs. When CLASSIFICATION_MODEL_PROCESSES_PER_GPU > 1,
            # multiple workers will share the same GPU.
            future = executor.submit(
                classify_batch_of_items,
                item_ids=item_ids,
                model_filepath=model_filepath,
                cuda_device=cuda_gpus[cuda_gpus_i],
                cpus_limit=per_task_cpus_limit,
            )
            futures[future] = cuda_gpus[cuda_gpus_i]

        for future in as_completed(futures):
            cuda_gpu: str = futures[future]
            try:
                future.result()
            except Exception:
                logger.debug(traceback.format_exc())
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
    """
    Classify all detections for a batch of PipelineBatchItem IDs on a single worker process.
    """
    import os

    # set CUDA_VISIBLE_DEVICES before importing torch/ultralytics - avoids defaulting to cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device.replace("cuda:", "")
    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    device = "cuda:0"  # since the corresponding device is the only one visible, it becomes cuda:0

    # Load model
    model = YOLO(model_filepath, verbose=False)
    model.to(device)
    class_names = list(model.names.values())

    # Collect all classifications from all items
    all_classified_entries: list[Classification] = []

    # Track timing for all items
    total_time_preproc = timedelta(0)
    total_time_infer = timedelta(0)
    total_time_clear_gpu = timedelta(0)
    total_n_crops = 0
    total_failed_crops = 0

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

        if not item_detections.exists():
            logger.warning(f"Volume {volume_barcode} has no detection. Skipping.")
            continue

        # images = dict: filename -> bytes
        image_bytes_by_filename = {str(k): v for k, v in item.data.images.items()}

        # Aggregators for recording stats
        classified_entries: list[Classification] = []

        time_preproc = timedelta(0)
        time_infer = timedelta(0)
        time_clear_gpu = timedelta(0)

        # Preprocessing: decode scan images, do crop according to bbox
        start = get_time()

        # Decode only the scans referenced by this item's detections
        loaded_images = load_scans_for_detections(
            volume_barcode=volume_barcode,
            detections=item_detections,
            image_bytes_by_filename=image_bytes_by_filename,
            max_workers=cpus_limit,
        )

        # Build (Detection, crop, filename) records for this item
        crop_image_records, failed_crops = build_detection_crops(
            volume_barcode=volume_barcode,
            detections=item_detections,
            loaded_images=loaded_images,
            with_filename=True,
        )
        n_crops = len(crop_image_records)

        end = get_time()
        time_preproc += end - start

        if n_crops == 0:
            logger.info(f"{volume_barcode}: All crops failed, skipping.")
            total_failed_crops += failed_crops
            continue

        crop_batches = list(chunked(crop_image_records, CLASSIFICATION_MAX_BATCH))

        for batch in crop_batches:
            detections_batch, images_batch, filenames_batch = zip(*batch)

            # Inference!
            start_inf = get_time()
            results = model(
                list(images_batch),
                imgsz=CLASSIFICATION_MODEL_IMGSZ,
                device=device,
                verbose=False,
            )
            end_inf = get_time()
            time_infer += end_inf - start_inf

            for idx, result in enumerate(results):
                probs = getattr(result, "probs", None)
                if probs is None:
                    logger.warning(
                        f"No probs for crop of {filenames_batch[idx]} in {volume_barcode}: skipping."
                    )
                    continue
                pred_idx = probs.top1
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

                # Apply confidence threshold - if below threshold, set to class 0 <- "Other" class
                if pred_conf < CLASSIFICATION_MODEL_CONF:
                    pred_class = "0"
                    pred_idx = 0

                now = get_time()

                classified_entries.append(
                    Classification(
                        detection_id=detections_batch[idx].id_detection,
                        pred_idx=int(pred_idx),
                        pred_class=pred_class,
                        probs=probs.data.cpu().tolist(),
                        pipeline_batch_item=id_pipeline_batch_item,
                        scan_filename=filenames_batch[idx],
                        pred_conf=pred_conf,
                        created=now,
                    )
                )

        # Clear GPU cache and run GC
        start_gpu = get_time()
        with torch.cuda.device(0):
            torch.cuda.empty_cache()
        gc.collect()
        end_gpu = get_time()
        time_clear_gpu += end_gpu - start_gpu

        # Add to totals
        total_time_preproc += time_preproc
        total_time_infer += time_infer
        total_time_clear_gpu += time_clear_gpu
        total_n_crops += n_crops
        total_failed_crops += failed_crops

        # Add classifications to batch
        all_classified_entries.extend(classified_entries)

        # Stats for this volume
        logger.info(
            f"{volume_barcode} | Crops: {n_crops} - "
            f"Failed crops: {failed_crops} - "
            f"Device: {cuda_device} - "
            f"Preprocessing: {time_preproc} - "
            f"Inference: {time_infer} - "
            f"Clear GPU: {time_clear_gpu} - "
            f"Total: {time_preproc + time_infer + time_clear_gpu}"
        )

    #
    # Save all classifications to the database (batch operation)
    #
    # Delete previous Classification entries for all items in this batch
    Classification.delete().where(Classification.pipeline_batch_item.in_(item_ids)).execute()

    # Batch write all classifications
    process_db_write_batch(
        model=Classification,
        entries_to_create=all_classified_entries,
    )

    return True
