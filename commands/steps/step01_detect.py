import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
import gc
import multiprocessing as mp

import click
from loguru import logger
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
from more_itertools import chunked

from utils import get_db, process_db_write_batch
from models import PipelineBatchItem, Detection, Classification

from const import (
    DETECTION_MODEL_REPO,
    DETECTION_MODEL_FILEPATH,
    DETECTION_MODEL_IMGSZ,
    DETECTION_MODEL_CONF,
    DETECTION_MODEL_IOU,
    DETECTION_MODEL_PROCESSES_PER_GPU,
    CUDA_GPUS,
    CPUS_LIMIT,
)


@click.command("step01-detect")
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
def step01_detect(id_pipeline_batch: int, cpus_limit: int, cuda_gpus: list[str]):
    """
    Runs the visual elements detection model against a batch of volumes.
    """
    model_filepath: Path | None = None

    cuda_gpus_total = len(cuda_gpus)
    processes_per_gpu = DETECTION_MODEL_PROCESSES_PER_GPU
    processes_total = cuda_gpus_total * processes_per_gpu

    logger.info(f"Number of GPUs: {cuda_gpus_total}")
    logger.info(f"Number of processes: {processes_per_gpu}")
    logger.info(f"Total processes: {processes_total}")
    logger.info(f"Total cpus: {cpus_limit}")

    # 1 batch of items per GPU process (processes_total)
    item_id_batches: list[list[int]] = [[] for i in range(0, processes_total)]

    # Set CPU concurrency for each process. Reduce if we're running more than 1 process per GPU.
    per_task_cpus_limit = int(round(cpus_limit / cuda_gpus_total))

    if processes_per_gpu > 1:
        per_task_cpus_limit = max(2, per_task_cpus_limit // 2)

    #
    # Make sure model is present on disk
    #
    logger.info(f"Pulling {DETECTION_MODEL_REPO} from HuggingFace or cache ...")

    model_filepath = Path(
        hf_hub_download(
            repo_id=DETECTION_MODEL_REPO,
            filename=DETECTION_MODEL_FILEPATH,
        ),
    )

    #
    # Make sure at least one GPU has been assigned
    #
    try:
        assert cuda_gpus

        for device in cuda_gpus:
            assert device.startswith("cuda:")
    except AssertionError:
        logger.error("No CUDA GPU assigned to the current task when at least 1 is needed. Exiting.")
        click.get_current_context().exit(0)

    #
    # Populate per-process batches
    #
    for i, item in enumerate(
        PipelineBatchItem.select(PipelineBatchItem)
        .where(PipelineBatchItem.pipeline_batch == id_pipeline_batch)
        .order_by(PipelineBatchItem.id_pipeline_batch_item)
        .iterator()
    ):
        process_i = i % processes_total
        item_id_batches[process_i].append(item.id_pipeline_batch_item)

    #
    # Start parallel processing pool
    #
    # Use "spawn" to avoid CUDA / fork issues; this removes the need for per-process delays.
    mp_ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=processes_total,
        initializer=get_db,
        mp_context=mp_ctx,  # <-- key change
    ) as executor:
        futures = {}

        # Schedule tasks, assign each batch for a CUDA device
        for i, item_ids in enumerate(item_id_batches):
            cuda_gpus_i = i % cuda_gpus_total

            future = executor.submit(
                process_batch_of_items,
                item_ids=item_ids,
                model_filepath=model_filepath,
                cuda_device=cuda_gpus[cuda_gpus_i],
                cpus_limit=per_task_cpus_limit,
            )

            futures[future] = cuda_gpus[cuda_gpus_i]

        # Grab results
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


def process_batch_of_items(
    item_ids: list[int],
    model_filepath: Path,
    cuda_device: str,
    cpus_limit: int,
):
    """
    Processes a batch of PipelineBatchItem instances (1 item = 1 volume).
    """
    import torch
    from ultralytics import YOLO, settings
    from datetime import datetime, timedelta

    settings.update({"sync": False})

    # Estimate: image size in GPU once processed by YOLO
    approx_image_nbytes = DETECTION_MODEL_IMGSZ * DETECTION_MODEL_IMGSZ * 3 * 4

    # Total VRAM of CUDA device
    vram_total_nbytes = torch.cuda.get_device_properties(cuda_device).total_memory

    # Estimated optimal `batch` value for `model.predict()`
    yolo_batch_size = max(4, int(vram_total_nbytes / approx_image_nbytes / 100 / 2))

    #
    # Load model and assign it to device
    #
    model = YOLO(model_filepath, verbose=False)
    model.to(cuda_device)
    classes = list(model.names.values())

    try:
        assert len(classes) == 1
    except Exception as err:
        raise Exception("Object detection model must be single-class.") from err

    # Collect all detections from all items
    all_detections: list[Detection] = []

    # Track timing for all items
    total_pre_processing_time = timedelta(seconds=0)
    total_inference_time = timedelta(seconds=0)
    total_clear_gpu_time = timedelta(seconds=0)

    #
    # Process 1 volume at a time
    #
    for id_pipeline_batch_item in item_ids:
        item = PipelineBatchItem.get(id_pipeline_batch_item=id_pipeline_batch_item)
        volume_barcode = item.ib_volume.barcode
        scans_total = 0

        detections: list[Detection] = []

        pre_processing_time = timedelta(seconds=0)  # Cumulative
        inference_time = timedelta(seconds=0)  # Cumulative
        clear_gpu_time = timedelta(seconds=0)  # Cumulative

        #
        # Process `yolo_batch_size` images from current volume at a time
        #
        for encoded_images_batch in chunked(item.data.images.items(), yolo_batch_size):

            #
            # Pre-process batch of images
            #
            start = datetime.now()

            images_filenames, images_ndarrays = preprocess_images_batch(
                volume_barcode=volume_barcode,
                encoded_images_batch=encoded_images_batch,
                cpus_limit=cpus_limit,
            )

            for filename in images_filenames:  # Remove from RAM encoded scans that were processed
                item.data.images[filename] = None

            scans_total += len(images_filenames)

            end = datetime.now()
            pre_processing_time += end - start

            #
            # Run inference on batch
            #
            start = datetime.now()

            try:
                detections += run_detection_on_images_batch(
                    model=model,
                    id_pipeline_batch_item=id_pipeline_batch_item,
                    images_filenames=images_filenames,
                    images_ndarrays=images_ndarrays,
                )
            except Exception as err:
                logger.debug(traceback.print_exc())
                logger.warning(
                    f"Could not process part of {volume_barcode}, "
                    + f"skipping batch of {len(images_filenames)} scans."
                )
            finally:
                del images_ndarrays
                del images_filenames

            end = datetime.now()
            inference_time += end - start

        #
        # Clear GPU cache and run GC
        #
        start = datetime.now()

        with torch.cuda.device(int(cuda_device.replace("cuda:", ""))):
            torch.cuda.empty_cache()

        gc.collect()

        end = datetime.now()
        clear_gpu_time = end - start

        # Add to totals
        total_pre_processing_time += pre_processing_time
        total_inference_time += inference_time
        total_clear_gpu_time += clear_gpu_time

        # Add detections to batch
        all_detections.extend(detections)

        #
        # Print stats for volume
        #
        logger.info(
            f"{volume_barcode} | "
            + f"Scans: {scans_total} - "
            + f"Device: {cuda_device} - "
            + f"Preprocessing: {pre_processing_time} - "
            + f"Inference: {inference_time} - "
            + f"Clear GPU cache: {clear_gpu_time} - "
            + f"Total: {pre_processing_time + inference_time + clear_gpu_time}"
        )

    #
    # Record all detections in db. Replace existing records for all pipeline batch items
    #
    # Delete classifications first (batch delete for all items)
    Classification.delete().where(Classification.pipeline_batch_item.in_(item_ids)).execute()

    # Delete existing detections (batch delete for all items)
    Detection.delete().where(
        Detection.pipeline_batch_item.in_(item_ids),
    ).execute()

    # Batch write all detections
    process_db_write_batch(
        model=Detection,
        entries_to_create=all_detections,
    )

    return True


def preprocess_images_batch(
    volume_barcode: str,
    encoded_images_batch: list[tuple[str, bytes]],
    cpus_limit: int = 1,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Pre-processes a batch of image data from `PipelineBatchItem.data.images`

    Returns a tuple containing:
    - A list of filenames
    - A list of np.ndarrays

    NOTE: This is where our main bottleneck is!
    """
    images_ndarrays: list[np.ndarray] = []
    images_filenames: list[str] = []

    with ThreadPoolExecutor(max_workers=cpus_limit) as executor:
        futures = {}

        for filename, image_bytes in encoded_images_batch:
            future = executor.submit(decode_image_bytes, image_bytes)
            futures[future] = filename

        done, _ = wait(futures)

        for future in done:
            filename = futures[future]
            try:
                image: np.ndarray = future.result()
                images_ndarrays.append(image)
                images_filenames.append(filename)
            except Exception:
                logger.debug(traceback.print_exc())
                logger.warning(f"Could not decode {volume_barcode}.{filename}. Skipping.")

    return (images_filenames, images_ndarrays)


def decode_image_bytes(image_bytes) -> np.ndarray:
    """
    Decodes image bytes and returns an ndarray
    """
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR_RGB)


def run_detection_on_images_batch(
    model: object,
    id_pipeline_batch_item: int,
    images_filenames: list[str],
    images_ndarrays: list[np.ndarray],
) -> list[Detection]:
    """
    Performs inference on a batch of images using the detection model.
    Model must have been pre-loaded and is passed as reference.
    """
    detections: list[Detection] = []

    results = model.predict(
        images_ndarrays,
        conf=DETECTION_MODEL_CONF,
        iou=DETECTION_MODEL_IOU,
        imgsz=DETECTION_MODEL_IMGSZ,
        stream=True,
        batch=len(images_filenames),
        verbose=False,
    )

    for i, result in enumerate(results):
        for bbox in result.boxes:

            detection = Detection(
                pipeline_batch_item=id_pipeline_batch_item,
                scan_filename=images_filenames[i],
                bbox_xyxy=bbox.xyxy.tolist()[0],
                bbox_xywh=bbox.xywh.tolist()[0],
                bbox_conf=float(bbox.conf),
            )

            detections.append(detection)

    del results

    return detections
