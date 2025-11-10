import traceback
import time
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, wait

import numpy as np
import click
from loguru import logger
from huggingface_hub import hf_hub_download

from utils import get_db, process_db_write_batch
from models import (
    PipelineBatch,
    PipelineBatchItem,
    IBVolume,
    Detection,
    Classification,
)

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
    Runs the visual elements classification model over detection crops for a batch of volumes.
    - Assumes detection (step01) already complete.
    - Does NOT write crop images to disk.
    - Writes back to Classification model in the DB.
    """
    model_filepath: Path | None = None

    cuda_gpus_total = len(cuda_gpus)
    # TODO: change processes per gpu? - add to .env as variable
    processes_total = cuda_gpus_total

    # 1 batch of items per GPU process
    item_id_batches: list[list[int]] = [[] for i in range(0, processes_total)]
    per_task_cpus_limit = int(round(cpus_limit / cuda_gpus_total))

    # Download model from HF if not cached
    logger.info(f"Pulling {CLASSIFICATION_MODEL_REPO} from HuggingFace or cache ...")
    model_filepath = Path(
        hf_hub_download(
            repo_id=CLASSIFICATION_MODEL_REPO,
            filename=CLASSIFICATION_MODEL_FILEPATH,
        ),
    )

    # Populate per-process batches
    items = (
        PipelineBatchItem.select(PipelineBatchItem)
        .where(PipelineBatchItem.pipeline_batch == id_pipeline_batch)
        .order_by(PipelineBatchItem.id_pipeline_batch_item)
        .iterator()
    )
    for i, item in enumerate(items):
        process_i = i % processes_total
        item_id_batches[process_i].append(item.id_pipeline_batch_item)

    # Start parallel executor pool
    with ProcessPoolExecutor(max_workers=processes_total, initializer=get_db) as executor:
        futures = {}
        for i, item_ids in enumerate(item_id_batches):
            cuda_gpus_i = i % cuda_gpus_total
            future = executor.submit(
                process_classification_batch_of_items,
                item_ids=item_ids,
                model_filepath=model_filepath,
                cuda_device=cuda_gpus[cuda_gpus_i],
                cpus_limit=per_task_cpus_limit,
            )
            futures[future] = cuda_gpus[cuda_gpus_i]

            # HACK: Helps prevent process collisions
            time.sleep(CLASSIFICATION_MODEL_PROCESSES_FORK_DELAY)

        for future in as_completed(futures):
            cuda_gpu = futures[future]
            try:
                future.result()
            except Exception as err:
                logger.debug(traceback.print_exc())
                logger.error(
                    f"A blocking error occured while processing classification batch on {cuda_gpu}. Exiting."
                )
                executor.shutdown(wait=False, cancel_futures=True)
                click.get_current_context().exit(1)
            except KeyboardInterrupt as err:
                logger.warning("Received interrupt signal")
                raise err


def process_classification_batch_of_items(
    item_ids: list[int],
    model_filepath: Path,
    cuda_device: str,
    cpus_limit: int,
):
    """
    Processes a batch of PipelineBatchItem instances for classification.
    For each detection, crops from the original image, classifies the crop, and writes outputs.
    """
    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False})

    # Load model and assign to device
    model = YOLO(model_filepath, task="classify", verbose=False)
    model.to(cuda_device)
    class_map = model.names if hasattr(model, "names") else None
    if isinstance(class_map, dict):
        class_map = [class_map[i] for i in sorted(class_map.keys())]

    for id_pipeline_batch_item in item_ids:
        item = PipelineBatchItem.get(id_pipeline_batch_item=id_pipeline_batch_item)
        volume_barcode = item.ib_volume.barcode

        # --- Load all Detection objects for this item ---
        detections = (
            Detection.select()
            .where(Detection.pipeline_batch_item == id_pipeline_batch_item)
            .execute()
        )
        logger.info(f"{volume_barcode} | {len(detections)} detections to classify.")

        # Preload all original scans available in RAM
        scan_images = {}
        for fname, b in item.data.images.items():
            if b is not None:
                scan_images[fname] = decode_image_bytes(b)

        # Classify in batches - TODO - chunk this by batch size?
        classified: list[Classification] = []
        classification_times = []
        preproc_times = []
        batch_size = 16  # TODO - pass as argument

        from more_itertools import chunked
        from datetime import datetime

        # Chunk detections for batching
        for det_batch in chunked(detections, batch_size):
            crops = []
            meta = []
            t0_proc = datetime.now()
            for d in det_batch:
                if d.scan_filename not in scan_images:
                    logger.warning(
                        f"scan image for {d.scan_filename} not found in RAM. Skipping detection id={d.id_detection}."
                    )
                    continue
                try:
                    crop = d.crop(scan_images[d.scan_filename])
                except Exception as e:
                    logger.warning(f"Crop failed for detection id={d.id_detection}: {e}; skipping.")
                    continue
                crops.append(crop)
                meta.append(d)
            t1_proc = datetime.now()
            preproc_times.append((t1_proc - t0_proc).total_seconds())

            t0_inf = datetime.now()
            yolo_preds = model(
                crops,
                conf=CLASSIFICATION_MODEL_CONF,
                imgsz=CLASSIFICATION_MODEL_IMGSZ,
                device=cuda_device,
                batch=len(crops),
                verbose=False,
            )
            t1_inf = datetime.now()
            classification_times.append((t1_inf - t0_inf).total_seconds())

            for i, result in enumerate(yolo_preds):
                d = meta[i]
                probs = result.probs
                pred_idx = int(probs.top1)
                pred_conf = (
                    float(probs[pred_idx]) if hasattr(probs, "__getitem__") else float(probs.max())
                )

                pred_class = (
                    class_map[pred_idx]
                    if (class_map and pred_idx < len(class_map))
                    else str(pred_idx)
                )

                classified.append(
                    Classification(
                        pipeline_batch_item=id_pipeline_batch_item,
                        detection=d,  # ForeignKey to detection
                        pred_idx=pred_idx,
                        pred_class=pred_class,
                        pred_conf=pred_conf,
                        scan_filename=d.scan_filename,
                        bbox_xyxy=d.bbox_xyxy,
                    )
                )

            del crops
            del meta
            del yolo_preds
            torch.cuda.empty_cache()

        # --- Replace existing for this batch item ---
        start_db = datetime.now()
        Classification.delete().where(
            Classification.pipeline_batch_item == id_pipeline_batch_item,
        ).execute()
        process_db_write_batch(
            model=Classification,
            entries_to_create=classified,
        )
        end_db = datetime.now()
        logger.info(
            f"{volume_barcode} | Classified {len(classified)} crops"
            + f" | Preproc {sum(preproc_times):.2f}s"
            + f" | Inference {sum(classification_times):.2f}s"
            + f" | DB write {(end_db - start_db).total_seconds():.2f}s"
        )
        gc.collect()

    return True


def decode_image_bytes(image_bytes) -> np.ndarray:
    """
    Decodes image bytes and returns an ndarray.
    """
    import numpy as np
    import cv2

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR_RGB)
