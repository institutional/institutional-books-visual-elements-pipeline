import gc
import gzip
import io
import json
import multiprocessing as mp
import os
import random
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import cv2
import numpy as np
from loguru import logger

from const import (
    CUDA_GPUS,
    DEFAULT_DB_BATCH_SIZE,
    HF_EXPORT_NETWORK_BASE_DELAY,
    HF_EXPORT_NETWORK_MAX_RETRIES,
    ORIENTATION_CONFIDENCE_THRESHOLD,
    ORIENTATION_INFERENCE_BATCH_SIZE,
    ORIENTATION_MODEL_FILEPATH,
    ORIENTATION_MODEL_REPO,
    ORIENTATION_PROCESSES_PER_GPU,
    OUTPUT_STORAGE_BUCKET_NAME,
)
from models import Detection
from utils import get_db, process_db_write_batch
from utils.get_s3_client import get_s3_client

NUM_CLASSES = 4
CLASS_MAP = {
    0: "upright",
    1: "rotate_90_clockwise",
    2: "rotate_180",
    3: "rotate_90_counterclockwise",
}
CLASS_INDEX_ORDER = [CLASS_MAP[i] for i in range(NUM_CLASSES)]

CV2_ROTATION_MAP = {
    "rotate_90_clockwise": cv2.ROTATE_90_CLOCKWISE,
    "rotate_180": cv2.ROTATE_180,
    "rotate_90_counterclockwise": cv2.ROTATE_90_COUNTERCLOCKWISE,
}

VACUUM_EVERY_N_CHUNKS = 50


def _retry(fn, *args, max_retries=HF_EXPORT_NETWORK_MAX_RETRIES, label="network call", **kwargs):
    import requests.exceptions

    retryable = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
        ConnectionError,
        TimeoutError,
        OSError,
    )
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            is_retryable = isinstance(e, retryable)
            if not is_retryable:
                cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
                is_retryable = cause and isinstance(cause, retryable)
            if is_retryable and attempt < max_retries:
                delay = HF_EXPORT_NETWORK_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2)
                logger.warning(f"{label} failed ({type(e).__name__}), retry {attempt}/{max_retries} in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise


def ensure_orientation_columns():
    db = get_db()
    migrations = [
        "ALTER TABLE detection ADD COLUMN IF NOT EXISTS orientation_correction_gen VARCHAR",
        "ALTER TABLE detection ADD COLUMN IF NOT EXISTS orientation_correction_confidence_gen DOUBLE PRECISION",
        "ALTER TABLE detection ADD COLUMN IF NOT EXISTS orientation_correction_probs_gen JSONB",
        "ALTER TABLE detection ADD COLUMN IF NOT EXISTS orientation_hf_corrected_gen BOOLEAN DEFAULT FALSE",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_orientation_gen ON detection (orientation_correction_gen) WHERE orientation_correction_gen IS NULL",
    ]
    for sql in migrations:
        db.execute_sql(sql)


def fetch_detection_groups(force: bool, limit: int | None) -> list[tuple[int, str, list[int]]]:
    """
    Query filtered_dataset for detections grouped by item.
    Returns list of (pipeline_batch_item_id, barcode, [id_detection, ...]).
    """
    db = get_db()
    conn = db.connection()
    try:
        conn.cursor().execute("SELECT 1")
        conn.rollback()
    except Exception:
        db.close()
        db.connect()
        conn = db.connection()

    cur = conn.cursor()

    target_classes = ("Image or Illustration", "Chart or Graph")

    if force:
        query = """
            SELECT id_detection, pipeline_batch_item_id, barcode
            FROM filtered_dataset
            WHERE pred_class IN %s
            ORDER BY pipeline_batch_item_id
        """
        params = (target_classes,)
    else:
        query = """
            SELECT fd.id_detection, fd.pipeline_batch_item_id, fd.barcode
            FROM filtered_dataset fd
            JOIN detection d ON d.id_detection = fd.id_detection
            WHERE d.orientation_correction_gen IS NULL
              AND fd.pred_class IN %s
            ORDER BY fd.pipeline_batch_item_id
        """
        params = (target_classes,)

    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    conn.rollback()

    groups: dict[int, tuple[str, list[int]]] = {}
    for id_detection, item_id, barcode in rows:
        if item_id not in groups:
            groups[item_id] = (barcode, [])
        groups[item_id][1].append(id_detection)

    return [(item_id, barcode, det_ids) for item_id, (barcode, det_ids) in groups.items()]


def build_orientation_model(device: str):
    import torch
    import torch.nn as nn
    import torchvision.models as models

    model = models.efficientnet_v2_m(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, NUM_CLASSES),
    )
    return model


def format_orientation_probs(probs_list: list[float]) -> list[dict]:
    prob_dicts = [
        {"label": CLASS_INDEX_ORDER[i], "prob": float(probs_list[i])}
        for i in range(NUM_CLASSES)
    ]
    prob_dicts.sort(key=lambda x: x["prob"], reverse=True)
    return prob_dicts


def download_and_extract_crops(item_id: int, barcode: str, det_ids: set[int]) -> dict[int, np.ndarray]:
    """Download crops tar.gz from R2 and extract PNGs for the given detection IDs."""
    s3 = get_s3_client("OUTPUT")
    s3_key = f"crops/{item_id}/{barcode}.tar.gz"

    def _s3_download():
        resp = s3.get_object(Bucket=OUTPUT_STORAGE_BUCKET_NAME, Key=s3_key)
        return resp["Body"].read()

    try:
        gz_bytes = _retry(_s3_download, label=f"OUTPUT download {barcode}")
    except Exception as e:
        logger.warning(f"Failed to download crops for item {item_id} ({barcode}): {e}")
        return {}

    crops: dict[int, np.ndarray] = {}
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes), mode="rb") as gz:
            tar_bytes = gz.read()
        del gz_bytes
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".png"):
                    continue
                stem = Path(member.name).stem
                parts = stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    det_id = int(parts[1])
                    if det_id not in det_ids:
                        continue
                    with tar.extractfile(member) as fh:
                        png_bytes = fh.read()
                    arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if arr is not None:
                        crops[det_id] = arr
        del tar_bytes
    except Exception as e:
        logger.warning(f"Failed to extract crops for item {item_id} ({barcode}): {e}")

    return crops


def orientation_worker(
    item_groups: list[tuple[int, str, list[int]]],
    model_filepath: str,
    cuda_device: str,
    inference_batch_size: int,
    db_batch_size: int,
) -> dict:
    """
    Worker process: loads orientation model, downloads images, runs inference, updates DB.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device.replace("cuda:", "")

    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    device = "cuda:0"
    get_db()

    model = build_orientation_model(device)
    state_dict = torch.load(model_filepath, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    del state_dict

    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    stats = {"processed": 0, "corrected": 0, "errors": 0, "items": 0}
    total_items_for_worker = len(item_groups)
    pending_updates: list[Detection] = []

    for item_id, barcode, det_ids in item_groups:
        det_id_set = set(det_ids)
        crops = download_and_extract_crops(item_id, barcode, det_id_set)

        if not crops:
            stats["errors"] += len(det_ids)
            stats["items"] += 1
            continue

        # Prepare batches for inference
        ordered_det_ids = [d for d in det_ids if d in crops]
        if not ordered_det_ids:
            stats["items"] += 1
            continue

        # Process in inference batches
        for batch_start in range(0, len(ordered_det_ids), inference_batch_size):
            batch_det_ids = ordered_det_ids[batch_start:batch_start + inference_batch_size]
            tensors = []

            for det_id in batch_det_ids:
                arr = crops[det_id]
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensor = val_transform(pil_img)
                tensors.append(tensor)

            batch_tensor = torch.stack(tensors).to(device)

            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1)

            probs_cpu = probs.cpu()
            del batch_tensor, logits, probs

            for i, det_id in enumerate(batch_det_ids):
                prob_values = probs_cpu[i].tolist()
                confidence = float(max(prob_values))
                pred_idx = int(probs_cpu[i].argmax())
                pred_label = CLASS_MAP[pred_idx]

                # Threshold: override to "upright" if below confidence threshold
                if confidence < ORIENTATION_CONFIDENCE_THRESHOLD:
                    correction_label = "upright"
                else:
                    correction_label = pred_label

                det = Detection()
                det.id_detection = det_id
                det.orientation_correction_gen = correction_label
                det.orientation_correction_confidence_gen = confidence
                det.orientation_correction_probs_gen = json.dumps(format_orientation_probs(prob_values))

                pending_updates.append(det)
                stats["processed"] += 1
                if correction_label != "upright":
                    stats["corrected"] += 1

            del probs_cpu

            # Flush to DB periodically
            if len(pending_updates) >= db_batch_size:
                process_db_write_batch(
                    Detection,
                    entries_to_update=pending_updates,
                    fields_to_update=[
                        Detection.orientation_correction_gen,
                        Detection.orientation_correction_confidence_gen,
                        Detection.orientation_correction_probs_gen,
                    ],
                )
                pending_updates = []

        del crops
        torch.cuda.empty_cache()
        gc.collect()
        stats["items"] += 1

        if stats["items"] % 100 == 0:
            logger.info(
                f"  [Worker {cuda_device}] {stats['items']}/{total_items_for_worker} items | "
                f"{stats['processed']} detections | {stats['corrected']} corrections"
            )

    # Final flush
    if pending_updates:
        process_db_write_batch(
            Detection,
            entries_to_update=pending_updates,
            fields_to_update=[
                Detection.orientation_correction_gen,
                Detection.orientation_correction_confidence_gen,
                Detection.orientation_correction_probs_gen,
            ],
        )

    return stats





def _run_vacuum():
    try:
        import psycopg2

        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            sslmode=os.getenv("POSTGRES_SSLMODE", "require"),
        )
        conn.autocommit = True
        logger.info("  Running VACUUM on detection table...")
        conn.cursor().execute("VACUUM detection")
        conn.close()
        logger.info("  VACUUM complete.")
    except Exception as e:
        logger.warning(f"  VACUUM failed (non-fatal): {e}")


@click.command("orientation-correction")
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_DB_BATCH_SIZE,
    help="Number of detections per DB write batch",
)
@click.option(
    "--inference-batch-size",
    type=int,
    default=ORIENTATION_INFERENCE_BATCH_SIZE,
    help="GPU inference batch size",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-process detections that already have orientation predictions",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit total number of detections to process (for testing)",
)
@click.option(
    "--cuda-gpus",
    type=click.Choice(CUDA_GPUS),
    multiple=True,
    default=CUDA_GPUS if CUDA_GPUS else ["cuda:0"],
    help="Which CUDA device(s) to use",
)
@click.option(
    "--processes-per-gpu",
    type=int,
    default=ORIENTATION_PROCESSES_PER_GPU,
    help="Number of model replicas per GPU",
)
def orientation_correction(batch_size, inference_batch_size, force, limit, cuda_gpus, processes_per_gpu):
    """
    Run orientation correction on filtered detections using the EfficientNet-V2-M model.

    Downloads crops from R2, runs batched GPU inference, and stores predictions in
    three columns on the detection table:
    - orientation_correction_gen: predicted correction (or "upright" if below threshold)
    - orientation_correction_confidence_gen: max softmax probability
    - orientation_correction_probs_gen: full 4-class probability distribution (JSONB)

    This should be run after the pipeline completes and before `export to-hf`, which
    reads orientation results from the DB and applies the rotation during image upload.
    """
    from huggingface_hub import hf_hub_download

    logger.info("Starting orientation correction...")

    ensure_orientation_columns()

    # Download model weights
    logger.info(f"  Pulling orientation model from {ORIENTATION_MODEL_REPO}...")
    model_filepath = hf_hub_download(
        repo_id=ORIENTATION_MODEL_REPO,
        filename=ORIENTATION_MODEL_FILEPATH,
    )
    logger.info(f"  Model cached at: {model_filepath}")

    # Fetch detections to process
    item_groups = fetch_detection_groups(force=force, limit=limit)
    total_detections = sum(len(det_ids) for _, _, det_ids in item_groups)
    total_items = len(item_groups)

    logger.info(f"  Detections to process: {total_detections} across {total_items} items")

    if total_detections == 0:
        logger.info("Nothing to process.")
        return

    # Set up multi-process inference
    cuda_gpus_total = len(cuda_gpus)
    processes_total = cuda_gpus_total * processes_per_gpu
    processes_total = min(processes_total, total_items)

    logger.info(f"  GPUs: {cuda_gpus_total}, processes per GPU: {processes_per_gpu}, total workers: {processes_total}")

    # Distribute items round-robin across workers
    worker_assignments: list[list[tuple[int, str, list[int]]]] = [[] for _ in range(processes_total)]
    for i, group in enumerate(item_groups):
        worker_assignments[i % processes_total].append(group)

    # Launch workers
    mp_ctx = mp.get_context("spawn")
    total_stats = {"processed": 0, "corrected": 0, "errors": 0, "items": 0}
    chunks_completed = 0

    with ProcessPoolExecutor(
        max_workers=processes_total,
        mp_context=mp_ctx,
    ) as executor:
        futures = {}
        for i, assignment in enumerate(worker_assignments):
            if not assignment:
                continue
            cuda_gpu_i = i % cuda_gpus_total
            future = executor.submit(
                orientation_worker,
                item_groups=assignment,
                model_filepath=model_filepath,
                cuda_device=cuda_gpus[cuda_gpu_i],
                inference_batch_size=inference_batch_size,
                db_batch_size=batch_size,
            )
            futures[future] = i

        for future in as_completed(futures):
            worker_i = futures[future]
            try:
                stats = future.result()
                for key in total_stats:
                    total_stats[key] += stats[key]
            except Exception:
                logger.error(f"Worker {worker_i} failed:\n{traceback.format_exc()}")
                executor.shutdown(wait=False, cancel_futures=True)
                click.get_current_context().exit(1)

            chunks_completed += 1
            logger.info(
                f"  Worker {worker_i} done. Progress: {total_stats['processed']}/{total_detections} "
                f"({total_stats['corrected']} corrections)"
            )

            if chunks_completed % VACUUM_EVERY_N_CHUNKS == 0:
                _run_vacuum()

    _run_vacuum()
    logger.success(
        f"Orientation correction complete. "
        f"Processed: {total_stats['processed']}, "
        f"Corrections: {total_stats['corrected']}, "
        f"Errors: {total_stats['errors']}"
    )

