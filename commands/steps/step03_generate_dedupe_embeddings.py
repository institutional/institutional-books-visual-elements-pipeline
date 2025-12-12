import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
import gc
import time

import click
from loguru import logger
import cv2
import numpy as np

from more_itertools import chunked
import imagehash

from utils import get_db, process_db_write_batch, get_s3_client
from models import PipelineBatchItem, Detection, Embedding, ImageHash

from const import (
    DEDUPE_EMBEDDING_MODEL_FILEPATH,
    DEDUPE_EMBEDDING_MODEL_STORAGE_PATH,
    DEDUPE_EMBEDDING_MODEL_PROCESSES_FORK_DELAY,
    DEDUPE_EMBEDDING_NUM_PROCESSES_PER_GPU,
    HASH_SIZE,
    CUDA_GPUS,
    CPUS_LIMIT,
    BUCKET_NAME,
    DEDUPE_EMBEDDING_BATCH_SIZE,
    DEDUPE_EMBEDDING_MODEL_NAME,
)


@click.command("step03-embed")
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
def step03_generate_dedupe_embeddings(
    id_pipeline_batch: int, cpus_limit: int, cuda_gpus: list[str]
):
    """
    Computes embeddings (and hashes) for all crops in all volumes with detections in this pipeline batch,
    and saves them to the database, per GPU.

    NOTE:
    - This command is intended to be run by the orchestrator. See `orchestration/execute.py` for details.
    - This is a batch-level step, which expects to process a batch for which images and text are already cached on disk.
    """
    model_filepath: Path | None = None
    cuda_gpus_total = len(cuda_gpus)
    processes_total = cuda_gpus_total * DEDUPE_EMBEDDING_NUM_PROCESSES_PER_GPU

    item_id_batches: list[list[int]] = [[] for _ in range(processes_total)]

    per_task_cpus_limit = int(round(cpus_limit / cuda_gpus_total))
    if processes_total > 1:
        per_task_cpus_limit = max(2, per_task_cpus_limit // 2)

    # Download model BEFORE spawning processes to avoid race conditions
    local_model_path = DEDUPE_EMBEDDING_MODEL_FILEPATH
    if not local_model_path.exists():
        logger.info(f"Downloading TorchScript model from S3 Storage...")
        download_model()
        logger.info(f"✓ Model downloaded and verified successfully")
    else:
        logger.info(f"Model file found at {local_model_path}, skipping download.")
    model_filepath = local_model_path

    # Only process volumes with detections
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

    with ProcessPoolExecutor(max_workers=processes_total, initializer=get_db) as executor:
        futures = {}
        for i, item_ids in enumerate(item_id_batches):
            cuda_gpus_i = i % cuda_gpus_total
            future = executor.submit(
                embed_batch_of_items,
                item_ids=item_ids,
                model_filepath=model_filepath,
                cuda_device=cuda_gpus[cuda_gpus_i],
                cpus_limit=per_task_cpus_limit,
            )
            futures[future] = cuda_gpus[cuda_gpus_i]
            time.sleep(
                DEDUPE_EMBEDDING_MODEL_PROCESSES_FORK_DELAY
            )  # mimic detection/classification fork delay
        for future in as_completed(futures):
            cuda_gpu: str = futures[future]
            try:
                future.result()
            except Exception as err:
                logger.debug(traceback.print_exc())
                logger.error(
                    f"A blocking error occured while embedding batch on {cuda_gpu}. Exiting."
                )
                executor.shutdown(wait=False, cancel_futures=True)
                click.get_current_context().exit(1)
            except KeyboardInterrupt as err:
                logger.warning("Received interrupt signal")
                raise err


def embed_batch_of_items(
    item_ids: list[int],
    model_filepath: Path,
    cuda_device: str,
    cpus_limit: int,
):
    import os

    # set CUDA_VISIBLE_DEVICES before importing torch/ultralytics - avoids defaulting to cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device.replace("cuda:", "")
    import torch
    from PIL import Image

    device = "cuda:0"  # since the corresponding device is the only one visible, it becomes cuda:0

    # Load TorchScript model only ONCE per process
    model = torch.jit.load(str(model_filepath), map_location=device)
    model.eval()

    def preprocess_for_model(crop: np.ndarray):
        img = Image.fromarray(crop.astype(np.uint8))
        img = img.convert("RGB").resize((224, 224))
        img = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(img)  # [C, H, W]

    from datetime import datetime, timezone

    # Collect all embeddings and hashes from all items
    all_embedding_entries = []
    all_imagehash_entries = []

    # Track statistics
    total_n_embeds = 0
    total_failed_embeds = 0

    for id_pipeline_batch_item in item_ids:
        item = PipelineBatchItem.get(id_pipeline_batch_item=id_pipeline_batch_item)
        volume_barcode = item.ib_volume.barcode

        item_detections = (
            Detection.select()
            .where(Detection.pipeline_batch_item == id_pipeline_batch_item)
            .order_by(Detection.id_detection)
        )

        if item_detections.count() == 0:
            logger.info(f"{volume_barcode}: No detections - skipping embedding for this item.")
            continue

        image_bytes_by_filename = dict(list(item.data.images.items()))
        image_bytes_by_filename = {str(k): v for k, v in image_bytes_by_filename.items()}

        # 1. Decode all scans needed for this item in parallel
        with ThreadPoolExecutor(max_workers=cpus_limit) as decode_executor:
            futures = {}
            used_filenames = set(str(det.scan_filename) for det in item_detections)
            loaded_images: dict[str, np.ndarray] = {}
            for fn in used_filenames:
                if fn not in image_bytes_by_filename:
                    logger.warning(
                        f"Missing image bytes for scan {volume_barcode}.{fn} - skipping this scan in embedding"
                    )
                    continue
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

        embedding_entries = []
        imagehash_entries = []
        n_embeds, failed_embeds = 0, 0

        # Compute embedding and hash for each crop (per detection)
        crops_and_meta = []
        for det in item_detections:
            scan_img = loaded_images.get(str(det.scan_filename))
            if scan_img is None:
                failed_embeds += 1
                continue
            try:
                crop = det.crop(scan_img)
                crops_and_meta.append((det, crop, str(det.scan_filename)))
                n_embeds += 1
            except Exception:
                logger.warning(
                    f"Could not crop detection in {volume_barcode}.{det.scan_filename}; skipping"
                )
                failed_embeds += 1

        # Prepare model inputs in minibatches
        batch_size = DEDUPE_EMBEDDING_BATCH_SIZE
        crop_batches = list(chunked(crops_and_meta, batch_size))

        for batch in crop_batches:
            detections_batch, crops_batch, filenames_batch = zip(*batch)
            # Prep
            prepped = [preprocess_for_model(crop) for crop in crops_batch]
            batch_tensor = torch.stack(prepped, dim=0).to(device)
            # Embedding inference
            with torch.no_grad():
                embeds = model(batch_tensor)  # [B, 512]
            embeds = embeds.cpu().numpy()
            # Normalize
            embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

            for idx, det in enumerate(detections_batch):
                embedding_entries.append(
                    Embedding(
                        detection_id=det.id_detection,
                        pipeline_batch_item=id_pipeline_batch_item,
                        scan_filename=filenames_batch[idx],
                        embedding=embeds[idx].tolist(),
                        created=datetime.now(timezone.utc),
                    )
                )
                # Hash (pHash)
                crop_img_pil = Image.fromarray(crops_batch[idx].astype(np.uint8))
                h = imagehash.phash(crop_img_pil, hash_size=HASH_SIZE)
                imagehash_val = str(h)  # hex string (e.g. 'feaf3452aaa21344')
                imagehash_entries.append(
                    ImageHash(
                        detection_id=det.id_detection,
                        pipeline_batch_item=id_pipeline_batch_item,
                        scan_filename=filenames_batch[idx],
                        image_hash=imagehash_val,
                        created=datetime.now(timezone.utc),
                    )
                )

        # Add to totals
        total_n_embeds += n_embeds
        total_failed_embeds += failed_embeds

        # Add to batch collections
        all_embedding_entries.extend(embedding_entries)
        all_imagehash_entries.extend(imagehash_entries)

        logger.info(
            f"{volume_barcode} | n_crops: {n_embeds} - failed crops: {failed_embeds} - embeddings: {len(embedding_entries)} - hashes: {len(imagehash_entries)}"
        )

        # GC/CUDA clear
        torch.cuda.empty_cache()
        gc.collect()

    #
    # Store all embeddings and hashes in DB (batch operation for all items)
    #
    from datetime import datetime

    # Delete previous entries for all items in this batch
    Embedding.delete().where(Embedding.pipeline_batch_item.in_(item_ids)).execute()
    ImageHash.delete().where(ImageHash.pipeline_batch_item.in_(item_ids)).execute()

    # Batch write all embeddings and hashes
    process_db_write_batch(
        model=Embedding,
        entries_to_create=all_embedding_entries,
    )
    process_db_write_batch(
        model=ImageHash,
        entries_to_create=all_imagehash_entries,
    )

    return True


# Use the detection/decode code as in detection pipeline
def decode_image_bytes(image_bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR_RGB)


def download_model():
    s3 = get_s3_client("OUTPUT")

    # Ensure the parent directory exists
    DEDUPE_EMBEDDING_MODEL_FILEPATH.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temporary file first, then rename (atomic operation)
    temp_filepath = DEDUPE_EMBEDDING_MODEL_FILEPATH.with_suffix(".pt.tmp")

    # Construct the full S3 key for the model file
    s3_key = DEDUPE_EMBEDDING_MODEL_STORAGE_PATH
    if not s3_key.endswith(".pt"):
        s3_key = f"{s3_key.rstrip('/')}/{DEDUPE_EMBEDDING_MODEL_NAME}"

    try:
        # Remove any existing files
        if temp_filepath.exists():
            temp_filepath.unlink()
        if DEDUPE_EMBEDDING_MODEL_FILEPATH.exists():
            DEDUPE_EMBEDDING_MODEL_FILEPATH.unlink()

        # First, check if the object exists and get its metadata
        click.echo(f"  Checking S3 object: {s3_key}")
        try:
            head_response = s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
            expected_size = head_response["ContentLength"]
            click.echo(f"  Expected file size: {expected_size:,} bytes")

            if expected_size < 1_000_000:
                raise RuntimeError(
                    f"S3 object is too small ({expected_size} bytes), likely not a valid model"
                )
        except s3.exceptions.NoSuchKey:
            logger.error(f"S3 object not found: s3://{BUCKET_NAME}/{s3_key}")
            raise
        except Exception as e:
            logger.error(f"Error checking S3 object: {e}")
            raise

        # Download the file
        click.echo(f"  Downloading {s3_key} from bucket {BUCKET_NAME}...")
        s3.download_file(BUCKET_NAME, s3_key, str(temp_filepath))

        # Verify file size matches
        actual_size = temp_filepath.stat().st_size
        click.echo(f"  Downloaded {actual_size:,} bytes")

        if actual_size != expected_size:
            raise RuntimeError(
                f"File size mismatch: expected {expected_size:,} bytes, got {actual_size:,} bytes"
            )

        # Move temp file to final location (atomic)
        temp_filepath.rename(DEDUPE_EMBEDDING_MODEL_FILEPATH)
        click.echo(
            f"  ✓ Model successfully downloaded and validated: {DEDUPE_EMBEDDING_MODEL_FILEPATH}"
        )

    except Exception as e:
        logger.error(f"Error downloading model from S3: {e}")
        # Clean up temp file
        if temp_filepath.exists():
            temp_filepath.unlink()
        # Clean up corrupted final file
        if DEDUPE_EMBEDDING_MODEL_FILEPATH.exists():
            DEDUPE_EMBEDDING_MODEL_FILEPATH.unlink()
        raise
