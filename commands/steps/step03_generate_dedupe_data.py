import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import gc
import multiprocessing as mp

import click
from loguru import logger
import numpy as np

from more_itertools import chunked
import imagehash

from utils import (
    get_db,
    process_db_write_batch,
    get_s3_client,
    get_time,
    load_scans_for_detections,
    build_detection_crops,
)
from models import PipelineBatchItem, Detection, ImageEmbedding, ImageHash

from const import (
    DEDUPE_EMBEDDING_MODEL_FILEPATH,
    DEDUPE_EMBEDDING_MODEL_STORAGE_PATH,
    DEDUPE_EMBEDDING_NUM_PROCESSES_PER_GPU,
    HASH_DEDUPE_LENGTH_BYTES,
    CUDA_GPUS,
    CPUS_LIMIT,
    OUTPUT_STORAGE_BUCKET_NAME,
    DEDUPE_EMBEDDING_BATCH_SIZE,
    DEDUPE_EMBEDDING_MODEL_NAME,
)


@click.command("step03-generate-dedupe-data")
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
def step03_generate_dedupe_data(id_pipeline_batch: int, cpus_limit: int, cuda_gpus: list[str]):
    """
    Computes embeddings (and hashes) for all crops in all volumes with detections in this pipeline batch,
    and saves them to the database, per GPU.
    """
    model_filepath: Path | None = None

    # Concurrency model:
    # - We launch processes_total = cuda_gpus_total * DEDUPE_EMBEDDING_NUM_PROCESSES_PER_GPU
    #   worker processes via ProcessPoolExecutor.
    # - Each worker process:
    #     * Initializes its own DB connection (initializer=get_db).
    #     * Loads the TorchScript embedding model once and pins it to one CUDA device.
    #     * Processes only the subset of PipelineBatchItem IDs assigned to it
    #       in item_id_batches.
    # - Items are assigned to workers round‑robin so the workload is roughly balanced.
    #   The CUDA device for worker i is chosen with cuda_gpus[i % cuda_gpus_total],
    #   so multiple workers can share the same GPU when
    #   DEDUPE_EMBEDDING_NUM_PROCESSES_PER_GPU > 1.
    # - Within each process, CPU‑bound work (image decode, crop, preprocessing)
    #   uses a small ThreadPoolExecutor bounded by cpus_limit so that the total
    #   number of active CPU threads across all processes stays close to the
    #   global cpus_limit and we avoid oversubscribing the host.

    cuda_gpus_total = len(cuda_gpus)
    processes_total = cuda_gpus_total * DEDUPE_EMBEDDING_NUM_PROCESSES_PER_GPU

    item_id_batches: list[list[int]] = [[] for _ in range(processes_total)]

    per_task_cpus_limit = int(round(cpus_limit / cuda_gpus_total))
    if processes_total > 1:
        per_task_cpus_limit = max(2, per_task_cpus_limit // 2)

    # Download model before spawning processes.
    # This is a TorchScript version of the SSCD (Self-Supervised Copy Detection) model
    # released by Facebook/Meta AI. We store the file in an S3-compatible object store
    # and pull it from there into local disk (DEDUPE_EMBEDDING_MODEL_FILEPATH) before
    # starting worker processes so that each process can load it from the local filesystem.

    local_model_path = DEDUPE_EMBEDDING_MODEL_FILEPATH
    if not local_model_path.exists():
        logger.info("Downloading TorchScript model from S3 Storage...")
        download_model()
        logger.info("✓ Model downloaded and verified successfully")
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

    # Use Peewee's iterator() to avoid loading the entire result set in memory.
    # We assign items round‑robin to the available worker processes.
    for i, item in enumerate(eligible_items_query.iterator()):
        process_i = i % processes_total
        item_id_batches[process_i].append(item.id_pipeline_batch_item)

    if not any(item_id_batches):
        logger.warning("No eligible items with detections found for this batch. Exiting.")
        click.get_current_context().exit(0)

    mp_ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=processes_total,
        initializer=get_db,
        mp_context=mp_ctx,
    ) as executor:
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
        for future in as_completed(futures):
            cuda_gpu: str = futures[future]
            try:
                future.result()
            except Exception:
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
    """
    Generate embeddings and perceptual hashes for all detections belonging to a batch
    of PipelineBatchItem IDs on a single worker process.
    """
    import os

    # set CUDA_VISIBLE_DEVICES before importing torch - avoids defaulting to physical cuda:0
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

    # Collect all embeddings and hashes from all items
    all_embedding_entries: list[ImageEmbedding] = []
    all_imagehash_entries: list[ImageHash] = []

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

        if not item_detections.exists():
            logger.info(f"{volume_barcode}: No detections - skipping embedding for this item.")
            continue

        image_bytes_by_filename = {str(k): v for k, v in item.data.images.items()}

        # 1) Decode all scans needed for this item (only those used by detections)
        loaded_images = load_scans_for_detections(
            volume_barcode=volume_barcode,
            detections=item_detections,
            image_bytes_by_filename=image_bytes_by_filename,
            max_workers=cpus_limit,
        )

        # 2) Build a list of (Detection, crop, filename) tuples
        crops_and_meta, failed_embeds = build_detection_crops(
            volume_barcode=volume_barcode,
            detections=item_detections,
            loaded_images=loaded_images,
            with_filename=True,
        )
        n_embeds = len(crops_and_meta)

        embedding_entries: list[ImageEmbedding] = []
        imagehash_entries: list[ImageHash] = []

        if n_embeds == 0:
            logger.info(f"{volume_barcode}: All crops failed; skipping embeddings.")
            total_failed_embeds += failed_embeds
            continue

        # Prepare model inputs in minibatches
        batch_size = DEDUPE_EMBEDDING_BATCH_SIZE
        crop_batches = list(chunked(crops_and_meta, batch_size))

        for batch in crop_batches:
            detections_batch, crops_batch, filenames_batch = zip(*batch)
            # Preprocess crops for the model
            prepped = [preprocess_for_model(crop) for crop in crops_batch]
            batch_tensor = torch.stack(prepped, dim=0).to(device)
            # Embedding inference
            with torch.no_grad():
                embeds = model(batch_tensor)  # [B, D]
            embeds = embeds.cpu().numpy()
            # Normalize
            embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

            for idx, det in enumerate(detections_batch):
                embedding_entries.append(
                    ImageEmbedding(
                        detection_id=det.id_detection,
                        pipeline_batch_item=id_pipeline_batch_item,
                        embedding=embeds[idx].tolist(),
                        created=get_time(),
                    )
                )
                # Hash (pHash)
                crop_img_pil = Image.fromarray(crops_batch[idx].astype(np.uint8))
                h = imagehash.phash(crop_img_pil, hash_size=HASH_DEDUPE_LENGTH_BYTES)
                imagehash_val = str(h)  # hex string (e.g. 'feaf3452aaa21344')
                imagehash_entries.append(
                    ImageHash(
                        detection_id=det.id_detection,
                        pipeline_batch_item=id_pipeline_batch_item,
                        image_hash=imagehash_val,
                        created=get_time(),
                    )
                )

        # Add to totals
        total_n_embeds += n_embeds
        total_failed_embeds += failed_embeds

        # Add to batch collections
        all_embedding_entries.extend(embedding_entries)
        all_imagehash_entries.extend(imagehash_entries)

        # Logging
        logger.info(
            f"{volume_barcode} | "
            f"n_crops: {n_embeds} - "
            f"failed_crops: {failed_embeds} - "
            f"Device: {cuda_device} - "
            f"embeddings: {len(embedding_entries)} - "
            f"hashes: {len(imagehash_entries)}"
        )

        # GC/CUDA clear
        torch.cuda.empty_cache()
        gc.collect()

    #
    # Store all embeddings and hashes in DB (batch operation for all items)
    #

    # Delete previous entries for all items in this batch
    ImageEmbedding.delete().where(ImageEmbedding.pipeline_batch_item.in_(item_ids)).execute()
    ImageHash.delete().where(ImageHash.pipeline_batch_item.in_(item_ids)).execute()

    # Batch write all embeddings and hashes
    process_db_write_batch(
        model=ImageEmbedding,
        entries_to_create=all_embedding_entries,
    )
    process_db_write_batch(
        model=ImageHash,
        entries_to_create=all_imagehash_entries,
    )

    return True


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
            # We proactively call head_object to:
            # - Fail fast if the key does not exist.
            # - Get the expected content length, so we can validate the download later.
            # - Check for partial/corrupted downloads.
            head_response = s3.head_object(Bucket=OUTPUT_STORAGE_BUCKET_NAME, Key=s3_key)
            expected_size = head_response["ContentLength"]
            click.echo(f"  Expected file size: {expected_size:,} bytes")

            if expected_size < 1_000_000:
                raise RuntimeError(
                    f"S3 object is too small ({expected_size} bytes), likely not a valid model"
                )
        except s3.exceptions.NoSuchKey:
            logger.error(f"S3 object not found: s3://{OUTPUT_STORAGE_BUCKET_NAME}/{s3_key}")
            raise
        except Exception as e:
            logger.error(f"Error checking S3 object: {e}")
            raise

        # Download the file
        click.echo(f"  Downloading {s3_key} from bucket {OUTPUT_STORAGE_BUCKET_NAME}...")
        s3.download_file(OUTPUT_STORAGE_BUCKET_NAME, s3_key, str(temp_filepath))

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
