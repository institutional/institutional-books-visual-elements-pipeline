import io
import traceback
import tarfile
import os

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import click
from loguru import logger
from PIL import Image
import numpy as np
import cv2

from utils import get_db, get_s3_client, load_scans_for_detections, build_detection_crops
from models import PipelineBatchItem, Detection

from const import OUTPUT_STORAGE_BUCKET_NAME, CPUS_LIMIT


@click.command("step05_store")
@click.option("--id-pipeline-batch", type=int, required=True)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT,
    help="Allows for limiting the number of CPU cores this command can use.",
)
def step05_store(
    id_pipeline_batch: int,
    cpus_limit: int,
):
    """
    Stores cropped detection regions to S3/R2 storage in full resolution PNG format.
    """

    # Concurrency model:
    # - This command uses a ProcessPoolExecutor where each worker process handles a disjoint
    #   subset of PipelineBatchItems. This avoids contention on the GIL for the CPU-bound
    #   parts (PNG encoding, image ops) and isolates per-process memory usage.
    # - Within each process we further use a ThreadPoolExecutor for:
    #     * I/O-bound scan decoding (load_scans_for_detections)
    #     * CPU-ish but relatively lightweight PNG encoding (create_tarball_parallel)
    #   The number of threads per process is capped and derived from worker_cpus so we
    #   don't oversubscribe CPU cores.
    # - Global BLAS/OpenMP-related env vars are pinned to 1 thread to prevent libraries
    #   like NumPy/OpenBLAS from spawning additional threads and defeating our own limits.

    # Number of worker processes
    processes_total = cpus_limit
    logger.info(f"Launching {processes_total} CPU processes ...")

    # Per-process CPU/thread "budget"
    per_process_cpus = max(2, cpus_limit // processes_total) if processes_total > 1 else cpus_limit

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

    if not eligible_items:
        logger.warning("No items with detections found. Exiting.")
        return

    # Round-robin distribute items across processes
    item_batches = [[] for _ in range(processes_total)]
    for i, item in enumerate(eligible_items):
        item_batches[i % processes_total].append(item.id_pipeline_batch_item)

    with ProcessPoolExecutor(max_workers=processes_total, initializer=get_db) as executor:
        futures = {}
        for idx, item_ids in enumerate(item_batches):
            if not item_ids:
                continue
            future = executor.submit(
                store_batch_of_items,
                item_ids=item_ids,
                worker_cpus=per_process_cpus,
            )
            futures[future] = idx

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                # Log a concise error at ERROR level, and the full traceback at DEBUG level
                logger.error("Error in a worker process; shutting down remaining workers.")
                logger.debug(traceback.format_exc())
                executor.shutdown(wait=False, cancel_futures=True)
                return

    logger.info(f"Completed storing crops for pipeline batch {id_pipeline_batch}")


def store_batch_of_items(item_ids: list[int], worker_cpus: int):
    """Process a batch of items and store their crops to S3."""

    s3_client = get_s3_client("OUTPUT")

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

        # Get image bytes mapping (as before)
        image_bytes_by_filename = {str(k): v for k, v in item.data.images.items()}

        # Decode scans (using shared utility)
        start_decode = datetime.now()
        # Use more threads for I/O-bound decoding, but cap it
        decode_threads = min(worker_cpus * 2, 16)
        loaded_images = load_scans_for_detections(
            volume_barcode=barcode,
            detections=dets,
            image_bytes_by_filename=image_bytes_by_filename,
            max_workers=decode_threads,
        )
        time_decode = (datetime.now() - start_decode).total_seconds()

        # Create crops using shared utility
        start_crop = datetime.now()
        records, failed_crops = build_detection_crops(
            volume_barcode=barcode,
            detections=dets,
            loaded_images=loaded_images,
            with_filename=True,  # we want the original scan filename
        )

        # Convert records -> crops_data for tar creation
        crops_data: list[tuple[str, np.ndarray]] = []
        for det, crop, fn in records:
            # Generate filename for inside the tar: {scan_base}_{detection_id}.png
            scan_base = fn.rsplit(".", 1)[0]
            filename_in_tar = f"{scan_base}_{det.id_detection}.png"
            crops_data.append((filename_in_tar, crop))

        time_crop = (datetime.now() - start_crop).total_seconds()

        if not crops_data:
            logger.info(f"{barcode}: All crops failed.")
            continue

        # Create tar.gz file with all crops (parallel PNG encoding)
        s3_key = f"crops/{id_pipeline_batch_item}/{barcode}.tar.gz"

        try:
            start_tar = datetime.now()
            # IMPORTANT: use worker_cpus, not global cpus_limit
            tar_bytes, time_encode = create_tarball_parallel(crops_data, worker_cpus)
            time_tar = (datetime.now() - start_tar).total_seconds()

            start_upload = datetime.now()
            success = upload_to_s3(s3_client, tar_bytes, s3_key, OUTPUT_STORAGE_BUCKET_NAME)
            time_upload = (datetime.now() - start_upload).total_seconds()

            if success:
                logger.info(
                    f"{barcode} | Crops: {len(crops_data)} | Failed: {failed_crops} | "
                    f"Decode: {time_decode:.2f}s | Crop: {time_crop:.2f}s | "
                    f"Tar (PNG encode: {time_encode:.2f}s): {time_tar:.2f}s | Upload: {time_upload:.2f}s"
                )
            else:
                logger.error(f"{barcode}: Failed to upload tar.gz")
        except Exception as e:
            logger.error(f"{barcode}: Exception creating/uploading tar.gz: {e}")
            traceback.print_exc()

    return True


def encode_crop_to_png(filename: str, crop_array: np.ndarray) -> tuple[str, bytes]:
    """
    Encode a single crop to PNG bytes.

    Returns:
        Tuple of (filename, png_bytes)
    """
    # Convert BGR (OpenCV) to RGB (PIL)
    crop_rgb = cv2.cvtColor(crop_array, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image and save as PNG
    img = Image.fromarray(crop_rgb)
    png_buffer = io.BytesIO()

    # compress_level=1 is much faster than 6 with minimal size difference
    # compress_level=0 is fastest (no compression)
    img.save(png_buffer, format="PNG", compress_level=1)

    return filename, png_buffer.getvalue()


def create_tarball_parallel(
    crops_data: list[tuple[str, np.ndarray]], worker_cpus: int
) -> tuple[bytes, float]:
    """
    Create a tar.gz file with parallel PNG encoding.

    Args:
        crops_data: List of (filename, crop_array) tuples
        worker_cpus: Per-process CPU/thread limit to use for parallel encoding

    Returns:
        Tuple of (tar_bytes, encoding_time_seconds)
    """
    start_encode = datetime.now()

    # Encode all crops to PNG in parallel
    encoded_crops = []
    # Use worker_cpus as the number of encoding threads in this process
    with ThreadPoolExecutor(max_workers=worker_cpus) as pool:
        futures = [
            pool.submit(encode_crop_to_png, filename, crop_array)
            for filename, crop_array in crops_data
        ]

        for fut in as_completed(futures):
            try:
                encoded_crops.append(fut.result())
            except Exception as e:
                logger.error(f"Failed to encode crop: {e}")

    time_encode = (datetime.now() - start_encode).total_seconds()

    # Create tar.gz file
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz", compresslevel=6) as tar:
        for filename, png_bytes in encoded_crops:
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(png_bytes)
            tarinfo.mtime = datetime.now().timestamp()
            tar.addfile(tarinfo, io.BytesIO(png_bytes))

    tar_buffer.seek(0)
    return tar_buffer.getvalue(), time_encode


def upload_to_s3(s3_client, tar_bytes: bytes, s3_key: str, bucket_name: str) -> bool:
    """
    Upload to S3.

    Args:
        s3_client: boto3 S3 client
        tar_bytes: Bytes to upload
        s3_key: S3 key path
        bucket_name: S3 bucket name

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        s3_client.put_object(
            Bucket=bucket_name, Key=s3_key, Body=tar_bytes, ContentType="application/gzip"
        )
        return True
    except Exception as e:
        logger.error(f"Error uploading to {s3_key}: {e}")
        return False
