import io
import traceback
import tarfile
import time
import threading
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
import click
from loguru import logger
import numpy as np
import cv2

from utils import get_db, get_s3_client
from models import PipelineBatchItem, Detection

from const import BUCKET_NAME, CPUS_LIMIT, MAX_S3_REQUESTS_PER_SECOND


# Global rate limiter for S3 requests
class RateLimiter:
    def __init__(self, max_requests_per_second):
        self.max_requests = max_requests_per_second
        self.tokens = max_requests_per_second
        self.lock = threading.Lock()
        self.last_update = time.monotonic()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update

                # Add tokens based on elapsed time
                self.tokens = min(self.max_requests, self.tokens + elapsed * self.max_requests)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

            # Wait a bit before trying again
            time.sleep(0.05)


# 1500 requests/min = 25 requests/sec
S3_RATE_LIMITER = RateLimiter(max_requests_per_second=MAX_S3_REQUESTS_PER_SECOND)


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

    processes_total = cpus_limit
    logger.info(f"Launching {processes_total} CPU processes ...")

    per_task_cpus_limit = (
        max(2, cpus_limit // processes_total) if processes_total > 1 else cpus_limit
    )

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

    logger.info(f"Completed storing crops for pipeline batch {id_pipeline_batch}")


def store_batch_of_items(item_ids: list[int], cpus_limit: int):
    """Process a batch of items and store their crops to S3."""

    s3_client = get_s3_client("OUTPUT")

    # Fetch all items at once
    items = list(
        PipelineBatchItem.select().where(PipelineBatchItem.id_pipeline_batch_item.in_(item_ids))
    )

    # Fetch ALL detections for these items at once
    detections_query = (
        Detection.select()
        .where(Detection.pipeline_batch_item.in_(item_ids))
        .order_by(Detection.id_detection)
    )

    # Group detections by pipeline_batch_item
    detections_by_item = {}
    for det in detections_query:
        item_id = det.pipeline_batch_item_id
        if item_id not in detections_by_item:
            detections_by_item[item_id] = []
        detections_by_item[item_id].append(det)

    for item in items:
        id_pipeline_batch_item = item.id_pipeline_batch_item
        volume = item.ib_volume
        barcode = volume.barcode

        # Access pre-fetched detections
        dets = detections_by_item.get(id_pipeline_batch_item, [])

        if not dets:
            logger.info(f"{barcode}: No detections - skipping.")
            continue

        # Get image bytes
        image_bytes_by_filename = {str(k): v for k, v in item.data.images.items()}

        # Decode scans in parallel
        loaded_images = {}
        used_filenames = set(det.scan_filename for det in dets)

        start_decode = datetime.now()
        # Use more threads for I/O-bound decoding
        with ThreadPoolExecutor(max_workers=min(cpus_limit * 2, 16)) as pool:
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

        # Create crops (numpy arrays only, no encoding yet)
        start_crop = datetime.now()
        crops_data = []  # List of (filename_in_tar, crop_array)
        failed_crops = 0

        for det in dets:
            img = loaded_images.get(det.scan_filename)
            if img is None:
                failed_crops += 1
                continue
            try:
                crop = det.crop(img)

                # Generate filename for inside the tar: {filename}_{detection_id}.png
                scan_base = det.scan_filename.rsplit(".", 1)[0]  # Remove extension
                filename_in_tar = f"{scan_base}_{det.id_detection}.png"

                crops_data.append((filename_in_tar, crop))

            except Exception as e:
                logger.warning(f"{barcode}: Failed to crop detection {det.id_detection}: {e}")
                failed_crops += 1

        time_crop = (datetime.now() - start_crop).total_seconds()

        if not crops_data:
            logger.info(f"{barcode}: All crops failed.")
            continue

        # Create tar.gz file with all crops (parallel PNG encoding)
        s3_key = f"crops/{id_pipeline_batch_item}/{barcode}.tar.gz"

        try:
            start_tar = datetime.now()
            tar_bytes, time_encode = create_tarball_parallel(crops_data, cpus_limit)
            time_tar = (datetime.now() - start_tar).total_seconds()

            start_upload = datetime.now()
            success = upload_to_s3_with_ratelimit(s3_client, tar_bytes, s3_key, BUCKET_NAME)
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


def decode_image_bytes(image_bytes):
    """Decode image bytes to numpy array."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def encode_crop_to_png(filename: str, crop_array: np.ndarray) -> tuple[str, bytes]:
    """
    Encode a single crop to PNG bytes using OpenCV.

    Returns:
        Tuple of (filename, png_bytes)
    """
    success, png_bytes = cv2.imencode(
        ".png",
        crop_array,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],  # 0 compression when encoding
    )

    if not success:
        raise ValueError(f"Failed to encode {filename} to PNG")

    return filename, png_bytes.tobytes()


def create_tarball_parallel(
    crops_data: list[tuple[str, np.ndarray]], cpus_limit: int
) -> tuple[bytes, float]:
    """
    Create a tar.gz file with parallel PNG encoding.

    Args:
        crops_data: List of (filename, crop_array) tuples
        cpus_limit: Number of CPUs to use for parallel encoding

    Returns:
        Tuple of (tar_bytes, encoding_time_seconds)
    """
    start_encode = datetime.now()

    # Encode all crops to PNG in parallel
    encoded_crops = []
    with ThreadPoolExecutor(max_workers=cpus_limit) as pool:
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


def upload_to_s3_with_ratelimit(s3_client, tar_bytes: bytes, s3_key: str, bucket_name: str) -> bool:
    """
    Upload to S3 with rate limiting.

    Args:
        s3_client: boto3 S3 client
        tar_bytes: Bytes to upload
        s3_key: S3 key path
        bucket_name: S3 bucket name

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Acquire rate limit token before making request
        S3_RATE_LIMITER.acquire()

        s3_client.put_object(
            Bucket=bucket_name, Key=s3_key, Body=tar_bytes, ContentType="application/gzip"
        )
        return True
    except Exception as e:
        logger.error(f"Error uploading to {s3_key}: {e}")
        return False
