import io
import traceback
import tarfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import click
from loguru import logger
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
    # - Use a small number of worker processes, each with multiple threads for
    #   decode/encode. This is closer to the old script's behavior (high parallelism),
    #   but avoids creating an excessive number of processes.
    #
    # - Each process:
    #     * Uses more threads for I/O-bound decode (load_scans_for_detections)
    #     * Uses multiple threads for PNG encoding (create_tarball_parallel)

    # Number of worker processes (cap at a small number)
    processes_total = cpus_limit  # THIS line
    logger.info(f"Launching {processes_total} CPU processes ...")

    # Per-process CPU/thread "budget"
    worker_cpus = max(1, cpus_limit // processes_total)

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
                worker_cpus=worker_cpus,
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
        # Use more threads for I/O-bound decoding, but cap it (similar to old script)
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
            # Use more threads for encode (similar in spirit to old script)
            encode_threads = max(1, worker_cpus * 2)
            tar_bytes, time_encode = create_tarball_parallel(crops_data, encode_threads)
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
    success, png_bytes = cv2.imencode(".png", crop_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not success:
        raise ValueError(f"Failed to encode {filename}")
    return filename, png_bytes.tobytes()


def create_tarball_parallel(
    crops_data: list[tuple[str, np.ndarray]], max_workers: int
) -> tuple[bytes, float]:
    """
    Create a tar.gz file with parallel PNG encoding.

    Args:
        crops_data: List of (filename, crop_array) tuples
        max_workers: Number of threads to use for parallel encoding

    Returns:
        Tuple of (tar_bytes, encoding_time_seconds)
    """
    start_encode = datetime.now()

    # Encode all crops to PNG in parallel
    encoded_crops = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
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
    with tarfile.open(fileobj=tar_buffer, mode="w:gz", compresslevel=1) as tar:
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
