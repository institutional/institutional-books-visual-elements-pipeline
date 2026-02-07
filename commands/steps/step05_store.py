import io
import traceback
import tarfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import click
from loguru import logger
import numpy as np
import cv2

from utils import get_db, get_s3_client, decode_image_bytes
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
    #     * Uses threads for per-scan decode + crop + encode
    #     * Creates a tarball per item in a memory-bounded way (no all-crops-in-RAM)

    # Number of worker processes
    processes_total = cpus_limit
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

        #
        # Group detections by scan filename for this item.
        # This lets us decode and process one scan at a time to limit memory.
        #
        dets_by_scan: dict[str, list[Detection]] = {}
        for det in dets:
            fn = str(det.scan_filename)
            dets_by_scan.setdefault(fn, []).append(det)

        # Create tar.gz file with all crops (parallel PNG encoding, but per-scan)
        s3_key = f"crops/{id_pipeline_batch_item}/{barcode}.tar.gz"

        # Timing / statistics
        total_decode_time = 0.0
        total_crop_time = 0.0
        total_encode_time = 0.0
        total_crops = 0
        total_failed_crops = 0

        # Use more threads for encode (similar in spirit to old script)
        encode_threads = max(1, worker_cpus * 2)

        try:
            start_tar = datetime.now()

            # Create tar.gz in memory and stream files into it
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w:gz", compresslevel=1) as tar:

                for fn, det_list in dets_by_scan.items():
                    img_bytes = image_bytes_by_filename.get(fn)
                    if img_bytes is None:
                        # Missing image -> all detections on this scan fail as crops
                        total_failed_crops += len(det_list)
                        logger.warning(
                            f"{barcode}: Missing image bytes for {fn}; skipping detections"
                        )
                        continue

                    # Decode this scan only once
                    start_decode = datetime.now()
                    try:
                        scan_img = decode_image_bytes(img_bytes)
                    except Exception:
                        total_failed_crops += len(det_list)
                        logger.warning(
                            f"{barcode}: Could not decode image for {fn}; skipping detections"
                        )
                        continue
                    decode_time = (datetime.now() - start_decode).total_seconds()
                    total_decode_time += decode_time

                    # Build crops for detections on this scan
                    start_crop = datetime.now()
                    crops_for_scan: list[tuple[str, np.ndarray]] = []
                    for det in det_list:
                        try:
                            crop = det.crop(scan_img)
                        except Exception:
                            total_failed_crops += 1
                            logger.warning(f"Could not crop detection in {barcode}.{fn}; skipping")
                            continue

                        scan_base = fn.rsplit(".", 1)[0]
                        filename_in_tar = f"{scan_base}_{det.id_detection}.png"
                        crops_for_scan.append((filename_in_tar, crop))

                    crop_time = (datetime.now() - start_crop).total_seconds()
                    total_crop_time += crop_time

                    # Done with this scan; free decoded image
                    del scan_img

                    if not crops_for_scan:
                        continue

                    total_crops += len(crops_for_scan)

                    # Encode crops for this scan in parallel and write directly into tar
                    encode_time = _encode_and_write_crops_to_tar(
                        crops_for_scan=crops_for_scan,
                        tar=tar,
                        encode_threads=encode_threads,
                    )
                    total_encode_time += encode_time

                    # Free per-scan crops list
                    del crops_for_scan

            time_tar = (datetime.now() - start_tar).total_seconds()

            # Upload to S3
            tar_buffer.seek(0)
            tar_bytes = tar_buffer.getvalue()
            start_upload = datetime.now()
            success = upload_to_s3(s3_client, tar_bytes, s3_key, OUTPUT_STORAGE_BUCKET_NAME)
            time_upload = (datetime.now() - start_upload).total_seconds()

            if success:
                logger.info(
                    f"{barcode} | Crops: {total_crops} | Failed: {total_failed_crops} | "
                    f"Decode: {total_decode_time:.2f}s | Crop: {total_crop_time:.2f}s | "
                    f"Tar (PNG encode: {total_encode_time:.2f}s): {time_tar:.2f}s | "
                    f"Upload: {time_upload:.2f}s"
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


def _encode_and_write_crops_to_tar(
    crops_for_scan: list[tuple[str, np.ndarray]],
    tar: tarfile.TarFile,
    encode_threads: int,
) -> float:
    """
    Encode a set of crops to PNG in parallel and write them directly into an open tarfile.

    Args:
        crops_for_scan: list of (filename_in_tar, crop_array)
        tar: open tarfile object in "w:gz" mode
        encode_threads: number of threads to use for parallel encoding

    Returns:
        encoding_time_seconds
    """
    start_encode = datetime.now()

    encoded_crops: list[tuple[str, bytes]] = []
    with ThreadPoolExecutor(max_workers=encode_threads) as pool:
        futures = [
            pool.submit(encode_crop_to_png, filename, crop_array)
            for filename, crop_array in crops_for_scan
        ]

        for fut in as_completed(futures):
            try:
                encoded_crops.append(fut.result())
            except Exception as e:
                logger.error(f"Failed to encode crop: {e}")

    encode_time = (datetime.now() - start_encode).total_seconds()

    # Write encoded crops into tar immediately, then free them
    now_ts = datetime.now().timestamp()
    for filename, png_bytes in encoded_crops:
        tarinfo = tarfile.TarInfo(name=filename)
        tarinfo.size = len(png_bytes)
        tarinfo.mtime = now_ts
        tar.addfile(tarinfo, io.BytesIO(png_bytes))

    # Help GC
    del encoded_crops

    return encode_time


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
