import io
import traceback
import tarfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
from loguru import logger
import numpy as np
import cv2

from utils import (
    get_db,
    get_s3_client,
    get_time,
    load_scans_for_detections,
    build_detection_crops,
)
from models import PipelineBatchItem, Detection

from const import OUTPUT_STORAGE_BUCKET_NAME, CPUS_LIMIT, MAX_S3_CONCURRENCY


@click.command("step05_store")
@click.option("--id-pipeline-batch", type=int, required=True)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT,
    help="Loose limit on the number of CPU cores this command should use.",
)
def step05_store(
    id_pipeline_batch: int,
    cpus_limit: int,
):
    """
    Stores cropped detection regions to S3/R2 storage in full resolution PNG format.

    This version tries to:
      - Keep total CPU-bound workers <= cpus_limit
      - Avoid massive oversubscription (too many processes * threads)
    """
    # NOTE on oversubscription:
    # The main risk is spawning too many *OS threads* (from both processes and
    # their internal thread pools) that sit blocked on I/O (S3, disk) or
    # waiting for futures to complete. That can lead to context-switch overhead
    # and worse throughput.
    #
    # To mitigate that:
    # - We cap the number of worker processes by both CPUS_LIMIT and
    #   MAX_S3_CONCURRENCY (since each process does S3 uploads).
    # - Inside each process we use a small, bounded number of threads for
    #   decode/encode work (threads_per_process).
    # - The total "effective" CPU workers is therefore roughly
    #   processes_total * threads_per_process, which we keep close to CPUS_LIMIT.
    #
    # If we still see long‑lived futures piling up, a further refinement would be
    # to chunk work at the process level (e.g., batches of CPUS_LIMIT or
    # MAX_S3_CONCURRENCY items) so we never have a huge number of unresolved
    # futures enqueued at once.

    logical_cpus = multiprocessing.cpu_count()
    target_workers = min(cpus_limit, logical_cpus)
    processes_total = min(target_workers, MAX_S3_CONCURRENCY) if target_workers > 0 else 1

    threads_per_process = max(1, target_workers // processes_total) if processes_total > 0 else 1

    effective_workers = processes_total * threads_per_process

    logger.info(
        f"CPU parallelism: CPUS_LIMIT={cpus_limit}, logical_cpus={logical_cpus}, "
        f"target_workers={target_workers}, processes_total={processes_total}, "
        f"threads_per_process={threads_per_process}, "
        f"effective_cpu_workers≈{effective_workers}"
    )

    # Concurrency model:
    # - We use a small pool of worker processes (ProcessPoolExecutor), each
    #   responsible for a disjoint subset of PipelineBatchItems.
    # - Within each process, we use a limited number of threads for:
    #     * Decoding input scans (decode_threads)
    #     * Encoding crops to PNG and building the tarball (encode_threads)
    # - This hybrid model (processes + a few threads) lets us:
    #     * Parallelize CPU-bound image work across cores.
    #     * Overlap I/O (S3 uploads, decoding) without spawning thousands of threads.
    # - We keep the total number of threads roughly bounded by CPUS_LIMIT to avoid
    #   oversubscribing the host and degrading performance.

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

    # Use Peewee's iterator() so we don't materialize the whole result set at once.
    item_batches = [[] for _ in range(processes_total)]
    got_any_items = False
    for i, item in enumerate(eligible_query.iterator()):
        got_any_items = True
        item_batches[i % processes_total].append(item.id_pipeline_batch_item)

    if not got_any_items:
        logger.warning("No items with detections found. Exiting.")
        return

    with ProcessPoolExecutor(max_workers=processes_total, initializer=get_db) as executor:
        futures = {}
        for idx, item_ids in enumerate(item_batches):
            if not item_ids:
                continue
            future = executor.submit(
                store_batch_of_items,
                item_ids=item_ids,
                threads_per_process=threads_per_process,
            )
            futures[future] = idx

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                # Detailed stack trace goes to debug so it can be enabled with --verbose
                logger.debug(traceback.format_exc())
                logger.error("Error in a worker process while storing crops.")
                executor.shutdown(wait=False, cancel_futures=True)
                return

    logger.info(f"Completed storing crops for pipeline batch {id_pipeline_batch}")


def store_batch_of_items(
    item_ids: list[int],
    threads_per_process: int = 1,
):
    """Process a batch of items and store their crops to S3."""
    decode_threads = threads_per_process
    encode_threads = threads_per_process

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
    detections_by_item: dict[int, list[Detection]] = {}
    for det in detections_query:
        item_id = det.pipeline_batch_item_id
        detections_by_item.setdefault(item_id, []).append(det)

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

        # Decode scans for this item
        start_decode = get_time()
        loaded_images = load_scans_for_detections(
            volume_barcode=barcode,
            detections=dets,
            image_bytes_by_filename=image_bytes_by_filename,
            max_workers=decode_threads,
        )
        time_decode = (get_time() - start_decode).total_seconds()

        # Create crops (numpy arrays only, no encoding yet)
        start_crop = get_time()

        # Build (Detection, crop, filename) from loaded images
        crop_records, failed_crops = build_detection_crops(
            volume_barcode=barcode,
            detections=dets,
            loaded_images=loaded_images,
            with_filename=True,
        )
        n_crops = len(crop_records)

        # Convert to (filename_in_tar, crop_array) tuples
        crops_data: list[tuple[str, np.ndarray]] = []
        for det, crop, scan_filename in crop_records:
            scan_base = scan_filename.rsplit(".", 1)[0]  # Remove extension
            filename_in_tar = f"{scan_base}_{det.id_detection}.png"
            crops_data.append((filename_in_tar, crop))

        time_crop = (get_time() - start_crop).total_seconds()

        if not crops_data:
            logger.info(f"{barcode}: All crops failed.")
            continue

        # Create tar.gz file with all crops
        s3_key = f"crops/{id_pipeline_batch_item}/{barcode}.tar.gz"

        try:
            start_tar = get_time()
            tar_bytes, time_encode = create_tarball_parallel(
                crops_data=crops_data,
                encode_threads=encode_threads,
            )
            time_tar = (get_time() - start_tar).total_seconds()

            start_upload = get_time()
            success = upload_to_s3(s3_client, tar_bytes, s3_key, OUTPUT_STORAGE_BUCKET_NAME)
            time_upload = (get_time() - start_upload).total_seconds()

            if success:
                logger.info(
                    f"{barcode} | "
                    f"Crops: {n_crops} - "
                    f"Failed_crops: {failed_crops} - "
                    f"Decode: {time_decode:.2f}s - "
                    f"Crop: {time_crop:.2f}s - "
                    f"Tar (PNG encode: {time_encode:.2f}s): {time_tar:.2f}s - "
                    f"Upload: {time_upload:.2f}s"
                )
            else:
                logger.error(f"{barcode}: Failed to upload tar.gz")
        except Exception as e:
            logger.error(f"{barcode}: Exception creating/uploading tar.gz: {e}")
            logger.debug(traceback.format_exc())

    return True


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
    crops_data: list[tuple[str, np.ndarray]],
    encode_threads: int = 1,
) -> tuple[bytes, float]:
    """
    Create a tar.gz file with parallel (or sequential) PNG encoding.

    Args:
        crops_data: List of (filename, crop_array) tuples
        encode_threads: Number of threads to use for parallel encoding (per process)

    Returns:
        Tuple of (tar_bytes, encoding_time_seconds)
    """
    start_encode = get_time()

    encoded_crops: list[tuple[str, bytes]] = []

    if encode_threads > 1:
        # Parallel encoding
        with ThreadPoolExecutor(max_workers=encode_threads) as pool:
            futures = [
                pool.submit(encode_crop_to_png, filename, crop_array)
                for filename, crop_array in crops_data
            ]

            for fut in as_completed(futures):
                try:
                    encoded_crops.append(fut.result())
                except Exception as e:
                    logger.error(f"Failed to encode crop: {e}")
    else:
        # Sequential encoding
        for filename, crop_array in crops_data:
            try:
                encoded_crops.append(encode_crop_to_png(filename, crop_array))
            except Exception as e:
                logger.error(f"Failed to encode crop: {e}")

    time_encode = (get_time() - start_encode).total_seconds()

    # Create tar.gz file
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz", compresslevel=6) as tar:
        now_ts = get_time().timestamp()
        for filename, png_bytes in encoded_crops:
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(png_bytes)
            tarinfo.mtime = now_ts
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
