import io
import traceback
import tarfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
import click
from loguru import logger
from PIL import Image
import numpy as np
import cv2

from utils import get_db, get_s3_client
from models import PipelineBatchItem, Detection

from const import BUCKET_NAME, CPUS_LIMIT


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

    if processes_total > 1:
        per_task_cpus_limit = max(2, cpus_limit // 2)

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

        # Get image bytes
        image_bytes_by_filename = {str(k): v for k, v in item.data.images.items()}

        # Decode scans in parallel
        loaded_images = {}
        used_filenames = set(det.scan_filename for det in dets)

        start_decode = datetime.now()
        with ThreadPoolExecutor(max_workers=cpus_limit) as pool:
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
        time_decode = datetime.now() - start_decode

        # Create crops and collect them for tar.gz
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

        if not crops_data:
            logger.info(f"{barcode}: All crops failed.")
            continue

        # Create tar.gz file with all crops
        start_upload = datetime.now()
        s3_key = f"crops/{id_pipeline_batch_item}/{barcode}.tar.gz"

        try:
            success = create_and_upload_tarball(s3_client, crops_data, s3_key, BUCKET_NAME)
            if success:
                logger.info(
                    f"{barcode} | Uploaded: {len(crops_data)} crops in tar.gz | Failed crops: {failed_crops} | Decode time: {time_decode} | Upload time: {datetime.now() - start_upload}"
                )
            else:
                logger.error(f"{barcode}: Failed to upload tar.gz")
        except Exception as e:
            logger.error(f"{barcode}: Exception creating/uploading tar.gz: {e}")

    return True


def decode_image_bytes(image_bytes):
    """Decode image bytes to numpy array."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def create_and_upload_tarball(
    s3_client, crops_data: list[tuple[str, np.ndarray]], s3_key: str, bucket_name: str
) -> bool:
    """
    Create a tar.gz file containing all crops and upload to S3.

    Args:
        s3_client: boto3 S3 client
        crops_data: List of (filename, crop_array) tuples
        s3_key: S3 key path
        bucket_name: S3 bucket name

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create tar.gz in memory
        tar_buffer = io.BytesIO()

        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            for filename, crop_array in crops_data:
                # Convert BGR (OpenCV) to RGB (PIL)
                crop_rgb = cv2.cvtColor(crop_array, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                img = Image.fromarray(crop_rgb)

                # Save to bytes buffer as PNG
                png_buffer = io.BytesIO()
                img.save(png_buffer, format="PNG", compress_level=6)
                png_bytes = png_buffer.getvalue()

                # Add to tar
                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = len(png_bytes)
                tarinfo.mtime = datetime.now().timestamp()
                tar.addfile(tarinfo, io.BytesIO(png_bytes))

        # Get the tar.gz bytes
        tar_buffer.seek(0)
        tar_bytes = tar_buffer.getvalue()

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name, Key=s3_key, Body=tar_bytes, ContentType="application/gzip"
        )

        return True

    except Exception as e:
        logger.error(f"Error creating/uploading tar.gz to {s3_key}: {e}")
        return False
