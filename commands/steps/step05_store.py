# NOTE:
# This is the last item of the batch-level steps series.
#
# Because every step after that is dataset-scale and will not have easy access to scans,
# the goal of this step would be to store intermediary objects to remote storage for easy access.
#
# In that case, we want to store crops on R2:
# - `bucket/crops/{id_pipeline_batch_item}/{barcode}/{filename}_{detection_id}.png`
#
# We should keep track of these crops and their properties in the database so they're easy to retrieve and analyze.

import io
import traceback
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
    default=175,
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

        # Create crops and prepare upload tasks
        crop_upload_tasks = []
        failed_crops = 0

        for det in dets:
            img = loaded_images.get(det.scan_filename)
            if img is None:
                failed_crops += 1
                continue
            try:
                crop = det.crop(img)

                # Generate S3 key path
                # Format: {id_pipeline_batch_item}/{barcode}/{filename}_{detection_id}.png
                scan_base = det.scan_filename.rsplit(".", 1)[0]  # Remove extension
                s3_key = (
                    f"crops/{id_pipeline_batch_item}/{barcode}/{scan_base}_{det.id_detection}.png"
                )

                crop_upload_tasks.append((det.id_detection, crop, s3_key))

            except Exception as e:
                logger.warning(f"{barcode}: Failed to crop detection {det.id_detection}: {e}")
                failed_crops += 1

        if not crop_upload_tasks:
            logger.info(f"{barcode}: All crops failed.")
            continue

        # Upload crops in parallel
        start_upload = datetime.now()
        uploaded_count = 0
        failed_uploads = 0

        with ThreadPoolExecutor(max_workers=cpus_limit) as upload_pool:
            upload_futures = {
                upload_pool.submit(upload_crop_to_s3, s3_client, crop, s3_key, BUCKET_NAME): (
                    det_id,
                    s3_key,
                )
                for det_id, crop, s3_key in crop_upload_tasks
            }

            for fut in as_completed(upload_futures):
                det_id, s3_key = upload_futures[fut]
                try:
                    success = fut.result()
                    if success:
                        uploaded_count += 1
                    else:
                        failed_uploads += 1
                        logger.warning(f"{barcode}: Failed to upload detection {det_id}")
                except Exception as e:
                    failed_uploads += 1
                    logger.warning(f"{barcode}: Exception uploading detection {det_id}: {e}")

        time_upload = datetime.now() - start_upload

        logger.info(
            f"{barcode} | Uploaded: {uploaded_count} | Failed crops: {failed_crops} | Failed uploads: {failed_uploads} | Decode time: {time_decode} | Upload time: {time_upload}"
        )

    return True


def decode_image_bytes(image_bytes):
    """Decode image bytes to numpy array."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def upload_crop_to_s3(s3_client, crop_array, s3_key: str, bucket_name: str) -> bool:
    """
    Convert crop array to PNG and upload to S3.

    Args:
        s3_client: boto3 S3 client
        crop_array: numpy array of the crop (OpenCV format, BGR)
        s3_key: S3 key path
        bucket_name: S3 bucket name

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert BGR (OpenCV) to RGB (PIL)
        crop_rgb = cv2.cvtColor(crop_array, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img = Image.fromarray(crop_rgb)

        # Save to bytes buffer as PNG (full resolution, lossless)
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=6)  # compress_level 6 is good balance
        buf.seek(0)

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name, Key=s3_key, Body=buf.getvalue(), ContentType="image/png"
        )

        return True

    except Exception as e:
        logger.error(f"Error uploading to {s3_key}: {e}")
        return False
