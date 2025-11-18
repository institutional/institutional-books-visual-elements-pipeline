import json
import openai
import click
from typing import Any, Optional, Dict
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
from models import PipelineBatch, OpenAIBatchObject
from utils import get_s3_client
from const import (
    CAPTION_JSONL_FILES_PATH,
    CAPTION_MAX_FILES_PROCESS_PER_DAY,
    CPUS_LIMIT_CAPTIONS,
    CAPTION_BUCKET_NAME,
)

client = openai.OpenAI()


@click.command("step04-1-process-caption-requests")
@click.option("--id-pipeline-batch", type=int, required=False, default=None)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT_CAPTIONS,
    help="Allows for limiting the number of CPU cores this command can use.",
)
def step04_1_process_caption_requests(id_pipeline_batch, cpus_limit):

    jsonl_dir = CAPTION_JSONL_FILES_PATH
    all_jsonl_files = sorted(glob(os.path.join(jsonl_dir, "*.jsonl")))

    # TODO: filter for pipeline_batch_id folder

    process_limit = min(CAPTION_MAX_FILES_PROCESS_PER_DAY, len(all_jsonl_files))
    files_to_process = all_jsonl_files[:process_limit]

    logger.info(f"Will process {len(files_to_process)} caption jsonl files.")

    with ProcessPoolExecutor(max_workers=cpus_limit) as executor:
        futures = [
            executor.submit(process_one_jsonl, f, id_pipeline_batch) for f in files_to_process
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as err:
                logger.error(f"Failed job: {err}")


def process_one_jsonl(batch_file, id_pipeline_batch):

    boto3_client = get_s3_client("OUTPUT")  # For saving jsonl files

    try:
        # 1. Upload to S3, in folder 'caption_jsonl_files/'
        s3_key = f"caption_jsonl_files/{id_pipeline_batch}/{os.path.basename(batch_file)}"
        boto3_client.upload_file(batch_file, CAPTION_BUCKET_NAME, s3_key)
        logger.info(f"Uploaded {batch_file} to s3://{CAPTION_BUCKET_NAME}/{s3_key}")

        # 2. Submit batch to OpenAI
        batch_obj = process_batch(batch_file, metadata=None)  # can pass metadata if desired

        # 3. Save batch info in DB
        OpenAIBatchObject.create(
            pipeline_batch_item=id_pipeline_batch,
            jsonl_file_name=os.path.basename(batch_file),
            s3_key=s3_key,
            batch_id=batch_obj.id,
            status=batch_obj.status,
            num_requests=batch_obj.request_counts.total if batch_obj.request_counts else 0,
            submitted_at=batch_obj.created_at,
            endpoint=batch_obj.endpoint,
        )
        logger.info(f"Logged OpenAI batch {batch_obj['id']} for {batch_file}")

        # 4. Delete local file (optional, safe)
        os.remove(batch_file)
        logger.info(f"Deleted local file {batch_file}")

    except Exception as e:
        logger.error(f"Error processing {batch_file}: {e}")


def process_batch(batch_file, metadata):
    """Upload BATCH_FILE and create a batch job with optional --metadata."""
    # Handle metadata
    metadata_dict = metadata or {}
    # Upload the file
    logger.info(f"Uploading {batch_file} to OpenAI...")
    file = upload_batch(client, batch_file)
    # Create the batch
    logger.info("Creating batch at OpenAI...")
    batch = create_batch(client, file, metadata=metadata_dict)
    logger.info(f"Batch created: {batch.id}")
    return batch


def upload_batch(client: Any, batch_filename: str) -> Any:
    """Uploads a batch file to the API client."""
    with open(batch_filename, "rb") as file_obj:
        batch_input_file = client.files.create(file=file_obj, purpose="batch")
    return batch_input_file


def create_batch(client: Any, batch_file: Any, metadata: Optional[Dict[str, str]] = None) -> Any:
    """Creates a new batch using the uploaded file's ID."""
    if metadata is None:
        metadata = {}
    batch_input_file_id = batch_file.id
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata,
    )
    return batch


def check_batch_status(client: Any, batch_name: str) -> Any:
    batch = client.batches.retrieve(batch_name)
    return batch
