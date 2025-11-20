# NOTE:
# This is the last item of the batch-level steps series.
#
# Because every step after that is dataset-scale and will not have easy access to scans,
# the goal of this step would be to store intermediary objects to remote storage for easy access.
#
# In that case, we want to store crops on R2:
# - `bucket/crops/barcode/page-filename/crop_*.jpg`
# - OR -
# - `bucket/crops/barcode/page-filename/crops.tar.gz`
#
# We should keep track of these crops and their properties in the database so they're easy to retrieve and analyze.

import click
from utils import get_s3_client
from loguru import logger
import os
from const import BUCKET_NAME, CPUS_LIMIT


@click.command("step05-store")
@click.option("--id-pipeline-batch", type=int, required=True)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT,
    help="Allows for limiting the number of CPU cores this command can use.",
)
def step05_store(
    id_pipeline_batch,
    cpus_limit,
):

    # get crops at original resolution

    # should store crop
    # should store crop dimensions
    pass


def save_to_storage(id_pipeline_batch, foldername: str, client, list_obj):
    boto3_client = get_s3_client("OUTPUT")  # For saving jsonl files
    try:
        for obj in list_obj:
            s3_key = f"{foldername}/{id_pipeline_batch}/{list_obj}"
            boto3_client.upload_file(obj, BUCKET_NAME, s3_key)
            logger.info(f"Uploaded {obj} to s3://{BUCKET_NAME}/{s3_key}")
    except Exception as e:
        logger.error(f"Error processing {obj}: {e}")
