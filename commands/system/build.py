import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from loguru import logger
from datasets import load_dataset, Dataset
import numpy as np

from utils import process_db_write_batch
from models import IBVolume
from const import DEFAULT_DB_BATCH_SIZE, CPUS_LIMIT, MAX_DB_CONCURRENCY, IB_10_METADATA_DATASET_REPO


@click.command("build")
@click.option(
    "--db-batch-size",
    type=int,
    required=False,
    default=DEFAULT_DB_BATCH_SIZE,
    help="Determines the frequency at which records will be written to the database (every X requests).",
)
@click.option(
    "--max-workers",
    type=click.IntRange(1, min(MAX_DB_CONCURRENCY, CPUS_LIMIT)),
    default=min(MAX_DB_CONCURRENCY, CPUS_LIMIT),
    help="Determines how many items can be processed in parallel. Increases pressure on the database.",
)
def build(db_batch_size: int, max_workers: int):
    """
    Prepares the pipeline by indexing the list of volumes that need to be processed from the metadata-only version of the IB 1.0 dataset.
    Stores the barcode and metadata of each volume in the database.
    """
    logger.info(f"Pulling latest version of {IB_10_METADATA_DATASET_REPO} ...")

    dataset = load_dataset(IB_10_METADATA_DATASET_REPO, split="train")

    total_rows = len(dataset)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Create and send batches of row_ids to `process_batch()`
        for row_ids in np.array_split(np.arange(total_rows), max_workers):
            future = executor.submit(process_batch, dataset, row_ids.tolist(), db_batch_size)
            futures.append(future)

        # Execute and get results
        for future in as_completed(futures):
            try:
                assert future.result()
            except Exception as err:
                logger.debug(traceback.print_exc())
                logger.error("An error occured while indexing the input dataset. Interrupting.")
                executor.shutdown(wait=False, cancel_futures=True)
                click.get_current_context().exit(1)


def process_batch(dataset: Dataset, row_ids: list[int], db_batch_size: int) -> bool:
    """
    Processes a batch of row ids.
    Creates / updates `IBVolume` records with data from the Institutional Books 1.0 dataset.
    """
    entries_to_create = []
    entries_to_update = []
    fields_to_update = list(IBVolume._meta.fields.keys())
    fields_to_update.pop(fields_to_update.index("barcode"))  # We don't want to update the barcode

    for i, row_id in enumerate(row_ids):
        row = dataset[row_id]
        barcode = row["barcode_src"]
        volume: IBVolume = None
        exists = False

        # Update record if it exists, create it otherwise
        try:
            volume = IBVolume.get(barcode=barcode)
            assert volume
            exists = True
        except Exception as err:
            # Does not exist
            pass

        volume = volume if exists else IBVolume(barcode=barcode)
        volume.pull_date = datetime.now(timezone.utc)
        volume.metadata = dict(row)

        if exists:
            entries_to_update.append(volume)
            logger.info(f"ib_volume record for {barcode} updated")
        else:
            entries_to_create.append(volume)
            logger.info(f"ib_volume record for {barcode} created")

        # Run batch of DB writes
        batched_updated = len(entries_to_create) + len(entries_to_update)

        if batched_updated >= db_batch_size or i + 1 >= len(row_ids):
            logger.info(f"Updating database (batch of {db_batch_size} records) ...")

            process_db_write_batch(
                IBVolume,
                entries_to_create=entries_to_create,
                entries_to_update=entries_to_update,
                fields_to_update=fields_to_update,
            )

    return True
