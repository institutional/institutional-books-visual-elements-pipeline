import traceback
import math
from datetime import datetime, timezone

import click
from loguru import logger
from humanize import naturalsize, intcomma

from utils import get_cache, process_db_write_batch
from models import IBVolume, PipelineRun, PipelineBatch, PipelineBatchItem


@click.command("prepare")
@click.option(
    "--offset",
    type=int,
    required=False,
    default=None,
    help="Allows for creating a run on a subset of volumes. Volumes are ordered by barcode.",
)
@click.option(
    "--limit",
    type=int,
    required=False,
    default=None,
    help="Allows for creating a run on a subset of volumes. Volumes are ordered by barcode.",
)
@click.option(
    "--items-per-batch",
    type=click.IntRange(10, 1000),
    default=1000,
)
@click.option(
    "--append-mode",
    is_flag=True,
    help="If set, will create a run for volumes that are not part of any other run.",
)
def prepare(offset: int, limit: int, items_per_batch: int, append_mode: bool):
    """
    Creates a pipeline run and its batches.
    This command returns an identifier that can be then passed to `orchestraction execute` to launch a run.
    """
    items_total = IBVolume.select().offset(offset).limit(limit).count()
    batches_total = 0
    volume_batches = []
    volumes_exclusion_list = set()

    #
    # If `--append-mode` exclude volumes that were already processed
    #
    if append_mode:
        logger.info("Listing items to exclude from run (--append-mode)")

        volumes_exclusion_list = {
            row.ib_volume.barcode
            for row in PipelineBatchItem.select(PipelineBatchItem.ib_volume).distinct()
        }

        items_total = items_total - len(volumes_exclusion_list)

    if items_total < 1:
        logger.info("No items left to process.")
        click.get_current_context().exit(0)

    batches_total = int(math.ceil(items_total / items_per_batch))

    # Ask for confirmation
    logger.info(f"Total volumes: {intcomma(items_total)}")
    logger.info(f"Volumes per batch: {intcomma(items_per_batch)}")
    logger.info(f"Total batches: {intcomma(batches_total)}")

    if not click.confirm("Proceed?"):
        click.get_current_context().exit(0)

    #
    # Create new pipeline run
    #
    pipeline_run = PipelineRun(
        items_total=items_total,
        items_per_batch=items_per_batch,
        batches_total=batches_total,
        created_date=datetime.now(timezone.utc),
    )
    pipeline_run.save()

    logger.info(f"Pipeline run id: {pipeline_run.id_pipeline_run}")

    #
    # Create and save pipeline batches
    #
    logger.info(f"Creating batches for run {pipeline_run.id_pipeline_run} ...")

    # Split volumes list into batches of volumes
    volume_batches.append([])

    for volume in (
        IBVolume.select().offset(offset).limit(limit).order_by(IBVolume.barcode).iterator()
    ):
        volumes_batch = volume_batches[-1]

        # If --append-mode: skip volumes that are already part of a run
        if append_mode and volume.barcode in volumes_exclusion_list:
            continue

        if len(volumes_batch) >= items_per_batch:
            volume_batches.append([])
            volumes_batch = volume_batches[-1]

        volumes_batch.append(volume)

    # Create pipeline batches and pipeline batches items for each volumes batch
    entries_to_create = []

    for volumes_batch in volume_batches:
        pipeline_batch = PipelineBatch(
            pipeline_run=pipeline_run,
            created_date=datetime.now(timezone.utc),
        )
        pipeline_batch.save()

        for ib_volume in volumes_batch:
            pipeline_batch_item = PipelineBatchItem(
                pipeline_batch=pipeline_batch,
                ib_volume=ib_volume,
            )

            entries_to_create.append(pipeline_batch_item)

    process_db_write_batch(PipelineBatchItem, entries_to_create)

    logger.info(f"Pipeline run ready. --id-pipeline-run={pipeline_run.id_pipeline_run}")
