import traceback
import os
from datetime import datetime, timezone, timedelta

import click
from loguru import logger

from utils import get_torch_devices
from models import PipelineRun, PipelineBatch, PipelineBatchItem
import commands.steps
from const import (
    CPUS_LIMIT,
    MAX_DB_CONCURRENCY,
    MAX_S3_CONCURRENCY,
    CUDA_GPUS,
    NODE_NAME,
    PIPELINE_BATCH_TIMEOUT_SECONDS,
)


@click.command("execute")
@click.option(
    "--id-pipeline-run",
    type=int,
    help="Identifier of the pipeline run to launch.",
)
@click.option(
    "--force-id-pipeline-batch",
    type=int,
    required=False,
    default=None,
    help="If specificed, only focuses on the specified batcs.",
)
@click.option(
    "--batch-processing-only",
    is_flag=True,
    help="If set, will only execute batch-level steps and skip dataset-wide steps.",
)
@click.option(
    "--ignore-locks",
    is_flag=True,
    help="If set, will ignore batch locks (batch running on another machine).",
)
def execute(
    id_pipeline_run: int,
    force_id_pipeline_batch: int | None,
    batch_processing_only: bool,
    ignore_locks: bool,
):
    """
    Executes a pipeline run.
    Runs all steps against all batches unless instructed otherwise.
    """
    pipeline_run: PipelineRun = None
    pipeline_run_start = datetime.now(timezone.utc)
    has_crashed = False

    batches_processed = 0
    batches_skipped = 0
    batches_crashed = 0

    #
    # Check: Pipeline run status
    #
    try:
        pipeline_run = PipelineRun.get(id_pipeline_run=id_pipeline_run)
    except:
        logger.error(f"Pipeline run {id_pipeline_run} does not exist.")
        click.get_current_context().exit(1)

    #
    # Check: CUDA-capable GPU availability
    #
    if not CUDA_GPUS:
        logger.debug(f"Available torch devices: {",".join(get_torch_devices())}")
        logger.error("The pipeline needs access to at least one CUDA-compatible GPU.")
        click.get_current_context().exit(1)

    #
    # Start of run
    #
    log_progress(pipeline_run=pipeline_run)

    #
    # BATCH-LEVEL: steps that can be run on a subset of records
    # NOTE: We use this cache mechanism to run all steps on subset of pre-cached images
    #
    for pipeline_batch in pipeline_run.batches:
        id_pipeline_batch: int = pipeline_batch.id_pipeline_batch
        pipeline_batch = PipelineBatch.get(id_pipeline_batch=id_pipeline_batch)  # Force refresh

        # Check if batch has not been excluded
        if force_id_pipeline_batch and id_pipeline_batch != force_id_pipeline_batch:
            logger.warning(
                f"RUN #{id_pipeline_run} BATCH#{id_pipeline_batch} was excluded via (`--force-id-pipeline-batch`)"
            )
            continue

        # Check whether batch is locked
        if not ignore_locks and not check_pipeline_batch_availability(pipeline_batch):
            logger.warning(
                f"RUN #{id_pipeline_run} BATCH#{id_pipeline_batch} is locked or complete. Skipping."
            )
            batches_skipped += 1
            continue

        if ignore_locks:
            logger.warning("Potential pipeline batch-level lock was ignored (--ignore-locks)")

        # Run steps
        try:
            # Lock pipeline batch
            pipeline_batch.node_name = NODE_NAME
            pipeline_batch.started_date = datetime.now(timezone.utc)
            pipeline_batch.ended = None
            pipeline_batch.has_crashed = False
            pipeline_batch.save()

            log_progress(
                step_name="",
                pipeline_run=pipeline_run,
                pipeline_batch=pipeline_batch,
            )

            # Step 0: Cache batch data
            # NOTE: Bandwidth is the bottleneck here. Will be much faster on cloud Node.
            has_crashed = not execute_batch_level_step(
                step_fn=pipeline_batch.cache_data,
                step_fn_kwargs={"max_workers": min(MAX_S3_CONCURRENCY, CPUS_LIMIT)},
                pipeline_run=pipeline_run,
                pipeline_batch=pipeline_batch,
            )

            # Step 1: Extraction
            if not has_crashed:
                has_crashed = not execute_batch_level_step(
                    step_fn=commands.steps.step01_detect,
                    step_fn_kwargs={"id_pipeline_batch": id_pipeline_batch},
                    pipeline_run=pipeline_run,
                    pipeline_batch=pipeline_batch,
                )

            # Step 2: Classification
            if not has_crashed:
                pass

            # Etc ...

        #
        # Exceptions catch-all
        #
        except Exception as err:
            has_crashed = True
            logger.debug(traceback.format_exc())
        #
        # Always unlock pipeline batch
        #
        finally:
            pipeline_batch.has_crashed = has_crashed
            pipeline_batch.ended_date = datetime.now(timezone.utc)
            pipeline_batch.save()

            log_progress(
                pipeline_run=pipeline_run,
                pipeline_batch=pipeline_batch,
                time=pipeline_batch.ended_date - pipeline_batch.started_date,
            )

            if not has_crashed:
                batches_processed += 1
            else:
                batches_crashed += 1
                logger.error("Pipeline has crashed!")
                break

    #
    # RUN-LEVEL: steps that need to run on all records
    #
    if batch_processing_only:
        logger.warning("Dataset-wide steps were skipped (--batch-processing-only)")
    else:
        pass
    #
    # End of run
    #
    log_progress(
        step_name="",
        pipeline_run=pipeline_run,
        time=(datetime.now(timezone.utc) - pipeline_run_start),
    )

    logger.info(f"Batches processed: {batches_processed}")
    logger.info(f"Batches crashed: {batches_crashed}")
    logger.info(f"Batches skipped: {batches_skipped}")


def check_pipeline_batch_availability(pipeline_batch: PipelineBatch) -> bool:
    """
    Assesses whether a given batch is available to be processed.

    Returns True (meaning: batch can be processed) if:
    - Batch was never processed
    - Batch has crashed
    - Batch has started but has timed out
    """
    # Batch has no date
    if not pipeline_batch.started_date and not pipeline_batch.ended_date:
        return True

    # Batch has crashed
    if pipeline_batch.has_crashed:
        return True

    # Batch is running and hasn't timed out yet
    if pipeline_batch.started_date:
        batch_lifetime_seconds = (datetime.now(timezone.utc) - pipeline_batch.started_date).seconds

        if batch_lifetime_seconds > PIPELINE_BATCH_TIMEOUT_SECONDS:
            return True

    # If we reached that point, the batch is not available
    return False


def execute_batch_level_step(
    step_fn,
    step_fn_kwargs: dict,
    pipeline_run: PipelineRun,
    pipeline_batch: PipelineBatch,
) -> bool:
    """
    Executes a batch-level step.
    Returns False if step exited with non-zero code.

    NOTE:
    - Runs GPU cache flushes and GC calls after each step
    """
    start = datetime.now(timezone.utc)
    end = None
    step_fn_name = f"{step_fn}"
    success = True
    ctx = click.get_current_context()

    # Try to pretty print function name, depending on its nature
    try:
        step_fn_name = step_fn.name.split(".")[-1]
    except:
        step_fn_name = step_fn.__name__.split(".")[-1]

    log_progress(f"{step_fn_name}", pipeline_run, pipeline_batch)

    # Run step
    try:
        ctx.invoke(step_fn, **step_fn_kwargs)
    except KeyboardInterrupt as err:
        logger.error(f"{step_fn_name} was interrupted.")
        success = False
    except click.exceptions.Exit as err:
        if err.exit_code != 0:
            logger.error(f"{step_fn_name} exited with code {err.exit_code}.")
            success = False

    # Stats
    end = datetime.now(timezone.utc)
    log_progress(f"{step_fn_name}", pipeline_run, pipeline_batch, time=(end - start))

    return success


def log_progress(
    step_name: str = "",
    pipeline_run: PipelineRun = None,
    pipeline_batch: PipelineBatch = None,
    time: timedelta = None,
):
    """
    Prepares and writes a log line for a specific step.
    If `time` is provided, generates an end-of-step log.
    """
    message = "START " if not time else "END "

    if step_name:
        message += f"{step_name} "

    if pipeline_run:
        message += f"RUN#{pipeline_run.id_pipeline_run} "

    if pipeline_batch:
        message += f"BATCH#{pipeline_batch.id_pipeline_batch} "

    if pipeline_batch and pipeline_batch.node_name:
        message += f"NODE#{pipeline_batch.node_name} "

    if time:
        message += f"- {time} "

    logger.info(message)
