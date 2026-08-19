import peewee
from playhouse.postgres_ext import *

from utils import get_db
from models import PipelineRun


class PipelineBatch(peewee.Model):
    """
    `pipeline_batch` table: Keeps track of pipeline run batches.
    """

    class Meta:
        table_name = "pipeline_batch"
        database = get_db()

    id_pipeline_batch = peewee.PrimaryKeyField()

    pipeline_run = peewee.ForeignKeyField(
        model=PipelineRun,
        field="id_pipeline_run",
        index=True,
    )

    node_name = peewee.CharField(max_length=128, null=True, index=False)  # Assigned at run time

    created_date = DateTimeTZField(null=True)

    started_date = DateTimeTZField(null=True)

    ended_date = DateTimeTZField(null=True)

    has_crashed = BooleanField(default=False)

    def cache_data(self, max_workers: int = 1) -> bool:
        """
        Pulls and caches on disk data (images and text) from the volumes listed in the current batch.
        Will not pull data from remote storage again if images are already cached.

        Note:
        - `max_workers` determine the number of threads that can be used. Increases pressure on both the CPU and S3.
        """
        from models import PipelineBatchItem
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _get_data(pipeline_batch_item: PipelineBatchItem):
            data = pipeline_batch_item.get_data()
            del data
            return True

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            # Queue tasks
            for pipeline_batch_item in (
                PipelineBatchItem.select()
                .where(PipelineBatchItem.pipeline_batch == self.id_pipeline_batch)
                .order_by(PipelineBatchItem.id_pipeline_batch_item)
                .iterator()
            ):
                future = executor.submit(_get_data, pipeline_batch_item)
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                assert future.result()
                del future

        return True

    @property
    def items(self) -> list:
        """
        Returns a sorted list of PipelineBatchItems instances from this batch.
        """
        from models import PipelineBatchItem, IBVolume

        items = []

        for pipeline_batch_item in (
            PipelineBatchItem.select(PipelineBatchItem, IBVolume)
            .where(PipelineBatchItem.pipeline_batch == self.id_pipeline_batch)
            .join(IBVolume)
            .order_by(PipelineBatchItem.id_pipeline_batch_item)
            .iterator()
        ):
            items.append(pipeline_batch_item)

        return items
