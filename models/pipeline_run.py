import os

import peewee
from playhouse.postgres_ext import *

from utils import get_db


class PipelineRun(peewee.Model):
    """
    `pipeline_run` table: Keeps track of pipeline runs.
    """

    class Meta:
        table_name = "pipeline_run"
        database = get_db()

    id_pipeline_run = peewee.PrimaryKeyField()

    items_total = peewee.IntegerField(null=False)

    items_per_batch = peewee.IntegerField(null=False)

    batches_total = peewee.IntegerField(null=False)

    created_date = DateTimeTZField(null=True)

    @property
    def batches(self) -> list:
        """
        Returns a sorted list of PipelineBatch instances from this run.
        """
        from models import PipelineBatch

        batches = []

        for pipeline_batch in (
            PipelineBatch.select()
            .where(PipelineBatch.pipeline_run == self.id_pipeline_run)
            .order_by(PipelineBatch.id_pipeline_batch)
            .iterator()
        ):
            batches.append(pipeline_batch)

        return batches
