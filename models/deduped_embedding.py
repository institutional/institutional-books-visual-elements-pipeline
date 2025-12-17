from peewee import *
from playhouse.postgres_ext import *

from models import PipelineBatchItem, Detection, ImageEmbedding


class DedupedEmbedding(Model):

    id = AutoField()

    embedding_id = ForeignKeyField(
        model=ImageEmbedding,
        field="id_embedding",
        index=True,
        on_delete="CASCADE",
    )

    group_id = IntegerField(index=True)  # dedupe group

    pipeline_batch_item = ForeignKeyField(
        PipelineBatchItem, field="id_pipeline_batch_item", index=True
    )
    detection = ForeignKeyField(
        Detection,
        field="id_detection",
        index=True,
        on_delete="CASCADE",
    )

    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])

    class Meta:
        table_name = "deduped_embedding"
        database = ImageEmbedding._meta.database
