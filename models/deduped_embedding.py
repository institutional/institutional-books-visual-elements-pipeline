from peewee import *
from playhouse.postgres_ext import *

from models import PipelineBatchItem, Detection, ImageEmbedding


class DedupedEmbedding(Model):
    """
    Deduplication groups for embeddings.
    """

    class Meta:
        table_name = "deduped_embedding"
        database = ImageEmbedding._meta.database

    id = AutoField()

    # backref: ImageEmbedding.deduped_entries
    embedding_id = ForeignKeyField(
        model=ImageEmbedding,
        field="id_embedding",
        index=True,
        on_delete="CASCADE",
        backref="deduped_entries",
    )

    group_id = IntegerField(index=True)  # dedupe group

    # backref: PipelineBatchItem.deduped_embeddings
    pipeline_batch_item = ForeignKeyField(
        PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
        backref="deduped_embeddings",
    )

    # backref: Detection.deduped_embeddings
    detection = ForeignKeyField(
        Detection,
        field="id_detection",
        index=True,
        on_delete="CASCADE",
        backref="deduped_embeddings",
    )

    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])
