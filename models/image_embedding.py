from playhouse.postgres_ext import *
from pgvector.peewee import VectorField

from utils import get_db
from models import PipelineBatchItem, Detection


class ImageEmbedding(Model):
    """
    Stores a deduplication embedding for a detected/cropped region.
    Embedding vectors are 512-dim (pgvector), referencing both the Detection (crop)
    and PipelineBatchItem (volume).
    """

    class Meta:
        table_name = "image_embedding"
        database = get_db()

    id_embedding = PrimaryKeyField()

    # backref: PipelineBatchItem.embeddings
    pipeline_batch_item = ForeignKeyField(
        model=PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
        backref="embeddings",
    )

    # backref: Detection.embeddings
    detection = ForeignKeyField(
        model=Detection,
        field="id_detection",
        index=True,
        backref="embeddings",
        on_delete="CASCADE",
    )

    embedding = VectorField(dimensions=512, null=False)

    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])
