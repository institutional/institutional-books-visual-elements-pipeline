import peewee
from playhouse.postgres_ext import *
from pgvector.peewee import VectorField
from utils import get_db
from models import PipelineBatchItem, Detection


class Embedding(peewee.Model):
    """
    Stores a deduplication embedding for a detected/cropped region.
    Embedding vectors are 512-dim (pgvector), referencing both the Detection (crop) and PipelineBatchItem (volume).
    """

    class Meta:
        table_name = "embedding"
        database = get_db()

    id_embedding = peewee.PrimaryKeyField()

    pipeline_batch_item = peewee.ForeignKeyField(
        model=PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
    )

    detection = peewee.ForeignKeyField(
        model=Detection,
        field="id_detection",
        index=True,
        backref="embeddings",
        on_delete="CASCADE",
    )

    # Scan file that produced the embedding (e.g., 00000123.jp2)
    scan_filename = peewee.CharField(index=True)

    # The actual vector!
    embedding = VectorField(dimensions=512, null=False)

    created = peewee.DateTimeField(constraints=[peewee.SQL("DEFAULT CURRENT_TIMESTAMP")])
