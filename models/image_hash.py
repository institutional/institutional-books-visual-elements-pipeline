import peewee
from playhouse.postgres_ext import *

from utils import get_db
from models import PipelineBatchItem, Detection


class ImageHash(Model):
    """
    Stores an integer image hash per crop (for deduplication).
    """

    class Meta:
        table_name = "image_hash"
        database = get_db()

    id_imagehash = PrimaryKeyField()

    # backref: PipelineBatchItem.imagehashes
    pipeline_batch_item = ForeignKeyField(
        model=PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
        backref="imagehashes",
    )

    # backref: Detection.imagehashes
    detection = ForeignKeyField(
        model=Detection,
        field="id_detection",
        index=True,
        backref="imagehashes",
        on_delete="CASCADE",
    )

    image_hash = CharField(max_length=100, index=True)  # Hex string

    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])
