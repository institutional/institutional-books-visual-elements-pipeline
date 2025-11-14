import peewee
from playhouse.postgres_ext import *
from utils import get_db
from models import PipelineBatchItem, Detection


class ImageHash(peewee.Model):
    """
    Stores an integer image hash per crop (for deduplication).
    """

    class Meta:
        table_name = "imagehash"
        database = get_db()

    id_imagehash = peewee.PrimaryKeyField()

    pipeline_batch_item = peewee.ForeignKeyField(
        model=PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
    )

    detection = peewee.ForeignKeyField(
        model=Detection,
        field="id_detection",
        index=True,
        backref="imagehashes",
        on_delete="CASCADE",
    )

    scan_filename = peewee.CharField(index=True)

    image_hash = peewee.CharField(max_length=32, index=True)  # Hex string

    created = peewee.DateTimeField(constraints=[peewee.SQL("DEFAULT CURRENT_TIMESTAMP")])
