from peewee import *
from playhouse.postgres_ext import *

from models import PipelineBatchItem, Detection, ImageHash


class DedupedHash(Model):
    """
    Deduplication groups for image hashes.
    """

    class Meta:
        table_name = "deduped_hash"
        database = ImageHash._meta.database

    id = AutoField()

    # backref: ImageHash.deduped_entries
    hash_id = ForeignKeyField(
        model=ImageHash,
        field="id_imagehash",
        index=True,
        on_delete="CASCADE",
        backref="deduped_entries",
    )

    group_id = IntegerField(index=True)  # dedupe group

    # backref: PipelineBatchItem.deduped_hashes
    pipeline_batch_item = ForeignKeyField(
        PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
        backref="deduped_hashes",
    )

    # backref: Detection.deduped_hashes
    detection = ForeignKeyField(
        Detection,
        field="id_detection",
        index=True,
        on_delete="CASCADE",
        backref="deduped_hashes",
    )

    image_hash = CharField(index=True)

    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])
