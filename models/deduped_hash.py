from peewee import *
from playhouse.postgres_ext import *

from models import PipelineBatchItem, Detection, ImageHash


class DedupedHash(Model):

    id = AutoField()

    hash_id = ForeignKeyField(
        model=ImageHash,
        field="id_imagehash",
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

    image_hash = CharField(index=True)

    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])

    class Meta:
        table_name = "deduped_hash"
        database = ImageHash._meta.database
