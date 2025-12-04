from peewee import *
from playhouse.postgres_ext import *
from utils import get_db
from models import PipelineBatchItem, Detection


class DedupedHash(Model):
    id = AutoField()
    hash_id = IntegerField(index=True)  # original ImageHash pk
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
    scan_filename = CharField()
    image_hash = CharField(index=True)
    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])

    class Meta:
        table_name = "deduped_hash"
        database = get_db()
