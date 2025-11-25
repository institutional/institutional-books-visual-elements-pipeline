from peewee import *
from playhouse.postgres_ext import *
from utils import get_db
from models import PipelineBatchItem, Detection


class DedupedEmbedding(Model):
    id = AutoField()
    embedding_id = IntegerField(index=True)  # original Embedding pk
    group_id = IntegerField(index=True)  # dedupe group
    pipeline_batch_item = ForeignKeyField(
        PipelineBatchItem, field="id_pipeline_batch_item", index=True
    )
    detection = ForeignKeyField(Detection, field="id_detection", index=True)
    scan_filename = CharField()
    embedding = ArrayField(FloatField, dimensions=1)
    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])

    class Meta:
        table_name = "deduped_embedding"
        database = get_db()
