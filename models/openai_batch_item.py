from peewee import *
from playhouse.postgres_ext import *
from utils import get_db
from models.pipeline_batch_item import PipelineBatchItem


class OpenAIBatchObject(Model):
    id = AutoField()

    # The batch item (volume or batch item id) this batch file is associated with
    pipeline_batch_item = ForeignKeyField(
        PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
        backref="openai_batches",
        on_delete="CASCADE",  # clean up if batch item is deleted
    )

    jsonl_file_name = CharField()
    s3_key = CharField()  # path in S3 bucket
    batch_id = CharField(index=True)  # OpenAI returned batch ID
    status = CharField()
    num_requests = IntegerField(default=0)
    submitted_at = DateTimeField(null=True)
    endpoint = CharField(null=True)
    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])

    class Meta:
        table_name = "openai_batch_object"
        database = get_db()
