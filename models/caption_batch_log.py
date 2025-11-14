import peewee
from playhouse.postgres_ext import *
from utils import get_db


class CaptionBatchLog(peewee.Model):
    """
    Table to record each OpenAI batch job for captioning.
    """

    class Meta:
        table_name = "caption_batch_log"
        database = get_db()

    id_log = peewee.PrimaryKeyField()
    batch_file = peewee.CharField()
    openai_batch_id = peewee.CharField(null=True)  # assigned when returned
    submitted = peewee.DateTimeField(index=True)
    n_requests = peewee.IntegerField()
    pipeline_batch = peewee.IntegerField(index=True)
    status = peewee.CharField(default="created")  # e.g. created, submitted, completed, failed
