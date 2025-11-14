import peewee
from playhouse.postgres_ext import *
from utils import get_db


class CaptionTokenLedger(peewee.Model):
    """
    Table to record total OpenAI tokens consumed per day.
    """

    class Meta:
        table_name = "caption_token_ledger"
        database = get_db()

    id_ledger = peewee.PrimaryKeyField()
    date = peewee.DateField(index=True)
    tokens_used = peewee.BigIntegerField(default=0)

    @classmethod
    def tokens_for_date(cls, d):
        rec, _ = cls.get_or_create(date=d)
        return rec.tokens_used
