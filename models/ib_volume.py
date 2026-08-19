import peewee
from playhouse.postgres_ext import *

from utils import get_db


class IBVolume(peewee.Model):
    """
    `ib_volume` table: Inventory and metadata of volumes present in the Institutional Books dataset.
    """

    class Meta:
        table_name = "ib_volume"
        database = get_db()

    barcode = peewee.CharField(
        max_length=64,
        null=False,
        unique=True,
        index=True,
        primary_key=True,
    )

    metadata = JSONField(
        null=True,
        unique=False,
        index=False,
    )

    pull_date = DateTimeField(null=True)
