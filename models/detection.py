import peewee
from playhouse.postgres_ext import *

from utils import get_db
from models import IBVolume, PipelineBatchItem


class Detection(peewee.Model):
    """
    `detection` table: Keeps track of visual elements detected by the visual elements detection model.
    """

    class Meta:
        table_name = "detection"
        database = get_db()

    id_detection = peewee.PrimaryKeyField()

    pipeline_batch_item = peewee.ForeignKeyField(
        model=PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
    )
    """ Allows for tracking back the volume of origin, as well as pipeline run/batch of origin."""

    scan_filename = peewee.CharField(index=True)

    bbox_xyxy = ArrayField(field_class=FloatField, dimensions=4, null=True)

    bbox_xywh = ArrayField(field_class=FloatField, dimensions=4, null=True)

    bbox_conf = FloatField(null=True)
