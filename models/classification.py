import peewee
from playhouse.postgres_ext import *

from utils import get_db
from models import PipelineBatchItem, Detection


class Classification(peewee.Model):
    """
    `classification` table: Stores image-level or crop-level class predictions made during the pipeline.
    Each record links to a Detection (i.e., a cropped region) belonging to a PipelineBatchItem.
    """

    class Meta:
        table_name = "classification"
        database = get_db()

    id_classification = peewee.PrimaryKeyField()

    # Reference to the parent batch item (volume)
    pipeline_batch_item = peewee.ForeignKeyField(
        model=PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
    )

    # Reference to the specific Detection this classifies
    detection = peewee.ForeignKeyField(
        model=Detection, field="id_detection", index=True, backref="classifications"
    )

    scan_filename = peewee.CharField(index=True)
    bbox_xyxy = ArrayField(field_class=peewee.FloatField, dimensions=4, null=True)
    pred_idx = peewee.IntegerField()
    pred_class = peewee.CharField()
    pred_conf = peewee.FloatField()  # Confidence value for predicted class

    created = peewee.DateTimeField(constraints=[peewee.SQL("DEFAULT CURRENT_TIMESTAMP")])

    # Store all class probabilities as an array
    probs = ArrayField(field_class=peewee.FloatField, null=True)

    # Other stuff?
