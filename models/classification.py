from playhouse.postgres_ext import *

from utils import get_db
from models import PipelineBatchItem, Detection


class Classification(Model):
    """
    `classification` table: Stores image-level or crop-level class predictions made during the pipeline.
    Each record links to a Detection (i.e., a cropped region) belonging to a PipelineBatchItem.
    """

    class Meta:
        table_name = "classification"
        database = get_db()

    id_classification = PrimaryKeyField()

    # backref: PipelineBatchItem.classifications
    pipeline_batch_item = ForeignKeyField(
        model=PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
        backref="classifications",
    )

    # backref: Detection.classifications
    detection = ForeignKeyField(
        model=Detection,
        field="id_detection",
        index=True,
        backref="classifications",
        on_delete="CASCADE",
    )

    pred_idx = IntegerField()
    pred_class = CharField()
    pred_conf = FloatField()  # Confidence value for predicted class

    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])

    # Store all class probabilities as an array
    probs = ArrayField(field_class=FloatField, null=True)
