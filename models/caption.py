from peewee import *
from playhouse.postgres_ext import *

from utils import get_db
from models import PipelineBatchItem, Detection


class Caption(Model):
    """
    `caption` table: Stores generated captions for an image or a detection crop.
    Each record links to a Detection (a cropped region) belonging to a PipelineBatchItem.
    """

    class Meta:
        table_name = "caption"
        database = get_db()

    id_caption = PrimaryKeyField()

    # backref: PipelineBatchItem.captions
    pipeline_batch_item = ForeignKeyField(
        model=PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
        backref="captions",
    )

    # backref: Detection.captions
    detection = ForeignKeyField(
        model=Detection,
        field="id_detection",
        index=True,
        backref="captions",
        on_delete="CASCADE",
    )

    text = TextField()
    lang = CharField(max_length=50)

    # Store the logprobs from the model response as JSON
    logprobs = BinaryJSONField(null=True, index=False)

    # Backfilled columns (computed from text/logprobs after initial caption creation)
    lang_detected = CharField(max_length=10, null=True)
    linear_prob = FloatField(null=True)
    thesaurus_matches = BinaryJSONField(null=True)

    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])
