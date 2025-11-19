from peewee import *
from playhouse.postgres_ext import *

from utils import get_db
from models import PipelineBatchItem, Detection


class Caption(Model):
    """
    caption table: Stores generated captions for an image or a detection crop.
    Each record links to a Detection (a cropped region) belonging to a PipelineBatchItem.
    """

    class Meta:
        table_name = "caption"
        database = get_db()

    id_caption = PrimaryKeyField()

    # Reference to the parent batch item (volume)
    pipeline_batch_item = ForeignKeyField(
        model=PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
    )

    # Reference to the specific Detection this caption describes
    detection = ForeignKeyField(
        model=Detection,
        field="id_detection",
        index=True,
        backref="captions",
        on_delete="CASCADE",
    )

    scan_filename = CharField(index=True)

    # Bounding box associated with caption (optional)
    bbox_xyxy = ArrayField(field_class=FloatField, dimensions=4, null=True)

    # The generated caption text
    caption = TextField()

    # Confidence score for the caption (if your captioner provides one)
    caption_conf = FloatField(null=True)

    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])

    # TODO sperate out the caption logprobs, etc from the caption text
