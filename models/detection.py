import peewee
from playhouse.postgres_ext import *
import numpy as np

from utils import get_db
from models import PipelineBatchItem


class Detection(Model):
    """
    `detection` table: Keeps track of visual elements detected by the visual elements detection model.
    """

    class Meta:
        table_name = "detection"
        database = get_db()

    id_detection = PrimaryKeyField()

    # backref: PipelineBatchItem.detections
    pipeline_batch_item = ForeignKeyField(
        model=PipelineBatchItem,
        field="id_pipeline_batch_item",
        index=True,
        backref="detections",
        on_delete="CASCADE",
    )
    """ Allows for tracking back the volume of origin, as well as pipeline run/batch of origin."""

    scan_filename = CharField(index=True)

    bbox_xyxy = ArrayField(field_class=FloatField, dimensions=4, null=True)
    bbox_xywh = ArrayField(field_class=FloatField, dimensions=4, null=True)
    bbox_conf = FloatField(null=True)

    def crop(self, scan_image: np.ndarray):
        """
        Returns the crop defined by bbox_xyxy from the given scan image (an np.ndarray).
        """
        if self.bbox_xyxy is None:
            raise ValueError("bbox_xyxy is None for this detection.")
        x1, y1, x2, y2 = [int(round(v)) for v in self.bbox_xyxy]
        h, w = scan_image.shape[:2]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        return scan_image[y1:y2, x1:x2, :]
