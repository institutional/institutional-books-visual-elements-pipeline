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

    def crop(self, scan_image):
        """
        Returns the crop defined by bbox_xyxy from the given scan image (an np.ndarray).
        """
        # should be able to call images directly (self.pipeline_batch_item.images) instead of passing by argument
        if self.bbox_xyxy is None:
            raise ValueError("bbox_xyxy is None for this detection.")
        x1, y1, x2, y2 = [int(round(v)) for v in self.bbox_xyxy]
        # Be defensive: clamp to image shape
        h, w = scan_image.shape[:2]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        return scan_image[y1:y2, x1:x2, :]
