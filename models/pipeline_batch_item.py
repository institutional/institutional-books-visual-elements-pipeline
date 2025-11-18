import io
import os
import gzip
import tarfile
from pathlib import Path
from dataclasses import dataclass, field

import peewee
from playhouse.postgres_ext import *
from loguru import logger

from utils import get_db, get_cache, get_s3_client
from models import IBVolume, PipelineBatch


@dataclass
class PipelineBatchItemData:
    images: dict[str, bytes] = field(default_factory=dict)
    texts: dict[str, bytes] = field(default_factory=dict)


class PipelineBatchItem(peewee.Model):
    """
    `pipeline_batch_item` table: Keeps track of individual items within a pipeline run batch.
    """

    class Meta:
        table_name = "pipeline_batch_item"
        database = get_db()

    id_pipeline_batch_item = peewee.PrimaryKeyField()

    pipeline_batch = peewee.ForeignKeyField(
        model=PipelineBatch,
        field="id_pipeline_batch",
        index=True,
    )

    ib_volume = peewee.ForeignKeyField(
        model=IBVolume,
        field="barcode",
        index=True,
    )

    def get_data(self) -> PipelineBatchItemData:
        """
        Retrieves data (images and OCR-extracted) text from the volume raw archive.
        Pull data from either remote storage or disk cache.

        NOTES:
        - Automatically adds items to disk cache.
        - Returns an empty PipelineBatchItemData if the archive could not be found
        """
        s3 = None
        id_pipeline_batch_item = self.id_pipeline_batch_item
        volume_barcode = self.ib_volume.barcode
        cache_key = f"PipelineBatchItem-{id_pipeline_batch_item}-{volume_barcode}"

        images_by_filename = {}
        texts_by_filename = {}

        gz_bytes = None

        #
        # Return images from cache if readily available
        #
        with get_cache() as cache:
            cached_data = cache.get(cache_key, None)

            if cached_data is not None:
                return cached_data

        #
        # Pull data from S3 otherwise
        #
        s3 = get_s3_client("GRIN_DATA")

        remote_key = f"{os.getenv('GRIN_DATA_RUN_NAME')}/{volume_barcode}.tar.gz"

        try:
            response = s3.get_object(
                Bucket=os.environ["GRIN_DATA_RAW_BUCKET"],
                Key=remote_key,
            )

            gz_bytes = response["Body"].read()
        except Exception as err:
            logger.warning(f"Raw archive for {volume_barcode} could not be found.")
            return PipelineBatchItemData()

        # Extract from tar.gz
        with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes), mode="rb") as gz:

            tar_bytes = gz.read()

            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
                for member in tar.getmembers():
                    filename = Path(member.name)

                    # Scans
                    if filename.suffix in [".jpg", ".tif", ".tiff", ".jp2"]:
                        with tar.extractfile(member) as fh:
                            image_bytes = fh.read()

                        # Normalize to string to prevent type mismatch
                        images_by_filename[str(filename.name)] = image_bytes

                    # OCR-extracted text
                    if filename.suffix == ".txt":
                        with tar.extractfile(member) as fh:
                            text_bytes = fh.read()

                        # Normalize to string to prevent type mismatch
                        texts_by_filename[str(filename.name)] = text_bytes.decode("utf-8")

        # Sort by filename
        images_by_filename = {k: images_by_filename[k] for k in sorted(images_by_filename)}
        texts_by_filename = {k: texts_by_filename[k] for k in sorted(texts_by_filename)}

        #
        # Create, cache and return container
        #
        container = PipelineBatchItemData(images=images_by_filename, texts=texts_by_filename)

        with get_cache() as cache:
            cache.set(cache_key, container)

        return container

    @property
    def data(self) -> PipelineBatchItemData:
        return self.get_data()
