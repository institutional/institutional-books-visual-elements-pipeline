from typing import Iterable
import numpy as np
from loguru import logger

from models import Detection


def build_detection_crops(
    volume_barcode: str,
    detections: Iterable[Detection],
    loaded_images: dict[str, np.ndarray],
    with_filename: bool = False,
):
    """
    Build crops for each detection given pre-decoded scans.

    Returns:
        list[(Detection, np.ndarray)] or list[(Detection, np.ndarray, filename)]

    NOTE: Assumes Detection model exists
    """
    records = []
    failed = 0
    for det in detections:
        fn = str(det.scan_filename)
        scan_img = loaded_images.get(fn)
        if scan_img is None:
            failed += 1
            continue
        try:
            crop = det.crop(scan_img)
            if with_filename:
                records.append((det, crop, fn))
            else:
                records.append((det, crop))
        except Exception:
            logger.warning(f"Could not crop detection in {volume_barcode}.{fn}; skipping")
            failed += 1

    return records, failed
