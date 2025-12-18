from concurrent.futures import ThreadPoolExecutor, wait
from loguru import logger
import numpy as np
from typing import Iterable

from utils import decode_image_bytes
from models import Detection


def load_scans_for_detections(
    volume_barcode: str,
    detections: Iterable[Detection],
    image_bytes_by_filename: dict[str, bytes],
    max_workers: int,
) -> dict[str, np.ndarray]:
    """
    Decode only the scan images referenced by the given detections.

    Returns:
        dict[filename -> np.ndarray]

    NOTE: Assumes Detection model exists
    """
    used_filenames = {str(det.scan_filename) for det in detections}
    loaded_images: dict[str, np.ndarray] = {}

    if max_workers <= 1:
        for fn in used_filenames:
            if fn not in image_bytes_by_filename:
                logger.warning(
                    f"Missing image bytes for scan {volume_barcode}.{fn} - skipping this scan"
                )
                continue
            try:
                loaded_images[fn] = decode_image_bytes(image_bytes_by_filename[fn])
            except Exception:
                logger.warning(f"Could not decode scan {volume_barcode}.{fn}")
        return loaded_images

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(decode_image_bytes, image_bytes_by_filename[fn]): fn
            for fn in used_filenames
            if fn in image_bytes_by_filename
        }
        done, _ = wait(futures)
        for fut in done:
            fn = futures[fut]
            try:
                loaded_images[fn] = fut.result()
            except Exception:
                logger.warning(f"Could not decode scan {volume_barcode}.{fn}")

    return loaded_images
