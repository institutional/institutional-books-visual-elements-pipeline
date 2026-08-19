import cv2
import numpy as np


def decode_image_bytes(image_bytes) -> np.ndarray:
    """
    Decodes image bytes and returns an ndarray
    """
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR_RGB)
