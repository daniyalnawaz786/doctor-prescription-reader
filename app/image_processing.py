"""Image preprocessing helpers for uploaded prescription images."""

import numpy as np
from PIL import Image


def to_pil(image):
    """Normalize any input (path, bytes, ndarray, PIL) into an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, bytes)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def deskew_and_denoise(pil_image):
    """Light cleanup: grayscale -> denoise -> adaptive threshold -> back to RGB.

    Kept intentionally simple; TrOCR is fairly robust to raw crops, so this is
    an optional enhancement path for noisy uploads.
    """
    import cv2  # lazy: only needed if this cleanup path is actually used

    img = np.array(pil_image.convert("L"))
    img = cv2.fastNlMeansDenoising(img, h=10)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)
