"""Word/region detection for a full prescription page.

Two backends:

* ``"numpy"`` — a dependency-free projection-profile segmenter (only needs
  numpy + Pillow). Rough, but runs anywhere and is great for previewing how a
  page gets carved into word crops.
* ``"doctr"`` — uses a pretrained docTR text detector (needs ``python-doctr``
  and torch). Much more robust on real, messy prescriptions.

Both return the same shape: a list of ``{"bbox": (x0, y0, x1, y1), "crop": PIL.Image}``
with pixel coordinates relative to the input image.
"""

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def _to_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, bytes)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _crop_boxes(pil_image, boxes):
    """Turn (x0, y0, x1, y1) pixel boxes into result dicts with crops."""
    results = []
    for (x0, y0, x1, y1) in boxes:
        crop = pil_image.crop((x0, y0, x1, y1))
        results.append({"bbox": (int(x0), int(y0), int(x1), int(y1)), "crop": crop})
    return results


# ----------------------------------------------------------------------
# Backend 1: pure-numpy projection profile (no extra deps)
# ----------------------------------------------------------------------
def _otsu_threshold(gray):
    """Compute an Otsu threshold for a uint8 grayscale array."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > max_var:
            max_var = between
            threshold = t
    return threshold


def _bands(mask_1d, min_gap, min_size):
    """Given a 1-D boolean 'has-ink' vector, return (start, end) bands.

    Small gaps (< min_gap) are bridged; bands smaller than min_size are dropped.
    """
    bands = []
    start = None
    gap = 0
    for i, on in enumerate(mask_1d):
        if on:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= min_gap:
                    bands.append((start, i - gap + 1))
                    start = None
                    gap = 0
    if start is not None:
        bands.append((start, len(mask_1d)))
    return [(s, e) for s, e in bands if e - s >= min_size]


def detect_words_numpy(image, min_word_width=12, min_line_height=8):
    """Projection-profile word segmentation. Returns list of crop dicts.

    Strategy: binarize (Otsu) -> split into text lines via horizontal
    projection -> split each line into words via vertical projection.
    """
    pil = _to_rgb(image)
    gray = np.asarray(pil.convert("L"))
    h, w = gray.shape

    threshold = _otsu_threshold(gray)
    # Ink is dark text on light background -> True where pixel is darker.
    ink = gray < threshold

    # --- horizontal projection -> line bands ---
    row_has_ink = ink.sum(axis=1) > (w * 0.005)
    line_gap = max(3, h // 100)
    line_bands = _bands(row_has_ink, min_gap=line_gap, min_size=min_line_height)

    boxes = []
    for (y0, y1) in line_bands:
        line_ink = ink[y0:y1, :]
        col_has_ink = line_ink.sum(axis=0) > 0
        # Gap between words is wider than gap between characters. Bias toward
        # splitting: over-segmenting is harmless (junk crops get filtered out),
        # while merged words match no single medicine.
        line_h = y1 - y0
        word_gap = max(3, line_h // 3)
        word_bands = _bands(col_has_ink, min_gap=word_gap, min_size=min_word_width)
        for (x0, x1) in word_bands:
            # tighten vertical extent to actual ink inside this word
            sub = ink[y0:y1, x0:x1]
            rows = np.where(sub.any(axis=1))[0]
            if rows.size == 0:
                continue
            ty0 = y0 + rows[0]
            ty1 = y0 + rows[-1] + 1
            # small padding
            pad = 2
            boxes.append(
                (
                    max(0, x0 - pad),
                    max(0, ty0 - pad),
                    min(w, x1 + pad),
                    min(h, ty1 + pad),
                )
            )

    return _crop_boxes(pil, boxes)


# ----------------------------------------------------------------------
# Backend 2: docTR pretrained detector (lazy import)
# ----------------------------------------------------------------------
_doctr_model = None


def _get_doctr(arch="db_resnet50"):
    global _doctr_model
    if _doctr_model is None:
        from doctr.models import detection_predictor  # lazy: heavy deps

        _doctr_model = detection_predictor(arch=arch, pretrained=True)
    return _doctr_model


def detect_words_doctr(image, arch="db_resnet50"):
    """Detect words with a pretrained docTR detector. Returns crop dicts."""
    pil = _to_rgb(image)
    arr = np.asarray(pil)
    h, w = arr.shape[:2]

    model = _get_doctr(arch)
    out = model([arr])[0]

    # docTR versions differ: dict with 'words' key, or a raw ndarray.
    preds = out["words"] if isinstance(out, dict) else out

    boxes = []
    for row in preds:
        xmin, ymin, xmax, ymax = row[:4]  # relative coords in [0, 1]
        boxes.append((xmin * w, ymin * h, xmax * w, ymax * h))
    return _crop_boxes(pil, boxes)


# ----------------------------------------------------------------------
# Public wrapper
# ----------------------------------------------------------------------
def doctr_available():
    try:
        import doctr  # noqa: F401

        return True
    except Exception:
        return False


class WordDetector:
    """Detect word regions on a full page, using the best available backend."""

    def __init__(self, backend="auto"):
        if backend == "auto":
            backend = "doctr" if doctr_available() else "numpy"
        self.backend = backend

    def detect(self, image):
        if self.backend == "doctr":
            return detect_words_doctr(image)
        return detect_words_numpy(image)
