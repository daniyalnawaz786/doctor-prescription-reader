"""Thin wrapper around the model for the Streamlit app, with caching."""

_reader = None


def get_reader():
    """Lazily create and cache a single PrescriptionReader instance.

    The heavy import (torch/transformers via src.predict) happens here rather
    than at module load, so the frontend can render before the model/deps exist.
    """
    global _reader
    if _reader is None:
        from src.predict import PrescriptionReader

        _reader = PrescriptionReader()
    return _reader


def predict_medicine(image):
    """Predict the handwritten medicine name from a PIL image or path."""
    return get_reader().predict(image)
