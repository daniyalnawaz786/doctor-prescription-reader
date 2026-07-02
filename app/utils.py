"""Small shared helpers for the Streamlit app."""

from pathlib import Path

from src import config


def model_is_available():
    """True if the fine-tuned model files exist locally."""
    return (config.MODEL_DIR / "model.safetensors").exists() or (
        config.MODEL_DIR / "pytorch_model.bin"
    ).exists()


def list_sample_images():
    """Return paths of bundled sample images, if any."""
    sample_dir = Path(config.DATA_DIR) / "sample_images"
    if not sample_dir.exists():
        return []
    return sorted(
        p for p in sample_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
