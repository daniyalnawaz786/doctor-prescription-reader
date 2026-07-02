"""Central configuration: paths, model name, and hyperparameters."""

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Fine-tuned model output / load location
MODEL_DIR = MODELS_DIR / "trocr_prescription_model"

# ------------------------------------------------------------------
# Dataset (Doctor's Handwritten Prescription BD Dataset)
# ------------------------------------------------------------------
KAGGLE_DATASET = "mamun1113/doctors-handwritten-prescription-bd-dataset"

# CSV column names
IMAGE_COL = "IMAGE"
MEDICINE_COL = "MEDICINE_NAME"
GENERIC_COL = "GENERIC_NAME"

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
BASE_MODEL_NAME = "microsoft/trocr-base-handwritten"

# ------------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_TARGET_LENGTH = 32
