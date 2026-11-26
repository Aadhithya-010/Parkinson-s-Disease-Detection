"""Centralized configuration for the Parkinson's Disease Detection project."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_REPORTS_DIR = REPORTS_DIR / "eda"
EVAL_REPORTS_DIR = REPORTS_DIR / "evaluation"
GRADCAM_REPORTS_DIR = REPORTS_DIR / "gradcam"

# Dataset & split configuration
DEFAULT_KAGGLE_DATASET = "irfansheriff/parkinsons-brain-mri-dataset"
SPLITS = ("train", "validate", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, validate, test
CLASSES = ("healthy", "parkinson")  # 0 = healthy, 1 = parkinson

# Image & pipeline settings
IMAGE_SIZE = (224, 224)
CHANNELS = 3
BATCH_SIZE = 16
SEED = 42
USE_CLAHE = False  # Set True only after comparing intensity histograms in EDA

# Training hyperparameters
CNN_EPOCHS = 30
TRANSFER_HEAD_EPOCHS = 15
TRANSFER_FINETUNE_EPOCHS = 15
CNN_LEARNING_RATE = 1e-3
HEAD_LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-5
FINE_TUNE_UNFREEZE_LAST_N = 20

# Supported file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}


def ensure_project_dirs() -> None:
    """Create standard output directories if they do not exist."""
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        EDA_REPORTS_DIR,
        EVAL_REPORTS_DIR,
        GRADCAM_REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
