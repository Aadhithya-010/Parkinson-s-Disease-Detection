# Parkinson's Disease Detection Using Deep Learning

Experimental deep-learning image classification pipeline comparing a custom CNN from scratch against transfer-learning models (**VGG16** and **ResNet50**) with **OpenCV** preprocessing, **tf.data** augmentation, clinical-style evaluation metrics, and **Grad-CAM** explainability.

> **Disclaimer:** This project is an experimental academic and engineering implementation. It is **not** a clinically validated diagnostic system and must not be used for medical decision-making.

## Project Structure

```text
parkinsons-disease-detection/
├── configs/
│   └── config.py                  # Paths, image size (224x224), split ratios, hyperparameters
├── notebooks/
│   └── parkinsons_walkthrough.ipynb  # Interactive step-by-step notebook (VS Code / Colab)
├── scripts/
│   ├── 01_download_and_audit.py   # Kaggle download, integrity & MD5 deduplication, clean splits
│   ├── 02_run_eda.py              # Class balance, dimensions, sample grid, intensity checks
│   ├── 03_train_and_evaluate.py   # Trains CustomCNN, VGG16, ResNet50 + plots + Grad-CAM
│   └── 04_predict_single.py       # Single-image inference + Grad-CAM overlay
├── src/
│   ├── data/
│   │   ├── audit.py               # Corrupt checks, DICOM support, MD5 dedup, leakage-safe split
│   │   ├── eda.py                 # EDA summary & visualization outputs
│   │   └── pipeline.py            # OpenCV preprocessing, augmentation, class weights, tf.data
│   ├── models/
│   │   ├── cnn.py                 # 3-block Custom CNN (BatchNorm + GAP + Dropout + Sigmoid)
│   │   ├── transfer.py            # VGG16 & ResNet50 builders + safe two-phase fine-tuning
│   │   └── gradcam.py             # Grad-CAM heatmaps & OpenCV overlays
│   └── evaluation/
│       └── metrics.py             # Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, CM, errors
├── requirements.txt
└── STUDY_GUIDE.md                 # Concept-by-concept learning and interview guide
```

## Setup (macOS Apple Silicon / VS Code)

TensorFlow requires Python 3.9–3.12 (not Python 3.13). Create a dedicated Conda environment:

```bash
conda create -n parkinsons python=3.11 -y
conda activate parkinsons
pip install -r requirements.txt
```

## Quickstart Pipeline

```bash
# 1. Download dataset, verify readability, remove MD5 duplicates, create train/validate/test splits
python scripts/01_download_and_audit.py --dataset irfansheriff/parkinsons-brain-mri-dataset

# 2. Generate EDA charts and statistics in reports/eda/
python scripts/02_run_eda.py

# 3. Train CustomCNN, VGG16, and ResNet50, evaluate on test set, and generate Grad-CAMs
python scripts/03_train_and_evaluate.py

# 4. Run single-image prediction + Grad-CAM overlay
python scripts/04_predict_single.py --image data/processed/test/parkinson/parkinson_0001.png --arch resnet50
```

## Methodology Summary

- **Data Integrity & Leakage Prevention:** Verifies image/DICOM readability, converts DICOMs safely if present, removes exact MD5 duplicates prior to splitting, and uses subject-grouped splitting when `PatientID` metadata is available.
- **OpenCV Preprocessing:** Aspect-ratio-preserving resize (`224x224`) with border-matched padding and optional LAB-space CLAHE contrast enhancement.
- **Architectures Compared:**
  - `CustomCNN` (Conv2D `32->64->128`, BatchNorm, ReLU, MaxPool, GlobalAveragePooling2D, Dropout `0.5`, Sigmoid)
  - `VGG16` (ImageNet backbone, Phase 1 frozen head @ `1e-3`, Phase 2 top-layer fine-tuning @ `1e-5`)
  - `ResNet50` (ImageNet backbone, Phase 1 frozen head @ `1e-3`, Phase 2 top-layer fine-tuning @ `1e-5` with `BatchNormalization` layers kept frozen)
- **Evaluation & Explainability:** Unshuffled test inference reporting Accuracy, Precision, Recall (Sensitivity), F1-score, Specificity, ROC-AUC, confusion matrices, error-analysis grids, and Grad-CAM heatmaps.
