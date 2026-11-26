"""Run single-image inference and save a Grad-CAM overlay using a trained .keras model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import CLASSES, GRADCAM_REPORTS_DIR, IMAGE_SIZE, MODELS_DIR, USE_CLAHE
from src.data.pipeline import get_model_preprocess_fn, preprocess_image_opencv
from src.models.gradcam import compute_gradcam_heatmap, overlay_gradcam_opencv


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict single image and save Grad-CAM visualization.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--arch", required=True, choices=["cnn", "vgg16", "resnet50"])
    parser.add_argument("--model-path", default=None, help="Optional explicit path to .keras file.")
    parser.add_argument("--clahe", action="store_true", default=USE_CLAHE)
    args = parser.parse_args()

    model_filename = "best_cnn.keras" if args.arch == "cnn" else f"best_{args.arch}.keras"
    model_path = Path(args.model_path) if args.model_path else (MODELS_DIR / model_filename)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    preprocess_fn = get_model_preprocess_fn(args.arch)

    rgb_float = preprocess_image_opencv(args.image, target_size=IMAGE_SIZE, use_clahe=args.clahe)
    rgb_uint8 = np.clip(rgb_float, 0, 255).astype(np.uint8)
    batch = preprocess_fn(tf.expand_dims(rgb_float, axis=0))

    heatmap, prob = compute_gradcam_heatmap(model, batch)
    overlay = overlay_gradcam_opencv(rgb_uint8, heatmap)
    pred_label = CLASSES[1] if prob >= 0.5 else CLASSES[0]

    GRADCAM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GRADCAM_REPORTS_DIR / f"single_gradcam_{Path(args.image).stem}_{args.arch}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Image: {args.image}")
    print(f"Predicted class: {pred_label} (P(parkinson) = {prob:.4f})")
    print(f"Saved Grad-CAM overlay: {out_path}")


if __name__ == "__main__":
    main()
