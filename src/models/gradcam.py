"""Grad-CAM explainability and OpenCV heatmap overlays for CNN, VGG16, and ResNet50."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from configs.config import CLASSES, GRADCAM_REPORTS_DIR, IMAGE_SIZE
from src.data.pipeline import get_model_preprocess_fn, preprocess_image_opencv


def infer_last_conv_layer_name(model: tf.keras.Model) -> str:
    """Detect the target last convolutional layer name from model name or layers."""
    name = model.name.lower()
    if "vgg16" in name:
        return "block5_conv3"
    if "resnet50" in name:
        return "conv5_block3_out"
    return "last_conv"


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    preprocessed_batch: np.ndarray | tf.Tensor,
    last_conv_layer_name: str | None = None,
) -> tuple[np.ndarray, float]:
    """Compute a normalized [0, 1] Grad-CAM heatmap for the Parkinson's output neuron."""
    conv_name = last_conv_layer_name or infer_last_conv_layer_name(model)
    x_tensor = tf.cast(preprocessed_batch, tf.float32)

    nested_backbone = next(
        (layer for layer in model.layers if isinstance(layer, tf.keras.Model)),
        None,
    )

    if nested_backbone is not None:
        conv_extractor = tf.keras.Model(
            inputs=nested_backbone.inputs,
            outputs=[nested_backbone.get_layer(conv_name).output, nested_backbone.output],
        )
        head_layers = [
            layer
            for layer in model.layers
            if layer is not nested_backbone and not isinstance(layer, tf.keras.layers.InputLayer)
        ]
        with tf.GradientTape() as tape:
            conv_outputs, features = conv_extractor(x_tensor, training=False)
            tape.watch(conv_outputs)
            x = features
            for layer in head_layers:
                x = layer(x, training=False)
            preds = x
            target_score = preds[:, 0]
    else:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(conv_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(x_tensor, training=False)
            tape.watch(conv_outputs)
            target_score = preds[:, 0]

    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        raise RuntimeError(f"Failed to compute Grad-CAM gradients for layer '{conv_name}'.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_map = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_map * pooled_grads, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0.0)
    max_val = float(heatmap.max())
    if max_val > 1e-8:
        heatmap = heatmap / max_val
    else:
        heatmap = np.zeros_like(heatmap)

    prob_parkinson = float(preds.numpy()[0, 0])
    return heatmap, prob_parkinson


def overlay_gradcam_opencv(
    rgb_uint8: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    """Resize Grad-CAM heatmap with OpenCV and blend onto the original RGB image."""
    h, w = rgb_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(np.clip(heatmap_resized * 255.0, 0, 255))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_uint8.astype(np.uint8), 1.0 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def save_gradcam_grid(
    model: tf.keras.Model,
    arch: str,
    test_df: pd.DataFrame,
    last_conv_layer_name: str,
    output_dir: Path = GRADCAM_REPORTS_DIR,
    max_images: int = 8,
) -> Path:
    """Generate and save a side-by-side Grad-CAM visualization grid on test samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_fn = get_model_preprocess_fn(arch)

    # Pick balanced examples across classes when available
    half = max(1, max_images // 2)
    chosen_frames = []
    for cls in CLASSES:
        sub = test_df[test_df["label_name"] == cls]
        if not sub.empty:
            chosen_frames.append(sub.iloc[:half])
    sample_df = pd.concat(chosen_frames, ignore_index=True)

    n = len(sample_df)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.6, 5.5))
    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx, row in sample_df.iterrows():
        raw_rgb_float = preprocess_image_opencv(row["path"], target_size=IMAGE_SIZE)
        raw_rgb_uint8 = np.clip(raw_rgb_float, 0, 255).astype(np.uint8)
        batch = preprocess_fn(tf.expand_dims(raw_rgb_float, axis=0))

        heatmap, prob = compute_gradcam_heatmap(model, batch, last_conv_layer_name=last_conv_layer_name)
        overlay = overlay_gradcam_opencv(raw_rgb_uint8, heatmap)
        pred_label = CLASSES[1] if prob >= 0.5 else CLASSES[0]

        axes[0, idx].imshow(raw_rgb_uint8)
        axes[0, idx].set_title(f"True: {row['label_name']}", fontsize=9)
        axes[0, idx].axis("off")

        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f"Pred: {pred_label} ({prob:.2f})", fontsize=9)
        axes[1, idx].axis("off")

    plt.suptitle(f"Grad-CAM (Parkinson Evidence) — {model.name}", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / f"gradcam_{arch.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
