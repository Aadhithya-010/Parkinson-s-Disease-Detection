"""OpenCV preprocessing, label-preserving augmentation, and tf.data pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from configs.config import BATCH_SIZE, CLASSES, IMAGE_SIZE, PROCESSED_DATA_DIR, SEED, USE_CLAHE

LABEL_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


def apply_clahe_rgb(rgb_img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE on the luminance (L) channel in LAB color space."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_chan)
    merged = cv2.merge((l_enhanced, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_image_opencv(
    image_path: str | Path,
    target_size: tuple[int, int] = IMAGE_SIZE,
    use_clahe: bool = USE_CLAHE,
) -> np.ndarray:
    """Load an image via OpenCV, optionally enhance contrast, and resize with aspect-ratio padding."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if use_clahe:
        rgb = apply_clahe_rgb(rgb)

    target_w, target_h = target_size
    src_h, src_w = rgb.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)

    # Estimate border color from image edges so padding matches background (dark MRI or light paper)
    edge_pixels = np.concatenate(
        [resized[0, :, :], resized[-1, :, :], resized[:, 0, :], resized[:, -1, :]],
        axis=0,
    )
    pad_color = [int(v) for v in np.median(edge_pixels, axis=0)]

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    canvas = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )
    return canvas.astype(np.float32)


def get_model_preprocess_fn(arch: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return the normalization function expected by each model architecture."""
    key = arch.lower()
    if key == "cnn":
        return lambda x: x / 255.0
    if key == "vgg16":
        return tf.keras.applications.vgg16.preprocess_input
    if key == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input
    raise ValueError(f"Unsupported architecture '{arch}'. Expected one of: cnn, vgg16, resnet50.")


def build_augmentation_layer(seed: int = SEED) -> tf.keras.Sequential:
    """Create label-preserving augmentations operating on [0, 255] RGB images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.03, fill_mode="nearest", seed=seed),
            tf.keras.layers.RandomZoom(0.08, fill_mode="nearest", seed=seed),
            tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="nearest", seed=seed),
            tf.keras.layers.RandomContrast(0.10, seed=seed),
        ],
        name="train_augmentation",
    )


def load_split_dataframe(split: str, processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Load file paths and numeric labels for a given split ('train', 'validate', or 'test')."""
    records: list[dict] = []
    for cls in CLASSES:
        cls_dir = processed_dir / split / cls
        if not cls_dir.exists():
            continue
        for file_path in sorted(cls_dir.iterdir()):
            if file_path.is_file() and not file_path.name.startswith("."):
                records.append({"path": str(file_path), "label_name": cls, "label": LABEL_TO_IDX[cls]})
    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(f"No images found for split '{split}' under {processed_dir}.")
    return df


def compute_balanced_class_weights(labels: np.ndarray | pd.Series) -> dict[int, float]:
    """Compute balanced binary class weights to counter split imbalance."""
    y = np.asarray(labels, dtype=np.int32)
    unique_classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y)
    return {int(cls): float(w) for cls, w in zip(unique_classes, weights)}


def build_tf_dataset(
    df_split: pd.DataFrame,
    arch: str,
    training: bool = False,
    batch_size: int = BATCH_SIZE,
    use_clahe: bool = USE_CLAHE,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, np.ndarray, np.ndarray]:
    """Build a deterministic or augmented tf.data.Dataset from OpenCV-preprocessed images."""
    images = np.stack(
        [preprocess_image_opencv(p, target_size=IMAGE_SIZE, use_clahe=use_clahe) for p in df_split["path"]],
        axis=0,
    )
    labels = df_split["label"].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df_split), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    if training:
        aug_layer = build_augmentation_layer(seed=seed)
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    preprocess_fn = get_model_preprocess_fn(arch)
    ds = ds.map(
        lambda x, y: (preprocess_fn(tf.cast(x, tf.float32)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, images, labels
