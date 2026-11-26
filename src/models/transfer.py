"""Transfer-learning builders for VGG16 and ResNet50 with safe two-phase fine-tuning."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from configs.config import CHANNELS, FINE_TUNE_UNFREEZE_LAST_N, IMAGE_SIZE


def build_transfer_model(
    arch: str,
    input_shape: tuple[int, int, int] = (*IMAGE_SIZE, CHANNELS),
    dropout_rate: float = 0.5,
) -> tuple[tf.keras.Model, tf.keras.Model, str]:
    """Build a frozen ImageNet backbone (VGG16 or ResNet50) with a binary classification head."""
    key = arch.lower()
    if key == "vgg16":
        base = tf.keras.applications.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
        last_conv_name = "block5_conv3"
        model_name = "VGG16"
    elif key == "resnet50":
        base = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
        last_conv_name = "conv5_block3_out"
        model_name = "ResNet50"
    else:
        raise ValueError(f"Unsupported transfer architecture '{arch}'. Use 'vgg16' or 'resnet50'.")

    base.trainable = False  # Phase 1: train only the new classification head

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model, base, last_conv_name


def unfreeze_top_layers_safely(
    base_model: tf.keras.Model,
    unfreeze_last_n: int = FINE_TUNE_UNFREEZE_LAST_N,
) -> int:
    """Unfreeze the last N backbone layers while keeping early layers and all BatchNorm frozen."""
    base_model.trainable = True
    cutoff = max(0, len(base_model.layers) - unfreeze_last_n)

    trainable_count = 0
    for idx, layer in enumerate(base_model.layers):
        if idx < cutoff or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable_count += 1
    return trainable_count
