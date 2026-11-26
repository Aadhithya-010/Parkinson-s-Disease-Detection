"""Custom CNN baseline architecture built with the Keras Functional API."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from configs.config import CHANNELS, IMAGE_SIZE


def build_custom_cnn(
    input_shape: tuple[int, int, int] = (*IMAGE_SIZE, CHANNELS),
    dropout_rate: float = 0.5,
) -> tuple[tf.keras.Model, str]:
    """Build a 3-block CNN with BatchNorm, GlobalAveragePooling2D, and sigmoid output."""
    inputs = layers.Input(shape=input_shape, name="image_input")
    x = inputs

    for block_idx, filters in enumerate((32, 64, 128), start=1):
        conv_name = "last_conv" if block_idx == 3 else f"conv_block_{block_idx}"
        x = layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=conv_name,
        )(x)
        x = layers.BatchNormalization(name=f"bn_block_{block_idx}")(x)
        x = layers.ReLU(name=f"relu_block_{block_idx}")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"pool_block_{block_idx}")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(64, activation="relu", name="dense_head")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CustomCNN")
    return model, "last_conv"
