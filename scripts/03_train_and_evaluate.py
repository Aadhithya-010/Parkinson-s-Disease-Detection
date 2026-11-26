"""Stage 3: Train Custom CNN, VGG16, and ResNet50, evaluate on test set, and save Grad-CAMs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    BATCH_SIZE,
    CNN_EPOCHS,
    CNN_LEARNING_RATE,
    EVAL_REPORTS_DIR,
    FINETUNE_LEARNING_RATE,
    GRADCAM_REPORTS_DIR,
    HEAD_LEARNING_RATE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    TRANSFER_FINETUNE_EPOCHS,
    TRANSFER_HEAD_EPOCHS,
    USE_CLAHE,
    ensure_project_dirs,
)
from src.data.pipeline import build_tf_dataset, compute_balanced_class_weights, load_split_dataframe
from src.evaluation.metrics import (
    evaluate_model_on_test,
    plot_evaluation_suite,
    save_comparison_table,
    save_error_analysis_grid,
)
from src.models.cnn import build_custom_cnn
from src.models.gradcam import save_gradcam_grid
from src.models.transfer import build_transfer_model, unfreeze_top_layers_safely


def set_reproducibility(seed: int = SEED) -> None:
    """Fix NumPy and TensorFlow random seeds."""
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_callbacks(
    checkpoint_path: Path,
    patience_es: int = 6,
    initial_best_val_loss: float | None = None,
) -> list[tf.keras.callbacks.Callback]:
    """Create standard EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            initial_value_threshold=initial_best_val_loss,
            verbose=0,
        ),
    ]


def merge_histories(h1: dict, h2: dict) -> dict[str, list[float]]:
    """Concatenate phase 1 and phase 2 Keras history dictionaries."""
    merged: dict[str, list[float]] = {}
    for key in set(h1.keys()) | set(h2.keys()):
        merged[key] = list(h1.get(key, [])) + list(h2.get(key, []))
    return merged


def compile_binary_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Compile model with Adam and binary classification metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate CNN, VGG16, and ResNet50.")
    parser.add_argument("--clahe", action="store_true", default=USE_CLAHE, help="Enable OpenCV CLAHE.")
    parser.add_argument("--cnn-epochs", type=int, default=CNN_EPOCHS)
    parser.add_argument("--head-epochs", type=int, default=TRANSFER_HEAD_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=TRANSFER_FINETUNE_EPOCHS)
    args = parser.parse_args()

    ensure_project_dirs()
    set_reproducibility(SEED)

    train_df = load_split_dataframe("train", PROCESSED_DATA_DIR)
    val_df = load_split_dataframe("validate", PROCESSED_DATA_DIR)
    test_df = load_split_dataframe("test", PROCESSED_DATA_DIR)
    class_weights = compute_balanced_class_weights(train_df["label"])

    print("Split sizes:", {"train": len(train_df), "validate": len(val_df), "test": len(test_df)})
    print("Balanced class weights:", class_weights)

    trained_models: dict[str, tuple[tf.keras.Model, str, str]] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    results: dict[str, dict] = {}

    # 1. Custom CNN from scratch
    print("\n--- Training CustomCNN ---")
    train_ds, _, _ = build_tf_dataset(train_df, arch="cnn", training=True, batch_size=BATCH_SIZE, use_clahe=args.clahe)
    val_ds, _, _ = build_tf_dataset(val_df, arch="cnn", training=False, batch_size=BATCH_SIZE, use_clahe=args.clahe)
    test_ds, _, y_test = build_tf_dataset(test_df, arch="cnn", training=False, batch_size=BATCH_SIZE, use_clahe=args.clahe)

    cnn_model, cnn_conv = build_custom_cnn()
    compile_binary_model(cnn_model, CNN_LEARNING_RATE)
    t0 = time.perf_counter()
    cnn_hist = cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.cnn_epochs,
        callbacks=make_callbacks(MODELS_DIR / "best_cnn.keras"),
        class_weight=class_weights,
        verbose=1,
    )
    cnn_time = time.perf_counter() - t0
    histories["CustomCNN"] = cnn_hist.history
    results["CustomCNN"] = evaluate_model_on_test(cnn_model, test_ds, y_test, cnn_time)
    trained_models["CustomCNN"] = (cnn_model, "cnn", cnn_conv)

    # 2. Transfer learning: VGG16 & ResNet50
    for arch, display_name in (("vgg16", "VGG16"), ("resnet50", "ResNet50")):
        print(f"\n--- Training {display_name} (Phase 1: Head, Phase 2: Fine-tune) ---")
        train_ds, _, _ = build_tf_dataset(train_df, arch=arch, training=True, batch_size=BATCH_SIZE, use_clahe=args.clahe)
        val_ds, _, _ = build_tf_dataset(val_df, arch=arch, training=False, batch_size=BATCH_SIZE, use_clahe=args.clahe)
        test_ds, _, y_test = build_tf_dataset(test_df, arch=arch, training=False, batch_size=BATCH_SIZE, use_clahe=args.clahe)

        model, base, conv_name = build_transfer_model(arch=arch)
        ckpt_path = MODELS_DIR / f"best_{arch}.keras"

        t0 = time.perf_counter()
        # Phase 1: train only classification head
        compile_binary_model(model, HEAD_LEARNING_RATE)
        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.head_epochs,
            callbacks=make_callbacks(ckpt_path),
            class_weight=class_weights,
            verbose=1,
        )

        # Phase 2: unfreeze top layers while keeping BatchNorm frozen
        best_p1_loss = float(min(h1.history["val_loss"]))
        unfrozen = unfreeze_top_layers_safely(base)
        print(f"{display_name}: unfroze {unfrozen} top non-BatchNorm layers for fine-tuning.")
        compile_binary_model(model, FINETUNE_LEARNING_RATE)
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.finetune_epochs,
            callbacks=make_callbacks(ckpt_path, initial_best_val_loss=best_p1_loss),
            class_weight=class_weights,
            verbose=1,
        )
        if ckpt_path.exists() and min(h2.history["val_loss"]) > best_p1_loss:
            model = tf.keras.models.load_model(ckpt_path)
        elapsed = time.perf_counter() - t0

        histories[display_name] = merge_histories(h1.history, h2.history)
        results[display_name] = evaluate_model_on_test(model, test_ds, y_test, elapsed)
        trained_models[display_name] = (model, arch, conv_name)

    # 3. Save comparison table, plots, error analysis, and Grad-CAM overlays
    table = save_comparison_table(results, output_dir=EVAL_REPORTS_DIR)
    print("\n=== TEST SET MODEL COMPARISON ===")
    print(table.round(4))

    plot_evaluation_suite(results, histories, table, output_dir=EVAL_REPORTS_DIR)

    best_model_name = table["f1"].idxmax()
    print(f"\nBest model by F1-score: {best_model_name} (Recall={results[best_model_name]['recall']:.4f})")

    for name, (model, arch, conv_name) in trained_models.items():
        save_error_analysis_grid(name, test_df, results[name], output_dir=EVAL_REPORTS_DIR)
        save_gradcam_grid(
            model=model,
            arch=arch,
            test_df=test_df,
            last_conv_layer_name=conv_name,
            output_dir=GRADCAM_REPORTS_DIR,
        )

    print(
        "\nDone. Note: This system is experimental and not clinically validated. "
        "Review reports/evaluation/ and reports/gradcam/ before reporting results."
    )


if __name__ == "__main__":
    main()
