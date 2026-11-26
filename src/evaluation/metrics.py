"""Test evaluation metrics, diagnostic checks, comparison plots, and error analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from configs.config import CLASSES, EVAL_REPORTS_DIR


def evaluate_model_on_test(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    y_true: np.ndarray,
    train_time_sec: float,
    threshold: float = 0.5,
) -> dict:
    """Evaluate a model on an unshuffled test dataset and compute clinical-style metrics."""
    y_true_int = np.asarray(y_true, dtype=np.int32)
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(np.int32)

    cm = confusion_matrix(y_true_int, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true_int)) == 2:
        auc_val = float(roc_auc_score(y_true_int, y_prob))
        fpr, tpr, _ = roc_curve(y_true_int, y_prob)
    else:
        auc_val = float("nan")
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    healthy_mask = y_true_int == 0
    pd_mask = y_true_int == 1

    return {
        "accuracy": float(accuracy_score(y_true_int, y_pred)),
        "precision": float(precision_score(y_true_int, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_int, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_int, y_pred, zero_division=0)),
        "specificity": specificity,
        "roc_auc": auc_val,
        "params": int(model.count_params()),
        "train_time_sec": round(float(train_time_sec), 2),
        "prob_std": float(np.std(y_prob)),
        "healthy_prob_mean": float(np.mean(y_prob[healthy_mask])) if np.any(healthy_mask) else float("nan"),
        "parkinson_prob_mean": float(np.mean(y_prob[pd_mask])) if np.any(pd_mask) else float("nan"),
        "conf_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "y_true": y_true_int,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def save_comparison_table(results: dict[str, dict], output_dir: Path = EVAL_REPORTS_DIR) -> pd.DataFrame:
    """Build and export the model comparison table to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "roc_auc",
        "prob_std",
        "healthy_prob_mean",
        "parkinson_prob_mean",
        "params",
        "train_time_sec",
    ]
    rows = {name: {k: res[k] for k in cols} for name, res in results.items()}
    table = pd.DataFrame.from_dict(rows, orient="index")[cols]
    table.to_csv(output_dir / "model_comparison.csv", index=True)
    return table


def plot_evaluation_suite(
    results: dict[str, dict],
    histories: dict[str, dict[str, list[float]]],
    table: pd.DataFrame,
    output_dir: Path = EVAL_REPORTS_DIR,
) -> dict[str, Path]:
    """Save grouped metrics bar chart, confusion matrices, ROC curves, and training curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # 1. Grouped metric comparison
    metric_cols = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]
    ax = table[metric_cols].plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_ylim(0, 1.08)
    ax.set_title("Model Performance Comparison (Test Set)")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", ncol=3)
    plt.tight_layout()
    p_bar = output_dir / "metrics_comparison.png"
    plt.savefig(p_bar, dpi=150, bbox_inches="tight")
    plt.close()
    saved["metrics_comparison"] = p_bar

    # 2. Confusion matrices side by side
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(
            res["conf_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            ax=ax,
        )
        ax.set_title(f"{name}\nAcc={res['accuracy']:.3f} | Recall={res['recall']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    p_cm = output_dir / "confusion_matrices.png"
    fig.savefig(p_cm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["confusion_matrices"] = p_cm

    # 3. ROC curves on one axis
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, res in results.items():
        ax.plot(res["fpr"], res["tpr"], lw=2, label=f"{name} (AUC = {res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Baseline (AUC = 0.500)")
    ax.set_title("ROC Curves — Test Set")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    p_roc = output_dir / "roc_curves.png"
    fig.savefig(p_roc, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["roc_curves"] = p_roc

    # 4. Training curves
    fig, axes = plt.subplots(2, n_models, figsize=(4.8 * n_models, 7))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=1)
    for col, (name, hist) in enumerate(histories.items()):
        axes[0, col].plot(hist.get("loss", []), label="train_loss")
        axes[0, col].plot(hist.get("val_loss", []), label="val_loss")
        axes[0, col].set_title(f"{name} — Loss")
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].legend()

        axes[1, col].plot(hist.get("accuracy", []), label="train_acc")
        axes[1, col].plot(hist.get("val_accuracy", []), label="val_acc")
        axes[1, col].set_title(f"{name} — Accuracy")
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].legend()
    plt.tight_layout()
    p_curves = output_dir / "training_curves.png"
    fig.savefig(p_curves, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["training_curves"] = p_curves

    return saved


def save_error_analysis_grid(
    model_name: str,
    test_df: pd.DataFrame,
    res: dict,
    output_dir: Path = EVAL_REPORTS_DIR,
    max_errors: int = 8,
) -> Path | None:
    """Save a grid of misclassified test images for qualitative error analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wrong_indices = np.where(res["y_pred"] != res["y_true"])[0]
    if len(wrong_indices) == 0:
        return None

    selected = wrong_indices[:max_errors]
    cols = min(4, len(selected))
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for plot_idx, test_idx in enumerate(selected):
        row = test_df.iloc[test_idx]
        pred_cls = CLASSES[int(res["y_pred"][test_idx])]
        prob = float(res["y_prob"][test_idx])
        with Image.open(row["path"]) as img:
            axes[plot_idx].imshow(img)
        axes[plot_idx].set_title(
            f"True: {row['label_name']}\nPred: {pred_cls} (p={prob:.2f})",
            fontsize=9,
        )

    plt.suptitle(f"Misclassified Test Samples — {model_name}", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / f"error_analysis_{model_name.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
