"""Exploratory Data Analysis (EDA) for processed image splits."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from configs.config import CLASSES, EDA_REPORTS_DIR, PROCESSED_DATA_DIR, SEED, SPLITS


def load_processed_dataframe(processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Scan data/processed/{split}/{class} and collect metadata for each image."""
    records: list[dict] = []
    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = processed_dir / split / cls
            if not cls_dir.exists():
                continue
            for file_path in sorted(cls_dir.iterdir()):
                if not file_path.is_file() or file_path.name.startswith("."):
                    continue
                with Image.open(file_path) as img:
                    width, height = img.size
                    channels = len(img.getbands())
                    fmt = img.format or file_path.suffix.lstrip(".").upper()
                records.append(
                    {
                        "path": str(file_path),
                        "split": split,
                        "label": cls,
                        "width": width,
                        "height": height,
                        "aspect_ratio": width / height if height else np.nan,
                        "channels": channels,
                        "format": fmt,
                    }
                )
    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(f"No processed images found under {processed_dir}.")
    return df


def plot_class_distribution(df: pd.DataFrame, output_dir: Path = EDA_REPORTS_DIR) -> Path:
    """Save overall and split-wise class balance bar charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    overall = df["label"].value_counts().reindex(CLASSES, fill_value=0)
    axes[0].bar(overall.index, overall.values, color=["#2a9d8f", "#e76f51"])
    axes[0].set_title("Overall Class Distribution")
    axes[0].set_ylabel("Image Count")
    for idx, val in enumerate(overall.values):
        axes[0].text(idx, val + max(overall.values) * 0.02, str(val), ha="center", fontweight="bold")

    split_counts = (
        df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(index=SPLITS, columns=CLASSES, fill_value=0)
    )
    split_counts.plot(kind="bar", ax=axes[1], color=["#2a9d8f", "#e76f51"])
    axes[1].set_title("Class Distribution by Split")
    axes[1].set_ylabel("Image Count")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    out_path = output_dir / "class_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_image_dimensions(df: pd.DataFrame, output_dir: Path = EDA_REPORTS_DIR) -> Path:
    """Save width/height histograms and width-vs-height scatter plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(df["width"], bins=20, color="#264653", alpha=0.85)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (px)")

    axes[1].hist(df["height"], bins=20, color="#2a9d8f", alpha=0.85)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (px)")

    for cls, color in zip(CLASSES, ("#2a9d8f", "#e76f51")):
        sub = df[df["label"] == cls]
        axes[2].scatter(sub["width"], sub["height"], label=cls, alpha=0.6, s=22, color=color)
    axes[2].set_title("Width vs Height by Class")
    axes[2].set_xlabel("Width (px)")
    axes[2].set_ylabel("Height (px)")
    axes[2].legend()

    plt.tight_layout()
    out_path = output_dir / "image_dimensions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sample_grid(
    df: pd.DataFrame,
    n_per_class: int = 8,
    output_dir: Path = EDA_REPORTS_DIR,
    seed: int = SEED,
) -> Path:
    """Save a grid of random healthy and Parkinson's sample images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(CLASSES), n_per_class, figsize=(n_per_class * 2.1, len(CLASSES) * 2.4))

    for row_idx, cls in enumerate(CLASSES):
        sub = df[df["label"] == cls]
        sample_n = min(n_per_class, len(sub))
        sampled = sub.sample(n=sample_n, random_state=seed)
        for col_idx in range(n_per_class):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < sample_n:
                rec = sampled.iloc[col_idx]
                with Image.open(rec["path"]) as img:
                    ax.imshow(img)
                ax.set_title(f"{cls}\n{rec['width']}x{rec['height']}", fontsize=8)

    plt.suptitle("Sample Images by Class", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = output_dir / "sample_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_intensity_distribution(
    df: pd.DataFrame,
    max_samples_per_class: int = 100,
    output_dir: Path = EDA_REPORTS_DIR,
    seed: int = SEED,
) -> tuple[Path, dict]:
    """Compare grayscale pixel histograms by class to check for brightness shortcuts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    stats: dict[str, dict[str, float]] = {}

    for cls, color in zip(CLASSES, ("#2a9d8f", "#e76f51")):
        sub = df[(df["split"] == "train") & (df["label"] == cls)]
        if sub.empty:
            sub = df[df["label"] == cls]
        sampled = sub.sample(n=min(max_samples_per_class, len(sub)), random_state=seed)
        pixels = []
        for path in sampled["path"]:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("L"), dtype=np.uint8)
                pixels.append(arr.ravel())
        flat = np.concatenate(pixels)
        stats[cls] = {"mean": float(flat.mean()), "std": float(flat.std())}
        ax.hist(flat, bins=64, range=(0, 255), density=True, alpha=0.45, label=f"{cls} (mean={flat.mean():.1f})", color=color)

    ax.set_title("Training Pixel Intensity Distribution (Shortcut Check)")
    ax.set_xlabel("Pixel Intensity (0-255)")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    out_path = output_dir / "intensity_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path, stats


def run_eda(processed_dir: Path = PROCESSED_DATA_DIR, output_dir: Path = EDA_REPORTS_DIR) -> dict:
    """Run all EDA computations and save visualization figures."""
    df = load_processed_dataframe(processed_dir)
    overall_counts = df["label"].value_counts().reindex(CLASSES, fill_value=0)
    min_count = max(int(overall_counts.min()), 1)
    imbalance_ratio = float(overall_counts.max() / min_count)

    class_plot = plot_class_distribution(df, output_dir)
    dim_plot = plot_image_dimensions(df, output_dir)
    sample_plot = plot_sample_grid(df, output_dir=output_dir)
    intensity_plot, intensity_stats = plot_intensity_distribution(df, output_dir=output_dir)

    summary = {
        "total_images": int(len(df)),
        "overall_counts": overall_counts.to_dict(),
        "split_counts": df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
        "imbalance_ratio": round(imbalance_ratio, 3),
        "unique_channels": sorted(df["channels"].unique().tolist()),
        "width_summary": df["width"].describe().round(2).to_dict(),
        "height_summary": df["height"].describe().round(2).to_dict(),
        "aspect_ratio_summary": df["aspect_ratio"].describe().round(3).to_dict(),
        "intensity_stats": intensity_stats,
        "plots": {
            "class_distribution": str(class_plot),
            "image_dimensions": str(dim_plot),
            "sample_grid": str(sample_plot),
            "intensity_distribution": str(intensity_plot),
        },
    }
    return summary
