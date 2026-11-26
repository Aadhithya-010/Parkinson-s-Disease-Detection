"""Dataset discovery, integrity checks, deduplication, and leakage-safe splitting."""

from __future__ import annotations

import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from configs.config import (
    CLASSES,
    DICOM_EXTENSIONS,
    IMAGE_EXTENSIONS,
    PROCESSED_DATA_DIR,
    SEED,
    SPLIT_RATIOS,
    SPLITS,
)

HEALTHY_TOKENS = {"healthy", "normal", "control", "controls", "hc", "non_parkinson", "negative"}
PARKINSON_TOKENS = {"parkinson", "parkinsons", "parkinson_disease", "pd", "patient", "patients", "positive"}


def _tokenize_name(name: str) -> set[str]:
    """Split a path component into lowercase alphanumeric tokens."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    tokens = set(cleaned.split("_")) if cleaned else set()
    if cleaned:
        tokens.add(cleaned)
    return tokens


def infer_label_from_path(path: Path, root: Path) -> Optional[str]:
    """Infer binary class label ('healthy' or 'parkinson') from relative directory names."""
    rel_parts = path.relative_to(root).parts[:-1]
    for part in reversed(rel_parts):
        tokens = _tokenize_name(part)
        if tokens & HEALTHY_TOKENS:
            return "healthy"
        if tokens & PARKINSON_TOKENS:
            return "parkinson"
    return None


def compute_file_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute exact MD5 hash of file bytes."""
    digest = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _dicom_to_uint8_rgb(dicom_path: Path) -> tuple[Image.Image, Optional[str]]:
    """Load a DICOM file, return an 8-bit RGB PIL image and optional PatientID."""
    import pydicom

    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        # Take middle slice if a 3D volume is stored in a single DICOM file
        arr = arr[arr.shape[0] // 2]
    lo, hi = np.percentile(arr, (1.0, 99.5))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = np.zeros_like(arr, dtype=np.uint8)

    pil_img = Image.fromarray(arr).convert("RGB")
    patient_id = getattr(ds, "PatientID", None)
    subject_id = str(patient_id).strip() if patient_id else None
    return pil_img, subject_id


def verify_and_collect_raw_files(raw_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Walk raw_dir, verify image/DICOM readability, compute MD5, and extract labels."""
    supported = IMAGE_EXTENSIONS | DICOM_EXTENSIONS
    records: list[dict] = []
    corrupt_or_unlabeled: list[dict] = []

    all_files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported)
    if not all_files:
        raise FileNotFoundError(
            f"No supported image or DICOM files found under {raw_dir}. "
            f"Supported extensions: {sorted(supported)}"
        )

    for file_path in all_files:
        label = infer_label_from_path(file_path, raw_dir)
        if label is None:
            corrupt_or_unlabeled.append({"path": str(file_path), "reason": "unrecognized_class_folder"})
            continue

        ext = file_path.suffix.lower()
        subject_id: Optional[str] = None
        try:
            if ext in DICOM_EXTENSIONS:
                pil_img, subject_id = _dicom_to_uint8_rgb(file_path)
                width, height = pil_img.size
                fmt = "DICOM"
                mode = "RGB"
            else:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    width, height = img.size
                    fmt = img.format or ext.lstrip(".").upper()
                    mode = img.mode
            md5 = compute_file_md5(file_path)
            records.append(
                {
                    "path": str(file_path),
                    "label": label,
                    "ext": ext,
                    "format": fmt,
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "md5": md5,
                    "subject_id": subject_id,
                }
            )
        except Exception as exc:
            corrupt_or_unlabeled.append({"path": str(file_path), "reason": f"corrupt: {exc}"})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(
            f"Found {len(all_files)} candidate files under {raw_dir}, "
            "but none passed label mapping and integrity checks."
        )
    return df, corrupt_or_unlabeled


def deduplicate_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact MD5 duplicate files while keeping the first occurrence."""
    before = len(df)
    dedup_df = df.drop_duplicates(subset=["md5"], keep="first").reset_index(drop=True)
    removed = before - len(dedup_df)
    return dedup_df, removed


def create_leakage_safe_splits(
    df: pd.DataFrame,
    split_ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, bool]:
    """Assign train/validate/test splits using subject grouping if available, else stratified."""
    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"SPLIT_RATIOS must sum to 1.0, got {split_ratios}")

    df = df.copy()
    has_subjects = df["subject_id"].notna().all() and df["subject_id"].nunique() >= 6

    if has_subjects:
        # Group-aware split ensures no subject's slices cross split boundaries
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(df, df["label"], groups=df["subject_id"]))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)

        val_rel = val_ratio / (train_ratio + val_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
        train_idx, val_idx = next(
            gss_val.split(train_val_df, train_val_df["label"], groups=train_val_df["subject_id"])
        )

        df.loc[:, "split"] = "train"
        df.iloc[test_idx, df.columns.get_loc("split")] = "test"
        val_original_indices = df.index[train_val_idx][val_idx]
        df.loc[val_original_indices, "split"] = "validate"
        return df, True

    # Stratified split on deduplicated image records
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(df, df["label"]))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)

    val_rel = val_ratio / (train_ratio + val_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
    train_idx, val_idx = next(sss_val.split(train_val_df, train_val_df["label"]))

    df.loc[:, "split"] = "train"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    val_original_indices = df.index[train_val_idx][val_idx]
    df.loc[val_original_indices, "split"] = "validate"
    return df, False


def materialize_processed_splits(
    split_df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
    clean_existing: bool = True,
) -> pd.DataFrame:
    """Copy or convert audited raw files into data/processed/{split}/{class}/."""
    if clean_existing and processed_dir.exists():
        shutil.rmtree(processed_dir)

    for split in SPLITS:
        for cls in CLASSES:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    exported_rows: list[dict] = []
    per_bucket_counter: Counter[tuple[str, str]] = Counter()

    for _, row in split_df.iterrows():
        split = row["split"]
        cls = row["label"]
        per_bucket_counter[(split, cls)] += 1
        idx = per_bucket_counter[(split, cls)]
        src_path = Path(row["path"])
        dest_name = f"{cls}_{idx:04d}.png"
        dest_path = processed_dir / split / cls / dest_name

        if row["ext"] in DICOM_EXTENSIONS:
            pil_img, _ = _dicom_to_uint8_rgb(src_path)
            pil_img.save(dest_path, format="PNG")
        else:
            with Image.open(src_path) as img:
                img.convert("RGB").save(dest_path, format="PNG")

        exported_rows.append(
            {
                "raw_path": str(src_path),
                "processed_path": str(dest_path),
                "split": split,
                "label": cls,
                "md5": row["md5"],
                "subject_id": row["subject_id"],
            }
        )

    manifest_df = pd.DataFrame(exported_rows)
    manifest_df.to_csv(processed_dir / "manifest.csv", index=False)
    return manifest_df


def run_dataset_audit(raw_dir: Path, processed_dir: Path = PROCESSED_DATA_DIR) -> dict:
    """Execute end-to-end raw dataset verification, deduplication, and splitting."""
    raw_df, issues = verify_and_collect_raw_files(raw_dir)
    dedup_df, duplicates_removed = deduplicate_records(raw_df)
    split_df, used_subject_grouping = create_leakage_safe_splits(dedup_df)
    manifest_df = materialize_processed_splits(split_df, processed_dir=processed_dir)

    split_counts_raw = manifest_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    split_counts = {
        str(col): {str(row_idx): int(val) for row_idx, val in series.items()}
        for col, series in split_counts_raw.items()
    }

    summary = {
        "raw_valid_files": int(len(raw_df)),
        "corrupt_or_unlabeled": int(len(issues)),
        "exact_duplicates_removed": int(duplicates_removed),
        "retained_unique_files": int(len(dedup_df)),
        "used_subject_grouping": bool(used_subject_grouping),
        "formats": {str(k): int(v) for k, v in dedup_df["format"].value_counts().items()},
        "split_counts": split_counts,
    }
    return summary
