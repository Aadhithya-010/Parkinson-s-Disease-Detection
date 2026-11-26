"""Stage 1: Download a Kaggle dataset (or audit local raw data) and build clean splits."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import DEFAULT_KAGGLE_DATASET, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_dirs
from src.data.audit import run_dataset_audit


def download_kaggle_dataset(dataset_slug: str, target_dir: Path) -> Path:
    """Download a Kaggle dataset using kagglehub and copy files into target_dir."""
    import kagglehub

    print(f"Downloading Kaggle dataset: {dataset_slug} ...")
    cache_path = Path(kagglehub.dataset_download(dataset_slug))
    print(f"Downloaded to kagglehub cache: {cache_path}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(cache_path, target_dir)
    print(f"Copied raw dataset into: {target_dir}")
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and audit Parkinson's image dataset.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_KAGGLE_DATASET,
        help=f"Kaggle dataset slug (default: {DEFAULT_KAGGLE_DATASET})",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Audit an already downloaded folder inside data/raw/ instead of downloading.",
    )
    parser.add_argument(
        "--raw-subdir",
        default=None,
        help="Subdirectory name inside data/raw/ (defaults to sanitized dataset slug).",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    subdir_name = args.raw_subdir or args.dataset.replace("/", "__")
    raw_target = RAW_DATA_DIR / subdir_name

    if not args.skip_download:
        download_kaggle_dataset(args.dataset, raw_target)
    elif not raw_target.exists():
        raise FileNotFoundError(f"--skip-download set, but raw directory does not exist: {raw_target}")

    summary = run_dataset_audit(raw_dir=raw_target, processed_dir=PROCESSED_DATA_DIR)
    print("\n=== DATASET AUDIT SUMMARY ===")
    print(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
