"""Stage 2: Run Exploratory Data Analysis on data/processed/ and save plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import EDA_REPORTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from src.data.eda import run_eda


def main() -> None:
    ensure_project_dirs()
    summary = run_eda(processed_dir=PROCESSED_DATA_DIR, output_dir=EDA_REPORTS_DIR)
    summary_path = EDA_REPORTS_DIR / "eda_summary.json"
    summary_json = json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    summary_path.write_text(summary_json, encoding="utf-8")
    print("\n=== EDA SUMMARY ===")
    print(summary_json)
    print(f"\nSaved EDA summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
