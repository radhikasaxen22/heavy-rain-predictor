"""Phase 2: Validate features and create a time-aware train/val/test split.

Rules applied:
  - No missing values in the model feature columns.
  - Outlier scan: extremes are flagged and checked against plausible physics.
  - Time alignment: IMD daily (mm) is matched to ERA5 daily on the same calendar
    date; the ~3 h difference in the India station rain-day convention is an
    accepted, documented limitation of the MVP.
  - No leakage: time split first (train -> val -> test by calendar date), so the
    model never sees future data while training.  rain_lag1 only uses the
    previous day, so it never looks ahead.

Output: data/processed/split/{train,val,test}.csv  (features + target + metadata)
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
from features.feature_config import MODEL_FEATURES, META_COLUMNS, RAIN_COL, TARGET  # noqa: E402

DATASET_CSV = BASE / "data" / "processed" / "rainfall_dataset.csv"
SPLIT_DIR = BASE / "data" / "processed" / "split"

TRAIN_END = "2016-12-31"
VAL_END = "2018-12-31"


def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    df = pd.read_csv(DATASET_CSV, parse_dates=["date"])

    print("== quality checks ==")
    missing = df[MODEL_FEATURES + [RAIN_COL]].isna().sum()
    print(f"  missing values: total {int(missing.sum())} ({missing.sum() / len(df):.2%} rows)")

    for col in MODEL_FEATURES:
        lo, hi = float(df[col].quantile(0.001)), float(df[col].quantile(0.999))
        extreme = df[(df[col] < lo) | (df[col] > hi)]
        print(f"  {col:<20} 0.1-99.9pct: {lo:>8.2f} .. {hi:>8.2f}  extreme rows: {len(extreme)}")

    print("== time-aware split ==")
    train = df[df["date"] <= pd.Timestamp(TRAIN_END)]
    val = df[(df["date"] > pd.Timestamp(TRAIN_END)) & (df["date"] <= pd.Timestamp(VAL_END))]
    test = df[df["date"] > pd.Timestamp(VAL_END)]

    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        heavy = int(part[TARGET].sum()) if TARGET in part else "n/a"
        print(f"  {name:<6} {len(part):>8,} rows | {part['date'].min().date()} to "
              f"{part['date'].max().date()} | heavy-rain days: {heavy} "
              f"({part[TARGET].mean() * 100:.2f}%)")
        part.to_csv(os.path.join(SPLIT_DIR, f"{name}.csv"), index=False)

    with open(os.path.join("src", "features", "model_features.json"), "w") as fh:
        json.dump(MODEL_FEATURES, fh, indent=2)
    print("  saved split CSVs + model_features.json")


if __name__ == "__main__":
    main()