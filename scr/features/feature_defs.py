from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype


@dataclass(frozen=True)
class Config:
    parquet_path: Path = Path(__file__).resolve().parents[2] / "data" / "interim" / "train_clean_baseline.parquet"
    report_dir: Path = Path(__file__).resolve().parents[2] / "data" / "interim" / "reports"
    drop_cols: tuple[str, ...] = ("TransactionID", "TransactionID_1", "isFraud", "dt", "TransactionDT", "hour", "day", "weekday")


def load_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.parquet_path)
    # Normalize placeholders to NaN
    df = df.replace("missing", np.nan)
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    df[num_cols] = df[num_cols].replace([-1, -999], np.nan)
    return df


def drop_features(df: pd.DataFrame, drop_cols: tuple[str, ...]) -> pd.DataFrame:
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def columns_type(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in df.columns if is_numeric_dtype(df[c]) or is_bool_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


def main():
    cfg = Config()
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    # Load once
    df = load_dataset(cfg)

    # Figure out what exists *before* dropping
    original_cols = set(df.columns)
    found_to_drop = sorted([c for c in cfg.drop_cols if c in original_cols])
    not_found = sorted([c for c in cfg.drop_cols if c not in original_cols])

    # Now drop
    df = drop_features(df, cfg.drop_cols)

    # Split types
    numeric, categorical = columns_type(df)
    numeric = sorted(numeric)
    categorical = sorted(categorical)

    # Integrity checks
    assert len(numeric) + len(categorical) == len(df.columns)
    assert set(numeric).isdisjoint(set(categorical))

    # Logs
    print(f"[OK] Columns dropped: {found_to_drop}")
    if not_found:
        print(f"[INFO] Drop candidates not present: {not_found}")
    print(f"[INFO] numeric={len(numeric)} categorical={len(categorical)} total={len(df.columns)}")
    print(f"[INFO] sample numeric: {numeric[:5]}")
    print(f"[INFO] sample categorical: {categorical[:5]}")

    # Report
    report = {
        "dataset_path": str(cfg.parquet_path),
        "dropped_features": found_to_drop,      # ✅ now correct
        "drop_candidates_not_found": not_found, # optional but useful
        "n_features_total": len(df.columns),
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "numeric": numeric,
        "categorical": categorical,
    }

    out_path = cfg.report_dir / "features_v0.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[DONE] Wrote report -> {out_path}")

if __name__ == "__main__":
    main()

