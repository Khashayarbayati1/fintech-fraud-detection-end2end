from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype


@dataclass(frozen=True)
class Config:
    parquet_path: Path = Path(__file__).resolve().parents[2] / "data" / "interim" / "train_clean_baseline.parquet"
    report_dir: Path = Path(__file__).resolve().parents[2] / "data" / "interim" / "reports"
    drop_cols: tuple[str, ...] = ("TransactionID", "isFraud", "dt", "TransactionDT")

def load_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.parquet_path)
    # Normalize string placeholders to NaN
    df = df.replace("missing", np.nan)
    return df

def drop_features(df: pd.DataFrame, drop_cols: tuple[str, ...]) -> pd.DataFrame:
    return df.drop(columns=list(drop_cols), errors="ignore")

def columns_type(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numerical = [c for c in df.columns if is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if not is_numeric_dtype(df[c])]
    return numerical, categorical
    


def main():
    cfg = Config()
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg)
    df = drop_features(df, cfg.drop_cols)
    numerical, categorical = columns_type(df)
    
    numerical = sorted(numerical)
    categorical = sorted(categorical)

    # Console summary (concise)
    print(f"[OK] Columns dropped: {cfg.drop_cols}")
    print(f"[INFO] numeric={len(numerical)} categorical={len(categorical)} total={len(df.columns)}")
    
    report = {
        "dataset_path": str(cfg.parquet_path),
        "dropped_features": list(cfg.drop_cols),
        "n_features_total": len(df.columns),
        "n_numeric": len(numerical),
        "n_categorical": len(categorical),
        "numeric": numerical,
        "categorical": categorical,
    }
    

    # Write JSON artifact
    out_path = cfg.report_dir / "features_v0.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[DONE] Wrote report -> {out_path}")

if __name__ == "__main__":
    main()
    
