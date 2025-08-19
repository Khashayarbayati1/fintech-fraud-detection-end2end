from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype


@dataclass(frozen=True)
class Config:
    parquet_path: Path = Path(__file__).resolve().parents[2] / "data" / "interim" / "train_clean_baseline.parquet"
    report_dir: Path = Path(__file__).resolve().parents[2] / "data" / "interim" / "reports"
    drop_cols: tuple[str, ...] = ("TransactionID", "isFraud", "dt", "TransactionDT", "hour", "day", "weekday")
    time_col_candidates: tuple[str, ...] = ("dt", "TransactionDT")

def load_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.parquet_path)
    # Normalize placeholders to NaN
    df = df.replace("missing", np.nan)
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    df[num_cols] = df[num_cols].replace([-1, -999], np.nan)
    return df

def pick_time_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            if c == "dt" and not is_datetime64_any_dtype(df[c]):
                df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
            return c
    raise ValueError(f"No time column found among {candidates}")

def main():
    cfg = Config()
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    # Load once
    df = load_dataset(cfg)
    time_col = pick_time_col(df, cfg.time_col_candidates)
    print(time_col)
    # Sort ascending and split
    df = df.sort_values(by="dt", ascending=True)
    
    return df
    
    
    
    
if __name__ == "__main__":
    df = main()
