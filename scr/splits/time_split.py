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

def prevalence(s: pd.Series) -> float:
    pos = int((s == 1).sum())
    tot = int(len(s))
    return pos / tot if tot else float("nan")

def main():
    # cfg = Config()
    # cfg.report_dir.mkdir(parents=True, exist_ok=True)

    # # Load once
    # df = load_dataset(cfg)
    # time_col = pick_time_col(df, cfg.time_col_candidates)
    # print(time_col)
    # # Sort ascending and split
    # df = df.sort_values(by="dt", ascending=True)
    
    # t_cut = df[time_col].quantile(80)
    # print(t_cut)
    return df
    
    
    
    
if __name__ == "__main__":
    cfg = Config()
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    # Load once
    df = load_dataset(cfg)
    time_col = pick_time_col(df, cfg.time_col_candidates)

    # Sort ascending and split
    df = df.sort_values(by="dt", ascending=True)
    
    t_cut = df[time_col].quantile(0.8)
    train_df = df[df[time_col] <= t_cut]
    val_df = df[df[time_col] > t_cut]
    
    report = {
        "dataset_path": str(cfg.parquet_path),
        "time_column": time_col,
        "t_cut": str(pd.Timestamp(t_cut)),
        "train": {
            "rows": int(len(_df)),
            "time_min": str(train_df[time_col].iloc[0]),
            "time_max": str(train_df[time_col].iloc[-1]),
            "pos": int((train_df["isFraud"] == 1).sum()),
            "neg": int(len(train_df) - (train_df["isFraud"] == 1).sum()),
            "prevalence": prevalence(train_df["isFraud"]),
        },
         "val": {
            "rows": int(len(val_df)),
            "time_min": str(val_df[cfg.time_col].iloc[0]),
            "time_max": str(val_df[cfg.time_col].iloc[-1]),
            "pos": int((val_df["isFraud"] == 1).sum()),
            "neg": int(len(val_df) - (val_df["isFraud"] == 1).sum()),
            "prevalence": prevalence(val_df["isFraud"]),
        },
    }