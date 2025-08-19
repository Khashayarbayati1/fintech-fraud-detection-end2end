# scr/splits/time_split.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


@dataclass(frozen=True)
class Config:
    parquet_path: Path = Path(__file__).resolve().parents[2] / "data" / "interim" / "train_clean_baseline.parquet"
    report_dir: Path   = Path(__file__).resolve().parents[2] / "data" / "interim" / "reports"
    time_col_candidates: tuple[str, ...] = ("dt", "TransactionDT")


def load_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.parquet_path)
    # Normalize placeholders to NaN (keeps consistency with earlier steps)
    df = df.replace("missing", np.nan)
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].replace([-1, -999], np.nan)
    return df


def pick_time_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            if c == "dt" and not is_datetime64_any_dtype(df[c]):
                df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
            return c
    raise ValueError(f"No time column found among {candidates}")


def prevalence(y: pd.Series) -> float:
    tot = int(y.shape[0])
    if tot == 0:
        return float("nan")
    pos = int((y == 1).sum())
    return pos / tot


def main() -> None:
    cfg = Config()
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg)
    time_col = pick_time_col(df, cfg.time_col_candidates)

    # Drop rows with missing time
    before = len(df)
    df = df[df[time_col].notna()].copy()
    dropped = before - len(df)

    # Sort ascending
    df = df.sort_values(time_col).reset_index(drop=True)

    # Compute 80th percentile cutoff
    t_cut = df[time_col].quantile(0.80)

    # Split
    train_df = df[df[time_col] <= t_cut]
    val_df   = df[df[time_col]  > t_cut]

    # Sanity checks
    assert not train_df.empty and not val_df.empty, "One of the splits is empty."
    assert train_df[time_col].max() <= t_cut < val_df[time_col].min(), "Overlap around cutoff."
    assert df[time_col].is_monotonic_increasing, "Data not sorted by time."

    # Format cutoff for report
    if is_datetime64_any_dtype(df[time_col]):
        t_cut_out = str(pd.Timestamp(t_cut))
    else:
        # numeric (e.g., TransactionDT seconds)
        t_cut_out = float(t_cut)

    report = {
        "dataset_path": str(cfg.parquet_path),
        "time_column": time_col,
        "dropped_rows_missing_time": int(dropped),
        "t_cut": t_cut_out,
        "train": {
            "rows": int(len(train_df)),
            "time_min": str(train_df[time_col].iloc[0]),
            "time_max": str(train_df[time_col].iloc[-1]),
            "pos": int((train_df["isFraud"] == 1).sum()),
            "neg": int(len(train_df) - (train_df["isFraud"] == 1).sum()),
            "prevalence": prevalence(train_df["isFraud"]),
        },
        "val": {
            "rows": int(len(val_df)),
            "time_min": str(val_df[time_col].iloc[0]),
            "time_max": str(val_df[time_col].iloc[-1]),
            "pos": int((val_df["isFraud"] == 1).sum()),
            "neg": int(len(val_df) - (val_df["isFraud"] == 1).sum()),
            "prevalence": prevalence(val_df["isFraud"]),
        },
    }

    # Console summary
    print(f"[INFO] time_col={time_col}")
    print(f"[INFO] t_cut={report['t_cut']}")
    tr, vr = report["train"]["rows"], report["val"]["rows"]
    tp, vp = report["train"]["prevalence"], report["val"]["prevalence"]
    print(f"[INFO] train rows={tr} prev={tp:.5f}")
    print(f"[INFO]   time range: {report['train']['time_min']} -> {report['train']['time_max']}")
    print(f"[INFO] val   rows={vr} prev={vp:.5f}")
    print(f"[INFO]   time range: {report['val']['time_min']} -> {report['val']['time_max']}")

    # Write JSON artifact
    out_path = cfg.report_dir / "split_v0.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[DONE] Wrote report -> {out_path}")


if __name__ == "__main__":
    main()
