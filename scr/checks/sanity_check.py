from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]  # repo root
    parquet_path: Path = Path(__file__).resolve().parents[2] / "data" / "interim" / "train_clean_baseline.parquet"
    report_dir: Path = Path(__file__).resolve().parents[2] / "data" / "interim" / "reports"
    top_missing_n: int = 10
    head_cols_k: int = 20
    required_cols: tuple[str, ...] = ("isFraud", "dt", "TransactionDT")


def load_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.parquet_path)
    # Normalize string placeholders to NaN
    df = df.replace("missing", np.nan)
    return df


def verify_columns(df: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    # target must be binary
    assert df["isFraud"].dropna().isin([0, 1]).all(), "isFraud must be binary 0/1"

    # dt: if present and not datetime, try to coerce
    if "dt" in df.columns:
        if not is_datetime64_any_dtype(df["dt"]):
            df["dt"] = pd.to_datetime(df["dt"], errors="coerce", utc=False)
        assert is_datetime64_any_dtype(df["dt"]), "dt exists but is not datetime after coercion"
        nat_rate = df["dt"].isna().mean()
        assert nat_rate < 0.001, f"Too many invalid datetimes in dt after coercion (NaT rate={nat_rate:.3f})"

    # TransactionDT: should be numeric seconds
    if "TransactionDT" in df.columns:
        assert is_numeric_dtype(df["TransactionDT"]), "TransactionDT must be numeric"

    return df


def class_balance(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    pos = int(df["isFraud"].sum())
    neg = int(total - pos)
    prevalence = float(pos / total) if total else float("nan")
    return {"total": total, "pos": pos, "neg": neg, "prevalence": prevalence}


def missing_summary(df: pd.DataFrame, n: int) -> dict[str, float]:
    frac = df.isna().mean().sort_values(ascending=False)
    return {k: float(v) for k, v in frac.head(n).items()}


def first_n_columns(df: pd.DataFrame, k: int) -> List[str]:
    return df.columns[:k].tolist()


def main() -> None:
    cfg = Config()
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg)
    verify_columns(df, cfg.required_cols)

    # Decide time column
    time_col = "dt" if "dt" in df.columns else "TransactionDT"

    report = {
        "dataset_path": str(cfg.parquet_path),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "head_columns": first_n_columns(df, cfg.head_cols_k),
        "time_column": time_col,
        "class_balance": class_balance(df),
        "top_missing": missing_summary(df, cfg.top_missing_n),
    }

    # Console summary (concise)
    print(f"[OK] Columns verified: {cfg.required_cols}")
    print(f"[INFO] Using time column: {report['time_column']}")
    cb = report["class_balance"]
    print(f"[INFO] Class balance: pos={cb['pos']}, neg={cb['neg']}, prevalence={cb['prevalence']:.5f}")
    print(f"[INFO] Columns: {report['n_cols']}, Rows: {report['n_rows']}")
    print("[INFO] Top missing:")
    for k, v in report["top_missing"].items():
        print(f"   {k}: {v:.3f}")

    # Write JSON artifact
    out_path = cfg.report_dir / "sanity_v0.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[DONE] Wrote report -> {out_path}")


if __name__ == "__main__":
    main()
