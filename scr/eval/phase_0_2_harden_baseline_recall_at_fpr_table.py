import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
import joblib

from datetime import datetime

@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    interim_path: Path = root / "data" / "interim" 
    reports_dir: Path = root / "data" / "interim" / "reports"
    model_dir: Path = root / "models"
    model_choice_file: str = "model_choice_v0.json"
    harden_setup_file: str = "harden_baseline_metadata.json"
    
def _read_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def deterministic_sort(df: pd.DataFrame) -> pd.DataFrame:
    # ensure time is datetime for tie-breaker + alerts/day
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    # optional: ensure 'id' is sortable (cast to string if mixed types)
    if df["id"].dtype == "object":
        try:
            # if all look numeric, keep as-is; otherwise cast to string
            pd.to_numeric(df["id"])
        except Exception:
            df["id"] = df["id"].astype(str)
    return df.sort_values(
        by=["score", "time", "id"],
        ascending=[False, True, True],
        kind="mergesort"  # stable
    )    

def recall_at_fpr_rows(df_sorted: pd.DataFrame, budgets=(0.1, 0.5, 1, 2)):
    """
    df_sorted: already sorted deterministically by score↓, time↑, id↑
    returns: list of dict rows for mode='fpr'
    """
    s = pd.to_numeric(df_sorted["score"], errors="coerce").values
    y = pd.to_numeric(df_sorted["y"], errors="coerce").fillna(0).astype(int).values
    if s.shape[0] == 0:
        return []

    # Build staircase cumulatives over the sorted order
    is_pos = (y == 1).astype(int)
    is_neg = (y == 0).astype(int)

    cum_TP = np.cumsum(is_pos)
    cum_FP = np.cumsum(is_neg)

    pos_total = is_pos.sum()
    neg_total = is_neg.sum()

    # Guards to avoid 0/0
    recall = np.divide(cum_TP, pos_total, out=np.zeros_like(cum_TP, dtype=float), where=pos_total > 0)
    denom_pf = cum_TP + cum_FP
    precision = np.divide(cum_TP, denom_pf, out=np.zeros_like(cum_TP, dtype=float), where=denom_pf > 0)
    fpr = np.divide(cum_FP, neg_total, out=np.zeros_like(cum_FP, dtype=float), where=neg_total > 0)

    # Alerts & per-day
    alerts = np.arange(1, len(s) + 1, dtype=int)
    unique_days = df_sorted["time"].dt.date.nunique()
    alerts_per_day = alerts / unique_days if unique_days > 0 else np.zeros_like(alerts, dtype=float)

    rows = []
    # For each budget b, choose the largest index with fpr <= b%
    for b in budgets:
        cap = b / 100.0
        ok_idxs = np.where(fpr <= cap)[0]
        if ok_idxs.size == 0:
            # Even the strictest threshold violates the cap → pick none (NaNs/zeros as appropriate)
            rows.append({
                "mode": "fpr",
                "K_or_FPR": float(b),
                "threshold": float("nan"),
                "precision": 0.0,
                "recall": 0.0,
                "fpr": float(fpr[0]) if len(fpr) else float("nan"),
                "alerts": 0,
                "alerts_per_day": 0.0
            })
            continue

        idx = ok_idxs[-1]  # largest τ (most conservative) under the cap
        tau = float(s[idx])  # threshold score at that index

        rows.append({
            "mode": "fpr",
            "K_or_FPR": float(b),
            "threshold": tau,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "fpr": float(fpr[idx]),
            "alerts": int(alerts[idx]),
            "alerts_per_day": float(alerts_per_day[idx]),
        })

    return rows


def main():
    cfg = Config()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    
    meta = _read_json(cfg.reports_dir / cfg.harden_setup_file)
    run_id = meta["run_id"]

    # Frozen Phase-0 file
    parquet_path: Path = cfg.interim_path / f"run_{run_id}" / "raw_scores.parquet"

    # read
    df = pd.read_parquet(parquet_path)

    # sanity
    required = {"id", "time", "y", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"raw_scores.parquet missing columns: {missing}")

    # deterministic sort (score ↓, time ↑, id ↑) and ensure datetime
    df_sorted = deterministic_sort(df)

    # build Recall@FPR rows
    budgets = (0.1, 0.5, 1, 2)
    rows = recall_at_fpr_rows(df_sorted, budgets)

    # list[dict] -> DataFrame
    out = pd.DataFrame(rows, columns=[
        "mode", "K_or_FPR", "threshold",
        "precision", "recall", "fpr",
        "alerts", "alerts_per_day"
    ])

    # write a standalone file for sanity checks:
    out_path = cfg.reports_dir / "baseline_recall_at_fpr.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    
if __name__ == "__main__":
    main()
    