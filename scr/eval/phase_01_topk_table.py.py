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
    
def select_topk_indices(df_sorted: pd.DataFrame, K_percent: float) -> pd.Index:
    """
    Select exactly N_K rows by position after the deterministic sort.
    Returns the index of selected rows (length == N_K).
    """
    N = len(df_sorted)
    N_K = int(np.ceil((K_percent / 100.0) * N))
    return df_sorted.index[:N_K]

def compute_topk_metrics(df: pd.DataFrame, selected_idx: pd.Index) -> dict:
    """
    Compute precision/recall/FPR/alerts/alerts_per_day given the exact N_K selection.
    This guarantees alerts == len(selected_idx) even if many ties share the same score.
    """
    sel = df.index.isin(selected_idx)
    # Predictions: mark only the selected rows as 1 (by position), everything else 0.
    y_true = df['y'].astype(int).values
    y_pred = sel.astype(int)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr       = FP / (TN + FP) if (TN + FP) > 0 else 0.0

    # Alerts & alerts/day
    alerts = int(sel.sum())
    # unique days (YYYY-MM-DD from `time`)
    # assumes df['time'] is already datetime (done in deterministic_sort)
    unique_days = df['time'].dt.date.nunique()
    alerts_per_day = alerts / unique_days if unique_days > 0 else 0.0

    # Report the *score* threshold as the score on the boundary row:
    # that's the last selected row in the deterministic order.
    # (We still compute metrics from exact-position selection to keep alerts == N_K.)
    boundary_score = float(df.loc[selected_idx[-1], 'score'])

    return dict(threshold=boundary_score,
                precision=precision,
                recall=recall,
                fpr=fpr,
                alerts=alerts,
                alerts_per_day=alerts_per_day)
    
    
def main():
    cfg = Config()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    
    harden_setup_rep = _read_json(cfg.reports_dir / cfg.harden_setup_file)
    
    run_id = harden_setup_rep["run_id"]
    parquet_path: Path =cfg. interim_path / ("run_" + run_id)
    
    # read + deterministic sort
    df = pd.read_parquet(parquet_path)
    # the file must have columns: id, time, y, score
    missing = {"id", "time", "y", "score"} - set(df.columns)
    if missing:
        raise ValueError(f"raw_scores.parquet missing columns: {missing}")

    df_sorted = deterministic_sort(df)
    
    Ks = np.array([0.5, 1, 2, 5])
    rows = []
    for K in Ks:
        top_idx = select_topk_indices(df_sorted, K)
        metrics = compute_topk_metrics(df_sorted, top_idx)
        rows.append({
            "mode": "topk",
            "K_or_FPR": float(K),
            **metrics
        })
    out = pd.DataFrame(rows, columns=[
        "mode", "K_or_FPR", "threshold",
        "precision", "recall", "fpr",
        "alerts", "alerts_per_day"
    ])
    
    out_path = cfg.reports_dir / "baseline_top_k_table.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    
    
if __name__ == "__main__":
    main()
    