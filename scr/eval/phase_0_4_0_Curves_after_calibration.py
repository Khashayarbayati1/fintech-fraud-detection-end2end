import os  
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score



# ---------------------------
# Config & utilities
# ---------------------------

@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    interim_path: Path = root / "data" / "interim"
    reports_dir: Path = root / "data" / "interim" / "reports"
    figures_dir: Path = root / "data" / "interim" / "figures"
    models_dir: Path = root / "models"
    harden_setup_file: str = "harden_baseline_metadata.json"  # has run_id


def _read_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def _time_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort ascending by time; ensure datetime."""
    out = df.copy()
    if not np.issubdtype(out["time"].dtype, np.datetime64):
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
    return out.sort_values("time", kind="mergesort")
    
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

    # Report the *p_cal* threshold as the score on the boundary row:
    # that's the last selected row in the deterministic order.
    # (We still compute metrics from exact-position selection to keep alerts == N_K.)
    boundary_p = float(df.loc[selected_idx[-1], 'p_cal'])

    return dict(threshold_prob=float(boundary_p),
                precision=float(precision),
                recall=float(recall),
                fpr=float(fpr),
                alerts=int(alerts),
                alerts_per_day=float(alerts_per_day))

def recall_at_fpr_rows(
    df_sorted: pd.DataFrame,
    budgets=(0.1, 0.5, 1, 2),
    *,
    run_id: str,
    cal_type: str,
    window: str,
):
    """
    df_sorted must be sorted deterministically by:
        p_cal ↓, time ↑, id ↑
    Returns list[dict] rows for mode='fpr', evaluating thresholds only at
    group boundaries of equal p_cal (i.e., unique thresholds).
    """
    # Probabilities and labels
    p = pd.to_numeric(df_sorted["p_cal"], errors="coerce").values
    y = pd.to_numeric(df_sorted["y"], errors="coerce").fillna(0).astype(int).values

    n = len(p)
    if n == 0:
        return []

    # Build staircase cumulatives over the sorted order
    is_pos = (y == 1).astype(int)
    is_neg = (y == 0).astype(int)

    cum_TP = np.cumsum(is_pos)
    cum_FP = np.cumsum(is_neg)

    pos_total = int(is_pos.sum())
    neg_total = int(is_neg.sum())

    # Guards
    recall = np.divide(cum_TP, pos_total, out=np.zeros_like(cum_TP, dtype=float), where=pos_total > 0)
    denom_pf = cum_TP + cum_FP
    precision = np.divide(cum_TP, denom_pf, out=np.zeros_like(cum_TP, dtype=float), where=denom_pf > 0)
    fpr = np.divide(cum_FP, neg_total, out=np.zeros_like(cum_FP, dtype=float), where=neg_total > 0)

    # Alerts & per-day
    alerts = np.arange(1, n + 1, dtype=int)
    unique_days = df_sorted["time"].dt.date.nunique() if "time" in df_sorted.columns else 0
    alerts_per_day = alerts / unique_days if unique_days > 0 else np.zeros_like(alerts, dtype=float)

    # --------- Evaluate only at unique p_cal boundaries ---------
    # Find last index of each contiguous block of equal p values (tie block)
    # We exclude NaN p_cal from candidate thresholds (treated as lowest scores).
    # boundaries: array of indices (sorted asc) where each is the last row of a tie block.
    # Treat near-equal floats as equal to avoid micro-splits
    eq = np.isclose(p[1:], p[:-1], rtol=0, atol=1e-12)
    changes = np.ones(n, dtype=bool)
    changes[1:] = ~eq                  # True where a NEW value starts

    # indices of last items in each block == (start_of_next_block - 1)
    block_starts = np.flatnonzero(changes)
    boundaries = np.r_[block_starts[1:] - 1, n - 1] if len(block_starts) > 0 else np.array([n - 1])

    # Keep only boundaries with non-NaN probabilities (valid thresholds)
    valid_boundaries_mask = ~np.isnan(p[boundaries])
    boundaries = boundaries[valid_boundaries_mask]

    # Edge case: if all p are NaN, there are no valid thresholds; fall back to empty boundaries
    if boundaries.size == 0:
        boundaries = np.array([], dtype=int)

    # Precompute views at boundaries
    fpr_b = fpr[boundaries]
    prec_b = precision[boundaries]
    rec_b = recall[boundaries]
    alerts_b = alerts[boundaries]
    alerts_per_day_b = alerts_per_day[boundaries]
    tau_b = p[boundaries]  # threshold scores at boundaries

    rows = []

    for b in budgets:
        cap = float(b) / 100.0

        if boundaries.size == 0:
            # No valid thresholds (e.g., all p_cal were NaN)
            rows.append({
                "run_id": run_id,
                "calibrator": cal_type,
                "window": window,
                "mode": "fpr",
                "budget_type": "FPR_percent",
                "budget_value": float(b),
                "threshold_prob": float("nan"),
                "precision": 0.0,
                "recall": 0.0,
                "fpr": float(fpr[0]) if len(fpr) else float("nan"),
                "alerts": 0,
                "alerts_per_day": 0.0,
            })
            continue

        # Find the largest boundary with FPR ≤ cap
        ok = np.where(fpr_b <= cap)[0]
        if ok.size == 0:
            # Even the strictest boundary exceeds the budget → select none
            rows.append({
                "run_id": run_id,
                "calibrator": cal_type,
                "window": window,
                "mode": "fpr",
                "budget_type": "FPR_percent",
                "budget_value": float(b),
                "threshold_prob": float("nan"),
                "precision": 0.0,
                "recall": 0.0,
                "fpr": float(fpr_b[0]),  # first boundary's fpr (most lenient threshold) for reference
                "alerts": 0,
                "alerts_per_day": 0.0,
            })
            continue

        j = ok[-1]  # choose the largest feasible boundary (most conservative under cap)
        rows.append({
            "run_id": run_id,
            "calibrator": cal_type,
            "window": window,
            "mode": "fpr",
            "budget_type": "FPR_percent",
            "budget_value": float(b),
            "threshold_prob": float(tau_b[j]),
            "precision": float(prec_b[j]),
            "recall": float(rec_b[j]),
            "fpr": float(fpr_b[j]),
            "alerts": int(alerts_b[j]),
            "alerts_per_day": float(alerts_per_day_b[j]),
        })

    return rows

     
def main():
    cfg = Config()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    # Run ID and frozen Phase-0 file
    run_id = _read_json(cfg.reports_dir / cfg.harden_setup_file)["run_id"]
    parquet_path = cfg.interim_path / f"run_{run_id}" / "raw_scores.parquet"
    cal_path = cfg.models_dir / f"calibrator_{run_id}.pkl"

    # Read & basic checks
    df = pd.read_parquet(parquet_path)
    required = {"id", "time", "y", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw_scores.parquet: {missing}")
    
     # Coerce types and split by time (80/20)
    df = _time_sort(df)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["score"]).reset_index(drop=True)

    cal_obj = joblib.load(cal_path)
    cal_type = cal_obj.get("type", None)
    model = cal_obj.get("model", None)
    if cal_type not in {"isotonic", "platt"} or model is None:
        raise ValueError("Calibrator file must contain {'type', 'model'} with type in {'isotonic','platt'}.")

    s = df["score"].to_numpy(dtype=float)
    
    # 2) Apply calibrator (NO refitting here)
    if cal_type == "isotonic":
        # IsotonicRegression expects 1D input
        p_cal = model.predict(s)
    else:  # platt (LogisticRegression)
        # LogisticRegression expects 2D input
        p_cal = model.predict_proba(s.reshape(-1, 1))[:, 1]

    # 3) Attach probabilities and keep needed columns
    df["p_cal"] = p_cal

     # 4) Sanity checks
    if (df["p_cal"].min() < -1e-12) or (df["p_cal"].max() > 1 + 1e-12):
        raise AssertionError("p_cal out of [0,1] bounds.")
    
    # Monotonicity vs score (non-decreasing with score)
    order = np.argsort(s)
    if np.any(np.diff(df["p_cal"].to_numpy()[order]) < -1e-10):
        raise AssertionError("Calibrated probabilities are not monotone in score.")

    # At this point, df has: id, time, y, score, p_cal
    # You can now subset to holdout or full window as needed in later steps.
    out_path = cfg.interim_path / f"run_{run_id}" / "raw_scores_with_pcal.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved calibrated probabilities to: {out_path}\nCalibrator={cal_type}, Run={run_id}")
    
    df_sorted = df.sort_values(
        by=["p_cal", "time", "id"],
        ascending=[False, True, True],  
        kind="mergesort"  # stable
    )   
    
    # Pick the same window as the label we chose "full"
    y_true = df["y"].to_numpy(dtype=int)
    p_prob = df["p_cal"].to_numpy(dtype=float)

    # --- ROC from calibrated probabilities (handles ties + endpoints) ---
    fpr, tpr, _ = roc_curve(y_true, p_prob)
    roc_auc = auc(fpr, tpr)

    # --- PR points from calibrated probabilities (no refit) ---
    prec, rec, _ = precision_recall_curve(y_true, p_prob)
    ap = average_precision_score(y_true, p_prob)

    # --- Side-by-side subplots: ROC (left) and PR (right) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title(f"ROC — Calibrator={cal_type}, Run={run_id}\nAUC={roc_auc:.4f}")

    # PR
    ax = axes[1]
    ax.plot(rec, prec, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR — Calibrator={cal_type}, Run={run_id}\nAP={ap:.4f}")

    out_fig = cfg.figures_dir / f"roc_pr_calibrated_run_{run_id}.png"
    fig.savefig(out_fig, dpi=180)
    print(f"Saved ROC+PR to: {out_fig}")
    
    # Top-K% Precision 
    Ks = np.array([0.5, 1, 2, 5])
    rows_topk = []
    for K in Ks:
        top_idx = select_topk_indices(df_sorted, K)
        metrics = compute_topk_metrics(df_sorted, top_idx)
        rows_topk.append({
            "run_id": run_id,
            "calibrator": cal_type,
            "window": "full",       # "holdout" or "full"
            "mode": "topk",
            "budget_type": "K_percent",
            "budget_value": float(K),          # e.g., 0.5, 1, 2, 5
            **metrics
        })

    # Recall@FPR
    budgets = (0.1, 0.5, 1, 2)
    rows_fpr = recall_at_fpr_rows(
    df_sorted, budgets, run_id=run_id, cal_type=cal_type, window="full"
    
    )
    
    # Combine and persist (single tidy CSV)
    rows_all = rows_topk + rows_fpr
    op = pd.DataFrame(rows_all)
    cols = ["run_id","calibrator","window","mode","budget_type","budget_value",
            "threshold_prob","precision","recall","fpr","alerts","alerts_per_day"]
    op = op[cols].sort_values(["window","mode","budget_type","budget_value"]).reset_index(drop=True)

    csv_path = cfg.reports_dir / f"baseline_operating_points_{cal_type}_calibrated_{run_id}.csv"

    if csv_path.exists():
        base = pd.read_csv(csv_path)
        # Drop existing rows for this run/calibrator/window to avoid duplicates
        mask = (
            (base.get("run_id","") == op["run_id"].iloc[0]) &
            (base.get("calibrator","") == op["calibrator"].iloc[0]) &
            (base.get("window","") == op["window"].iloc[0])
        )
        base = base.loc[~mask]
        out_df = pd.concat([base, op], ignore_index=True)
    else:
        out_df = op

    out_df.to_csv(csv_path, index=False)
    print(f"Updated operating points → {csv_path}")

if __name__ == "__main__":
    main()
