import os  
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


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


def _deterministic_sort_for_head(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """
    Deterministic tie-break for Top-K:
      1) prob/score ↓
      2) time ↑
      3) id   ↑
    """
    out = df.copy()
    if not np.issubdtype(out["time"].dtype, np.datetime64):
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
    if out["id"].dtype == "object":
        try:
            pd.to_numeric(out["id"])
        except Exception:
            out["id"] = out["id"].astype(str)
    return out.sort_values([prob_col, "time", "id"],
                           ascending=[False, True, True],
                           kind="mergesort")


def _topk_precisions(df: pd.DataFrame, prob_col: str, Ks=(0.5, 1, 2, 5)) -> dict:
    """Precision at Top-K% using deterministic ordering on the given probability column."""
    d = _deterministic_sort_for_head(df, prob_col)
    y = pd.to_numeric(d["y"], errors="coerce").fillna(0).astype(int).values
    N = len(d)
    out = {}
    for K in Ks:
        Nk = int(np.ceil((K / 100.0) * N))
        out[K] = (y[:Nk].sum() / Nk) if Nk > 0 else np.nan
    return out


def _ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform") -> float:
    """
    Expected Calibration Error (ECE) with binning.
    ECE = sum_b (n_b / N) * |mean(p)_b - mean(y)_b|
    """
    prob_true, prob_pred = calibration_curve(y_true, p_pred, n_bins=n_bins, strategy=strategy)
    # sklearn returns only non-empty bins, but not counts; recompute counts for weighting:
    # Use the same bin edges to count.
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return 0.0
    # widen edges and digitize
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(p_pred, edges[1:-1], right=True)
    counts = np.bincount(bins, minlength=len(edges) - 1)
    # keep only bins that calibration_curve kept (non-empty bins)
    non_empty_mask = counts > 0
    counts = counts[non_empty_mask]
    if counts.sum() == 0:
        return 0.0
    weights = counts / counts.sum()
    # prob_true/prob_pred align with non-empty bins only
    ece = np.sum(weights * np.abs(prob_pred - prob_true))
    return float(ece)


def _save_single_reliability(y, p, path, n_bins=10, strategy="quantile"):
    """Save a single reliability scatter against the diagonal."""
    # Compute decile/quantile bins; dedupe edges to avoid empty duplicates
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            # Degenerate (all same prob); draw trivial plot
            plt.figure()
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Empirical fraud rate")
            plt.title("Reliability")
            plt.savefig(path, bbox_inches="tight"); plt.close()
            return
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(p, edges[1:-1], right=True)
    dfb = (pd.DataFrame({"y": y, "p": p, "bin": bins})
           .groupby("bin")
           .agg(y_bar=("y", "mean"), p_bar=("p", "mean"), n=("y", "size"))
           .reset_index())

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(dfb["p_bar"], dfb["y_bar"])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical fraud rate")
    plt.title("Reliability")
    plt.savefig(path, bbox_inches="tight"); plt.close()


def _save_overlay_reliability(y, p_raw, p_platt, p_iso, path, n_bins=10, strategy="uniform"):
    """Overlay Raw, Platt, Isotonic on the same reliability plot."""
    x_raw, y_raw = calibration_curve(y, p_raw, n_bins=n_bins, strategy=strategy)
    x_pl, y_pl = calibration_curve(y, p_platt, n_bins=n_bins, strategy=strategy)
    x_iso, y_iso = calibration_curve(y, p_iso, n_bins=n_bins, strategy=strategy)

    plt.figure()
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
    plt.plot(x_raw, y_raw, marker="o", label="Raw")
    plt.plot(x_pl, y_pl, marker="o", label="Platt")
    plt.plot(x_iso, y_iso, marker="o", label="Isotonic")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical fraud rate")
    plt.title("Reliability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight"); plt.close()


# ---------------------------
# Main
# ---------------------------

def main():
    cfg = Config()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    # Run ID and frozen Phase-0 file
    run_id = _read_json(cfg.reports_dir / cfg.harden_setup_file)["run_id"]
    parquet_path = cfg.interim_path / f"run_{run_id}" / "raw_scores.parquet"

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

    split_idx = int(0.8 * len(df))
    cal_train = df.iloc[:split_idx].copy()
    cal_hold = df.iloc[split_idx:].copy()

    # Drop NaN scores
    cal_train = cal_train.dropna(subset=["score"])
    cal_hold = cal_hold.dropna(subset=["score"])

    s_tr = cal_train["score"].to_numpy(dtype=float)
    y_tr = cal_train["y"].to_numpy(dtype=int)
    s_ho = cal_hold["score"].to_numpy(dtype=float)
    y_ho = cal_hold["y"].to_numpy(dtype=int)

    if y_tr.sum() == 0 or (len(y_tr) - y_tr.sum()) == 0:
        raise ValueError("cal_train lacks positives or negatives; cannot calibrate reliably.")

    # --- Fit calibrators ---
    # Isotonic (monotone, piecewise-constant)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(s_tr, y_tr)
    p_iso = iso.predict(s_ho)
    iso_brier = brier_score_loss(y_ho, p_iso)
    iso_ece = _ece(y_ho, p_iso, n_bins=10, strategy="uniform")

    # Platt (logistic regression on score)
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000).fit(s_tr.reshape(-1, 1), y_tr)
    p_platt = lr.predict_proba(s_ho.reshape(-1, 1))[:, 1]
    platt_brier = brier_score_loss(y_ho, p_platt)
    platt_ece = _ece(y_ho, p_platt, n_bins=10, strategy="uniform")

    # Uncalibrated baseline
    p_raw = s_ho.copy()
    raw_brier = brier_score_loss(y_ho, p_raw)
    raw_ece = _ece(y_ho, p_raw, n_bins=10, strategy="uniform")

    # --- Head precision check (Top-K%) on holdout ---
    Ks = (0.5, 1, 2, 5)
    hold_raw = cal_hold.assign(prob_raw=p_raw)
    hold_iso = cal_hold.assign(prob_iso=p_iso)
    hold_platt = cal_hold.assign(prob_platt=p_platt)
    topk_raw = _topk_precisions(hold_raw, "prob_raw", Ks)
    topk_iso = _topk_precisions(hold_iso, "prob_iso", Ks)
    topk_platt = _topk_precisions(hold_platt, "prob_platt", Ks)

    # --- Choose calibrator by rule ---
    # Lower Brier wins, provided Top-1% precision drop ≤ 2 points vs raw
    best = "platt"
    chosen_model = lr
    chosen_probs = p_platt
    if (iso_brier + 1e-12) < (platt_brier - 1e-12):
        best = "isotonic"
        chosen_model = iso
        chosen_probs = p_iso

    drop_limit = 0.02  # absolute points at Top-1%
    def _p1(d): return d.get(1.0, np.nan)
    ok_head = (
        np.isnan(_p1(topk_raw)) or np.isnan(_p1(topk_iso)) or np.isnan(_p1(topk_platt))
        or ((best == "platt" and (_p1(topk_raw) - _p1(topk_platt) <= drop_limit))
            or (best == "isotonic" and (_p1(topk_raw) - _p1(topk_iso) <= drop_limit)))
    )
    if not ok_head:
        # Fallback to the other if it satisfies the head constraint
        if best == "platt" and (_p1(topk_raw) - _p1(topk_iso) <= drop_limit):
            best, chosen_model, chosen_probs = "isotonic", iso, p_iso
        elif best == "isotonic" and (_p1(topk_raw) - _p1(topk_platt) <= drop_limit):
            best, chosen_model, chosen_probs = "platt", lr, p_platt

    # --- Save reliability plots ---
    rel_iso_path = cfg.figures_dir / "reliability_iso.png"
    rel_platt_path = cfg.figures_dir / "reliability_platt.png"
    rel_overlay_path = cfg.figures_dir / "reliability_overlay.png"
    iso_rel     = os.path.relpath(rel_iso_path,     start=cfg.reports_dir)
    platt_rel   = os.path.relpath(rel_platt_path,   start=cfg.reports_dir)
    overlay_rel = os.path.relpath(rel_overlay_path, start=cfg.reports_dir)
    _save_single_reliability(y_ho, p_iso, rel_iso_path, n_bins=10, strategy="quantile")
    _save_single_reliability(y_ho, p_platt, rel_platt_path, n_bins=10, strategy="quantile")
    _save_overlay_reliability(y_ho, p_raw, p_platt, p_iso, rel_overlay_path, n_bins=10, strategy="uniform")

    # --- Persist chosen calibrator ---
    cal_start, cal_end = cal_train["time"].min(), cal_train["time"].max()
    hold_start, hold_end = cal_hold["time"].min(), cal_hold["time"].max()
    model_path = cfg.models_dir / f"calibrator_{run_id}.pkl"
    joblib.dump({
        "type": best,
        "model": chosen_model,
        "run_id": run_id,
        "fitted_on": {
            "train_time_start": str(cal_start),
            "train_time_end": str(cal_end),
            "hold_time_start": str(hold_start),
            "hold_time_end": str(hold_end),
            "n_train": int(len(s_tr)),
            "n_holdout": int(len(s_ho)),
        },
        "metrics": {
            "raw_brier": float(raw_brier), "raw_ece": float(raw_ece),
            "iso_brier": float(iso_brier), "iso_ece": float(iso_ece),
            "platt_brier": float(platt_brier), "platt_ece": float(platt_ece),
            "topk_raw": topk_raw, "topk_iso": topk_iso, "topk_platt": topk_platt,
        }
    }, model_path)

    # --- Write the report ---
    report_path = cfg.reports_dir / "calibration_report.md"
    with open(report_path, "w") as f:
        f.write("# Calibration Report\n\n")
        f.write(f"- Run ID: `{run_id}`\n")
        f.write(f"- Split: 80% cal_train / 20% cal_holdout (by time)\n")
        f.write(f"- Brier (holdout): raw={raw_brier:.5f}, isotonic={iso_brier:.5f}, platt={platt_brier:.5f}\n")
        f.write(f"- ECE (holdout, 10 uniform bins): raw={raw_ece:.5f}, isotonic={iso_ece:.5f}, platt={platt_ece:.5f}\n")
        f.write(f"- Top-K% precision (holdout, K in {{0.5,1,2,5}}):\n\n")
        def line(name, d): return "  - " + name + ": " + ", ".join([f"{k}%={d[k]:.3f}" for k in (0.5,1,2,5)]) + "\n"
        f.write(line("raw", topk_raw))
        f.write(line("isotonic", topk_iso))
        f.write(line("platt", topk_platt))
        f.write(f"\n- Chosen calibrator: **{best}**\n")
        f.write(f"- Figures: `{rel_iso_path.name}`, `{rel_platt_path.name}`, `{rel_overlay_path.name}`\n")
        ### Curves
        f.write('<p style="display:flex;gap:10px;flex-wrap:wrap;">\n')
        f.write(f'  <img src="{iso_rel}" alt="Isotonic" width="320"/>\n')
        f.write(f'  <img src="{platt_rel}" alt="Platt" width="320"/>\n')
        f.write(f'  <img src="{overlay_rel}" alt="Overlay" width="640"/>\n')
        f.write("</p>\n")

        f.write("- Notes:\n")
        f.write("  - Monotone calibration preserves ranking (ROC/PR unchanged up to ties).\n")
        f.write("  - Use calibrated probabilities for policy thresholds, SLAs, and cost curves.\n")

    print(f"Saved calibrator: {model_path}")
    print(f"Report: {report_path}")
    print(f"Figures: {rel_iso_path}, {rel_platt_path}, {rel_overlay_path}")


if __name__ == "__main__":
    main()
