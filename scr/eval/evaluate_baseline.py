# scr/eval/evaluate_baseline.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
)

@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    parquet_path: Path = root / "data" / "interim" / "train_clean_baseline.parquet"
    reports_dir: Path = root / "data" / "interim" / "reports"
    figures_dir: Path = root / "data" / "interim" / "figures"
    model_dir: Path = root / "models"
    preprocessor_file: str = "preprocessor_baseline.pkl"
    model_file: str = "baseline_lgbm.pkl"
    split_file: str = "split_v0.json"
    preprocess_file: str = "preprocess_v0.json"

def _read_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)

def file_md5(path: Path, chunk=1<<20) -> str:
    m = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            m.update(b)
    return m.hexdigest()

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_pct: float) -> float:
    k = max(1, int(len(y_score) * k_pct))
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].mean())

def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0

# Threshold @0.5 and best-F1
def thr_metrics(y_true, y_score, thr: float) -> dict:
    y_hat = (y_score >= thr).astype(int)
    prec = float(precision_score(y_true, y_hat, zero_division=0))
    rec  = float(recall_score(y_true, y_hat, zero_division=0))
    f1   = float(f1_score(y_true, y_hat, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
    return {"threshold": thr, "precision": prec, "recall": rec, "f1": f1,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
        
def main():
    cfg = Config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    # Load split & features info
    split_rep = _read_json(cfg.reports_dir / cfg.split_file)
    prep_rep  = _read_json(cfg.reports_dir / cfg.preprocess_file)
    features  = prep_rep["feature_order"]
    time_col  = split_rep["time_column"]
    t_cut     = pd.to_datetime(split_rep["t_cut"])

    # Load data
    df = pd.read_parquet(cfg.parquet_path)
    # enforce time and sort
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].sort_values(time_col).reset_index(drop=True)

    # Split
    train_df = df[df[time_col] <= t_cut]
    val_df   = df[df[time_col]  > t_cut]

    X_train = train_df[features]
    X_val   = val_df[features]
    y_train = train_df["isFraud"].to_numpy()
    y_val   = val_df["isFraud"].to_numpy()

    # Load artifacts
    preproc = joblib.load(cfg.model_dir / cfg.preprocessor_file)
    model   = joblib.load(cfg.model_dir / cfg.model_file)

    # Transform (no fit)
    Xtr = preproc.transform(X_train)
    Xva = preproc.transform(X_val)

    # Optional: wrap arrays w/ names to avoid sklearn warning
    names = getattr(preproc, "get_feature_names_out", lambda *_: None)(features)
    if names is not None and len(names) == Xva.shape[1]:
        Xva_infer = pd.DataFrame(Xva, columns=list(names))
        Xtr_infer = pd.DataFrame(Xtr, columns=list(names))
    else:
        Xva_infer, Xtr_infer = Xva, Xtr

    # Predict on val using best_iteration_
    best_iter = getattr(model, "best_iteration_", None)
    p_val = model.predict_proba(Xva_infer, num_iteration=best_iter)[:, 1]

    # --- Metrics ---
    val_auc = float(roc_auc_score(y_val, p_val))
    val_ap  = float(average_precision_score(y_val, p_val))
    prevalence = float(y_val.mean())

    pr_prec, pr_rec, pr_thr = precision_recall_curve(y_val, p_val)
    den = (pr_prec[:-1] + pr_rec[:-1]).copy()
    den[den == 0.0] = 1e-12
    f1s = 2 * pr_prec[:-1] * pr_rec[:-1] / den
    ix = int(np.argmax(f1s))
    best_thr = float(pr_thr[ix])

    at_05 = thr_metrics(y_val, p_val, 0.5)
    at_best = thr_metrics(y_val, p_val, best_thr)
    at_best.update({"precision_curve": float(pr_prec[ix]), "recall_curve": float(pr_rec[ix])})

    # Precision@K slices (connects to analyst capacity)
    prec_at = {pct: precision_at_k(y_val, p_val, pct) for pct in [0.005, 0.01, 0.02, 0.05]}
    # Recall@FPR slices (customer impact)
    rec_at_fpr = {fpr: recall_at_fpr(y_val, p_val, fpr) for fpr in [0.001, 0.005, 0.01, 0.02]}

    # --- Plots ---
    # ROC
    fpr, tpr, _ = roc_curve(y_val, p_val)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={val_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Validation)")
    plt.legend(loc="lower right")
    roc_path = cfg.figures_dir / "roc_val.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight"); plt.close()

    # PR
    plt.figure()
    plt.plot(pr_rec, pr_prec, label=f"PR-AUC={val_ap:.3f}")
    plt.hlines(prevalence, xmin=0, xmax=1, linestyles="--", label=f"baseline={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Validation)")
    plt.legend(loc="upper right")
    pr_path = cfg.figures_dir / "pr_val.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight"); plt.close()

    # Precision@K bar
    plt.figure()
    ks = list(prec_at.keys()); vals = [prec_at[k] for k in ks]
    labels = [f"{int(k*100)}%" for k in ks]
    plt.bar(labels, vals)
    plt.ylim(0, 1)
    plt.ylabel("Precision")
    plt.title("Precision@Top-K% (Validation)")
    p_at_k_path = cfg.figures_dir / "precision_at_k_val.png"
    plt.savefig(p_at_k_path, dpi=150, bbox_inches="tight"); plt.close()

    # Recall@FPR bar
    plt.figure()
    fs = list(rec_at_fpr.keys()); vals = [rec_at_fpr[f] for f in fs]
    labels = [f"{int(f*1000)/10:.1f}%" for f in fs]  # e.g., 0.5%
    plt.bar(labels, vals)
    plt.ylim(0, 1)
    plt.ylabel("Recall")
    plt.title("Recall@FPR (Validation)")
    r_at_fpr_path = cfg.figures_dir / "recall_at_fpr_val.png"
    plt.savefig(r_at_fpr_path, dpi=150, bbox_inches="tight"); plt.close()

    # --- Save evaluation summary ---
    eval_summary = {
        "dataset": {
            "path": str(cfg.parquet_path),
            "md5": file_md5(cfg.parquet_path),
            "val_rows": int(len(y_val)),
            "val_prevalence": prevalence,
        },
        "metrics": {
            "roc_auc": val_auc,
            "pr_auc": val_ap,
            "pr_baseline": prevalence,
            "thr_0_5": at_05,
            "thr_best_f1": at_best,
            "precision_at_k": prec_at,
            "recall_at_fpr": rec_at_fpr,
        },
        "figures": {
            "roc": str(roc_path),
            "pr": str(pr_path),
            "precision_at_k": str(p_at_k_path),
            "recall_at_fpr": str(r_at_fpr_path),
        },
        "best_iteration": int(best_iter) if best_iter is not None else None,
    }
    out_path = cfg.reports_dir / "eval_summary_v0.json"
    with open(out_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Saved:\n - {roc_path}\n - {pr_path}\n - {p_at_k_path}\n - {r_at_fpr_path}\n - {out_path}")

if __name__ == "__main__":
    main()
