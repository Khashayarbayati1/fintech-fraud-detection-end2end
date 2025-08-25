import json
import time
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score,
                            precision_recall_curve, precision_score, 
                            recall_score, f1_score, confusion_matrix)

import lightgbm as lgb
from lightgbm import LGBMClassifier

@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    parquet_path: Path = root / "data" / "interim" / "train_clean_baseline.parquet"
    reports_dir: Path = root / "data" / "interim" / "reports"
    model_dir: Path = root / "models"
    features_file: str = "features_v0.json"
    split_file: str = "split_v0.json"
    preprocess_file: str = "preprocess_v0.json"
    preprocessor_file: str = "preprocessor_baseline.pkl"
    model_choice_file: str = "model_choice_v0.json"  
    out_model_file: str = "baseline_lgbm.pkl"
    out_metrics_file: str = "baseline_summary_v0.json"
    
    time_col_candidates: tuple[str, ...] = ("dt", "TransactionDT")

@dataclass(frozen=True)
class ReportContracts:
    features_rep: Dict[str, Any]
    split_rep: Dict[str, Any]
    preprocess_rep: Dict[str, Any]
    model_choice_rep: Dict[str, Any]

@dataclass(frozen=True)
class UpstreamContracts:
    feature_order: list[str]
    time_column: str
    t_cut: str
    scale_pos_weight: float

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing report file: {path}")
    with open(path, "r") as f:
        return json.load(f)

def read_reports(cfg: Config) -> ReportContracts:
    d = cfg.reports_dir
    return ReportContracts(
        features_rep=_read_json(d / cfg.features_file),
        split_rep=_read_json(d / cfg.split_file),
        preprocess_rep=_read_json(d / cfg.preprocess_file),
        model_choice_rep=_read_json(d / cfg.model_choice_file),
    )

def extract_contracts(reports: ReportContracts) -> UpstreamContracts:
    pr = reports.preprocess_rep
    mr = reports.model_choice_rep

    # Fail fast with clear messages if structure changes
    for k in ("feature_order", "time_column", "t_cut"):
        if k not in pr:
            raise KeyError(f"preprocess report missing '{k}'")

    if "imbalance_handling" not in mr or "scale_pos_weight" not in mr["imbalance_handling"]:
        raise KeyError("model_choice report missing 'imbalance_handling.scale_pos_weight'")

    feature_order = pr["feature_order"]
    if not isinstance(feature_order, list) or not all(isinstance(c, str) for c in feature_order):
        raise TypeError("feature_order must be a list[str]")

    return UpstreamContracts(
        feature_order=feature_order,
        time_column=str(pr["time_column"]),
        t_cut=str(pr["t_cut"]),
        scale_pos_weight=float(mr["imbalance_handling"]["scale_pos_weight"]),
    )

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

def reconstruct_split(df: pd.DataFrame, c: UpstreamContracts, time_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors="raise", utc=False)
        
    # Sort ascending
    df = df.sort_values(time_col).reset_index(drop=True)
    
    t_cut = pd.to_datetime(c.t_cut, utc=False)
    train = df[df[time_col] <= t_cut]
    val   = df[df[time_col]  > t_cut]

    return train, val

def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    
    num_pip = Pipeline([
        ("impute", SimpleImputer(strategy="median"))
        ])
    cat_pip = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    
    return ColumnTransformer([
        ("num", num_pip, numeric),
        ("cat", cat_pip, categorical),
    ], remainder="drop")

def try_feature_names(preproc, feature_order: list[str]) -> list[str] | None:
    """
    Try to recover transformed feature names; fall back to None if not available.
    This depends on sklearn version & transformers exposing get_feature_names_out.
    """
    try:
        return list(preproc.get_feature_names_out(feature_order))
    except Exception:
        return None

# ---- Metrics at threshold = 0.5 ----
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
    reports = read_reports(cfg)
    contracts = extract_contracts(reports)

    df = load_dataset(cfg)

    # Enforce the contract’s time column
    time_col = pick_time_col(df, cfg.time_col_candidates)
    assert time_col == contracts.time_column, f"time column mismatch: picked {time_col} vs contract {contracts.time_column}"

    # Ensure datetime and drop missing times
    if not is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    # Split strictly via the contract (handles t_cut conversion)
    train_df, val_df = reconstruct_split(df, contracts, time_col)

    # Features in the exact contracted order
    features = contracts.feature_order
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing contracted columns in parquet: {missing}")

    X_train = train_df[features]
    X_val   = val_df[features]

    if "isFraud" not in df.columns:
        raise KeyError("'isFraud' column not found in data.")
    y_train = train_df["isFraud"].to_numpy()
    y_val   = val_df["isFraud"].to_numpy()

    # Load frozen preprocessor
    preproc_path = cfg.model_dir / cfg.preprocessor_file
    if not preproc_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {preproc_path}")
    preproc: ColumnTransformer = joblib.load(preproc_path)

    # Transform (NO fitting)
    Xtr = preproc.transform(X_train)
    Xva = preproc.transform(X_val)

    # Quick sanity checks
    print(f"Train transformed shape: {getattr(Xtr, 'shape', None)}")
    print(f"Val   transformed shape: {getattr(Xva, 'shape', None)}")

    # Load parameters
    mcr = reports.model_choice_rep # Model Choice Report
    base_params = mcr["lightgbm_params"].copy()
    # extract early stopping from JSON, then remove from base_params
    stopping_rounds = mcr["lightgbm_params"].get("early_stopping_rounds", 100)
    # Optional speed/repro tweaks
    base_params.setdefault("force_row_wise", True)
    base_params.setdefault("n_jobs", -1)
    # drop early stopping keys if they exist
    for bad_key in ("early_stopping_round", "early_stopping_rounds"):
        base_params.pop(bad_key, None)

    # mirror canonicals → aliases (explicit assignment)
    if "feature_fraction" in base_params:
        base_params["colsample_bytree"] = base_params["feature_fraction"]
    if "bagging_fraction" in base_params:
        base_params["subsample"] = base_params["bagging_fraction"]
    if "bagging_freq" in base_params:
        base_params["subsample_freq"] = base_params["bagging_freq"]
    if "min_data_in_leaf" in base_params:
        base_params["min_child_samples"] = base_params["min_data_in_leaf"]
        
    # Instantiate model
    clf = LGBMClassifier(**base_params)

    # Train with early stopping (monitor ROC-AUC first; log both AUC & AP)
    callbacks = [
        lgb.early_stopping(stopping_rounds=stopping_rounds, first_metric_only=True, verbose=True),
        lgb.log_evaluation(period=50),  # print every 50 rounds
    ]
    
    names = getattr(preproc, "get_feature_names_out", lambda *_: None)(features)

    # Wrap arrays as DataFrames if names are available
    if names is not None:
        import pandas as pd
        # some sklearn versions return a numpy array of dtype object
        names = list(names)
        if len(names) != Xtr.shape[1]:
            # fallback if the preprocessor can't map names 1:1
            Xtr_df, Xva_df = Xtr, Xva
        else:
            Xtr_df = pd.DataFrame(Xtr, columns=names)
            Xva_df = pd.DataFrame(Xva, columns=names)
    else:
        Xtr_df, Xva_df = Xtr, Xva

    t0 = time.perf_counter()
    clf.fit(
        Xtr_df, y_train,
        eval_set=[(Xva_df, y_val)],
        eval_metric=["auc", "average_precision"],  # put AUC first for early stopping
        callbacks=callbacks,
    )
    
    t1 = time.perf_counter()
    train_time_s = t1 - t0
    
    best_iter = int(clf.best_iteration_) if getattr(clf, "best_iteration_", None) else int(base_params["n_estimators"])
    print(f"Best iteration: {best_iter}  |  Train time: {train_time_s:.2f}s")
    
    # Evaluate at best iteration (threshold-free metrics)
    p_val = clf.predict_proba(Xva, num_iteration=best_iter)[:, 1]
    val_auc = float(roc_auc_score(y_val, p_val))
    val_ap  = float(average_precision_score(y_val, p_val))
    print(f"VAL ROC-AUC: {val_auc:.6f}  |  VAL PR-AUC (AP): {val_ap:.6f}")
    
    # ---- Prevalence & PR "no-skill" baseline ----
    val_prevalence = float(y_val.mean())        # share of positives in val
    pr_baseline = val_prevalence                # no-skill PR-AUC baseline
    
    # ---- Metrics at threshold = 0.5 ----
    at_05 = thr_metrics(y_val, p_val, 0.5)
    
    # ---- Best-F1 on validation ----
    prec, rec, thr = precision_recall_curve(y_val, p_val)   # len(thr) = len(prec) - 1
    # avoid division by zero
    den = (prec[:-1] + rec[:-1])
    den[den == 0.0] = 1e-12
    f1s = 2 * prec[:-1] * rec[:-1] / den
    ix = int(np.argmax(f1s))
    best_thr = float(thr[ix])
    best_f1_metrics = thr_metrics(y_val, p_val, best_thr)
    best_f1_metrics.update({"precision_curve": float(prec[ix]), "recall_curve": float(rec[ix])})

    print(f"Val prevalence (PR baseline): {val_prevalence:.4f}")
    print(f"@0.5  P={at_05['precision']:.3f} R={at_05['recall']:.3f} F1={at_05['f1']:.3f}")
    print(f"@bestF1 thr={best_thr:.4f}  P={best_f1_metrics['precision']:.3f} "
        f"R={best_f1_metrics['recall']:.3f} F1={best_f1_metrics['f1']:.3f}")

    # Persist model & metrics
    model_out_path   = cfg.model_dir / cfg.out_model_file
    metrics_out_path = cfg.reports_dir / cfg.out_metrics_file
    
    joblib.dump(clf, model_out_path)
    
    metrics_report = {
        "best_iteration": best_iter,
        "val_auc_roc": val_auc,
        "val_pr_auc": val_ap,
        "train_time_seconds": train_time_s,
        "n_trees_cap": int(base_params["n_estimators"]),
        "early_stopping_rounds": stopping_rounds,
        "scale_pos_weight": contracts.scale_pos_weight,
        "val_prevalence": val_prevalence,
        "pr_baseline": pr_baseline,
        "thr_0_5": at_05,
        "thr_best_f1": best_f1_metrics,
    }
    
    # Top feature importances (may not align 1:1 with original names after encoding)
    try:
        booster = clf.booster_
        gains = booster.feature_importance(importance_type="gain")
        names = try_feature_names(preproc, contracts.feature_order) or [f"f{i}" for i in range(len(gains))]
        order = np.argsort(gains)[::-1]
        topk = 25 if len(gains) > 25 else len(gains)
        metrics_report["top_importances_gain"] = [
            {"feature": names[i], "gain": float(gains[i])} for i in order[:topk]
        ]
    except Exception as e:
        print(f"(Skipping importances; reason: {e})")

    with open(metrics_out_path, "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"Saved model → {model_out_path}")
    print(f"Saved metrics → {metrics_out_path}")


if __name__ == "__main__":
    out = main()
    
