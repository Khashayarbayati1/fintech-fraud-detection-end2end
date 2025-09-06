# scr/eval/cv_time_splits.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    parquet_path: Path = root / "data" / "interim" / "train_clean_baseline.parquet"
    reports_dir: Path = root / "data" / "interim" / "reports"
    model_dir: Path = root / "models"
    preprocessor_file: str = "preprocessor_baseline.pkl"
    preprocess_file: str = "preprocess_v0.json"

def run_fold(df, preproc, features, time_col, train_end, val_end, base_params):
    # Train: <= train_end, Val: (train_end, val_end]
    train = df[df[time_col] <= train_end]
    val   = df[(df[time_col] > train_end) & (df[time_col] <= val_end)]
    Xtr, ytr = preproc.transform(train[features]), train["isFraud"].to_numpy()
    Xva, yva = preproc.transform(val[features]),   val["isFraud"].to_numpy()

    clf = LGBMClassifier(**base_params)
    callbacks = [lgb.early_stopping(stopping_rounds=100, first_metric_only=True)]
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=["auc","average_precision"], callbacks=callbacks)
    p = clf.predict_proba(Xva, num_iteration=clf.best_iteration_)[:,1]
    return float(roc_auc_score(yva, p)), float(average_precision_score(yva, p)), int(clf.best_iteration_)

def main():
    cfg = Config()
    prep_rep = json.load(open(cfg.reports_dir / cfg.preprocess_file))
    features = prep_rep["feature_order"]
    time_col = json.load(open(cfg.reports_dir / "split_v0.json"))["time_column"]
    df = pd.read_parquet(cfg.parquet_path)
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].sort_values(time_col).reset_index(drop=True)

    preproc = joblib.load(cfg.model_dir / cfg.preprocessor_file)

    # base params (match your training)
    base_params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "n_estimators": 5000,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "scale_pos_weight": 27.46,
        "random_state": 42,
        "n_jobs": -1,
        "force_row_wise": True,
    }

    # Define rolling folds (example: 3 folds)
    cuts = pd.to_datetime([
        "2017-04-15", "2017-05-01", "2017-05-15", "2017-05-30"
    ])
    results = []
    for i in range(len(cuts)-1):
        auc, ap, it = run_fold(df, preproc, features, time_col, cuts[i], cuts[i+1], base_params)
        results.append({"fold": i, "train_end": str(cuts[i]), "val_end": str(cuts[i+1]),
                        "roc_auc": auc, "pr_auc": ap, "best_iter": it})
        print(results[-1])

    out = cfg.reports_dir / "cv_time_splits_v0.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"Saved CV results → {out}")

if __name__ == "__main__":
    main()
