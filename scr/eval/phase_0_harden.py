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
    parquet_path: Path = root / "data" / "interim" / "train_clean_baseline.parquet"
    reports_dir: Path = root / "data" / "interim" / "reports"
    model_dir: Path = root / "models"
    model_choice_file: str = "model_choice_v0.json"
    split_file: str = "split_v0.json"
    features_file: str = "features_v0.json"
    preprocess_file: str = "preprocess_v0.json"
    preprocessor_file: str = "preprocessor_baseline.pkl"
    model_file: str = "baseline_lgbm.pkl"
    
def _read_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)
    
def main():
    cfg = Config()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load split & features info
    split_rep = _read_json(cfg.reports_dir / cfg.split_file)
    prep_rep  = _read_json(cfg.reports_dir / cfg.preprocess_file)
    features  = prep_rep["feature_order"]
    time_col  = split_rep["time_column"]
    t_cut     = pd.to_datetime(split_rep["t_cut"])

    # run id
    run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = cfg.reports_dir / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Run ID = {run_id}")
    
    # model choice metadata    
    mc_rep  = _read_json(cfg.reports_dir / cfg.model_choice_file) # model_choice_v0 report.
    random_state = mc_rep["lightgbm_params"]["random_state"]

    # Load data
    
    df = pd.read_parquet(cfg.parquet_path)
    # enforce time and sort
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].sort_values(time_col).reset_index(drop=True)

    # Split
    val_df   = df[df[time_col]  > t_cut]
    assert "TransactionID" in val_df.columns, "Missing TransactionID in validation slice"
    assert val_df["TransactionID"].notna().all(), "TransactionID has NaNs"
    # If TransactionID should be unique per row:
    assert not val_df["TransactionID"].duplicated().any(), "Duplicate TransactionID in val"

    X_val   = val_df[features]
    y_val   = val_df["isFraud"].to_numpy()

    assert val_df[time_col].notna().all(), "NaNs in time column"
    assert np.isfinite(y_val).all(), "Non-finite values in y"

    prevalence = float(y_val.mean())
    print(f"[INFO] Class prevalence (val): {prevalence:.4%}")

    # load frozen artifacts
    preproc = joblib.load(cfg.model_dir / cfg.preprocessor_file)
    model   = joblib.load(cfg.model_dir / cfg.model_file)

    # Transform (no fit!)
    X_val_proc = preproc.transform(X_val)
    
    # score
    scores = model.predict_proba(X_val_proc)[:, 1]   # probability of fraud
    assert np.isfinite(scores).all(), "Non-finite values in scores"

    meta = {
        "run_id": run_id,
        "score_type": "probability",   # because you used predict_proba[:,1]
        "time_column": time_col,
        "t_cut": str(pd.to_datetime(split_rep["t_cut"])),
        "model_file": cfg.model_file,
        "preprocessor_file": cfg.preprocessor_file,
        "features_version": cfg.features_file,
        "preprocess_version": cfg.preprocess_file,
        "split_version": cfg.split_file,
        "rows_val": int(len(val_df)),
        "positives_val": int(y_val.sum()),
        "prevalence_val": prevalence,
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # write output
    out_df = pd.DataFrame({
        "id": val_df["TransactionID"].to_numpy(),
        "time": val_df[time_col].to_numpy(),
        "y": y_val,
        "score": scores,
    })
    out_path = out_df.to_parquet(out_dir / "raw_scores.parquet", index=False)
    print(f"[INFO] Saved raw scores to {out_dir / 'raw_scores.parquet'}")

if __name__ == "__main__":
    main()