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
    scores = df_sorted["score"].unique()
    
    print(scores.shape)
    print(scores)
    budget = np.array([0.1, 0.5, 1, 2])
    
    
if __name__ == "__main__":
    main()
    