import numpy as np

import json
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from pandas.api.types import is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    parquet_path: Path = root / "data" / "interim" / "train_clean_baseline.parquet"
    reports_dir: Path = root / "data" / "interim" / "reports"
    model_dir: Path = root / "models"
    time_col_candidates: tuple[str, ...] = ("dt", "TransactionDT")
    features_file: str = "features_v0.json"
    split_file: str = "split_v0.json"
    out_prep_file: str = "preprocessor_baseline.pkl"
    out_report_file: str = "preprocess_v0.json"

@dataclass(frozen=True)
class UpstreamContracts:
    numeric: list[str]
    categorical: list[str]
    t_cut: str  
    dropped: list[str]
    n_numeric: int
    n_categorical: int
    time_col: str
    t_cut: Any  
  
def load_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.parquet_path)
    # Normalize placeholders to NaN (keeps consistency with earlier steps)
    df = df.replace("missing", np.nan)
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].replace([-1, -999], np.nan)
    return df

def read_reports(cfg: Config) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load feature and split reports from JSON files inside `data_path`."""
    data_path = cfg.reports_dir
    features_report = json.load(open(data_path / cfg.features_file, "r"))
    split_report    = json.load(open(data_path / cfg.split_file, "r"))
    return features_report, split_report

def extract_contracts(features_report: Dict[str, Any], split_report: Dict[str, Any]) -> None:
    """Validate consistency between reported metadata and actual lists."""
    
    # Feature count checks
    assert len(features_report["numeric"]) == features_report["n_numeric"], \
        f"Expected {features_report['n_numeric']} numeric features, got {len(features_report['numeric'])}"
    assert len(features_report["categorical"]) == features_report["n_categorical"], \
        f"Expected {features_report['n_categorical']} categorical features, got {len(features_report['categorical'])}"

    # Split checks
    assert "train" in split_report and "val" in split_report, "Split report missing 'train' or 'val' keys"

    train = split_report["train"]
    val  = split_report["val"]

    print(f"Train set: {train['rows']} rows, {train['time_min']} → {train['time_max']}")
    print(f"Test set:  {val['rows']} rows, {val['time_min']} → {val['time_max']}")
    
    time_col = split_report["time_column"] if "time_column" in split_report else split_report.get("time_col", "dt")

    return UpstreamContracts(
        numeric=features_report["numeric"],
        categorical=features_report["categorical"],
        dropped=features_report["dropped_features"],
        n_numeric=features_report["n_numeric"],
        n_categorical=features_report["n_categorical"],
        time_col=time_col,
        t_cut=split_report['t_cut'],
    )
    
def reconstruct_split(df: pd.DataFrame, c: UpstreamContracts) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.issubdtype(df[c.time_col].dtype, np.datetime64):
        df[c.time_col] = pd.to_datetime(df[c.time_col], errors="raise", utc=False)
        
    # Sort ascending
    df = df.sort_values(c.time_col).reset_index(drop=True)
    
    t_cut = pd.to_datetime(c.t_cut, utc=False)
    train = df[df[c.time_col] <= t_cut]
    val   = df[df[c.time_col]  > t_cut]

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

def main():
    cfg = Config()
    
    features_report, split_report = read_reports(cfg)
    contracts = extract_contracts(features_report, split_report)
    
    df = load_dataset(cfg)
    # Restrict to the fixed feature set (order matters)
    features = contracts.numeric + contracts.categorical
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing contracted columns in parquet: {missing}")
    
    train_df, val_df = reconstruct_split(df, contracts)
    X_train = train_df[features]
    X_val = val_df[features]
    y_train = train_df["isFraud"]
    y_val = val_df["isFraud"]
    
    preproc = build_preprocessor(contracts.numeric, contracts.categorical)
    preproc.fit(X_train)
    Xtr = preproc.transform(X_train)
    Xva = preproc.transform(X_val)
    
    joblib.dump(preproc, cfg.model_dir / cfg.out_prep_file)

    
if __name__ == "__main__":
    main()
