import json
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parents[2]
    parquet_path: Path = root / "data" / "interim" / "train_clean_baseline.parquet"
    reports_dir: Path = root / "data" / "interim" / "reports"
    model_dir: Path = root / "models"
    features_file: str = "features_v0.json"
    split_file: str = "split_v0.json"
    preprocess_file: str = "preprocess_v0.json"
    out_report_file: str = "model_choice_v0.json"

@dataclass(frozen=True)
class ReportContracts:
    features_rep: Dict[str, Any]
    split_rep: Dict[str, Any]
    preprocess_rep: Dict[str, Any]
    
def read_reports(cfg: Config) -> ReportContracts:
    data_path = cfg.reports_dir
    return ReportContracts(
        features_rep=json.load(open(data_path / cfg.features_file, "r")),
        split_rep=json.load(open(data_path / cfg.split_file, "r")), 
        preprocess_rep=json.load(open(data_path / cfg.preprocess_file, "r"))
    )

def write_baseline_model_choice_report(cfg: Config):
    reports = read_reports(cfg)

    scale_pos_weight = round(
        reports.split_rep["train"]["neg"] / reports.split_rep["train"]["pos"], 2
    )

    report = {
        "t_cut": reports.split_rep["t_cut"],
        "train_rows": reports.split_rep["train"]["rows"],
        "val_rows": reports.split_rep["val"]["rows"],
        "train_prevalence": reports.split_rep["train"]["prevalence"],
        "val_prevalence": reports.split_rep["val"]["prevalence"],

        "n_features_total": reports.features_rep["n_features_total"],
        "categorical_handling": "ordinal_encoded, unknown=-1",

        "imbalance_handling": {
            "pos": reports.split_rep["train"]["pos"],
            "neg": reports.split_rep["train"]["neg"],
            "scale_pos_weight": scale_pos_weight,
        },

        "lightgbm_params": {
            "objective": "binary",
            "metric": ["auc", "average_precision"],
            "learning_rate": 0.05,
            "n_estimators": 5000,
            "early_stopping_rounds": 100,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42
        },

        "eval_protocol": "eval_set = val, keep best_iteration, log AUC and PR-AUC"
    }

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.reports_dir / cfg.out_report_file, "w") as f:
        json.dump(report, f, indent=2)

    
if __name__ == "__main__":
    cfg = Config()
    write_baseline_model_choice_report(cfg)
