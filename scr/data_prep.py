# scr/data_prep.py
from __future__ import annotations

from pathlib import Path as _Path
import duckdb
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

PROJECT_ROOT = _Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "ieee_fraud.duckdb"
INTERIM = PROJECT_ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

DROP_MISSING_THRESHOLD = 0.90


def load_train_joined(limit: int | None = None) -> pd.DataFrame:
    """Load the joined training table from DuckDB as a DataFrame."""
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        q = "SELECT * FROM train_joined"
        if limit:
            q += f" LIMIT {int(limit)}"
        df = conn.execute(q).df()

    # Normalize sentinel missing values
    df = df.replace("missing", np.nan)
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].replace([-1, -999], np.nan)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning for baseline parquet: drop ultra-sparse cols, add time features."""
    miss = df.isna().mean()
    to_drop = miss[miss > DROP_MISSING_THRESHOLD].index.tolist()
    df = df.drop(columns=to_drop)

    # build all new columns first
    dt0 = pd.Timestamp("2017-01-01")
    dt = pd.to_timedelta(df["TransactionDT"].fillna(0), unit="s")
    dt_series = dt0 + dt

    new_cols = pd.DataFrame({
        "dt": dt_series,
        "hour": dt_series.dt.hour,
        "day": (dt_series.dt.normalize() - dt0.normalize()).dt.days,
        "weekday": dt_series.dt.weekday,
    }, index=df.index)

    # add them all at once (prevents fragmentation)
    df = pd.concat([df, new_cols], axis=1)
    
    return df


def save_parquet(df: pd.DataFrame, name: str) -> _Path:
    out = INTERIM / f"{name}.parquet"
    df.to_parquet(out, index=False)
    print(f"[DONE] wrote {out}")
    return out


if __name__ == "__main__":
    df = load_train_joined()             # add limit=200_000 if memory is tight
    clean = basic_clean(df)
    save_parquet(clean, "train_clean_baseline")
