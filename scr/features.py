from pathlib import Path as _Path
import duckdb
import pandas as pd
import numpy as np

PROJECT_ROOT = _Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "ieee_fraud.duckdb"
INTERIM = PROJECT_ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

DROP_MISSING_THRESHOLD = 0.90

def load_train_joined(limit: int | None = None) -> pd.DataFrame:
    import duckdb
    from pathlib import Path

    # read_only=True avoids write lock
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        q = "SELECT * FROM train_joined"
        if limit:
            q += f" LIMIT {int(limit)}"
        return conn.execute(q).df()

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns with >90% missing
    miss = df.isna().mean()
    to_drop = miss[miss > DROP_MISSING_THRESHOLD].index.tolist()
    df = df.drop(columns=to_drop)

    # Minimal time features (as in EDA notebook)
    dt0 = pd.Timestamp("2017-01-01")
    dt = pd.to_timedelta(df["TransactionDT"].fillna(0), unit="s")
    df["dt"] = dt0 + dt
    df["hour"] = df["dt"].dt.hour
    df["day"] = (df["dt"].dt.normalize() - dt0.normalize()).dt.days
    df["weekday"] = df["dt"].dt.weekday

    # Separate numeric/categorical
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Impute
    df[num_cols] = df[num_cols].fillna(-999)
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("missing")

    return df

def save_parquet(df: pd.DataFrame, name: str):
    out = INTERIM / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out

def add_time_split(df, cutoff_day: int):
    # cutoff_day from the 'day' feature we created
    df = df.copy()
    df["split"] = np.where(df["day"] < cutoff_day, "train", "val")
    return df