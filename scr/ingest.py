# scr/ingest.py
from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd  # keep if you prefer the pandas->register path

REPO = Path(__file__).resolve().parents[1]
RAW  = REPO / "data" / "raw"
DB_PATH = REPO / "data" / "ieee_fraud.duckdb"

FILES = {
    "train_transaction": RAW / "train_transaction.csv",
    "train_identity":    RAW / "train_identity.csv",
    "test_transaction":  RAW / "test_transaction.csv",
    "test_identity":     RAW / "test_identity.csv",
}

def load_csv_to_duckdb(conn: duckdb.DuckDBPyConnection, table: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file for {table}: {path}")
    print(f"Loading {table}: {path.name}")
    # Option A (your original approach via pandas)
    df = pd.read_csv(path)
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.register(f"df_{table}", df)
    conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df_{table}")
    conn.unregister(f"df_{table}")
    # --- Option B (recommended: faster, less memory)
    # conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto('{path.as_posix()}')")

def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Context manager ensures the DB is closed even if an error happens
    with duckdb.connect(str(DB_PATH)) as conn:
        # Load train & test tables
        for t in ["train_transaction", "train_identity", "test_transaction", "test_identity"]:
            load_csv_to_duckdb(conn, t, FILES[t])

        # Basic sanity checks
        print("\nSanity Checks:")
        for t in ["train_transaction", "train_identity", "test_transaction", "test_identity"]:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"{t}: {n:,} rows")

        # Create convenient joined views for modeling
        conn.execute("DROP VIEW IF EXISTS train_joined")
        conn.execute("""
            CREATE VIEW train_joined AS
            SELECT t.*, i.*
            FROM train_transaction t
            LEFT JOIN train_identity i USING (TransactionID)
        """)

        conn.execute("DROP VIEW IF EXISTS test_joined")
        conn.execute("""
            CREATE VIEW test_joined AS
            SELECT t.*, i.*
            FROM test_transaction t
            LEFT JOIN test_identity i USING (TransactionID)
        """)

        print("\nCreated views: train_joined, test_joined")

    print(f"Done. DuckDB at: {DB_PATH}")

if __name__ == "__main__":
    main()
