import duckdb
import pandas as pd
from pathlib import Path

RAW = Path("../data/raw")
DB_PATH = Path("../data/ieee_fraud.duckdb")

FILES = {
    "train_transation": RAW / "train_transaction.csv",
    "train_identity":   RAW / "train_identity.csv",
    "test_transation":  RAW / "test_transaction.csv",
    "test_identity":    RAW / "test_identity.csv",
}

def load_csv_to_duckdb(conn, table, path):
    print(f"Loading {table} from {path}")
    df = pd.read_csv(path)
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.register(f"df_{table}", df)
    conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df_{table}")
    conn.unregister(f"df_{table}")
    
def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    
    # Load train & test tables
    load_csv_to_duckdb(conn, "train_transation", FILES["train_transation"])
    load_csv_to_duckdb(conn, "train_identity", FILES["train_identity"])
    load_csv_to_duckdb(conn, "test_transation", FILES["test_transation"])
    load_csv_to_duckdb(conn, "test_identity", FILES["test_identity"])
    
    # Basic sanity checks
    print("\nSanity Checks:")
    for t in ["train_transaction","train_identity","test_transaction","test_identity"]:
        n = conn.execute(f"SELECT COUNT (*) FROM {t}").fetchone()[0]
        print(f"{t}: {n:,} rows")
    
    # Create a convenient joined view for modeling
    conn.execute("DROP VIEW IF EXISTS train_joined")
    conn.execute("""
                 CREATE VIEW train_joined AS
                 SELECT *
                 FROM train_transation t
                 LEFT JOIN train_identity i USING (TransactionID)
                 """)
    
    conn.execute("DROP VIEW IF EXISTS test_joined")
    conn.execute("""
                 CREATE VIEW test_joined
                 SELECT *
                 FROM test_transaction t
                 LEFT JOIN test_identity i USING (TransactionID)
                 """)
        
    print("\nCreated views: train_joined, test_joined")
    conn.close()
    print(f"Done. DuckDB at: {DB_PATH}")
    
if __name__ == "__main__":
    main()