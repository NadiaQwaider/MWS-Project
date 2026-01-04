# create_db.py
import sqlite3
import pandas as pd
from utils import load_wdbc, BENIGN, MALIGNANT, LABEL_MAP_INT2STR

DB_PATH = "wdbc.db"

def create_tables(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        diagnosis_int INTEGER NOT NULL,
        diagnosis_text TEXT NOT NULL
    )
    """)
    conn.execute("DELETE FROM samples")

def insert_samples(conn, X, y, feat_names):
    # جدول الخصائص
    cols_def = ", ".join([f'"{c}" REAL' for c in feat_names])
    conn.execute(f'CREATE TABLE IF NOT EXISTS features (sample_id INTEGER, {cols_def})')
    conn.execute("DELETE FROM features")

    for i in range(X.shape[0]):
        d_int = int(y[i])
        d_txt = LABEL_MAP_INT2STR[d_int]
        conn.execute("INSERT INTO samples (diagnosis_int, diagnosis_text) VALUES (?, ?)", (d_int, d_txt))
        sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        placeholders = ", ".join(["?"] * X.shape[1])
        conn.execute(f'INSERT INTO features (sample_id, {", ".join([f"""\"{c}\"""" for c in feat_names])}) VALUES (?, {placeholders})',
                     tuple([sid] + list(map(float, X[i, :]))))
    conn.commit()

if __name__ == "__main__":
    X, y, feat_names = load_wdbc()
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    insert_samples(conn, X, y, feat_names)
    conn.close()
    print(f"Created {DB_PATH} with {X.shape[0]} rows, {len(feat_names)} features.")
