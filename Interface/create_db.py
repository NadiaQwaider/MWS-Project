# create_db.py
import os
import sqlite3
from sklearn.datasets import load_breast_cancer

DB_NAME = "medical_data.db"

def create_wdbc_db(force_recreate=False):
    data = load_breast_cancer()
    feature_names = [f.replace(" ", "_") for f in data.feature_names]
    X, y = data.data, data.target

    if force_recreate and os.path.exists(DB_NAME):
        os.remove(DB_NAME)

    # إذا موجودة وما بنعيد الإنشاء: نوقف
    if os.path.exists(DB_NAME) and not force_recreate:
        print(f"{DB_NAME} already exists — skipping creation.")
        return

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    cols_def = ", ".join([f'"{col}" REAL' for col in feature_names])
    c.execute("DROP TABLE IF EXISTS medical_dataset")
    c.execute(f"""
        CREATE TABLE medical_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {cols_def},
            Diagnosis TEXT
        )
    """)

    cols_quoted = ", ".join([f'"{col}"' for col in feature_names])
    placeholders = ", ".join(["?"] * (len(feature_names) + 1))
    insert_sql = f'INSERT INTO medical_dataset ({cols_quoted}, "Diagnosis") VALUES ({placeholders})'

    for i in range(len(X)):
        diag = "Malignant" if y[i] == 0 else "Benign"
        vals = list(X[i]) + [diag]
        c.execute(insert_sql, tuple(vals))

    conn.commit()
    conn.close()
    print("✅ WDBC database created at", DB_NAME)

if __name__ == "__main__":
    # شغلي الملف: python create_db.py
    # لو بدك تعيد الإنشاء استخدمي create_wdbc_db(force_recreate=True)
    create_wdbc_db(force_recreate=False)
