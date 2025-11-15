# app.py
import os
import sqlite3
import pandas as pd
import streamlit as st
from create_db import create_wdbc_db
from genetic_algorithm import run_ga
from utils import plot_fitness_interactive, display_selected_features

DB_NAME = "medical_data.db"
st.set_page_config(page_title="GA Feature Selection — WDBC", layout="wide")

# ----- CSS للـ Tabs و أيقونات -----
st.markdown("""
<style>
.stTabs [role="tablist"] button[aria-selected="true"] {
    background-color:#1c7ed6; color:white; border-radius:10px; font-weight:600;
    box-shadow: 0 4px 10px rgba(0,0,0,0.12);
}
.stTabs [role="tablist"] button:hover {
    background-color:#4dabf7; color:white; transform:scale(1.03);
}
.icon-btn{border:none; background:transparent; font-size:18px;}
.cell{overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
</style>
""", unsafe_allow_html=True)

# ----- دوال DB مساعدة -----
def conn():
    return sqlite3.connect(DB_NAME)

def db_exists():
    return os.path.exists(DB_NAME)

def ensure_db():
    if not db_exists():
        st.warning("Database not found. Create it from WDBC.")
        if st.button("Initialize WDBC Database"):
            create_wdbc_db(force_recreate=False)
            st.experimental_rerun()

def get_feature_names():
    with conn() as c:
        cur = c.cursor()
        cur.execute("PRAGMA table_info(medical_dataset)")
        cols = [r[1] for r in cur.fetchall()]
    return [c for c in cols if c not in ("id", "Diagnosis")]

def read_db():
    with conn() as c:
        df = pd.read_sql("SELECT * FROM medical_dataset", c)
    return df

def insert_row(values_dict):
    cols = ", ".join([f'"{k}"' for k in values_dict.keys()])
    placeholders = ", ".join(["?"]*len(values_dict))
    sql = f'INSERT INTO medical_dataset ({cols}) VALUES ({placeholders})'
    with conn() as c:
        c.execute(sql, tuple(values_dict.values()))

def update_row(row_id, updated):
    set_clause = ", ".join([f'"{k}"=?' for k in updated.keys()])
    sql = f'UPDATE medical_dataset SET {set_clause} WHERE id=?'
    with conn() as c:
        c.execute(sql, tuple(updated.values()) + (row_id,))

def delete_row(row_id):
    with conn() as c:
        c.execute('DELETE FROM medical_dataset WHERE id=?', (row_id,))

# ----- Session state -----
if "ga_results" not in st.session_state:
    st.session_state.ga_results = None
if "fitness_history" not in st.session_state:
    st.session_state.fitness_history = None

# ----- Tabs -----
tabs = st.tabs(["🏠 Home", "📊 Database", "📂 Upload CSV", "⚙️ GA Settings", "📈 Results"])

# ----- Home -----
with tabs[0]:
    st.title("Genetic Algorithm Feature Selection — WDBC")
    st.write("Interactive app to perform feature selection on the Wisconsin Diagnostic dataset (WDBC).")
    st.image("https://images.unsplash.com/photo-1581091870622-5f241b36dfb1?auto=format&fit=crop&w=1600&q=80", use_container_width=True)
    st.markdown("<p style='text-align:center;color:gray;'>Created by Nadia Qwaider - 2025</p>", unsafe_allow_html=True)

# ----- Database (CRUD, table with icons) -----
with tabs[1]:
    st.header("Database Management (WDBC)")
    ensure_db()
    if not db_exists():
        st.stop()

    df = read_db()
    if df.empty:
        st.info("Database is empty.")
    else:
        # Insert expander
        with st.expander("➕ Insert New Row", expanded=False):
            feat_names = get_feature_names()
            new_vals = {}
            cols3 = st.columns(3)
            for i, f in enumerate(feat_names):
                with cols3[i % 3]:
                    v = st.text_input(f"{f}", key=f"ins_{f}")
                    new_vals[f] = None if v == "" else float(v)
            diag = st.selectbox("Diagnosis", ["Benign","Malignant"], key="ins_diag")
            new_vals["Diagnosis"] = diag
            if st.button("Insert Row"):
                try:
                    insert_row(new_vals)
                    st.success("Row inserted.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Insert failed: {e}")

        st.subheader("Records (quick view)")
        # show subset columns for readability
        preferred = ["mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness"]
        show_cols = ["id"] + [c for c in preferred if c in df.columns] + ["Diagnosis"]
        header = st.columns(len(show_cols)+2)
        for i, name in enumerate(show_cols):
            header[i].markdown(f"**{name}**")
        header[-2].markdown("**Edit**")
        header[-1].markdown("**Delete**")

        for _, row in df.iterrows():
            rcols = st.columns(len(show_cols)+2)
            for i, name in enumerate(show_cols):
                rcols[i].write(f"<div class='cell'>{row[name]}</div>", unsafe_allow_html=True)

            # Edit icon (opens inline expander)
            if rcols[-2].button("✏️", key=f"edit_{row['id']}"):
                st.session_state[f"edit_open_{row['id']}"] = True
            # Delete icon
            if rcols[-1].button("🗑️", key=f"del_{row['id']}"):
                delete_row(int(row["id"]))
                st.warning(f"Row {int(row['id'])} deleted.")
                st.experimental_rerun()

            # Edit form
            if st.session_state.get(f"edit_open_{row['id']}", False):
                with st.expander(f"Edit Row ID {int(row['id'])}", expanded=True):
                    edit_cols = [c for c in df.columns if c not in ("id",)]
                    updated = {}
                    grid = st.columns(3)
                    for j, col in enumerate(edit_cols):
                        with grid[j % 3]:
                            if col == "Diagnosis":
                                val = st.selectbox("Diagnosis", ["Benign","Malignant"], index=0 if row[col]=="Benign" else 1, key=f"diag_{row['id']}")
                                updated[col] = val
                            else:
                                val = st.text_input(col, value=str(row[col]), key=f"edit_{row['id']}_{col}")
                                updated[col] = None if val=="" else float(val)
                    c1, c2 = st.columns([1,1])
                    if c1.button("💾 Save", key=f"save_{row['id']}"):
                        update_row(int(row["id"]), updated)
                        st.success(f"Row {int(row['id'])} updated.")
                        st.session_state[f"edit_open_{row['id']}"] = False
                        st.experimental_rerun()
                    if c2.button("Cancel", key=f"cancel_{row['id']}"):
                        st.session_state[f"edit_open_{row['id']}"] = False
                        st.experimental_rerun()

# ----- Upload CSV -----
with tabs[2]:
    st.header("Upload CSV (optional)")
    st.write("Upload a CSV that matches WDBC columns (features names with underscores) and a 'Diagnosis' column.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            up_df = pd.read_csv(uploaded)
            st.dataframe(up_df.head(), use_container_width=True)
            if st.button("Append to DB"):
                cols_db = get_feature_names() + ["Diagnosis"]
                common = [c for c in cols_db if c in up_df.columns]
                if not common:
                    st.error("Uploaded CSV has no matching columns.")
                else:
                    with conn() as c:
                        up_df[common].to_sql("medical_dataset", c, if_exists="append", index=False)
                    st.success("Data appended to DB.")
        except Exception as e:
            st.error(f"Upload error: {e}")

# ----- GA Settings -----
with tabs[3]:
    st.header("GA Settings (runs on Local DB)")
    ensure_db()
    if not db_exists():
        st.stop()
    df = read_db().drop(columns=["id"])
    cols = list(df.columns)
    target_col = "Diagnosis"
    feature_candidates = [c for c in cols if c != target_col]

    features = st.multiselect("Select features to include in GA", feature_candidates, default=feature_candidates[:min(12,len(feature_candidates))])
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        population_size = st.number_input("Population size", min_value=10, value=50)
    with c2:
        generations = st.number_input("Generations", min_value=1, value=25)
    with c3:
        mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.05)
    with c4:
        crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.8)

    if st.button("Run Genetic Algorithm"):
        if not features:
            st.error("Select at least one feature.")
        else:
            best_features, fitness_history = run_ga(df, features, target_col,
                                                    population_size=population_size,
                                                    generations=generations,
                                                    mutation_rate=mutation_rate,
                                                    crossover_rate=crossover_rate)
            st.session_state.ga_results = best_features
            st.session_state.fitness_history = fitness_history
            st.success("GA finished — check Results tab.")

# ----- Results -----
with tabs[4]:
    st.header("Results")
    if not st.session_state.ga_results:
        st.info("Run GA first (in GA Settings).")
    else:
        display_selected_features(st.session_state.ga_results)
        st.subheader("Fitness Evolution")
        plot_fitness_interactive(st.session_state.fitness_history)
