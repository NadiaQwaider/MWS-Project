# app.py
# ================================================================
# Breast Cancer – GA Feature Selection App (English UI) + DB CRUD
# ================================================================

import io
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import time
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# 3rd-party grid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# Project internals (as in your repo)
from utils import (
    load_wdbc, outer_split, clf_pipeline, compute_metrics, calibration_xy,
    save_json, jaccard, eval_model
)
from genetic_algorithm import GeneticAlgorithmFS, GAConfig
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "patients.db"

def db_connect():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

# --------------------- Page setup ---------------------
st.set_page_config(page_title="Medical Data Feature Selection Using Genetic Algorithm", layout="wide")

# --- run-after-rerun scroll to top ---
if st.session_state.get("__scroll_top_next_run", False):
    components.html(
        """
        <script>
          setTimeout(() => { window.scrollTo(0,0); }, 50);
        </script>
        """,
        height=0,
    )
    st.session_state["__scroll_top_next_run"] = False
    
st.title("Medical Data Feature Selection Using Genetic Algorithm App")
#    "Improved GA with inner-CV composite fitness. "
#   "Fitness = 0.5 × (Mean CV_Accuracy + Mean CV_F1) − α × (#selected / total_features). "
#   "Outer hold-out/CV evaluation, compact plots, and a dataset/DB manager."


# --------------------- Helpers ---------------------
def fmt_float(v):
    try:
        if v is None:
            return "NA"
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return "NA"
        return f"{float(v):.4f}"
    except Exception:
        return "NA"

def to_serializable(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, )):
        return int(o)
    if isinstance(o, (np.floating, )):
        return float(o)
    if isinstance(o, (np.bool_, )):
        return bool(o)
    return str(o)

def download_fig(fig, filename="figure.png", dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    st.download_button("⬇️ Download figure (PNG)", data=buf, file_name=filename, mime="image/png")

def download_df(df, filename="table.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download table (CSV)", data=csv, file_name=filename, mime="text/csv")


def _auto_compare_text(baseline_view: pd.DataFrame, ga_view: pd.DataFrame) -> str:
    """
    Compares baseline (all-features) vs GA-selected results using fuzzy model-family matching
    (LR / SVM / RF) even if model names differ and don't include (All)/(GA).
    Requires columns: Model, Accuracy, F1-macro, ROC-AUC (or close).
    """

    def _num(v):
        try:
            if v is None: return None
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
            return float(v)
        except Exception:
            return None

    def _get_metric(row, col_candidates):
        for c in col_candidates:
            if c in row and _num(row[c]) is not None:
                return c, _num(row[c])
        return None, None

    def _pick_metric(row):
        # prefer ROC-AUC, then F1, then Accuracy
        c, v = _get_metric(row, ["ROC-AUC", "ROC_AUC", "AUC", "auc", "roc_auc"])
        if v is not None: return "ROC-AUC", v
        c, v = _get_metric(row, ["F1-macro", "F1_macro", "F1", "f1", "f1_macro"])
        if v is not None: return "F1-macro", v
        c, v = _get_metric(row, ["Accuracy", "Acc", "accuracy", "acc"])
        return "Accuracy", v

    def _family(name: str) -> str:
        s = str(name).lower()

        # Logistic Regression
        if ("logistic" in s) or re.search(r"\blr\b", s):
            return "LR"

        # SVM
        if ("svm" in s) or ("support vector" in s):
            return "SVM"

        # Random Forest
        if ("random forest" in s) or re.search(r"\brf\b", s):
            return "RF"

        return "OTHER"

    if "name" not in baseline_view.columns:
        return "Auto-summary: could not find 'Model' column in baseline results."
    if "name" not in ga_view.columns:
        return "Auto-summary: could not find 'Model' column in GA results."

    # pick the first row for each family in baseline & GA
    b_byfam = {}
    for _, r in baseline_view.iterrows():
        fam = _family(r["name"])
        if fam != "OTHER" and fam not in b_byfam:
            b_byfam[fam] = r.to_dict()

    g_byfam = {}
    for _, r in ga_view.iterrows():
        fam = _family(r["name"])
        if fam != "OTHER" and fam not in g_byfam:
            g_byfam[fam] = r.to_dict()

    # build lines
    fam_pretty = {"LR": "Logistic Regression", "SVM": "SVM (RBF)", "RF": "Random Forest"}
    lines = []

    for fam in ["LR", "SVM", "RF"]:
        if fam not in b_byfam or fam not in g_byfam:
            continue

        rb = b_byfam[fam]
        rg = g_byfam[fam]

        metric_name_b, base_val = _pick_metric(rb)
        metric_name_g, ga_val   = _pick_metric(rg)

        # if metric types differ (rare), force same preference order by re-picking from both
        # based on baseline chosen metric
        preferred = metric_name_b
        if preferred == "ROC-AUC":
            _, base_val = _get_metric(rb, ["ROC-AUC", "ROC_AUC", "AUC", "auc", "roc_auc"])
            _, ga_val   = _get_metric(rg, ["ROC-AUC", "ROC_AUC", "AUC", "auc", "roc_auc"])
            metric_name = "ROC-AUC"
        elif preferred == "F1-macro":
            _, base_val = _get_metric(rb, ["F1-macro", "F1_macro", "F1", "f1", "f1_macro"])
            _, ga_val   = _get_metric(rg, ["F1-macro", "F1_macro", "F1", "f1", "f1_macro"])
            metric_name = "F1-macro"
        else:
            _, base_val = _get_metric(rb, ["Accuracy", "Acc", "accuracy", "acc"])
            _, ga_val   = _get_metric(rg, ["Accuracy", "Acc", "accuracy", "acc"])
            metric_name = "Accuracy"

        if base_val is None or ga_val is None:
            continue

        diff = ga_val - base_val
        if abs(diff) <= 0.01:
            verdict = "comparable"
        elif diff > 0.01:
            verdict = "higher"
        else:
            verdict = "slightly lower"
        lines.append(
            f"Using the GA-selected features, **{fam_pretty[fam]}** achieved **{verdict} {metric_name}** "
            f"({ga_val:.2f}) compared to using all features ({base_val:.2f})."
        )

    if not lines:
        return (
            "Auto-summary: results were generated, but model names could not be matched to LR/SVM/RF "
            "or there was not enough metric data to produce a comparison."
        )

    closing = (
        "Overall, the results suggest effective dimensionality reduction, where performance remains competitive "
        "while using fewer features—supporting better interpretability and potentially lower computational cost."
        " Marginal reduction is commonly observed in medical datasets "
        "such as Heart Disease or Diabetes, where predictive information is "
        "distributed across multiple weakly-informative features, making strict "
        "feature reduction more challenging without minor performance trade-offs."
    )
    return "\n\n".join(lines + [closing])

# --------------------- Load base data ---------------------
X, y, feat_names = load_wdbc()   # X: (569,30), y: (569,), feat_names: list of 30
P = X.shape[1]


# --------------------- Sidebar: GA config ---------------------
presets = {
    "Fast (demo)":  {"pop": 40, "gens": 25, "pc": 0.90, "pm": 0.03, "tk": 3, "elit": 2, "inner_k": 3, "lam": 0.5, "alpha": 0.10, "early": 8},
    "Balanced":     {"pop": 60, "gens": 50, "pc": 0.85, "pm": 0.04, "tk": 3, "elit": 2, "inner_k": 5, "lam": 0.5, "alpha": 0.10, "early": 10},
    "Thorough":     {"pop": 80, "gens": 70, "pc": 0.80, "pm": 0.05, "tk": 4, "elit": 3, "inner_k": 5, "lam": 0.5, "alpha": 0.12, "early": 12},
}
st.sidebar.header("GA Settings")
choice = st.sidebar.selectbox("Preset", list(presets.keys()))
cfg0 = presets[choice]

pop = st.sidebar.number_input("Population Size", 10, 500, cfg0["pop"], 5)
gens = st.sidebar.number_input("Generations", 5, 300, cfg0["gens"], 5)
pc = st.sidebar.slider("Crossover Probability", 0.0, 1.0, float(cfg0["pc"]), 0.01)
pm = st.sidebar.slider("Mutation Probability", 0.0, 1.0, float(cfg0["pm"]), 0.01)
tk = st.sidebar.number_input("Tournament k", 2, 10, cfg0["tk"], 1)
elit = st.sidebar.number_input("Elitism", 0, 10, cfg0["elit"], 1)
inner_k = st.sidebar.number_input("Inner CV folds (fitness)", 2, 10, cfg0["inner_k"], 1)
lam = st.sidebar.slider("λ (weight of F1 in fitness)", 0.0, 1.0, float(cfg0["lam"]), 0.05)  # kept for completeness; we use 0.5 in caption/formula
alpha = st.sidebar.slider("α penalty", 0.0, 0.5, float(cfg0["alpha"]), 0.01)
early_stop = st.sidebar.number_input("Early-stopping rounds", 0, 100, cfg0["early"], 1)
seed = st.sidebar.number_input("Random Seed", 0, 10_000, 42, 1)
lock_seed = st.sidebar.checkbox("Lock seed (reproducible)", value=True)

# Figure size
fig_w = st.sidebar.slider("Figure width (inch)", 3.0, 8.0, 4.2, 0.1)
fig_h = st.sidebar.slider("Figure height (inch)", 2.0, 6.0, 3.0, 0.1)

if lock_seed:
    np.random.seed(int(seed))


# --------------------- SQLite (DB) ---------------------
from pathlib import Path
import sqlite3
import streamlit as st

def db_init():
    with db_connect() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE,
            label INTEGER,           -- 0 benign, 1 malignant
            features_json TEXT,      -- dict of feature_name: value
            notes TEXT,
            created_at TEXT
        )
        """)
        con.commit()

def db_insert(patient_id: str, label: int, features: Dict[str, Any], notes: str = "") -> Tuple[bool, str]:
    try:
        with db_connect() as con:
            cur = con.cursor()
            cur.execute("""
            INSERT INTO patients (patient_id, label, features_json, notes, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (patient_id, int(label), json.dumps(features, default=to_serializable), notes, datetime.utcnow().isoformat()))
            con.commit()
        return True, "Added."
    except sqlite3.IntegrityError:
        return False, "⚠️ Duplicate patient_id."
    except Exception as e:
        return False, f"Error: {e}"

def db_update(row_id: int, label: int, features: Dict[str, Any], notes: str = "") -> Tuple[bool, str]:
    try:
        with db_connect() as con:
            cur = con.cursor()
            cur.execute("""
            UPDATE patients SET label=?, features_json=?, notes=? WHERE id=?
            """, (int(label), json.dumps(features, default=to_serializable), notes, int(row_id)))
            con.commit()
        return True, "Updated."
    except Exception as e:
        return False, f"Error: {e}"

def db_delete(row_id: int):
    try:
        with db_connect() as con:
            cur = con.cursor()

            cur.execute("DELETE FROM patients WHERE id = ?", (int(row_id),))
            con.commit()

            # number of rows deleted (SQLite reliable)
            cur.execute("SELECT changes()")
            deleted = cur.fetchone()[0]

            if deleted == 0:
                return False, f"No row deleted. (id={row_id}) not found in DB."
            return True, f"Deleted row id={row_id}"
    except Exception as e:
        return False, f"Error: {e}"

def db_fetch_all() -> pd.DataFrame:
    with db_connect() as con:
        df = pd.read_sql_query(
            "SELECT id, patient_id, label, features_json, notes, created_at "
            "FROM patients ORDER BY id DESC",
            con
        )
    return df

# ---------- Build BASE & ADDED dataframes ----------
def build_base_df() -> pd.DataFrame:
    base = pd.DataFrame(X, columns=list(map(str, feat_names)))
    base.insert(0, "patient_id", [f"WDBC_{i+1:04d}" for i in range(len(y))])
    base.insert(1, "label", y.astype(int))
    base["created_at"] = "—"
    base["source"] = "Base"
    base["id"] = None
    cols = ["source", "id", "patient_id", "label"] + list(map(str, feat_names)) + ["created_at"]
    base = base[cols]
    return base

def build_added_df() -> pd.DataFrame:
    df_db = db_fetch_all()
    if df_db is None or len(df_db) == 0:
        empty = pd.DataFrame(columns=["source", "id", "patient_id", "label"] + list(map(str, feat_names)) + ["created_at"])
        return empty
    def expand(js):
        try:
            d = json.loads(js) if isinstance(js, str) else (js or {})
            return pd.Series({str(k): d.get(str(k), np.nan) for k in map(str, feat_names)})
        except Exception:
            return pd.Series({str(k): np.nan for k in map(str, feat_names)})
    feat_expanded = df_db["features_json"].apply(expand)
    out = pd.concat([df_db[["id", "patient_id", "label", "created_at"]].reset_index(drop=True),
                     feat_expanded.reset_index(drop=True)], axis=1)
    out.insert(0, "source", "Added")
    cols = ["source", "id", "patient_id", "label"] + list(map(str, feat_names)) + ["created_at"]
    out = out[cols]
    return out

def merged_view() -> pd.DataFrame:
    add = build_added_df()
    base = build_base_df()
    df = pd.concat([add, base], ignore_index=True)
    return df

def ui_refresh(grid_key: str = "db_grid"):
    # 1) reload data from DB (THIS is the missing part)
    st.session_state["full_df"] = merged_view()

    # 2) clear action/pending
    st.session_state[f"{grid_key}__pending_action"] = None
    st.session_state[f"{grid_key}__pending_row"] = None
    st.session_state[f"{grid_key}__last_token"] = ""

    # 3) force AgGrid rebuild
    st.session_state.setdefault(f"{grid_key}__grid_version", 0)
    st.session_state[f"{grid_key}__grid_version"] += 1
    st.session_state["__scroll_top_next_run"] = True
    # 4) rerun
    st.rerun()



# ---------- AgGrid control table with inline icons ----------
def paginated_actions_table(df: pd.DataFrame, page_size: int = 20, key: str = "db_grid"):
    """
    AgGrid-based table (Streamlit 1.50 compatible):
      - 20 rows/page
      - Single 'Control' column with inline icons: 👁 | ✏️ | 🗑️
      - Clicks detected via hidden columns (__action, __token) written by JS
      - Base rows are view-only; Added rows editable/deletable
      - Delete confirmation uses red background (st.error)
    """


    if df is None or len(df) == 0:
        st.info("No data to show.")
        return

    show_cols = ["id", "patient_id", "label"] + [str(f) for f in feat_names] + ["source", "created_at"]
    show_cols = [c for c in show_cols if c in df.columns]
    data = df[show_cols].copy()

    # Hidden columns used as a signal channel from JS → Python
    data.insert(0, "__action", "")
    data.insert(1, "__token", "")
    data.insert(2, "Control", "")

    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_grid_options(
        pagination=True, paginationAutoPageSize=False, paginationPageSize=page_size,
        suppressRowClickSelection=True, rowSelection="single",
        getRowId=JsCode("function(p){ return p.data.patient_id ? p.data.patient_id : (p.data.id || p.node.id); }"),
    )

    # Column sizing
    gb.configure_column("__action", hide=True)
    gb.configure_column("__token", hide=True)
    gb.configure_column("Control", width=120, pinned=True)
    gb.configure_column("id", width=90)
    gb.configure_column("patient_id", width=160)
    gb.configure_column("label", width=90)
    gb.configure_column("source", width=100)
    gb.configure_column("created_at", width=160)

    # JS renderer writes action + token to hidden columns
    control_renderer = JsCode("""
    class ControlRenderer {
      init(params) {
        const isBase = (params.data && params.data.source === "Base");
        const wrap = document.createElement('div');
        wrap.style.display = 'flex';
        wrap.style.gap = '10px';
        wrap.style.alignItems = 'center';

        const mk = (txt, title, action, disabled) => {
          const a = document.createElement('a');
          a.textContent = txt;
          a.title = title;
          a.style.cursor = disabled ? 'not-allowed' : 'pointer';
          a.style.opacity = disabled ? '0.35' : '1';
          a.style.userSelect = 'none';
          a.onclick = (e) => {
            e.preventDefault();
            if (disabled) return;
            const now = Date.now().toString();
            params.node.setDataValue('__action', action);
            params.node.setDataValue('__token', now);
          };
          return a;
        };

        wrap.appendChild(mk("👁", "View row", "view", false));
        wrap.appendChild(mk("✏️", "Edit row", "edit", isBase));
        wrap.appendChild(mk("🗑️", "Delete row", "delete", isBase));
        this.eGui = wrap;
      }
      getGui(){ return this.eGui; }
    }
    """)
    gb.configure_column("Control", cellRenderer=control_renderer)

    grid_options = gb.build()
    st.session_state.setdefault(f"{key}__grid_version", 0)
    grid_version = st.session_state[f"{key}__grid_version"]
    grid = AgGrid(
        data,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=False,
        height=560,
        allow_unsafe_jscode=True,
        theme="material",
        key=f"{key}_ag_v{grid_version}",
    )

    df_after = grid["data"] if "data" in grid else pd.DataFrame()
    st.session_state.setdefault(f"{key}__last_token", "")
    last_token = st.session_state[f"{key}__last_token"]

    act_row, act_typ = None, None
    if not df_after.empty and "__action" in df_after.columns and "__token" in df_after.columns:
        clicks = df_after[df_after["__action"].astype(str).str.len() > 0].copy()
        if len(clicks):
            clicks["__token_num"] = pd.to_numeric(clicks["__token"], errors="coerce")
            clicks = clicks.dropna(subset=["__token_num"])
            if len(clicks):
                latest = clicks.sort_values("__token_num", ascending=False).iloc[0]
                token = str(int(latest["__token_num"]))
                if token != last_token:
                    st.session_state[f"{key}__last_token"] = token
                    act_typ = str(latest["__action"])
                    act_row = latest.to_dict()

    # --- Persist last action so Streamlit reruns (button click) won't lose it ---
    st.session_state.setdefault(f"{key}__pending_action", None)
    st.session_state.setdefault(f"{key}__pending_row", None)

    if act_row is not None and act_typ is not None:
        st.session_state[f"{key}__pending_action"] = act_typ
        st.session_state[f"{key}__pending_row"] = act_row
    else:
        # Use pending if available
        act_typ = st.session_state.get(f"{key}__pending_action")
        act_row = st.session_state.get(f"{key}__pending_row")

    if act_row is None or act_typ is None:
        return


    # Actions as inline expanders
    if act_typ == "view":
        with st.expander(f"Row details — {act_row.get('patient_id')} ({act_row.get('source')})", expanded=True):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.write(f"**Label:** {act_row.get('label')}")
                st.write(f"**Source:** {act_row.get('source')}")
                st.write(f"**Created:** {act_row.get('created_at')}")
            with c2:
                feats = {str(f): act_row.get(str(f)) for f in feat_names}
                st.dataframe(pd.DataFrame([feats]), use_container_width=True)

    elif act_typ == "edit":
        if act_row.get("source") == "Base":
            st.error("Base rows are locked (cannot edit).")
            return
        with st.expander(f"Edit — {act_row.get('patient_id')}", expanded=True):
            with st.form("ag_edit_form"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.text_input("Patient ID", value=act_row.get("patient_id"), disabled=True)
                with c2:
                    new_label = st.selectbox("Label", [0, 1], index=int(act_row.get("label", 0)))
                cols = st.columns(3)
                new_feats = {}
                for j, fname in enumerate(feat_names):
                    v = act_row.get(str(fname))
                    v = 0.0 if (v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))) else float(v)
                    with cols[j % 3]:
                        new_feats[str(fname)] = st.number_input(str(fname), value=v, format="%.6f", key=f"ag_edit_{j}")
                notes_edit = st.text_area("Notes", value="")
                saved = st.form_submit_button("Save changes")
            if saved:
                row_id = act_row.get("id")
                if row_id is None or (isinstance(row_id, float) and np.isnan(row_id)):
                    st.error("Invalid row id for update.")
                else:
                    ok, msg = db_update(int(row_id), int(new_label), new_feats, notes_edit)
                    (st.success if ok else st.error)(msg)
                    if ok:
                      time.sleep(1.5)  
                      ui_refresh("db_grid")

    elif act_typ == "delete":
     with st.expander("Confirm deletion", expanded=True):
        pid = act_row.get("patient_id")
        row_id = act_row.get("id")
        tok = act_row.get("__token", "t0")

        st.error(f'Are you sure you want to delete the patient "{pid}" details from the database? This action cannot be undone.')

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, delete", key=f"yes_del_{row_id}_{tok}"):
                if act_row.get("source") == "Base":
                    st.error("Base rows are locked (cannot delete).")
                else:
                    ok, msg = db_delete(int(float(row_id)))
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.session_state[f"{key}__pending_action"] = None
                        st.session_state[f"{key}__pending_row"] = None
                        time.sleep(1.5)
                        ui_refresh("db_grid")
        with c2:
            if st.button("Cancel", key=f"cancel_del_{row_id}_{tok}"):
                st.session_state[f"{key}__pending_action"] = None
                st.session_state[f"{key}__pending_row"] = None
                st.rerun()

# --------------------- Tabs ---------------------
tabs = st.tabs(["Dataset/DB", "Run GA", "Baselines", "General Dataset Runner", "Results & Plots", "Stability", "Export"])


# ============ Dataset / DB tab ============ #
with tabs[0]:
    st.header("Dataset")

    # Quick stats
    added_df = build_added_df()
    st.write(f"Base WDBC: 569 rows | Added: {len(added_df)} | Displayed: {569 + len(added_df)}")

    # Full merged view
    if "full_df" not in st.session_state:
     st.session_state["full_df"] = merged_view()

    paginated_actions_table(st.session_state["full_df"], page_size=20, key="db_grid")

    st.markdown("---")
    st.subheader("➕ Add new sample")
    st.session_state.setdefault("add_expanded", False)
    st.session_state.setdefault("add_box_ver", 0)

    expander_title = f"Add (Manual 30 features / CSV)"
    expander = st.expander(expander_title, expanded=st.session_state["add_expanded"])
    with expander:
        mode = st.radio(
            "Input mode",
            ["Manual (30 features)", "Upload CSV (batch)"],
            horizontal=True, key="add_mode_radio"
        )

        if mode == "Manual (30 features)":
            st.info("Enter all 30 WDBC features")
            with st.form("add_30"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    patient_id = st.text_input("Patient ID", value="")
                with c2:
                    label = st.selectbox("Label (0=Benign, 1=Malignant)", [0, 1], index=0)

                cols = st.columns(3)
                values_30: Dict[str, Any] = {}
                for i, fname in enumerate(feat_names):
                    with cols[i % 3]:
                        values_30[str(fname)] = st.number_input(
                            str(fname), value=0.0, format="%.6f", key=f"add30_{i}"
                        )
                notes = st.text_area("Notes (optional)", value="")
                submitted = st.form_submit_button("Add")
            if submitted:
                ok, msg = db_insert(patient_id, label, values_30, notes)
                (st.success if ok else st.error)(msg)
                if ok:
                   st.session_state["add_expanded"] = False
                   st.session_state["add_box_ver"] += 1
                   time.sleep(1.5)
                   ui_refresh("db_grid")

        else:
            st.info("CSV must include `patient_id`, `label`, and any subset/all of the 30 feature columns (use WDBC names).")
            file_csv = st.file_uploader("Upload CSV", type=["csv"], key="up_csv_patients_new")
            if file_csv is not None:
                try:
                    df_up = pd.read_csv(file_csv)
                    st.write("Preview:")
                    st.dataframe(df_up.head(5), use_container_width=True)
                    if st.button("Import all rows", key="btn_import_csv_new"):
                        added, skipped = 0, 0
                        for _, row in df_up.iterrows():
                            pid = str(row.get("patient_id", "")).strip()
                            if pid == "":
                                skipped += 1
                                continue
                            label_val = int(row.get("label", 0))
                            feats = {c: row[c] for c in df_up.columns if c not in ["patient_id", "label"]}
                            ok, _ = db_insert(pid, label_val, feats, "")
                            added += 1 if ok else 0
                        st.success(f"Imported: {added} | Skipped/Duplicates: {skipped}")
                        st.session_state["add_expanded"] = False
                        st.session_state["add_box_ver"] += 1
                        time.sleep(1.5)
                        ui_refresh("db_grid")

                except Exception as e:
                    st.error(f"Import failed: {e}")


# ============ Run GA tab ============ #
with tabs[1]:
    st.header("Run GA")

    cfg = GAConfig(
        population_size=int(pop),
        generations=int(gens),
        crossover_prob=float(pc),
        mutation_prob=float(pm),
        tournament_k=int(tk),
        elitism=int(elit),
        inner_cv_folds=int(inner_k),
        lambda_f1=float(0.5),        # match your formula in caption
        alpha_penalty=float(alpha),
        early_stopping_rounds=int(early_stop),
        random_state=int(seed),
    )
    st.session_state.setdefault("outer_mode", "Hold-out (70/30)")
    run_btn = st.button("Run GA now", key="btn_run_ga_main")

    if run_btn:
        with st.spinner("Running GA..."):
            # Hold-out outer split
            X_tr, X_te, y_tr, y_te = outer_split(X, y, test_size=0.3, seed=seed)
            ga = GeneticAlgorithmFS(X_tr, y_tr, cfg)
            mask, fit, history = ga.run()
            idx = np.where(mask == 1)[0]
            if idx.size == 0:
                idx = np.array([0])
            model = clf_pipeline()
            model.fit(X_tr[:, idx], y_tr)
            y_pred = model.predict(X_te[:, idx])
            y_prob = model.predict_proba(X_te[:, idx])[:, 1]
            m = compute_metrics(y_te, y_prob, y_pred)

        st.success("Done.")

        # Selected features + metrics
        st.markdown(f"**Selected features:** {idx.size}/{P}")
        st.write(", ".join([str(feat_names[i]) for i in idx]))

        st.markdown(
            f"**Composite Fitness (Inner-CV)** = 0.5 × (Mean CV_Acc + Mean CV_F1) − α × (k/{P}): **{fit:.4f}**  \n"
            f"**Outer Accuracy:** {m.accuracy:.4f}  \n"
            f"**F1-macro:** {m.f1_macro:.4f}  \n"
            f"**ROC-AUC:** {fmt_float(m.roc_auc)}"
        )

        st.info(
            "**Interpretation**  \n"
            "- Composite Fitness: inner-CV mean accuracy/F1 with feature penalty.  \n"
            "- Accuracy: hold-out correctness.  \n"
            "- F1-macro: balance of precision/recall across classes.  \n"
            "- ROC-AUC: separability across thresholds."
        )

        # Fitness history
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.plot(history, linewidth=2)
        ax.set_title("GA Best Fitness per Generation", fontsize=12, fontweight="semibold")
        ax.set_xlabel("Generation", fontsize=10); ax.set_ylabel("Best Fitness", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        download_fig(fig, "fitness.png")

        # Persist GA outputs for live thresholding
        st.session_state["ga_result"] = {
            "y_prob": np.asarray(y_prob, dtype=float),
            "y_true": np.asarray(y_te, dtype=int),
            "selected_idx": [int(i) for i in idx],
            "selected_names": [str(feat_names[i]) for i in idx],
            "fitness": float(fit),
        }
        st.session_state.setdefault("thr", 0.50)

        # Confusion matrix at default 0.50 (initial view)
        fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
        cm = m.cm
        total = cm.sum()
        bg = np.array([[0, 1], [1, 0]])
        ax2.imshow(bg, cmap=ListedColormap(["#A8E6A1", "#F5A3A3"]), vmin=0, vmax=1)
        labels = np.array([["True Negative (TN)", "False Positive (FP)"],
                           ["False Negative (FN)", "True Positive (TP)"]])
        for (i, j), val in np.ndenumerate(cm):
            pct = (val / total) * 100 if total > 0 else 0.0
            ax2.text(j, i, f"{val} ({pct:.1f}%)\n{labels[i, j]}",
                     ha="center", va="center", fontsize=6, fontweight="semibold", linespacing=1.2)
        ax2.set_title("Confusion Matrix — Benign (0) vs Malignant (1)", fontsize=11, fontweight="semibold")
        ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
        ax2.set_xticklabels(["Predicted Benign (0)", "Predicted Malignant (1)"], fontsize=7)
        ax2.set_yticklabels(["Actual Benign (0)", "Actual Malignant (1)"], fontsize=7)
        ax2.set_xlabel("Predicted", fontsize=9); ax2.set_ylabel("Actual", fontsize=9)
        ax2.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax2.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax2.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax2.tick_params(which="minor", bottom=False, left=False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=False)

    # === Live Decision-Threshold Panel (post-hoc; no retrain) ===
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    )

    def _metrics_at_threshold(y_true, y_prob, thr):
        y_pred = (y_prob >= thr).astype(int)
        m = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
            "cm": confusion_matrix(y_true, y_pred)
        }
        return m

    if "ga_result" in st.session_state:
        st.markdown("---")
        st.subheader("Decision threshold")

        gr = st.session_state["ga_result"]
        y_prob = gr["y_prob"]; y_true = gr["y_true"]

        thr = st.slider("Decision threshold", 0.05, 0.95, float(st.session_state.get("thr", 0.50)), 0.01, key="thr")

        m_thr = _metrics_at_threshold(y_true, y_prob, thr)

        st.markdown(
            f"**@threshold={thr:.2f}**  \n"
            f"- Accuracy: **{m_thr['accuracy']:.4f}**  \n"
            f"- F1-macro: **{m_thr['f1_macro']:.4f}**  \n"
            f"- Precision: **{m_thr['precision']:.4f}**  \n"
            f"- Recall (Sensitivity): **{m_thr['recall']:.4f}**  \n"
            f"- ROC-AUC (fixed): **{m_thr['roc_auc']:.4f}**"
        )

        # Confusion matrix with % inside cells
        fig_cm, ax_cm = plt.subplots(figsize=(fig_w, fig_h))
        cm2 = m_thr["cm"]; total2 = cm2.sum()
        bg = np.array([[0, 1], [1, 0]])
        ax_cm.imshow(bg, cmap=ListedColormap(["#A8E6A1", "#F5A3A3"]), vmin=0, vmax=1)
        labels = np.array([["True Negative (TN)", "False Positive (FP)"],
                           ["False Negative (FN)", "True Positive (TP)"]])
        for (i, j), val in np.ndenumerate(cm2):
            pct = (val / total2) * 100 if total2 > 0 else 0.0
            ax_cm.text(j, i, f"{val} ({pct:.1f}%)\n{labels[i, j]}",
                       ha="center", va="center", fontsize=6, fontweight="semibold", linespacing=1.2)
        ax_cm.set_title("Confusion Matrix — Benign (0) vs Malignant (1)", fontsize=11, fontweight="semibold")
        ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Predicted Benign (0)", "Predicted Malignant (1)"], fontsize=7)
        ax_cm.set_yticklabels(["Actual Benign (0)", "Actual Malignant (1)"], fontsize=7)
        ax_cm.set_xlabel("Predicted", fontsize=7); ax_cm.set_ylabel("Actual", fontsize=7)
        ax_cm.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax_cm.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax_cm.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax_cm.tick_params(which="minor", bottom=False, left=False)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=False)

        cols = st.columns([1, 3, 1])
        with cols[2]:
            if st.button("Clear stored results"):
                st.session_state.pop("ga_result", None)
                st.rerun()
    else:
        st.info("Run GA to enable live thresholding.")


# ============ Baselines tab ============ #
with tabs[2]:
    st.header("Baselines (All features vs. GA features)")
    st.caption("Run LR/SVM/RF on all 30 features, then on GA-selected features.")
    if st.button("Run baselines", key="btn_baselines"):
        with st.spinner("Running baselines..."):
            X_tr, X_te, y_tr, y_te = outer_split(X, y, test_size=0.3, seed=seed)
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline
            def eval_model(pipe, name, Xtr, Xte, ytr, yte, tag):
                pipe.fit(Xtr, ytr)
                y_pred = pipe.predict(Xte)
                y_prob = pipe.predict_proba(Xte)[:, 1] if hasattr(pipe, "predict_proba") else None
                m = compute_metrics(yte, y_prob, y_pred)
                st.write(f"**{name} {tag}** | Acc={m.accuracy:.4f} | F1={m.f1_macro:.4f} | ROC-AUC={fmt_float(m.roc_auc)}")
                return m
            lr = clf_pipeline()
            svm = Pipeline([
                ("scaler", __import__("sklearn").preprocessing.StandardScaler()),
                ("svc", SVC(kernel="rbf", probability=True))
            ])
            rf = Pipeline([
                ("scaler", __import__("sklearn").preprocessing.StandardScaler()),
                ("rf", __import__("sklearn").ensemble.RandomForestClassifier(n_estimators=300, random_state=seed))
            ])
            st.subheader("All 30 features")
            m_lr_all = eval_model(lr, "Logistic Regression", X_tr, X_te, y_tr, y_te, "(All)")
            m_svm_all = eval_model(svm, "SVM (RBF)", X_tr, X_te, y_tr, y_te, "(All)")
            m_rf_all  = eval_model(rf, "Random Forest", X_tr, X_te, y_tr, y_te, "(All)")

            st.subheader("GA-selected features")
            cfg_tmp = GAConfig(random_state=int(seed))
            ga = GeneticAlgorithmFS(X_tr, y_tr, cfg_tmp)
            mask, _, _ = ga.run()
            idx = np.where(mask == 1)[0]
            if idx.size == 0:
                idx = np.array([0])
            st.write(f"Selected {idx.size} features: " + ", ".join([str(feat_names[i]) for i in idx]))
            m_lr_ga = eval_model(lr, "Logistic Regression", X_tr[:, idx], X_te[:, idx], y_tr, y_te, "(GA)")
            m_svm_ga = eval_model(svm, "SVM (RBF)", X_tr[:, idx], X_te[:, idx], y_tr, y_te, "(GA)")
            m_rf_ga  = eval_model(rf, "Random Forest", X_tr[:, idx], X_te[:, idx], y_tr, y_te, "(GA)")

            rows = []
            def row_of(name, m, tag):
                rows.append({
                    "Model": f"{name} {tag}",
                    "Accuracy": float(m.accuracy),
                    "F1-macro": float(m.f1_macro),
                    "ROC-AUC": np.nan if m.roc_auc is None else float(m.roc_auc)
                })
            row_of("LR", m_lr_all, "(All)"); row_of("SVM", m_svm_all, "(All)"); row_of("RF", m_rf_all, "(All)")
            row_of("LR", m_lr_ga, "(GA)");  row_of("SVM", m_svm_ga, "(GA)");  row_of("RF", m_rf_ga, "(GA)")
            df_sum = pd.DataFrame(rows)
            st.markdown("**Summary:**")
            st.dataframe(df_sum, use_container_width=True)
            download_df(df_sum, "baseline_summary.csv")
        st.success("Done.")



# ============ General Dataset Runner tab ============
with tabs[3]:
    st.header("General Medical Dataset Runner (Upload & Experiment)")
    st.caption("Upload any medical CSV, select target, run baselines and GA on your dataset (Binary supported).")

    # --- Fixed to match Run GA ---
    TEST_SIZE = 0.30  # ✅ same as Run GA (70/30)

    up = st.file_uploader("Upload CSV", type=["csv"], key="uploader_general")
    if up is None:
        st.info("Upload a CSV file to start.")
        st.stop()

    # -------------------- Load CSV --------------------
    st.subheader(f"Dataset: {up.name}")
    df_raw = pd.read_csv(up)

    # Clean common junk columns (Excel artifacts)
    df_raw = df_raw.loc[:, ~df_raw.columns.astype(str).str.match(r"^Unnamed")]
    df_raw = df_raw.dropna(axis=1, how="all")

    # -------------------- Preview with pagination (10 rows/page) --------------------
    st.write("Preview (paged):")
    page_size = 10
    n_rows = len(df_raw)
    n_pages = max(1, int(np.ceil(n_rows / page_size)))

    cpg1, cpg2 = st.columns([1, 3])
    with cpg1:
        page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1, key="prev_page")
    start = (int(page) - 1) * page_size
    end = min(start + page_size, n_rows)
    with cpg2:
        st.caption(f"Rows {start+1}–{end} of {n_rows}")

    st.dataframe(df_raw.iloc[start:end], use_container_width=True)

    # -------------------- User inputs --------------------
    st.markdown("### Dataset setup")

    target_col = st.selectbox("Target column", options=list(df_raw.columns), key="target_col_general")

    # Binary only (avoid any accidental multiclass path)
    task_type = "binary"

    uniq = list(pd.unique(df_raw[target_col].dropna()))
    if len(uniq) == 0:
        st.error("Target column has no valid (non-null) values.")
        st.stop()

    positive_class = st.selectbox("Positive class (treated as 1)", options=uniq, key="pos_class_general")

    feature_mode = st.radio(
        "Feature columns mode",
        ["numeric_only", "manual"],
        horizontal=True,
        key="feat_mode_general"
    )

    selected_cols = None
    if feature_mode == "manual":
        feature_candidates = [c for c in df_raw.columns if c != target_col]
        selected_cols = st.multiselect("Select feature columns", feature_candidates, key="feat_cols_general")
        if selected_cols is not None and len(selected_cols) == 0:
            st.warning("Manual mode selected but no feature columns chosen. Please select at least 1 feature.")
            st.stop()

    # ✅ Keep Recall option in the UI (you asked to keep it)
    metric_priority = st.selectbox(
        "Metric priority (affects GA fitness)",
        ["Balanced (Accuracy+F1)", "Recall-focused (reduce FN)"],
        key="metric_priority_general"
    )

    seed_general = st.number_input("Random seed", value=int(seed), step=1, key="seed_general")  # default from sidebar seed

    # -------------------- Helpers --------------------

    def _clean_and_prepare_binary(df_in: pd.DataFrame, target: str, pos_class, feat_mode: str, manual_cols):
        """
        Returns: X (float ndarray), y (int ndarray 0/1), feat_names (list[str])
        """
        df = df_in.copy()

        # 1) treat '-' / ' - ' as missing
        df = df.replace(["-", " - ", "—", "NA", "N/A", ""], np.nan)

        # 2) drop ID-like columns automatically (do NOT let GA use them)
        id_like = []
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ["id", "patient_id", "case_id", "record_id", "sample_id"]:
                id_like.append(c)
            elif cl.endswith("id") and cl != str(target).strip().lower():
                id_like.append(c)
        df = df.drop(columns=id_like, errors="ignore")

        # 3) Build y (binary: positive_class => 1, else 0)
        y_raw = df[target]
        y = (y_raw == pos_class).astype(int).to_numpy()

        # 4) Feature columns
        if feat_mode == "manual" and manual_cols is not None:
            feat_cols = [c for c in manual_cols if c != target]
        else:
            # numeric_only
            feat_cols = [c for c in df.columns if c != target]
            # convert everything to numeric; non-numeric becomes NaN
            for c in feat_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            feat_cols = list(df[feat_cols].select_dtypes(include=["number"]).columns)

        if len(feat_cols) == 0:
            raise ValueError("No numeric feature columns found after cleaning. Try manual selection or check your file.")

        Xdf = df[feat_cols].copy()

        # 5) convert numeric (coerce) + impute
        for c in Xdf.columns:
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(Xdf)

        feat_names_local = list(map(str, feat_cols))
        return X.astype(float), y.astype(int), feat_names_local

    def _feature_hint(name: str, X_tr: np.ndarray, col_idx: int) -> str:
        """Small hint printed next to feature name (generic & always valid)."""
        col = X_tr[:, col_idx]
        mu = float(np.nanmean(col))
        sd = float(np.nanstd(col))
        return f"(μ={mu:.3f}, σ={sd:.3f})"

    def plot_ga_curve(history, title="GA Fitness Evolution (Best per Generation)"):
        if history is None or not isinstance(history, (list, tuple)) or len(history) < 2:
            st.info("No GA history to plot.")
            return None
        fig, ax = plt.subplots(figsize=(6.0, 2.6))
        gens_local = list(range(1, len(history) + 1))
        ax.plot(gens_local, history, linewidth=1.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Generation", fontsize=9)
        ax.set_ylabel("Fitness", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        return fig

    def plot_confusion_matrix(cm, title="Confusion Matrix (0/1)"):
        fig, ax = plt.subplots(figsize=(6.0, 2.6))
        total = cm.sum() if cm is not None else 0
        bg = np.array([[0, 1], [1, 0]])
        ax.imshow(bg, cmap=ListedColormap(["#A8E6A1", "#F5A3A3"]), vmin=0, vmax=1)

        labels = np.array([["TN", "FP"], ["FN", "TP"]])
        for (i, j), val in np.ndenumerate(cm):
            pct = (val / total) * 100 if total > 0 else 0.0
            ax.text(j, i, f"{val} ({pct:.1f}%)\n{labels[i, j]}",
                    ha="center", va="center", fontsize=8, fontweight="semibold")

        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        return fig

    # -------------------- Run Experiment --------------------
    if st.button("Run Experiment", type="primary", key="run_exp_general"):
        # 1) Prepare dataset (same preprocessing idea as Run GA: clean -> numeric -> impute)
        try:
            X2, y2, feat_names_local = _clean_and_prepare_binary(
                df_in=df_raw,
                target=target_col,
                pos_class=positive_class,
                feat_mode=feature_mode,
                manual_cols=selected_cols
            )
        except Exception as e:
            st.error(f"Dataset preparation failed: {e}")
            st.stop()

        # 2) Outer split (✅ fixed 70/30 like Run GA)
        X_tr2, X_te2, y_tr2, y_te2 = outer_split(X2, y2, test_size=TEST_SIZE, seed=int(seed_general))

        # 3) Baselines (All features)
        st.subheader("Baselines (All features)")
        lr_all = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=5000))])
        svm_all = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True))])
        rf_all  = RandomForestClassifier(n_estimators=300, random_state=int(seed_general))

        m_lr_all = eval_model(lr_all, "Logistic Regression (All)", X_tr2, X_te2, y_tr2, y_te2, "")
        m_svm_all = eval_model(svm_all, "SVM (RBF) (All)", X_tr2, X_te2, y_tr2, y_te2, "")
        m_rf_all  = eval_model(rf_all,  "Random Forest (All)", X_tr2, X_te2, y_tr2, y_te2, "")

        baseline_df = pd.DataFrame([m_lr_all, m_svm_all, m_rf_all])
        baseline_view = baseline_df.drop(columns=["y_pred", "y_prob"], errors="ignore")
        st.dataframe(baseline_view, use_container_width=True)

        st.divider()

        # 4) GA Feature Selection (✅ same GAConfig settings style as Run GA)
        st.subheader("GA Feature Selection + Baselines on selected features")

        cfg = GAConfig(
            population_size=int(pop),
            generations=int(gens),
            crossover_prob=float(pc),
            mutation_prob=float(pm),
            tournament_k=int(tk),
            elitism=int(elit),
            inner_cv_folds=int(inner_k),
            lambda_f1=float(0.5),                 # ✅ same as Run GA formula
            alpha_penalty=float(alpha),           # ✅ same penalty slider as Run GA
            early_stopping_rounds=int(early_stop),
            random_state=int(seed_general),
        )

        # ✅ Keep metric priority behavior (Balanced vs Recall-focused) if your GAConfig supports it
        if metric_priority.startswith("Balanced"):
            if hasattr(cfg, "w_acc"): setattr(cfg, "w_acc", 0.5)
            if hasattr(cfg, "w_f1"): setattr(cfg, "w_f1", 0.5)
            if hasattr(cfg, "w_recall_pos"): setattr(cfg, "w_recall_pos", 0.0)
        else:
            if hasattr(cfg, "w_acc"): setattr(cfg, "w_acc", 0.2)
            if hasattr(cfg, "w_f1"): setattr(cfg, "w_f1", 0.3)
            if hasattr(cfg, "w_recall_pos"): setattr(cfg, "w_recall_pos", 0.5)

        ga2 = GeneticAlgorithmFS(X_tr2, y_tr2, cfg)
        mask2, best_fit2, hist2 = ga2.run()

        idx2 = np.where(np.array(mask2) == 1)[0]
        if idx2.size == 0:
            st.warning("GA selected 0 features. Falling back to the first feature to avoid crash.")
            idx2 = np.array([0])

        st.write(f"Selected **{idx2.size}** features:")
        # ✅ feature name + hint
        selected_rows = []
        for i in idx2:
            nm = feat_names_local[i]
            mean_val = np.mean(X_tr2[:, i])
            std_val  = np.std(X_tr2[:, i])
            hint = f"(Mean={mean_val:.2f}, Standard Deviation={std_val:.2f})"
            selected_rows.append({"feature": f"{nm}  {hint}"})
        st.dataframe(pd.DataFrame(selected_rows), use_container_width=True)

        # 5) Baselines on GA-selected features
        X_tr_sel2 = X_tr2[:, idx2]
        X_te_sel2 = X_te2[:, idx2]

        lr_ga = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=5000))])
        svm_ga = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True))])
        rf_ga  = RandomForestClassifier(n_estimators=300, random_state=int(seed_general))

        m_lr_ga  = eval_model(lr_ga,  "Logistic Regression (GA)", X_tr_sel2, X_te_sel2, y_tr2, y_te2, "")
        m_svm_ga = eval_model(svm_ga, "SVM (RBF) (GA)",        X_tr_sel2, X_te_sel2, y_tr2, y_te2, "")
        m_rf_ga  = eval_model(rf_ga,  "Random Forest (GA)",     X_tr_sel2, X_te_sel2, y_tr2, y_te2, "")

        ga_df = pd.DataFrame([m_lr_ga, m_svm_ga, m_rf_ga])
        ga_view = ga_df.drop(columns=["y_pred", "y_prob"], errors="ignore")
        st.dataframe(ga_view, use_container_width=True)

        st.success("Done.")

        # ---- Auto explanatory comparison ----
        st.subheader("Auto Interpretation (Text Summary)")
        summary_txt = _auto_compare_text(baseline_view, ga_view)
        st.markdown(summary_txt)


        # 6) Fitness plot 
        st.subheader("GA Fitness Progress")
        st.success(f"Best fitness: {best_fit2:.4f}")
        plot_ga_curve(hist2, title="GA Fitness Evolution (Best per Generation)")

        # 7) ✅ Confusion matrix AFTER the fitness figure (use the same evaluation style as Run GA)
        # We'll compute it using clf_pipeline() for consistency with your thesis baseline
        st.subheader("Confusion Matrix (GA-selected features)")
        model_cm = clf_pipeline()
        model_cm.fit(X_tr_sel2, y_tr2)
        y_pred_cm = model_cm.predict(X_te_sel2)
        y_prob_cm = model_cm.predict_proba(X_te_sel2)[:, 1]
        m_cm = compute_metrics(y_te2, y_prob_cm, y_pred_cm)

        plot_confusion_matrix(m_cm.cm, title="Confusion Matrix — class 0 vs class 1 (GA features)")

# ============ Results & Plots tab ============ #
with tabs[4]:
    st.header("Results & Plots")
    up = st.file_uploader("Upload GA result JSON", type=["json"], key="up_json")
    if up is not None:
        res = json.load(up)
        if "history" in res:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.plot(res["history"], linewidth=2)
            ax.set_title("GA best fitness per generation")
            ax.set_xlabel("Generation"); ax.set_ylabel("Best fitness")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            download_fig(fig, "fitness_from_json.png")
        if "metrics" in res:
            m = pd.DataFrame([res["metrics"]])
            st.markdown("**Metrics:**")
            st.dataframe(m, use_container_width=True)
            download_df(m, "metrics_from_json.csv")
        if "selected_names" in res:
            st.markdown("**Selected features:**")
            st.dataframe(pd.DataFrame({"feature": res["selected_names"]}), use_container_width=True)


# ============ Stability tab ============ #
with tabs[5]:
    st.header("Stability across repeated runs")
    st.caption("Repeated GA runs with different seeds; report mean Jaccard similarity of selected sets.")
    n_runs_stability = st.number_input("Repeat runs", 1, 50, 10, 1, key="runs_stab")
    if st.button("Run repeated GA", key="btn_stability"):
        with st.spinner("Computing stability..."):
            X_tr, X_te, y_tr, y_te = outer_split(X, y, test_size=0.3, seed=seed)
            masks = []
            for i in range(n_runs_stability):
                cfg_i = GAConfig(random_state=seed + i)
                ga = GeneticAlgorithmFS(X_tr, y_tr, cfg_i)
                mask, _, _ = ga.run()
                masks.append(mask.tolist())
            jacs = []
            for i in range(len(masks)):
                for j in range(i+1, len(masks)):
                    jacs.append(jaccard(masks[i], masks[j]))
        st.success("Done.")
        if jacs:
            st.write(f"Mean Jaccard: {np.mean(jacs):.3f} | N={len(jacs)} pairs")
        else:
            st.write("Mean Jaccard: NA")


# ============ Export tab ============ #
with tabs[6]:
    st.header("Export")
    st.write("Results are saved under `results/` after running. Download config below:")
    cfg_json = {
        "population": int(pop), "generations": int(gens),
        "pc": float(pc), "pm": float(pm), "tournament_k": int(tk),
        "elitism": int(elit), "inner_cv": int(inner_k),
        "lambda": float(0.5), "alpha": float(alpha),
        "early_stop": int(early_stop), "seed": int(seed),
    }
    st.download_button("⬇️ Download GA config", data=json.dumps(cfg_json, indent=2, ensure_ascii=False), file_name="ga_config.json")
