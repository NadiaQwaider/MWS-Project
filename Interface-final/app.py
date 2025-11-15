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

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 3rd-party grid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# Project internals (as in your repo)
from utils import (
    load_wdbc, outer_split, clf_pipeline, compute_metrics, calibration_xy,
    save_json, jaccard
)
from genetic_algorithm import GeneticAlgorithmFS, GAConfig


# --------------------- Page setup ---------------------
st.set_page_config(page_title="Breast Cancer – GA Feature Selection", layout="wide")
st.title("Breast Cancer – GA Feature Selection App")
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
DB_PATH = "patients.db"

def db_connect():
    return sqlite3.connect(DB_PATH)

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

def db_delete(row_id: int) -> Tuple[bool, str]:
    try:
        with db_connect() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM patients WHERE id=?", (int(row_id),))
            con.commit()
        return True, "Deleted."
    except Exception as e:
        return False, f"Error: {e}"

def db_fetch_all() -> pd.DataFrame:
    with db_connect() as con:
        df = pd.read_sql_query("SELECT id, patient_id, label, features_json, notes, created_at FROM patients ORDER BY id DESC", con)
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
        key=f"{key}_ag",
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

    elif act_typ == "delete":
        with st.expander("Confirm deletion", expanded=True):
            pid = act_row.get("patient_id")
            st.error(f'Are you sure you want to delete the patient "{pid}" details from the database? This action cannot be undone.')
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete", key=f"yes_del_{pid}"):
                    if act_row.get("source") == "Base":
                        st.error("Base rows are locked (cannot delete).")
                    else:
                        row_id = act_row.get("id")
                        if row_id is None or (isinstance(row_id, float) and np.isnan(row_id)):
                            st.error("Invalid row id for deletion.")
                        else:
                            ok, msg = db_delete(int(row_id))
                            (st.success if ok else st.error)(msg)
            with c2:
                st.button("Cancel", key=f"cancel_del_{pid}")


# --------------------- Tabs ---------------------
tabs = st.tabs(["Dataset / DB", "Run GA", "Baselines", "Results & Plots", "Stability", "Export"])


# ============ Dataset / DB tab ============ #
with tabs[0]:
    st.header("Dataset")

    # Quick stats
    added_df = build_added_df()
    st.write(f"Base WDBC: 569 rows | Added: {len(added_df)} | Displayed: {569 + len(added_df)}")

    # Full merged view
    full_df = merged_view()
    paginated_actions_table(full_df, page_size=20, key="db_grid")

    st.markdown("---")
    st.subheader("➕ Add new sample")
    expander = st.expander("Add (Manual 30 features / CSV)", expanded=False)
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


# ============ Results & Plots tab ============ #
with tabs[3]:
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
with tabs[4]:
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
with tabs[5]:
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
