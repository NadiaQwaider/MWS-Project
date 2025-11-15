# app.py
import io
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# مبدّل لغة في الشريط الجانبي
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])

# قواميس نصوص
TXT = {
    "English": {
        "title": "Breast Cancer – GA Feature Selection App",
        "run": "Run GA now",
        "acc": "Accuracy",
        "f1": "F1-macro",
        "roc": "ROC-AUC",
        "fit": "Composite Fitness (Inner-CV)",
        "explain": (
            "**Interpretation:**\n"
            "- Composite Fitness: blended inner-CV score with a penalty on #features.\n"
            "- Accuracy: correct predictions on outer test.\n"
            "- F1-macro: balance of precision/recall across classes.\n"
            "- ROC-AUC: discrimination ability; ≥0.90 is excellent."
        ),
        "selected": "Selected features",
        "spinner": "Running the genetic algorithm… please wait.",
        "done": "Done. Results below ✅",
    },
    "العربية": {
        "title": "تطبيق اختيار الخصائص بالـ GA لسرطان الثدي",
        "run": "تنفيذ الخوارزمية الآن",
        "acc": "Accuracy (الدقة)",
        "f1": "F1-macro (معامل F1 الكلي)",
        "roc": "ROC-AUC",
        "fit": "Composite Fitness (التحقق الداخلي)",
        "explain": (
            "**تفسير القيم:**\n"
            "- Composite Fitness: مقياس مركّب داخل CV مع حدّ عقوبة على عدد الخصائص.\n"
            "- Accuracy: نسبة التنبؤات الصحيحة على الاختبار الخارجي.\n"
            "- F1-macro: توازن الدقّة والاستدعاء عبر الفئتين.\n"
            "- ROC-AUC: القدرة التمييزية؛ القيم ≥ 0.90 ممتازة."
        ),
        "selected": "الخصائص المختارة",
        "spinner": "جاري تنفيذ الخوارزمية الجينية... يرجى الانتظار.",
        "done": "تم التنفيذ وعرض النتائج ✅",
    }
}

# دعم RTL عند العربية (اختياري)
if lang == "العربية":
    st.markdown("""
        <style>
        html, body, [class*="css"]  { direction: rtl; text-align: right; }
        </style>
    """, unsafe_allow_html=True)

st.title(TXT[lang]["title"])

from utils import (
    load_wdbc, outer_split, clf_pipeline, compute_metrics, calibration_xy,
    BENIGN, MALIGNANT, LABEL_MAP_INT2STR, save_json, jaccard
)
from genetic_algorithm import GeneticAlgorithmFS, GAConfig

st.set_page_config(page_title= TXT[lang]["title"], layout="wide")

# ---------- helper: safe float formatting ----------
def fmt_float(v):
    try:
        if v is None:
            return "NA"
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return "NA"
        return f"{float(v):.4f}"
    except Exception:
        return "NA"

# ---------- helper: paginated table ----------
def paginated_table(df: pd.DataFrame, page_size: int = 5, key: str = "pg"):
    if df is None or len(df) == 0:
        st.warning("لا توجد بيانات للعرض.")
        return
    total = len(df)
    n_pages = max(1, (total + page_size - 1) // page_size)
    pg_key = f"_{key}_page"
    if pg_key not in st.session_state:
        st.session_state[pg_key] = 0

    cols_nav = st.columns([1,1,4,1,1])
    with cols_nav[0]:
        if st.button("⟵ السابق", disabled=(st.session_state[pg_key] <= 0), key=f"{key}_prev"):
            st.session_state[pg_key] = max(0, st.session_state[pg_key] - 1)
    with cols_nav[2]:
        st.write(f"صفحة {st.session_state[pg_key] + 1} / {n_pages} (إجمالي الصفوف: {total})")
    with cols_nav[4]:
        if st.button("التالي ⟶", disabled=(st.session_state[pg_key] >= n_pages - 1), key=f"{key}_next"):
            st.session_state[pg_key] = min(n_pages - 1, st.session_state[pg_key] + 1)

    start = st.session_state[pg_key] * page_size
    end = start + page_size
    st.dataframe(
        df.iloc[start:end],
        use_container_width=True
    )

# --- Sidebar: GA Settings ---
st.sidebar.header("GA Settings")
pop = st.sidebar.number_input("Population Size", 10, 500, 60, 5)
gens = st.sidebar.number_input("Generations", 5, 300, 50, 5)
pc = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.80, 0.01)
pm = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.05, 0.01)
tk = st.sidebar.number_input("Tournament k", 2, 10, 3, 1)
elit = st.sidebar.number_input("Elitism", 0, 10, 2, 1)
inner_k = st.sidebar.number_input("Inner CV folds (fitness)", 2, 10, 5, 1)
lam = st.sidebar.slider("λ (weight of F1 in fitness)", 0.0, 1.0, 0.5, 0.05)
alpha = st.sidebar.slider("α penalty (0–0.5)", 0.0, 0.5, 0.15, 0.01)
early_stop = st.sidebar.number_input("Early-stopping rounds", 0, 100, 10, 1)
seed = st.sidebar.number_input("Random Seed", 0, 10_000, 42, 1)

outer_mode = st.sidebar.selectbox("Outer evaluation", ["Hold-out (70/30)", "Outer CV (5-fold)"])
n_runs_stability = st.sidebar.number_input("Repeat runs (stability)", 1, 20, 3, 1)

# --- Data ---
X, y, feat_names = load_wdbc()
p = X.shape[1]

st.subheader("Dataset")
st.write(f"Samples: **{X.shape[0]}**, Features: **{p}**  | Labels: 0=Benign, 1=Malignant")

with st.expander("Preview rows"):
    df_prev = pd.DataFrame(X, columns=feat_names)
    df_prev["label"] = y
    paginated_table(df_prev, page_size=5, key="preview")

# --- Helper: evaluate a trained pipeline on given split ---
def evaluate_on_split(X_tr, X_te, y_tr, y_te, mask_idx: np.ndarray):
    model = clf_pipeline()
    model.fit(X_tr[:, mask_idx], y_tr)
    y_pred = model.predict(X_te[:, mask_idx])
    y_prob = model.predict_proba(X_te[:, mask_idx])[:, 1]
    return compute_metrics(y_te, y_prob, y_pred)

# --- Tabs ---
tabs = st.tabs(["Run GA", "Baselines", "Results & Plots", "Stability", "Export"])

with tabs[0]:
    st.header("Run GA (with Inner-CV in fitness)")
    cfg = GAConfig(
        population_size=int(pop),
        generations=int(gens),
        crossover_prob=float(pc),
        mutation_prob=float(pm),
        tournament_k=int(tk),
        elitism=int(elit),
        inner_cv_folds=int(inner_k),
        lambda_f1=float(lam),
        alpha_penalty=float(alpha),
        early_stopping_rounds=int(early_stop),
        random_state=int(seed),
    )
    run_btn = st.button("Run GA now" , key="btn_run_ga_main")
    if run_btn:
        with st.spinner("⏳ جاري تنفيذ الخوارزمية الجينية... يرجى الانتظار لحين اكتمال العملية."):
            if outer_mode.startswith("Hold-out"):
                X_tr, X_te, y_tr, y_te = outer_split(X, y, test_size=0.3, seed=seed)
                ga = GeneticAlgorithmFS(X_tr, y_tr, cfg)
                mask, fit, history = ga.run()
                idx = np.where(mask == 1)[0]
                if idx.size == 0:
                    idx = np.array([0])
                m = evaluate_on_split(X_tr, X_te, y_tr, y_te, idx)
            else:
                # Outer CV = 5-fold
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                outer_metrics = []
                selected_sets = []
                histories = []
                for oidx, (tr, te) in enumerate(skf.split(X, y), start=1):
                    ga = GeneticAlgorithmFS(X[tr], y[tr], cfg)
                    mask, fit, hist = ga.run()
                    idx = np.where(mask == 1)[0]
                    if idx.size == 0:
                        idx = np.array([0])
                    m_fold = evaluate_on_split(X[tr], X[te], y[tr], y[te], idx)
                    outer_metrics.append(m_fold.__dict__)
                    selected_sets.append(mask.tolist())
                    histories.append([float(v) for v in hist])
                # استقرار الاختيار عبر الطيات
                jacs = []
                for i in range(len(selected_sets)):
                    for j in range(i + 1, len(selected_sets)):
                        jacs.append(jaccard(selected_sets[i], selected_sets[j]))
        st.success("✅ تم تنفيذ الخوارزمية الجينية بنجاح وعرض النتائج أدناه.")

        if outer_mode.startswith("Hold-out"):
            st.markdown(f"**عدد الخصائص المختارة:** {idx.size}/{p}")
            st.write(", ".join([feat_names[i] for i in idx]))

            st.markdown(f"""
            **Composite Fitness (Inner-CV):** {fit:.4f}  
            **Outer hold-out Accuracy:** {m.accuracy:.4f}  
            **F1-macro:** {m.f1_macro:.4f}  
            **ROC-AUC:** {fmt_float(m.roc_auc)}
            """)

            st.markdown("""
            **توضيح النتائج:**  
            - **Composite Fitness (Inner-CV):**  مقياس مركّب يعكس نسبة جودة الحل     
            - **Accuracy:** تمثل نسبة العينات المصنّفة تصنيفًا صحيحًا ضمن مجموعة الاختبار الخارجية
            - **F1-macro:**  يعبّر عن توازن الأداء بين الدقة والاستدعاء
            - **ROC-AUC:**  القيم ≥ 0.90 تشير إلى تمييز ممتاز، والقيم القريبة من 0.5 تعكس ضعف القدرة التمييزي
            """, unsafe_allow_html=True)

            # history plot
            fig, ax = plt.subplots(figsize=(3.6, 2.2))  
            plt.tight_layout()            
            ax.plot(history, linewidth=2)
            ax.set_title("GA Best Fitness per Generation" , fontsize=9, fontweight="semibold")
            ax.set_xlabel("Generation" , fontsize=8)
            ax.set_ylabel("Best Fitness" , fontsize=8)
            st.pyplot(fig, use_container_width=False)

           # 🔹 مصفوفة الالتباس مع النسب والتصنيفات (Green=Correct, Red=Error)
            # 🔹 Confusion Matrix 
            fig2, ax2 = plt.subplots(figsize=(4.2, 3.0))   
            cm = m.cm
            total = cm.sum()

            # الألوان الخلفية
            bg = np.array([[0, 1],
                        [1, 0]])
            ax2.imshow(bg, cmap=ListedColormap(["#A8E6A1", "#F5A3A3"]), vmin=0, vmax=1)

            # التصنيفات والنِّسَب
            labels = np.array([
                ["True Negative (TN)", "False Positive (FP)"],
                ["False Negative (FN)", "True Positive (TP)"]
            ])

            for (i, j), val in np.ndenumerate(cm):
                pct = (val / total) * 100 if total > 0 else 0.0
                label = labels[i, j]
                ax2.text(
                    j, i,
                    f"{val} ({pct:.1f}%)\n{label}",
                    ha="center", va="center",
                    fontsize=6, color="black", linespacing=1.3
                )

            # عناوين ومحاور أصغر قليلًا
            ax2.set_title("Confusion Matrix — Benign (0) vs Malignant (1)", fontsize=9, fontweight="semibold")
            ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
            ax2.set_xticklabels(["Predicted Benign (0)", "Predicted Malignant (1)"], fontsize=6)
            ax2.set_yticklabels(["Actual Benign (0)", "Actual Malignant (1)"], fontsize=6)
            ax2.set_xlabel("Predicted", fontsize=8)
            ax2.set_ylabel("Actual", fontsize=8)

            # خطوط بيضاء خفيفة
            ax2.set_xticks(np.arange(-.5, 2, 1), minor=True)
            ax2.set_yticks(np.arange(-.5, 2, 1), minor=True)
            ax2.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
            ax2.tick_params(which="minor", bottom=False, left=False)

            plt.tight_layout()
            st.pyplot(fig2, use_container_width=False)


            # calibration
            if m.roc_auc is not None:
                model = clf_pipeline()
                model.fit(X_tr[:, idx], y_tr)
                y_prob = model.predict_proba(X_te[:, idx])[:, 1]
                cal_y, cal_x = calibration_xy(y_te, y_prob, n_bins=10)
                fig3, ax3 = plt.subplots(figsize=(4.2, 3.0))    
                plt.tight_layout()                
                ax3.plot([0, 1], [0, 1], linestyle="--")
                ax3.plot(cal_x, cal_y, marker="o")
                ax3.set_title("Calibration Curve", fontsize=9 ,fontweight="semibold")
                ax3.set_xlabel("Predicted Probability", fontsize=8)
                ax3.set_ylabel("Observed Frequency", fontsize=8)
                st.pyplot(fig3, use_container_width=False)

            # save
            out = {
                "mode": "holdout",
                "seed": seed,
                "selected_k": int(idx.size),
                "selected_names": [feat_names[i] for i in idx],
                "history": [float(v) for v in history],
                "metrics": m.__dict__,
                "config": cfg.__dict__,
            }
            Path("results").mkdir(exist_ok=True, parents=True)
            save_json(out, "results/ga_holdout_result.json")

            def to_serializable(o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, (np.integer, )):
                    return int(o)
                if isinstance(o, (np.floating, )):
                    return float(o)
                if isinstance(o, (np.bool_, )):
                    return bool(o)
                return str(o)

            st.download_button(
                "Download JSON result",
                data=json.dumps(out, indent=2, ensure_ascii=False, default=to_serializable),
                file_name="ga_holdout_result.json"
            )
        else:
            # عرض نتائج الـ Outer CV
            st.subheader("Outer CV (5-fold) metrics")
            df_outer = pd.DataFrame(outer_metrics)
            paginated_table(df_outer, page_size=5, key="outercv")
            if len(outer_metrics) > 0:
                st.info(f"Outer-CV Selection Stability (mean Jaccard): {np.mean(jacs):.3f}")
            out = {"mode": "outer_cv", "seed": seed, "metrics": outer_metrics, "config": cfg.__dict__}
            Path("results").mkdir(exist_ok=True, parents=True)
            save_json(out, "results/ga_outercv_result.json")
            st.download_button("Download JSON result", data=json.dumps(out, indent=2), file_name="ga_outercv_result.json")

with tabs[1]:
    st.header("Baselines (All features vs. GA features)")
    st.caption("تشغيل النماذج المرجعية على جميع الخصائص الأصلية (30) ومقارنتها بنفس النماذج على الخصائص المختارة بواسطة GA.")
    if st.button("Run baselines on all 30 features" , key="btn_baselines"):
        with st.spinner("⏳ جاري تشغيل النماذج المرجعية ومقارنة النتائج..."):
            X_tr, X_te, y_tr, y_te = outer_split(X, y, test_size=0.3, seed=seed)
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline

            # --- helpers ---
            def eval_model(pipe, name, Xtr, Xte, ytr, yte, tag):
                pipe.fit(Xtr, ytr)
                y_pred = pipe.predict(Xte)
                y_prob = pipe.predict_proba(Xte)[:, 1] if hasattr(pipe, "predict_proba") else None
                m = compute_metrics(yte, y_prob, y_pred)
                st.write(f"**{name} {tag}** | Acc={m.accuracy:.4f} | F1={m.f1_macro:.4f} | ROC-AUC={fmt_float(m.roc_auc)}")
                return m

            # Pipelines
            lr = clf_pipeline()
            svm = Pipeline([
                ("scaler", __import__("sklearn").preprocessing.StandardScaler()),
                ("svc", SVC(kernel="rbf", probability=True))
            ])
            rf = Pipeline([
                ("scaler", __import__("sklearn").preprocessing.StandardScaler()),
                ("rf", __import__("sklearn").ensemble.RandomForestClassifier(n_estimators=300, random_state=seed))
            ])

            # ---------- ALL 30 FEATURES ----------
            st.subheader("All 30 features")
            m_lr_all = eval_model(lr, "Logistic Regression", X_tr, X_te, y_tr, y_te, "(All)")
            m_svm_all = eval_model(svm, "SVM (RBF)", X_tr, X_te, y_tr, y_te, "(All)")
            m_rf_all  = eval_model(rf, "Random Forest", X_tr, X_te, y_tr, y_te, "(All)")

            st.markdown("""
            **توضيح النتائج:**  
            - **Accuracy:** تمثل نسبة العينات المصنّفة تصنيفًا صحيحًا ضمن مجموعة الاختبار الخارجية
            - **F1-macro:**  يعبّر عن توازن الأداء بين الدقة والاستدعاء
            - **ROC-AUC:**  القيم ≥ 0.90 تشير إلى تمييز ممتاز، والقيم القريبة من 0.5 تعكس ضعف القدرة التمييزي
            """, unsafe_allow_html=True)

            # ---------- GA-SELECTED FEATURES ----------
            st.subheader("GA-selected features")
            cfg_tmp = GAConfig(random_state=int(seed))
            ga = GeneticAlgorithmFS(X_tr, y_tr, cfg_tmp)
            mask, _, _ = ga.run()
            idx = np.where(mask == 1)[0]
            if idx.size == 0:
                idx = np.array([0])
            st.write(f"Selected {idx.size} features: " + ", ".join([feat_names[i] for i in idx]))

            m_lr_ga = eval_model(lr, "Logistic Regression", X_tr[:, idx], X_te[:, idx], y_tr, y_te, "(GA)")
            m_svm_ga = eval_model(svm, "SVM (RBF)", X_tr[:, idx], X_te[:, idx], y_tr, y_te, "(GA)")
            m_rf_ga  = eval_model(rf, "Random Forest", X_tr[:, idx], X_te[:, idx], y_tr, y_te, "(GA)")

            st.markdown("""
            **توضيح النتائج:**  
            - **Composite Fitness (Inner-CV):**  مقياس مركّب يعكس نسبة جودة الحل     
            - **Accuracy:** تمثل نسبة العينات المصنّفة تصنيفًا صحيحًا ضمن مجموعة الاختبار الخارجية
            - **F1-macro:**  يعبّر عن توازن الأداء بين الدقة والاستدعاء
            - **ROC-AUC:**  القيم ≥ 0.90 تشير إلى تمييز ممتاز، والقيم القريبة من 0.5 تعكس ضعف القدرة التمييزي
            """, unsafe_allow_html=True)


            # ---------- Summary Table (paginated) ----------
            rows = []
            def row_of(name, m, tag):
                rows.append({
                    "Model": f"{name} {tag}",
                    "Accuracy": float(m.accuracy),
                    "F1-macro": float(m.f1_macro),
                    "ROC-AUC": np.nan if m.roc_auc is None else float(m.roc_auc)
                })
            row_of("LR", m_lr_all, "(All)")
            row_of("SVM", m_svm_all, "(All)")
            row_of("RF", m_rf_all, "(All)")
            row_of("LR", m_lr_ga, "(GA)")
            row_of("SVM", m_svm_ga, "(GA)")
            row_of("RF", m_rf_ga, "(GA)")

            df_sum = pd.DataFrame(rows)
            st.markdown("**ملخّص مقارن (جدول – مع تنقّل 5 صفوف):**")
            # نعرض بنسخة منسّقة ثم نمررها للدالة المرقّمة
            styled = df_sum.style.format({"Accuracy": "{:.4f}", "F1-macro": "{:.4f}", "ROC-AUC": "{:.4f}"})
            # نحول الـ style إلى DataFrame بسيط للعرض المرقّم (Streamlit لا يدعم pagination مباشرة للـ Styler)
            paginated_table(df_sum, page_size=5, key="baseline_table")
        st.success("✅ اكتمل تشغيل النماذج المرجعية وعرض النتائج.")

with tabs[2]:
    st.header("Results & Plots")
    st.write("اعرض هنا JSON النتائج التي حفظتها لعرض الرسوم والجداول (مع تنقّل 5 صفوف).")
    uploaded = st.file_uploader("Upload GA result JSON", type=["json"])
    if uploaded is not None:
        res = json.load(uploaded)
        if "history" in res:
            fig, ax = plt.subplots(figsize=(3.6, 2.2))   # أصغر فعليًا
            plt.tight_layout()
            ax.plot(res["history"], linewidth=2)
            ax.set_title("GA best fitness per generation")
            ax.set_xlabel("Generation"); ax.set_ylabel("Best fitness")
            st.pyplot(fig, use_container_width=False)
        if "metrics" in res:
            m = pd.DataFrame([res["metrics"]])
            st.markdown("**Metrics (paginated):**")
            paginated_table(m, page_size=5, key="json_metrics")
        if "selected_names" in res:
            st.markdown("**Selected features (paginated):**")
            paginated_table(pd.DataFrame({"feature": res["selected_names"]}), page_size=5, key="json_feats")

with tabs[3]:
    st.header("Stability across repeated runs")
    st.caption("تشغيل GA عدة مرات ببذور مختلفة وحساب متوسط Jaccard لاختيار الخصائص (مع تنبيه أثناء التنفيذ).")
    if st.button("Run repeated GA" , key="btn_stability"):
        with st.spinner("⏳ جاري تكرار تشغيل GA لحساب الاستقرار..."):
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
        st.success("✅ اكتمل حساب الاستقرار (Jaccard).")
        if jacs:
            st.write(f"Mean Jaccard: {np.mean(jacs):.3f} | N={len(jacs)} pairs")
        else:
            st.write("Mean Jaccard: NA")

with tabs[4]:
    st.header("Export")
    st.write("سيتم حفظ النتائج تحت مجلد `results/` عند التشغيل. يمكنك تنزيل JSON مباشرة من أزرار التحميل في التبويبات السابقة.")
