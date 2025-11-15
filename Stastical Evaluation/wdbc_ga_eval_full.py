# wdbc_ga_eval_full.py
# ==========================================================
# Wisconsin Diagnostic Breast Cancer (WDBC)
# Baseline vs GA-Default vs GA-Improved
# Nested-ish CV, Accuracy + F1-macro, paired tests & plots
# ==========================================================

import os, math, random, time, json
import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from scipy import stats
import matplotlib.pyplot as plt

from deap import base, creator, tools

# -----------------------------
# إعدادات عامة
# -----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUT_DIR = "./outputs_wdbc"
os.makedirs(OUT_DIR, exist_ok=True)

# مفاتيح لتخفيف/توسيع الدراسة
ULTRAFAST = False   # True: لكل شيء إعدادات صغيرة جدًا
FAST       = True   # True: إعدادات متوسطة؛ False: إعدادات كاملة 5x5 + GA موسّعة

# -----------------------------
# تحميل البيانات وترميزها
# -----------------------------
data = load_breast_cancer()
X_full = data.data
y_full = data.target
feature_names = data.feature_names
n_features = X_full.shape[1]

# -----------------------------
# أدوات مساعدة
# -----------------------------
def make_scaler():
    return StandardScaler()

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def summarize_df_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"[saved] {path}")

def paired_tests(df_scores: pd.DataFrame, a_col: str, b_col: str):
    # تفترض الأعمدة هي قياسات مطابقة بالحجم
    a = df_scores[a_col].values
    b = df_scores[b_col].values
    t_stat, t_p = stats.ttest_rel(a, b, alternative="two-sided")
    w_stat, w_p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {"t_stat": t_stat, "t_p": t_p, "wilcoxon_stat": w_stat, "wilcoxon_p": w_p}

def boxplot_metric(df, metric, save_path):
    plt.figure(figsize=(7,6))
    ax = df.boxplot(column=[c for c in df.columns if c.endswith(metric)], grid=False)
    plt.title(f"Boxplot - {metric}")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"[saved] {save_path}")

def distplot_metric(df, metric, save_path):
    plt.figure(figsize=(7,6))
    for c in [c for c in df.columns if c.endswith(metric)]:
        vals = df[c].dropna().values
        if len(vals) == 0: 
            continue
        # رسم كثافة بسيط
        xs = np.linspace(min(vals), max(vals), 200)
        try:
            kde = stats.gaussian_kde(vals)
            plt.plot(xs, kde(xs), label=c)
        except Exception:
            # كاحتياط لو فشل KDE
            plt.hist(vals, bins=10, alpha=0.4, density=True, label=c)
    plt.title(f"Distribution - {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"[saved] {save_path}")

# -----------------------------
# مصنّفات الأساس Baseline
# -----------------------------
def baseline_models():
    return {
        "LR": Pipeline([("scaler", make_scaler()),
                        ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
        "SVM": Pipeline([("scaler", make_scaler()),
                         ("clf", SVC(kernel="rbf", probability=False, random_state=SEED))]),
        "RF": Pipeline([("scaler", make_scaler()),
                        ("clf", RandomForestClassifier(
                            n_estimators=300 if not ULTRAFAST and not FAST else (120 if FAST else 60),
                            max_depth=None, random_state=SEED))])
    }

# -----------------------------
# GA لانتقاء الخصائص (DEAP)
# -----------------------------
@dataclass
class GAConfig:
    pop_size: int
    n_gen: int
    cx_pb: float
    mut_pb: float
    indpb: float        # احتمال قلب البت لكل جين
    tourn_size: int
    elitism: int        # عدد الأفراد النخبة
    inner_cv_splits: int  # KFold داخل اللياقة
    alpha: float        # عقوبة نسبة الخصائص
    use_f1: bool        # إدخال F1 في اللياقة

def make_ga_config_default():
    if ULTRAFAST:
        return GAConfig(pop_size=18, n_gen=8, cx_pb=0.7, mut_pb=0.2, indpb=0.04,
                        tourn_size=3, elitism=1, inner_cv_splits=3, alpha=0.10, use_f1=True)
    if FAST:
        return GAConfig(pop_size=36, n_gen=16, cx_pb=0.75, mut_pb=0.2, indpb=0.04,
                        tourn_size=3, elitism=2, inner_cv_splits=5, alpha=0.10, use_f1=True)
    return GAConfig(pop_size=40, n_gen=20, cx_pb=0.7, mut_pb=0.2, indpb=0.05,
                    tourn_size=3, elitism=2, inner_cv_splits=5, alpha=0.10, use_f1=True)

def make_ga_config_improved():
    if ULTRAFAST:
        return GAConfig(pop_size=24, n_gen=10, cx_pb=0.8, mut_pb=0.18, indpb=0.05,
                        tourn_size=3, elitism=2, inner_cv_splits=3, alpha=0.12, use_f1=True)
    if FAST:
        return GAConfig(pop_size=54, n_gen=22, cx_pb=0.8, mut_pb=0.18, indpb=0.05,
                        tourn_size=3, elitism=3, inner_cv_splits=5, alpha=0.12, use_f1=True)
    return GAConfig(pop_size=80, n_gen=30, cx_pb=0.8, mut_pb=0.18, indpb=0.05,
                    tourn_size=3, elitism=4, inner_cv_splits=5, alpha=0.12, use_f1=True)

# اللياقة: متوسط CV-Accuracy و CV-F1 (macro) مع عقوبة الحجم
def ga_fitness_builder(X_tr, y_tr, alpha=0.1, inner_cv_splits=5, use_f1=True):
    # نستخدم LR كمقوم داخل اللياقة
    def eval_individual(individual):
        mask = np.array(individual, dtype=bool)
        if not mask.any():
            return (0.0,)
        X_sub = X_tr[:, mask]
        # CV داخل اللياقة
        kf = KFold(n_splits=inner_cv_splits, shuffle=True, random_state=SEED)
        accs, f1s = [], []
        for tr_idx, val_idx in kf.split(X_sub):
            X_t, X_v = X_sub[tr_idx], X_sub[val_idx]
            y_t, y_v = y_tr[tr_idx], y_tr[val_idx]
            pipe = Pipeline([
                ("scaler", make_scaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=SEED))
            ])
            pipe.fit(X_t, y_t)
            y_hat = pipe.predict(X_v)
            accs.append(accuracy_score(y_v, y_hat))
            f1s.append(f1_macro(y_v, y_hat))
        acc_cv = float(np.mean(accs))
        f1_cv  = float(np.mean(f1s))
        # عقوبة الحجم
        penalty = alpha * (np.sum(mask) / mask.size)
        if use_f1:
            # مزج بسيط بين Accuracy و F1
            score = 0.5 * acc_cv + 0.5 * f1_cv
        else:
            score = acc_cv
        return (score - penalty,)
    return eval_individual

def run_ga_once(X_tr, y_tr, cfg: GAConfig):
    # DEAP setup (إنشاء أنماط مرة واحدة آمن إذا لم تُنشأ سابقًا)
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    eval_func = ga_fitness_builder(X_tr, y_tr, alpha=cfg.alpha,
                                   inner_cv_splits=cfg.inner_cv_splits,
                                   use_f1=cfg.use_f1)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=cfg.indpb)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tourn_size)

    pop = toolbox.population(n=cfg.pop_size)

    # تقييم أولي
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fits = list(map(toolbox.evaluate, invalid))
    for ind, ft in zip(invalid, fits):
        ind.fitness.values = ft

    # تطور
    for gen in range(cfg.n_gen):
        # نخبوية
        elites = tools.selBest(pop, cfg.elitism) if cfg.elitism > 0 else []
        offspring = toolbox.select(pop, len(pop) - len(elites))
        offspring = list(map(toolbox.clone, offspring))

        # تقاطع
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cfg.cx_pb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # طفرة
        for mut in offspring:
            if random.random() < cfg.mut_pb:
                toolbox.mutate(mut)
                del mut.fitness.values

        # تقييم غير الصالحين
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, ft in zip(invalid, fits):
            ind.fitness.values = ft

        pop = offspring + elites

    best = tools.selBest(pop, 1)[0]
    mask = np.array(best, dtype=bool)
    return mask

# -----------------------------
# 5x5 Stratified CV خارجي
# -----------------------------
if ULTRAFAST:
    OUTER_SPLITS = 2
    REPEATS = 1
elif FAST:
    OUTER_SPLITS = 3
    REPEATS = 1
else:
    OUTER_SPLITS = 5
    REPEATS = 5  # 5x5

rows = []

# Baselines سنجمعها هنا
BASELINES = baseline_models()
SCENARIOS = ["Baseline_LR", "Baseline_SVM", "Baseline_RF", "GA_Default", "GA_Improved"]

t0_global = time.time()

for rep in range(REPEATS):
    skf = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED + rep)
    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X_full, y_full), start=1):
        X_tr, X_te = X_full[tr_idx], X_full[te_idx]
        y_tr, y_te = y_full[tr_idx], y_full[te_idx]

        # -------- Baseline (كلها تستخدم جميع الخصائص)
        for name, model in BASELINES.items():
            model_ = deepcopy(model)
            model_.fit(X_tr, y_tr)
            y_hat = model_.predict(X_te)
            acc = accuracy_score(y_te, y_hat)
            f1m = f1_macro(y_te, y_hat)
            rows.append({
                "rep": rep+1, "fold": fold_id, "scenario": f"Baseline_{name}",
                "n_features": n_features, "acc": acc, "f1": f1m
            })

        # -------- GA-Default
        cfg_def = make_ga_config_default()
        mask_def = run_ga_once(X_tr, y_tr, cfg_def)
        used_idx_def = np.where(mask_def)[0].tolist()
        # تدريب LR على الخصائص المنتقاة فقط (Nested-ish: CV داخل اللياقة + اختبار خارجي هنا)
        pipe_def = Pipeline([("scaler", make_scaler()),
                             ("clf", LogisticRegression(max_iter=5000, random_state=SEED))])
        pipe_def.fit(X_tr[:, mask_def], y_tr)
        y_hat = pipe_def.predict(X_te[:, mask_def])
        rows.append({
            "rep": rep+1, "fold": fold_id, "scenario": "GA_Default",
            "n_features": int(mask_def.sum()),
            "acc": accuracy_score(y_te, y_hat),
            "f1": f1_macro(y_te, y_hat)
        })

        # -------- GA-Improved
        cfg_imp = make_ga_config_improved()
        mask_imp = run_ga_once(X_tr, y_tr, cfg_imp)
        used_idx_imp = np.where(mask_imp)[0].tolist()
        pipe_imp = Pipeline([("scaler", make_scaler()),
                             ("clf", LogisticRegression(max_iter=5000, random_state=SEED))])
        pipe_imp.fit(X_tr[:, mask_imp], y_tr)
        y_hat = pipe_imp.predict(X_te[:, mask_imp])
        rows.append({
            "rep": rep+1, "fold": fold_id, "scenario": "GA_Improved",
            "n_features": int(mask_imp.sum()),
            "acc": accuracy_score(y_te, y_hat),
            "f1": f1_macro(y_te, y_hat)
        })

t1_global = time.time()
print(f"Total time: {t1_global - t0_global:.1f}s")

# -----------------------------
# حفظ النتائج التفصيلية
# -----------------------------
df = pd.DataFrame(rows)
scores_path = os.path.join(OUT_DIR, f"wdbc_ga_cv_scores_{'ULTRAFAST' if ULTRAFAST else ('FAST' if FAST else 'FULL')}.csv")
summarize_df_to_csv(df, scores_path)

# -----------------------------
# جداول الملخص
# -----------------------------
summary = (df.groupby("scenario")[["acc", "f1", "n_features"]]
             .agg(["mean","std","median","min","max","count"]))
summary.columns = ['_'.join(col) for col in summary.columns]
summary = summary.reset_index()
summary_path = os.path.join(OUT_DIR, f"wdbc_ga_cv_summary_{'ULTRAFAST' if ULTRAFAST else ('FAST' if FAST else 'FULL')}.csv")
summarize_df_to_csv(summary, summary_path)

# -----------------------------
# اختبارات زوجية (Baseline_LR ضد كل سيناريو مهم؛ وأيضًا بين GA-Default و GA-Improved)
# -----------------------------
# نحول النتائج إلى wide لكل طيّة/تكرار
def to_wide(metric):
    w = df.pivot_table(index=["rep","fold"], columns="scenario", values=metric)
    return w.dropna(axis=0, how="any")

pairs = [
    ("Baseline_LR", "GA_Default"),
    ("Baseline_LR", "GA_Improved"),
    ("GA_Default", "GA_Improved"),
    ("Baseline_SVM", "GA_Improved"),
    ("Baseline_RF", "GA_Improved"),
]

tests_rows = []
for metric in ["acc","f1"]:
    w = to_wide(metric)
    for a,b in pairs:
        if a in w.columns and b in w.columns:
            res = paired_tests(w[[a,b]].dropna(), a, b)
            tests_rows.append({
                "metric": metric, "pair": f"{a} vs {b}",
                "t_stat": res["t_stat"], "t_p": res["t_p"],
                "wilcoxon_stat": res["wilcoxon_stat"], "wilcoxon_p": res["wilcoxon_p"],
                "N_pairs": len(w)
            })

tests_df = pd.DataFrame(tests_rows)
tests_path = os.path.join(OUT_DIR, f"wdbc_ga_paired_tests_{'ULTRAFAST' if ULTRAFAST else ('FAST' if FAST else 'FULL')}.csv")
summarize_df_to_csv(tests_df, tests_path)

# -----------------------------
# رسومات
# -----------------------------
# نجهز نسخة wide لسهولة الرسم المتسق
w_acc = to_wide("acc")
w_f1  = to_wide("f1")

# Boxplots
boxplot_metric(w_acc, "acc", os.path.join(OUT_DIR, f"fig_box_acc_{'ULTRAFAST' if ULTRAFAST else ('FAST' if FAST else 'FULL')}.png"))
boxplot_metric(w_f1,  "f1",  os.path.join(OUT_DIR, f"fig_box_f1_{'ULTRAFAST' if ULTRAFAST else ('FAST' if FAST else 'FULL')}.png"))

# Distributions
distplot_metric(w_acc, "acc", os.path.join(OUT_DIR, f"fig_dist_acc_{'ULTRAFAST' if ULTRAFAST else ('FAST' if FAST else 'FULL')}.png"))
distplot_metric(w_f1,  "f1",  os.path.join(OUT_DIR, f"fig_dist_f1_{'ULTRAFAST' if ULTRAFAST else ('FAST' if FAST else 'FULL')}.png"))

print("\nDone. CSVs and figures are in:", OUT_DIR)
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(" -", f)
