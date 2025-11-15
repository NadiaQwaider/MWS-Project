import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def evaluate_subset_cv(X, y, mask, cv_splits=5, random_state=42):
    if mask.sum() == 0:
        return 0.0
    X_sub = X[:, mask]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, solver="lbfgs"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X_sub, y, cv=cv, scoring="accuracy")
    return float(scores.mean())

def initial_population(n_features, pop_size, rng):
    pop = rng.integers(0, 2, size=(pop_size, n_features), dtype=np.int8)
    # تأكيد وجود ميزة واحدة على الأقل بكل كروموسوم
    for i in range(pop.shape[0]):
        if pop[i].sum() == 0:
            pop[i, rng.integers(0, n_features)] = 1
    return pop

def tournament_selection(pop, fitness, k, rng):
    selected = []
    n = len(pop)
    for _ in range(n):
        idx = rng.choice(n, size=k, replace=False)
        best = idx[np.argmax(fitness[idx])]
        selected.append(pop[best].copy())
    return np.asarray(selected, dtype=np.int8)

def uniform_crossover(parents, crossover_rate, rng):
    n, m = parents.shape
    offspring = parents.copy()
    for i in range(0, n, 2):
        if i + 1 >= n: break
        if rng.random() < crossover_rate:
            mask = rng.integers(0, 2, size=m, dtype=np.int8).astype(bool)
            a, b = parents[i].copy(), parents[i+1].copy()
            offspring[i]   = np.where(mask, b, a)
            offspring[i+1] = np.where(mask, a, b)
    return offspring

def mutate(pop, mutation_rate, rng):
    flips = rng.random(size=pop.shape) < mutation_rate
    pop[flips] = 1 - pop[flips]
    # لا تسمح بكروموسوم بدون ميزات
    for i in range(pop.shape[0]):
        if pop[i].sum() == 0:
            j = rng.integers(0, pop.shape[1])
            pop[i, j] = 1
    return pop

def run_ga_feature_selection(X, y, *,
    label="GA run", population_size=40, generations=20,
    mutation_rate=0.08, crossover_rate=0.7,
    tournament_k=3, cv_splits=5, random_state=42):

    print(f"\n=== {label} ===")
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    pop = initial_population(n_features, population_size, rng)
    best_mask = None
    best_score = -1.0
    fitness_history = []

    for g in range(generations):
        fitness = np.array([evaluate_subset_cv(X, y, mask.astype(bool), cv_splits, random_state)
                            for mask in pop], dtype=float)
        elite_idx = int(np.argmax(fitness))
        elite_score = float(fitness[elite_idx])
        if elite_score > best_score or best_mask is None:
            best_score = elite_score
            best_mask = pop[elite_idx].copy()

        fitness_history.append(best_score)
        best_feats = int(best_mask.sum())

        # 👇 طباعة التقدّم مع عدد الخصائص
        print(f"Generation {g+1}/{generations} | Best fitness = {best_score:.4f} | Features = {best_feats}")

        parents = tournament_selection(pop, fitness, tournament_k, rng)
        children = uniform_crossover(parents, crossover_rate, rng)
        children = mutate(children, mutation_rate, rng)
        children[0] = best_mask  # elitism
        pop = children

    return best_mask.astype(bool), fitness_history, best_score

def main():
    data = load_breast_cancer()
    X_all, y_all = data.data, data.target
    feature_names = data.feature_names

    # Baseline (كل الميزات)
    print("\n=== Baseline (all features) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, stratify=y_all, random_state=42
    )
    baseline_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, solver="lbfgs"))
    baseline_pipe.fit(X_train, y_train)
    baseline_acc = float(baseline_pipe.score(X_test, y_test))
    baseline_feats = X_all.shape[1]
    print(f"Accuracy (test) = {baseline_acc:.4f} | Features = {baseline_feats}")

    # GA (default)
    mask_ga, fitness_ga, cv_best_ga = run_ga_feature_selection(
        X_all, y_all, label="GA (default params)"
    )
    ga_feats = int(mask_ga.sum())

    # GA (improved)
    mask_ga2, fitness_ga2, cv_best_ga2 = run_ga_feature_selection(
        X_all, y_all, label="GA (improved params)",
        population_size=80, generations=30,
        mutation_rate=0.05, crossover_rate=0.8, random_state=7
    )
    ga2_feats = int(mask_ga2.sum())

    # ملخص النتائج
    df_results = pd.DataFrame({
        "Scenario": ["Baseline (all)", "GA (default)", "GA (improved)"],
        "Accuracy (CV best or Test)": [baseline_acc, cv_best_ga, cv_best_ga2],
        "Features Used": [baseline_feats, ga_feats, ga2_feats]
    })
    print("\n=== Final Comparison ===")
    print(df_results.to_string(index=False))

    # حفظ CSV + رسم منحنى التقدم لكل تجربة GA
    df_results.to_csv("ga_progress_results.csv", index=False)
    plt.figure(figsize=(9,5))
    plt.plot(fitness_ga,  label="GA (default)")
    plt.plot(fitness_ga2, label="GA (improved)")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (accuracy)")
    plt.legend()
    plt.title("GA Progress Over Generations")
    plt.tight_layout()
    plt.savefig("ga_progress_plot.png", dpi=150)
    print("Saved: ga_progress_results.csv + ga_progress_plot.png")

    # حفظ أسماء الميزات المختارة (اختياري)
    sel_ga  = [feature_names[i] for i, m in enumerate(mask_ga)  if m]
    sel_ga2 = [feature_names[i] for i, m in enumerate(mask_ga2) if m]
    with open("ga_selected_features_default.txt", "w", encoding="utf-8") as f:
        f.write(", ".join(sel_ga))
    with open("ga_selected_features_improved.txt", "w", encoding="utf-8") as f:
        f.write(", ".join(sel_ga2))
    print("Saved: ga_selected_features_default.txt / ga_selected_features_improved.txt")

if __name__ == "__main__":
    main()
