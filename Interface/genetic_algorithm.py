# genetic_algorithm.py
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

def evaluate_subset(X, y, mask):
    if mask.sum() == 0:
        return 0.0
    X_sub = X[:, mask.astype(bool)]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, solver="lbfgs"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_sub, y, cv=cv, scoring="accuracy")
    return float(scores.mean())

def initial_population(n_features, pop_size, rng):
    pop = rng.integers(0, 2, size=(pop_size, n_features), dtype=np.int32)
    # ضمان اختيار عنصر واحد على الأقل في كل كروموسوم
    for i in range(pop.shape[0]):
        if pop[i].sum() == 0:
            pop[i, rng.integers(0, n_features)] = 1
    return pop

def tournament_selection(pop, fitness, k, rng):
    selected = []
    n = len(pop)
    for _ in range(n):
        inds = rng.choice(n, size=k, replace=False)
        best = inds[np.argmax(fitness[inds])]
        selected.append(pop[best].copy())
    return np.array(selected, dtype=np.int32)

def uniform_crossover(parents, crossover_rate, rng):
    n, m = parents.shape
    offspring = parents.copy()
    for i in range(0, n, 2):
        if i+1 >= n: break
        if rng.random() < crossover_rate:
            mask = rng.integers(0, 2, size=m).astype(bool)
            a, b = parents[i].copy(), parents[i+1].copy()
            a_new = np.where(mask, b, a)
            b_new = np.where(mask, a, b)
            offspring[i], offspring[i+1] = a_new, b_new
    return offspring

def mutate(pop, mutation_rate, rng):
    flips = rng.random(pop.shape) < mutation_rate
    pop[flips] = 1 - pop[flips]
    # تأكد لا يوجد كروموسوم بدون ميزة
    for i in range(pop.shape[0]):
        if pop[i].sum() == 0:
            pop[i, rng.integers(0, pop.shape[1])] = 1
    return pop

def run_ga(df, features, target_col,
           population_size=50, generations=25,
           mutation_rate=0.05, crossover_rate=0.8,
           random_state=42):
    rng = np.random.default_rng(random_state)
    X = df[features].values.astype(float)
    y = df[target_col].astype(str).str.lower().map({"benign":0, "malignant":1}).values
    n_features = len(features)

    pop = initial_population(n_features, population_size, rng)
    best_mask = None
    best_score = -1.0
    fitness_history = []

    for gen in range(generations):
        fitness = np.array([evaluate_subset(X, y, mask) for mask in pop])
        # تسجيل أفضل في الجيل
        gen_best_idx = np.argmax(fitness)
        gen_best_score = float(fitness[gen_best_idx])
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_mask = pop[gen_best_idx].copy()
        fitness_history.append(best_score)

        # اختيار، تقاطع، طفرة
        parents = tournament_selection(pop, fitness, k=3, rng=rng)
        children = uniform_crossover(parents, crossover_rate, rng)
        children = mutate(children, mutation_rate, rng)
        # حفظ النخبة في بداية الجيل الجديد
        children[0] = best_mask
        pop = children

    selected_features = [f for f, m in zip(features, best_mask) if m == 1]
    return selected_features, fitness_history
