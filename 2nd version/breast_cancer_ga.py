# ==========================================================
# Genetic Algorithm for Feature Selection (Improved + Plots Saved)
# - Inner CV in fitness
# - Outer train/test for unbiased final estimate
# - Seeds fixed, DEAP creator guard
# - Logs per generation
# - Plots are ALWAYS saved to ./figures and shown if possible
# ==========================================================

import os
import numpy as np
import random
import warnings
from deap import base, creator, tools

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------- Config -------------
SEED = 42
POP_SIZE = 60
NGEN = 50
CX_PB = 0.7          # crossover probability
MUT_PB = 0.2         # mutation probability (per individual)
INDPB = 0.05         # bit-flip probability per gene
ALPHA = 0.10         # penalty weight for number of features
N_SPLITS = 5         # inner CV folds

SHOW_PLOTS = True
SAVE_PLOTS = True
OUTPUT_DIR = "figures"
# ----------------------------------

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Data
data = load_breast_cancer()
X_full = data.data
y_full = data.target
feature_names = list(data.feature_names)
n_features = X_full.shape[1]

# Scaling + outer split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_full, test_size=0.30, random_state=SEED, stratify=y_full
)

# ----------------------------
# Fitness with inner CV (StratifiedKFold)
# ----------------------------
def fitness(individual, alpha=ALPHA, n_splits=N_SPLITS):
    selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected_idx) == 0:
        return (0.0,)

    Xs = X_train[:, selected_idx]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs = []

    clf = LogisticRegression(max_iter=5000, n_jobs=None)

    for tr, va in skf.split(Xs, y_train):
        clf.fit(Xs[tr], y_train[tr])
        yhat = clf.predict(Xs[va])
        accs.append(accuracy_score(y_train[va], yhat))

    acc = float(np.mean(accs))
    penalty = alpha * (len(selected_idx) / n_features)
    return (acc - penalty,)

# ----------------------------
# DEAP setup (with guard)
# ----------------------------
toolbox = base.Toolbox()

def _init_deap_creators():
    try:
        creator.FitnessMax
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.Individual
    except AttributeError:
        creator.create("Individual", list, fitness=creator.FitnessMax)

_init_deap_creators()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=3)

# ----------------------------
# GA runner
# ----------------------------
def run_ga():
    pop = toolbox.population(n=POP_SIZE)

    best_fitness_per_gen = []
    features_count_per_gen = []

    # initial evaluation
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)):
        ind.fitness.values = fit

    for gen in range(1, NGEN + 1):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PB:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # mutation
        for mut in offspring:
            if random.random() < MUT_PB:
                toolbox.mutate(mut)
                del mut.fitness.values

        # re-evaluate invalid
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)):
            ind.fitness.values = fit

        # replace population
        pop[:] = offspring

        # logging
        fits = [ind.fitness.values[0] for ind in pop]
        best = tools.selBest(pop, 1)[0]
        k = sum(best)

        best_fitness_per_gen.append(max(fits))
        features_count_per_gen.append(k)

        print(f"[Gen {gen:02d}] Best fitness={max(fits):.4f} | #features={k}")

    # final best
    best_ind = tools.selBest(pop, 1)[0]
    best_idx = [i for i, bit in enumerate(best_ind) if bit == 1]
    best_features = [feature_names[i] for i in best_idx]

    clf_final = LogisticRegression(max_iter=5000, n_jobs=None)
    clf_final.fit(X_train[:, best_idx], y_train)
    yhat_test = clf_final.predict(X_test[:, best_idx])
    test_acc = accuracy_score(y_test, yhat_test)

    print("\n" + "="*60)
    print("✅ Final Selected Feature Indices:", best_idx)
    print("✅ Final Selected Feature Names:")
    for name in best_features:
        print("   -", name)
    print(f"✅ Count of Features: {len(best_idx)} / {n_features}")
    print(f"✅ Outer Test Accuracy (unbiased): {test_acc:.4f}")
    print("="*60 + "\n")

    return best_fitness_per_gen, features_count_per_gen

# ----------------------------
# Plotting (save + show if possible)
# ----------------------------
def plot_curves(best_fitness_per_gen, features_count_per_gen):
    import matplotlib
    import matplotlib.pyplot as plt

    # For headless environments you can uncomment:
    # matplotlib.use("Agg")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Figure 1: fitness
    fig1 = plt.figure(figsize=(6.2,4.2))
    plt.plot(range(1, len(best_fitness_per_gen)+1), best_fitness_per_gen, marker='o')
    plt.title("Best Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (CV Accuracy - Penalty)")
    fig1.tight_layout()
    if SAVE_PLOTS:
        f1 = os.path.join(OUTPUT_DIR, "fitness_per_generation.png")
        fig1.savefig(f1, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {f1}")

    # Figure 2: #features
    fig2 = plt.figure(figsize=(6.2,4.2))
    plt.plot(range(1, len(features_count_per_gen)+1), features_count_per_gen, marker='o')
    plt.title("Selected Features per Generation")
    plt.xlabel("Generation")
    plt.ylabel("#Features")
    fig2.tight_layout()
    if SAVE_PLOTS:
        f2 = os.path.join(OUTPUT_DIR, "features_count_per_generation.png")
        fig2.savefig(f2, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {f2}")

    if SHOW_PLOTS:
        try:
            plt.show()
        except Exception as e:
            warnings.warn(f"Showing plots failed: {e}\nOpen the saved PNG files instead.")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    best_fit, feat_counts = run_ga()
    plot_curves(best_fit, feat_counts)
