# ==========================================================
# GA (Improved, Inner-CV + Plots): Fitness = mean(Acc, F1) - Penalty
# Saves plots and prints final Accuracy + F1 on outer test
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
from sklearn.metrics import accuracy_score, f1_score

# -------- Config --------
SEED = 42
TEST_SIZE = 0.30

POP_SIZE = 60
NGEN = 50
CX_PB = 0.7
MUT_PB = 0.2
INDPB = 0.05

ALPHA = 0.10     # penalty weight
N_SPLITS = 5     # inner CV folds

SHOW_PLOTS = True
SAVE_PLOTS = True
OUTPUT_DIR = "figures"
# ------------------------

random.seed(SEED)
np.random.seed(SEED)

# Data
data = load_breast_cancer()
X_full = data.data
y_full = data.target
feature_names = list(data.feature_names)
n_features = X_full.shape[1]

# Scale + outer split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_full, test_size=TEST_SIZE, random_state=SEED, stratify=y_full
)

# -------- Fitness: mean(CV-Acc, CV-F1) - penalty --------
def fitness(individual, alpha=ALPHA, n_splits=N_SPLITS):
    idx = [i for i, bit in enumerate(individual) if bit == 1]
    if not idx:
        return (0.0,)
    Xs = X_train[:, idx]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    accs, f1s = [], []
    clf = LogisticRegression(max_iter=5000)

    for tr, va in skf.split(Xs, y_train):
        clf.fit(Xs[tr], y_train[tr])
        yhat = clf.predict(Xs[va])
        accs.append(accuracy_score(y_train[va], yhat))
        f1s.append(f1_score(y_train[va], yhat, average="macro"))

    cv_acc = float(np.mean(accs))
    cv_f1  = float(np.mean(f1s))
    score  = 0.5 * (cv_acc + cv_f1)
    penalty = alpha * (len(idx) / n_features)
    return (score - penalty,)

# DEAP setup (safe creators)
toolbox = base.Toolbox()
try:
    creator.FitnessMax
except AttributeError:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
try:
    creator.Individual
except AttributeError:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga():
    pop = toolbox.population(n=POP_SIZE)

    best_fitness_per_gen = []
    features_count_per_gen = []

    # initial eval
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit

    for gen in range(1, NGEN+1):
        off = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # crossover
        for c1, c2 in zip(off[::2], off[1::2]):
            if random.random() < CX_PB:
                toolbox.mate(c1, c2)
                del c1.fitness.values; del c2.fitness.values

        # mutation
        for m in off:
            if random.random() < MUT_PB:
                toolbox.mutate(m)
                del m.fitness.values

        # re-eval
        invalid = [ind for ind in off if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = off

        fits = [ind.fitness.values[0] for ind in pop]
        best = tools.selBest(pop, 1)[0]
        k = sum(best)

        best_fitness_per_gen.append(max(fits))
        features_count_per_gen.append(k)

        print(f"[Gen {gen:02d}] Best fitness={max(fits):.4f} | #features={k}")

    # final best
    best = tools.selBest(pop, 1)[0]
    idx = [i for i, bit in enumerate(best) if bit == 1]
    names = [feature_names[i] for i in idx]

    # Final unbiased test
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train[:, idx], y_train)
    yhat = clf.predict(X_test[:, idx])
    acc = accuracy_score(y_test, yhat)
    f1  = f1_score(y_test, yhat, average="macro")

    print("\n" + "="*60)
    print("✅ Selected indices:", idx)
    print("✅ Selected features:")
    for n in names: print("  -", n)
    print(f"✅ Count: {len(idx)} / {n_features}")
    print(f"✅ Outer Test — Accuracy: {acc:.4f} | F1(macro): {f1:.4f}")
    print("="*60 + "\n")

    return best_fitness_per_gen, features_count_per_gen

def plot_curves(best_fitness_per_gen, features_count_per_gen):
    import matplotlib.pyplot as plt
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # fitness curve
    fig1 = plt.figure(figsize=(6.2,4.2))
    plt.plot(range(1, len(best_fitness_per_gen)+1), best_fitness_per_gen, marker='o')
    plt.title("Best Fitness per Generation (GA Improved)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (mean(CV Acc, CV F1) - Penalty)")
    fig1.tight_layout()
    if SAVE_PLOTS:
        f1 = os.path.join(OUTPUT_DIR, "fitness_per_generation.png")
        fig1.savefig(f1, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {f1}")

    # features curve
    fig2 = plt.figure(figsize=(6.2,4.2))
    plt.plot(range(1, len(features_count_per_gen)+1), features_count_per_gen, marker='o')
    plt.title("Selected Features per Generation (GA Improved)")
    plt.xlabel("Generation")
    plt.ylabel("#Features")
    fig2.tight_layout()
    if SAVE_PLOTS:
        f2 = os.path.join(OUTPUT_DIR, "features_count_per_generation.png")
        fig2.savefig(f2, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {f2}")

    if SHOW_PLOTS:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception as e:
            warnings.warn(f"Showing plots failed: {e}\nOpen the saved PNG files instead.")

if __name__ == "__main__":
    best_fit, feat_counts = run_ga()
    plot_curves(best_fit, feat_counts)
