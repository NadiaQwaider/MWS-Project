# ==========================================================
# خوارزمية جينية محسّنة (توازن بين الدقة وعدد الخصائص)
# ==========================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random
from deap import base, creator, tools

# 1. تحميل البيانات
data = load_breast_cancer()
X = data.data
y = data.target
n_features = X.shape[1]

# تطبيع البيانات
scaler = StandardScaler()
X = scaler.fit_transform(X)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------------
# دالة اللياقة (Fitness Function)
# ----------------------------
def fitness(individual, alpha=0.1):  
    selected_features = [i for i in range(n_features) if individual[i] == 1]
    if len(selected_features) == 0:
        return 0,
    
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train[:, selected_features], y_train)
    y_pred = clf.predict(X_test[:, selected_features])
    acc = accuracy_score(y_test, y_pred)
    
    # دقة - عقوبة بسيطة على عدد الخصائص
    penalty = alpha * (len(selected_features) / n_features)
    return acc - penalty,

# ----------------------------
# إعداد GA
# ----------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# ----------------------------
# تشغيل GA
# ----------------------------
def run_ga():
    pop = toolbox.population(n=60)   # حجم المجتمع أكبر
    NGEN = 50                        # عدد الأجيال أكبر
    for gen in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # تقاطع
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # طفرة
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # تقييم
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"الجيل {gen+1}: أفضل لياقة = {max(fits):.4f}")

    best_ind = tools.selBest(pop, 1)[0]
    best_features = [i for i in range(n_features) if best_ind[i] == 1]
    print("\n✅ أفضل مجموعة خصائص مختارة:", best_features)
    print("عدد الخصائص:", len(best_features))
    print("دقة النموذج النهائي (قبل العقوبة):", accuracy_score(
        y_test, LogisticRegression(max_iter=5000).fit(
            X_train[:, best_features], y_train).predict(X_test[:, best_features])
    ))

# تشغيل
run_ga()

