# ==========================================================
# خوارزمية جينية محسّنة + رسم بياني (دقة vs الأجيال)
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random
from deap import base, creator, tools
import arabic_reshaper
from bidi.algorithm import get_display
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
# تشغيل GA + تخزين النتائج
# ----------------------------
def run_ga():
    pop = toolbox.population(n=60)
    NGEN = 50
    best_acc_per_gen = []
    features_count = []

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
        
        # أفضل فرد في هذا الجيل
        best_ind = tools.selBest(pop, 1)[0]
        best_features = [i for i in range(n_features) if best_ind[i] == 1]
        acc = accuracy_score(
            y_test, LogisticRegression(max_iter=5000).fit(
                X_train[:, best_features], y_train).predict(X_test[:, best_features])
        )
        
        text1= "الجيل"
        reshaped_text1 = arabic_reshaper.reshape(text1)    # correct its shape
        bidi_text1 = get_display(reshaped_text1)   # correct its direction

        text2= "أفضل دقة "
        reshaped_text2 = arabic_reshaper.reshape(text2)    # correct its shape
        bidi_text2 = get_display(reshaped_text2)           # correct its direction

        text3= " عدد الخصائص "
        reshaped_text3 = arabic_reshaper.reshape(text3)    # correct its shape
        bidi_text3 = get_display(reshaped_text3)  # correct its direction

        text4= " تطور الدقة عبر الأجيال"
        reshaped_text4 = arabic_reshaper.reshape(text4)    # correct its shape
        bidi_text4 = get_display(reshaped_text4)  # correct its direction
        
        text5= " عدد الأجيال"
        reshaped_text5 = arabic_reshaper.reshape(text5)    # correct its shape
        bidi_text5 = get_display(reshaped_text5)  # correct its direction
        
        text6= "الدقة"
        reshaped_text6 = arabic_reshaper.reshape(text6)    # correct its shape
        bidi_text6 = get_display(reshaped_text6)  # correct its direction 
        
        text7= "عدد الخصائص المختارة عبر الأجيال"
        reshaped_text7 = arabic_reshaper.reshape(text7)    # correct its shape
        bidi_text7 = get_display(reshaped_text7)  # correct its direction 
        
        text8= "أفضل مجموعة خصائص مختارة:"
        reshaped_text8 = arabic_reshaper.reshape(text8)    # correct its shape
        bidi_text8 = get_display(reshaped_text8)  # correct its direction 
        
        text9= " دقة النموذج النهائي  "
        reshaped_text9 = arabic_reshaper.reshape(text9)    # correct its shape
        bidi_text9 = get_display(reshaped_text9)  # correct its direction
        
        best_acc_per_gen.append(acc)
        features_count.append(len(best_features))
        print(f"{bidi_text1}{gen+1}: {bidi_text2} {acc:.4f}, {bidi_text3} = {len(best_features)}")

    # ----------------------------
    # رسم النتائج
    # ----------------------------
    plt.figure(figsize=(12,5))

    # منحنى الدقة
    plt.subplot(1,2,1)
    plt.plot(range(1, NGEN+1), best_acc_per_gen, marker='o')
    plt.title(f"{bidi_text4}")
    plt.xlabel(f"{bidi_text5}")
    plt.ylabel(f"{bidi_text6}")
    plt.ylim(0.9, 1.0)

    # منحنى عدد الخصائص
    plt.subplot(1,2,2)
    plt.plot(range(1, NGEN+1), features_count, marker='o', color='orange')
    plt.title(f"{bidi_text7}")
    plt.xlabel(f"{bidi_text5}")
    plt.ylabel(f"{bidi_text3}")
    plt.ylim(0, n_features)

    plt.tight_layout()
    plt.show()

    # ----------------------------
    # أفضل حل نهائي
    # ----------------------------
    print(f"{bidi_text8}", best_features)
    print(f"{bidi_text3}", len(best_features))
    print(f"{bidi_text9}", acc)

# تشغيل
run_ga()
