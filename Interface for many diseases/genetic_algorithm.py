# genetic_algorithm.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone

from utils import clf_pipeline, compute_metrics

@dataclass
class GAConfig:
    population_size: int = 60
    generations: int = 50
    crossover_prob: float = 0.8
    mutation_prob: float = 0.05
    tournament_k: int = 3
    elitism: int = 2
    inner_cv_folds: int = 5
    lambda_f1: float = 0.5   # وزن F1 في الملاءمة
    alpha_penalty: float = 0.15  # حد عقابي على عدد الخصائص
    early_stopping_rounds: int = 10  # إيقاف مبكر
    random_state: int = 42
        # Fitness weights (default = balanced thesis behavior)
    w_acc: float = 0.5
    w_f1: float = 0.5
    w_recall_pos: float = 0.0  # only meaningful for binary


class GeneticAlgorithmFS:
    def __init__(self, X: np.ndarray, y: np.ndarray, config: GAConfig):
        self.X = X
        self.y = y
        self.p = X.shape[1]
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg.random_state)
        self.memo: Dict[Tuple[int, ...], float] = {}

    def _init_population(self) -> List[np.ndarray]:
        pop = []
        for _ in range(self.cfg.population_size):
            mask = self.rng.binomial(1, 0.5, size=self.p)
            if mask.sum() == 0:
                mask[self.rng.randint(0, self.p)] = 1
            pop.append(mask)
        return pop

    def _tournament(self, pop: List[np.ndarray], fitness: Dict[int, float]) -> int:
        contestants = self.rng.choice(len(pop), size=self.cfg.tournament_k, replace=False)
        best = max(contestants, key=lambda idx: fitness[idx])
        return best

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.rand() > self.cfg.crossover_prob:
            return a.copy(), b.copy()
        point = self.rng.randint(1, self.p - 1)
        c1 = np.concatenate([a[:point], b[point:]])
        c2 = np.concatenate([b[:point], a[point:]])
        return c1, c2

    def _mutation(self, x: np.ndarray) -> np.ndarray:
        for i in range(self.p):
            if self.rng.rand() < self.cfg.mutation_prob:
                x[i] = 1 - x[i]
        if x.sum() == 0:  # منع قناع فارغ
            x[self.rng.randint(0, self.p)] = 1
        return x

    def _fitness(self, mask: np.ndarray) -> float:
        key = tuple(mask.tolist())
        if key in self.memo:
            return self.memo[key]

        idx = np.where(mask == 1)[0]
        Xs = self.X[:, idx] if len(idx) > 0 else self.X[:, :1]  # ضمان بعد واحد على الأقل
        model = clf_pipeline()
        cv = StratifiedKFold(n_splits=self.cfg.inner_cv_folds, shuffle=True, random_state=self.cfg.random_state)

        # نستخدم تنبؤات CV للحصول على F1/Accuracy و احتمالات ROC/calibration عند الحاجة
        y_pred = cross_val_predict(model, Xs, self.y, cv=cv, method="predict")
        # للحصول على احتمالات إيجابي (malignant=1) نعيد تدريب داخل cross_val_predict مرة أخرى
        try:
            y_prob = cross_val_predict(model, Xs, self.y, cv=cv, method="predict_proba")[:, 1]
        except Exception:
            y_prob = None

        m = compute_metrics(self.y, y_prob, y_pred)
        # الملاءمة المركبة + عقوبة عدد الخصائص
        k = float(mask.sum())
        p = float(self.X.shape[1])
        penalty = self.cfg.alpha_penalty * (k / p)
        fit = self.cfg.lambda_f1 * m.f1_macro + (1 - self.cfg.lambda_f1) * m.accuracy - penalty

        self.memo[key] = fit
        return fit

    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        self.rng.seed(self.cfg.random_state)
        pop = self._init_population()
        best_mask = pop[0].copy()
        best_fit = -1e9
        history = []
        rounds_since_improve = 0

        for g in range(self.cfg.generations):
            fitness = {i: self._fitness(ind) for i, ind in enumerate(pop)}
            # حفظ أفضل
            gen_best_idx = max(fitness, key=lambda i: fitness[i])
            gen_best_fit = fitness[gen_best_idx]
            gen_best_mask = pop[gen_best_idx].copy()
            history.append(float(gen_best_fit))

            if gen_best_fit > best_fit + 1e-8:
                best_fit = float(gen_best_fit)
                best_mask = gen_best_mask
                rounds_since_improve = 0
            else:
                rounds_since_improve += 1

            # إيقاف مبكر
            if self.cfg.early_stopping_rounds and rounds_since_improve >= self.cfg.early_stopping_rounds:
                break

            # نخبوية
            sorted_idx = sorted(fitness.keys(), key=lambda i: fitness[i], reverse=True)
            new_pop = [pop[i].copy() for i in sorted_idx[: self.cfg.elitism]]

            # إنتاج بقية الأفراد
            while len(new_pop) < self.cfg.population_size:
                p1 = pop[self._tournament(pop, fitness)]
                p2 = pop[self._tournament(pop, fitness)]
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutation(c1)
                if len(new_pop) < self.cfg.population_size:
                    new_pop.append(c1)
                c2 = self._mutation(c2)
                if len(new_pop) < self.cfg.population_size:
                    new_pop.append(c2)
            pop = new_pop
        return best_mask, best_fit, history
