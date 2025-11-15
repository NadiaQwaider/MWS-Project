# utils.py
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

BENIGN = 0
MALIGNANT = 1
LABEL_MAP_STR2INT = {"benign": BENIGN, "malignant": MALIGNANT}
LABEL_MAP_INT2STR = {v: k for k, v in LABEL_MAP_STR2INT.items()}

def load_wdbc(as_dataframe: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = load_breast_cancer()
    X = data.data
    y = data.target.copy()
    # sklearn: benign=1, malignant=0 -> نعيد ترميزها إلى: benign=0, malignant=1
    y = np.where(y == 1, BENIGN, MALIGNANT)
    feat_names = list(data.feature_names)
    if as_dataframe:
        return X, y, feat_names
    return X, y, feat_names

def outer_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

def clf_pipeline(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    # خط أساس مطابق للأطروحة: StandardScaler + Logistic Regression
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs"))
    ])

@dataclass
class Metrics:
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    roc_auc: Optional[float]
    cm: np.ndarray
    brier: Optional[float]

def compute_metrics(y_true: np.ndarray, y_prob: Optional[np.ndarray], y_pred: np.ndarray) -> Metrics:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro")
    roc = None
    brier = None
    if y_prob is not None:
        # احتمال الفئة الإيجابية (malignant=1)
        roc = roc_auc_score(y_true, y_prob)
        try:
            brier = brier_score_loss(y_true, y_prob)
        except Exception:
            brier = None
    cm = confusion_matrix(y_true, y_pred, labels=[BENIGN, MALIGNANT])
    return Metrics(acc, f1m, prec, rec, roc, cm, brier)

def calibration_xy(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return prob_true, prob_pred

def save_json(obj, path: str):
    import os, json, numpy as np

    def to_serializable(o):
        # مصفوفات NumPy
        if isinstance(o, np.ndarray):
            return o.tolist()
        # أعداد NumPy
        if isinstance(o, (np.integer, )):
            return int(o)
        if isinstance(o, (np.floating, )):
            return float(o)
        if isinstance(o, (np.bool_, )):
            return bool(o)
        # أي شيء آخر غير مدعوم
        return str(o)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=to_serializable)


def jaccard(a: List[int], b: List[int]) -> float:
    A = set([i for i, v in enumerate(a) if v == 1])
    B = set([i for i, v in enumerate(b) if v == 1])
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))
