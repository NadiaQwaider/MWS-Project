# utils.py
# ================================================================
# Utilities for:
# - WDBC preset dataset
# - Metrics & evaluation helpers
# - CSV generic dataset loader (Binary classification)
# - Plot helpers (calibration)
# - JSON saving + stability (Jaccard)
# ================================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# -------------------------
# Labels for WDBC preset
# -------------------------
BENIGN = 0
MALIGNANT = 1
LABEL_MAP_STR2INT = {"benign": BENIGN, "malignant": MALIGNANT}
LABEL_MAP_INT2STR = {v: k for k, v in LABEL_MAP_STR2INT.items()}


# ================================================================
# WDBC loader
# ================================================================
def load_wdbc(as_dataframe: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load Breast Cancer Wisconsin Diagnostic dataset from sklearn.
    sklearn uses: benign=1, malignant=0
    We remap to: benign=0, malignant=1 for consistency.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target.copy()
    y = np.where(y == 1, BENIGN, MALIGNANT)  # benign=0, malignant=1
    feat_names = list(data.feature_names)
    return X, y, feat_names


def outer_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def clf_pipeline(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    # Baseline pipeline: StandardScaler + Logistic Regression
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs"))
    ])


# ================================================================
# Metrics
# ================================================================
@dataclass
class Metrics:
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    roc_auc: Optional[float]
    cm: np.ndarray
    brier: Optional[float]


def compute_metrics(
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray],
    y_pred: np.ndarray
) -> Metrics:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro")

    roc = None
    brier = None
    if y_prob is not None:
        roc = roc_auc_score(y_true, y_prob)
        try:
            brier = brier_score_loss(y_true, y_prob)
        except Exception:
            brier = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return Metrics(acc, f1m, prec, rec, roc, cm, brier)


def calibration_xy(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return prob_true, prob_pred


# ================================================================
# Generic CSV dataset loader (Binary Classification)
# ================================================================
def load_csv_dataset(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "binary",
    positive_class=None,
    feature_mode: str = "numeric_only",
    selected_feature_cols=None,
    test_size: float = 0.3,
    seed: int = 42,
):
    import numpy as np
    import pandas as pd

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    # --- Clean target column ---
    y_raw = df[target_col].copy()

    # Treat common placeholders as missing
    y_raw = y_raw.replace(["-", "—", "NA", "N/A", "null", "None", ""], np.nan)

    # Strip strings
    if y_raw.dtype == object:
        y_raw = y_raw.astype(str).str.strip()
        y_raw = y_raw.replace(["nan", "NaN"], np.nan)

    # Drop rows where target is missing
    keep = y_raw.notna()
    df2 = df.loc[keep].copy()
    y_clean = y_raw.loc[keep].copy()

    # --- Feature columns selection ---
    if feature_mode == "manual":
        if not selected_feature_cols:
            raise ValueError("Manual feature mode selected but no feature columns provided.")
        feat_cols = [c for c in selected_feature_cols if c in df2.columns and c != target_col]
    elif feature_mode in ("auto", "numeric_only"):
        # keep numeric columns only (safe default)
        feat_cols = [c for c in df2.columns if c != target_col and pd.api.types.is_numeric_dtype(df2[c])]
    else:
        raise ValueError(f"Unknown feature_mode='{feature_mode}'")

    if len(feat_cols) == 0:
        raise ValueError("No usable numeric feature columns found after filtering.")

    # --- Build y according to task_type ---
    task_type = (task_type or "binary").strip().lower()

    if task_type == "binary":
        # Force binary even if original target has more than 2 values
        if positive_class is None:
            # If no positive class, try infer from unique values
            uniq = pd.unique(y_clean.dropna())
            if len(uniq) != 2:
                raise ValueError(
                    f"Binary task selected but target has {len(uniq)} unique values. "
                    f"Please choose 'positive_class'. Unique={list(uniq)[:10]}"
                )
            neg, pos = uniq[0], uniq[1]
            y = (y_clean == pos).astype(int).to_numpy()
            meta = {"target_col": target_col, "task_type": "binary", "positive_class": pos, "negative_class": neg}
        else:
            # Map: positive_class -> 1, everything else -> 0
            y = (y_clean == positive_class).astype(int).to_numpy()
            meta = {"target_col": target_col, "task_type": "binary", "positive_class": positive_class}

    elif task_type in ("multi", "multi-class", "multiclass"):
        # Not supported for now
        raise NotImplementedError("Multi-class is not supported yet in this version.")
    else:
        raise ValueError(f"Unknown task_type='{task_type}'")

    # --- Build X ---
    X = df2[feat_cols].copy()

    # Replace inf and handle missing numeric values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    X = X.to_numpy(dtype=float)
    feat_names = feat_cols

    return X, y, feat_names, meta


    # -------------------------
    # Feature columns selection
    # -------------------------
    df_X = df.drop(columns=[target_col])

    # If user specified a subset of feature columns
    if selected_feature_cols is not None and len(selected_feature_cols) > 0:
        keep = [c for c in selected_feature_cols if c in df_X.columns]
        if len(keep) == 0:
            raise ValueError("None of the selected feature columns exist in the dataset.")
        df_X = df_X[keep]

    # Decide preprocessing strategy
    feature_mode_norm = (feature_mode or "").strip().lower()

    # Numeric-only mode (safe default)
    if feature_mode_norm in ["numeric-only", "numeric", "auto numeric", "auto-numeric", "numbers only", "numeric only"]:
        numeric_cols = df_X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError(
                "No numeric feature columns found. "
                "Either choose a different feature mode (one-hot) or ensure your dataset has numeric features."
            )

        # Keep only numeric
        df_X = df_X[numeric_cols].copy()

        # Impute numeric
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(df_X.values)

        feat_names = numeric_cols
        meta = {
            "task_type": task_type,
            "target_col": target_col,
            "label_map": label_map,
            "positive_class": pos,
            "negative_class": str(neg),
            "feature_mode": "Numeric-only",
            "n_samples": int(df.shape[0]),
            "n_features": int(len(feat_names)),
        }
        return X.astype(float), y, feat_names, meta

    # One-hot mode (numeric + categoricals)
    else:
        # Identify numeric/categorical
        numeric_cols = df_X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df_X.columns if c not in numeric_cols]

        if len(numeric_cols) == 0 and len(categorical_cols) == 0:
            raise ValueError("No usable feature columns found.")

        # Preprocessors
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_cols),
                ("cat", cat_pipe, categorical_cols),
            ],
            remainder="drop"
        )

        X = pre.fit_transform(df_X)

        # feature names (best-effort)
        feat_names: List[str] = []
        feat_names.extend(numeric_cols)
        try:
            ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["onehot"]
            ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
            feat_names.extend(ohe_names)
        except Exception:
            # fallback if sklearn version differs
            feat_names.extend([f"cat_{i}" for i in range(X.shape[1] - len(numeric_cols))])

        meta = {
            "task_type": task_type,
            "target_col": target_col,
            "label_map": label_map,
            "positive_class": pos,
            "negative_class": str(neg),
            "feature_mode": "One-hot",
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "n_samples": int(df.shape[0]),
            "n_features": int(len(feat_names)),
        }
        return X.astype(float), y, feat_names, meta


# ================================================================
# JSON helpers
# ================================================================
def save_json(obj, path: str):
    def to_serializable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=to_serializable)


# ================================================================
# Stability metric
# ================================================================
def jaccard(a: List[int], b: List[int]) -> float:
    """
    Jaccard similarity between two binary chromosomes.
    """
    A = set([i for i, v in enumerate(a) if v == 1])
    B = set([i for i, v in enumerate(b) if v == 1])
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))
# ================================================================
# Helper Methods
# ================================================================

def eval_model(
    model,
    name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    prefix: str = ""
):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")

    auc = None
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)

    return {
        "name": f"{prefix}{name}",
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

