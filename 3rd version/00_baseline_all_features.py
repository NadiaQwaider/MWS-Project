# ==========================================================
# Baseline (All Features): LR, SVM, RF  — Accuracy + F1
# ==========================================================
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
TEST_SIZE = 0.30

# Load
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Scale + split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

def eval_model(name, clf):
    clf.fit(X_tr, y_tr)
    yhat = clf.predict(X_te)
    acc = accuracy_score(y_te, yhat)
    f1  = f1_score(y_te, yhat, average="macro")
    print(f"{name:>18} | Accuracy={acc:.4f} | F1(macro)={f1:.4f}")

print("=== Baseline on ALL 30 features ===")
eval_model("LogisticRegression", LogisticRegression(max_iter=5000))
eval_model("SVM (RBF)", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False))
eval_model("RandomForest", RandomForestClassifier(n_estimators=300, random_state=SEED))
