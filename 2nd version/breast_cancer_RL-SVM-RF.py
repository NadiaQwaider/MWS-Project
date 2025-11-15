# ==========================================================
# مقارنة: Logistic Regression - SVM - Random Forest
# باستخدام كل الخصائص (بدون خوارزمية جينية)
# ==========================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import arabic_reshaper
from bidi.algorithm import get_display
# 1. تحميل البيانات
data = load_breast_cancer()
X = data.data
y = data.target

# 2. تطبيع البيانات (مهم لـ Logistic Regression و SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. تقسيم البيانات (70% تدريب - 30% اختبار)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# 4. Logistic Regression
# -------------------------------
clf_lr = LogisticRegression(max_iter=5000)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# -------------------------------
# 5. Support Vector Machine (SVM)
# -------------------------------
clf_svm = SVC(kernel='linear')  # kernel='linear' أسرع وأسهل للتفسير
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

# -------------------------------
# 6. Random Forest
# -------------------------------
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# -------------------------------
# 7. طباعة النتائج
# -------------------------------
text= " دقة النموذج باستخدام"
reshaped_text = arabic_reshaper.reshape(text)    # correct its shape
bidi_text = get_display(reshaped_text)           # correct its direction
print(f"{bidi_text} Logistic Regression:", acc_lr)
print(f"{bidi_text} SVM:", acc_svm)
print(f"{bidi_text} Random Forest:", acc_rf)
