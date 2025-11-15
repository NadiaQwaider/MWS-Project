# ==========================================================
# مقارنة: نموذج Logistic Regression باستخدام كل الخصائص
# ==========================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import arabic_reshaper
from bidi.algorithm import get_display
# 1. تحميل البيانات
data = load_breast_cancer()
X = data.data
y = data.target

# 2. تطبيع البيانات (تحسين الأداء)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. تقسيم البيانات (70% تدريب - 30% اختبار)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. تدريب النموذج باستخدام كل الخصائص
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train, y_train)

# 5. التنبؤ وحساب الدقة
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
text= "✅ دقة النموذج باستخدام كل الخصائص:"
reshaped_text = arabic_reshaper.reshape(text)    # correct its shape
bidi_text = get_display(reshaped_text)           # correct its direction

print(f"{bidi_text}", acc)
