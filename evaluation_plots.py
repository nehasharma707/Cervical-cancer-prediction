import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix
)

from xgboost import XGBClassifier

# Load Data
df = pd.read_csv("Data/risk_factors_cervical_cancer.csv")

df.replace("?", np.nan, inplace=True)

df.drop(
    columns=[
        "STDs: Time since first diagnosis",
        "STDs: Time since last diagnosis"
    ],
    inplace=True
)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.fillna(df.mean(), inplace=True)

X = df.drop("Biopsy", axis=1)
y = df["Biopsy"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ==========================
# ROC CURVE
# ==========================

fpr, tpr, _ = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("Results/roc_curve.png")
plt.show()

# ==========================
# PRECISION RECALL CURVE
# ==========================

precision, recall, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(6,5))

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")

plt.savefig("Results/precision_recall_curve.png")
plt.show()

print("ROC and Precision-Recall graphs saved.")