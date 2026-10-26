import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("Data/risk_factors_cervical_cancer.csv")

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Drop highly missing columns
df.drop(
    columns=[
        "STDs: Time since first diagnosis",
        "STDs: Time since last diagnosis"
    ],
    inplace=True
)

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Features and Target
X = df.drop("Biopsy", axis=1)
y = df["Biopsy"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Random Forest
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))