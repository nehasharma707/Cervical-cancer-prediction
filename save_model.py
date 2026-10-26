import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier

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

# Convert columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Features and Target
X = df.drop("Biopsy", axis=1)
y = df["Biopsy"]

# Train Final Model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X, y)

# Save Model
joblib.dump(model, "xgboost_model.pkl")

print("Model saved successfully!")