import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

# Load Dataset
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

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X, y)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(10))

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"][:10],
    feature_importance["Importance"][:10]
)

plt.title("Top 10 Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()