import pandas as pd
import numpy as np

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

# Convert everything to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values with column mean
df.fillna(df.mean(), inplace=True)

print("Shape:", df.shape)
print("\nMissing Values Remaining:")
print(df.isnull().sum().sum())