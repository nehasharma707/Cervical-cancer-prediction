import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Data/risk_factors_cervical_cancer.csv")

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Drop columns with >90% missing values
df.drop(
    columns=[
        "STDs: Time since first diagnosis",
        "STDs: Time since last diagnosis"
    ],
    inplace=True
)

print("Dataset Shape After Dropping Columns:")
print(df.shape)

print("\nRemaining Columns:")
print(df.columns.tolist())