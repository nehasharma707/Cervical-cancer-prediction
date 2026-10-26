import pandas as pd
import numpy as np

df = pd.read_csv("Data/risk_factors_cervical_cancer.csv")

df.replace("?", np.nan, inplace=True)

print("Missing Values Per Column:\n")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())