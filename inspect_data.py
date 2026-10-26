import pandas as pd

df = pd.read_csv("Data/risk_factors_cervical_cancer.csv")

print("Dataset loaded successfully\n")

print("Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())