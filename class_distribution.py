import pandas as pd

df = pd.read_csv("Data/risk_factors_cervical_cancer.csv")

print("Biopsy Counts:\n")
print(df["Biopsy"].value_counts())

print("\nPercentage:\n")
print(df["Biopsy"].value_counts(normalize=True) * 100)