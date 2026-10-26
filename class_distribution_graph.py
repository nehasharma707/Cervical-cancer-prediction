import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data/risk_factors_cervical_cancer.csv")

plt.figure(figsize=(6,4))

sns.countplot(x="Biopsy", data=df)

plt.title("Biopsy Class Distribution")
plt.xlabel("Biopsy Result")
plt.ylabel("Count")

plt.savefig("class_distribution.png")

plt.show()