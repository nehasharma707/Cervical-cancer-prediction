import matplotlib.pyplot as plt

models = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost"
]

accuracy = [
    95.35,
    95.35,
    95.35,
    96.51
]

plt.figure(figsize=(8,5))

bars = plt.bar(models, accuracy)

plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")

# Zoom into useful range
plt.ylim(94, 97)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig("model_comparison_improved.png")

plt.show()