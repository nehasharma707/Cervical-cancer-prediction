import pandas as pd

results = pd.DataFrame({
    "Model":[
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ],
    "Accuracy":[
        95.35,
        95.35,
        95.35,
        96.51
    ],
    "Recall":[
        0.55,
        0.45,
        0.64,
        0.64
    ],
    "F1":[
        0.60,
        0.56,
        0.64,
        0.70
    ]
})

print(results)

results.to_csv(
    "model_metrics.csv",
    index=False
)