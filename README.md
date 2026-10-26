# Cervical Cancer Prediction System

## Overview

This project predicts cervical cancer risk using Machine Learning techniques.

The dataset contains demographic, behavioral, and medical screening information from patients.

## Features

- Data Cleaning
- Missing Value Handling
- Feature Importance Analysis
- Correlation Heatmap
- Model Comparison
- Cross Validation
- Streamlit Web Application

## Models Tested

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

## Best Model

XGBoost

### Performance

- Accuracy: 96.51%
- Cross Validation Accuracy: 95.11% ± 1.13%
- Recall: 0.64
- F1 Score: 0.70

## Visualizations

- Class Distribution
- Correlation Heatmap
- Feature Importance
- Model Comparison
- Confusion Matrix
- ROC Curve
- Precision Recall Curve

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start application:

```bash
streamlit run app.py
```

## Project Structure

```text
Data/
Results/
app.py
xgboost_model.pkl
requirements.txt
README.md
```
