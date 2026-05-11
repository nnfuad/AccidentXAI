# AccidentXAI

## Temporal Explainable AI for Accident Severity Analysis

AccidentXAI is a research-oriented machine learning project focused on:

- Accident severity prediction
- Explainable AI (XAI)
- SHAP-based interpretation
- Temporal feature attribution analysis
- Imbalanced classification handling
- Comparative model evaluation

The project investigates how accident-related factors influence fatality prediction over time using interpretable machine learning techniques.

---

# Objectives

This project aims to:

1. Build robust classification models for accident severity prediction.
2. Handle class imbalance using SMOTE-based techniques.
3. Compare traditional and boosting-based ML models.
4. Analyze feature importance using SHAP.
5. Perform temporal SHAP analysis to observe how influential factors evolve over time.
6. Develop a reproducible research pipeline suitable for academic work.

---

# Models Used

## Baseline Models

- Logistic Regression
- Random Forest

## Boosting Models

- XGBoost
- LightGBM
- CatBoost

---

# Explainability Techniques

- SHAP Summary Plot
- SHAP Dependence Plot
- SHAP Waterfall Plot
- Temporal SHAP Comparison

---

# Evaluation Metrics

- F1-score (Macro)
- F1-score (Weighted)
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

---

# Project Structure

```text
AccidentXAI/
│
├── data/
├── notebooks/
├── src/
├── reports/
├── configs/
├── requirements/
├── tests/
│
├── README.md
├── pyproject.toml
└── main.py
```

---

# Research Motivation

Accident datasets are often:

- highly imbalanced
- noisy
- temporally dynamic
- difficult to interpret

Traditional machine learning models may achieve strong predictive performance while remaining difficult to explain.

This project combines explainable AI and temporal analysis to better understand how accident severity factors evolve over time.

---

# Planned Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Temporal Splitting
   ↓
SMOTE Balancing
   ↓
Model Training
   ↓
Cross Validation
   ↓
Evaluation
   ↓
SHAP Analysis
   ↓
Temporal SHAP Interpretation
```

---

# Future Extensions

- Drift analysis
- SHAP interaction values
- Temporal stability analysis
- Model calibration
- Dashboard deployment

---

# Environment

Designed for:

- MacBook Air M1
- Python 3.11+
- Jupyter Notebook
- Google Colab compatibility

---

# License

MIT License