# Heart Disease Detection Using Machine Learning

A machine learning project comparing Random Forest, Support Vector Machine, and Naive Bayes classifiers for detecting heart disease using clinical patient data.

---

## Overview

This project applies and evaluates three supervised classification models on the Cleveland Heart Disease dataset to predict the presence or absence of heart disease in patients. Models are assessed using accuracy, recall, ROC-AUC score, confusion matrices, and cross-validation.

---

## Repository Structure

```
Heart/
├── Heart.ipynb
├── heart.csv
└── README.md
```

---

## Dataset

- **Source:** [Heart Disease Cleveland UCI — Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci/data)
- **Total records:** 1,025 patients
- **Features:** 13 clinical attributes
- **Target:** Presence of heart disease (0 = No, 1 = Yes)

### Feature Description

| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Gender (0 = Female, 1 = Male) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (0/1) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (0/1) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment (0–2) |
| ca | Number of major vessels colored by fluoroscopy (0–4) |
| thal | Thalassemia status (0–3) |

---

## Methodology

1. **Exploratory Data Analysis** — Distribution plots, count plots, scatter plots, and correlation heatmap across all features
2. **Preprocessing** — No missing values detected; outliers removed using the IQR method on `trestbps`, `chol`, and `thalach`
3. **Feature Scaling** — StandardScaler applied to training and test sets
4. **Train/Test Split** — 80% training, 20% testing (random state = 9)
5. **Model Training** — Three classifiers trained independently with hyperparameter tuning via GridSearchCV
6. **Evaluation** — Accuracy, classification report, confusion matrix, ROC-AUC score, and Repeated K-Fold cross-validation (5 splits, 10 repeats)

---

## Models & Results

### Random Forest Classifier

- Tuned parameters: `n_estimators`, `max_depth`
- **Accuracy:** ~98%
- **Average Cross-Validation Recall:** 0.9818
- **ROC-AUC:** 1.00 (perfect discrimination on test set)

### Support Vector Machine

- Tuned parameters: `C=100`, `kernel='poly'`, `degree=5`, `coef0=0.1`
- Accuracy improved from 88% (default) to **98%** after hyperparameter tuning
- **Average Cross-Validation Recall:** 0.9790

### Naive Bayes (Gaussian)

- Tuned parameter: `var_smoothing=0.811`
- **Average Cross-Validation Recall:** 0.8333
- **ROC-AUC:** 0.90

### Model Comparison Summary

| Model | Accuracy | Avg. CV Recall | ROC-AUC |
|-------|----------|----------------|---------|
| Random Forest | ~98% | 0.9818 | 1.00 |
| SVM (tuned) | ~98% | 0.9790 | — |
| Naive Bayes (tuned) | — | 0.8333 | 0.90 |

> Random Forest achieved the highest recall score, making it the best model for this task where minimizing false negatives is critical.

---

## Key Findings

- Maximum heart rate (`thalach`) and ST depression (`oldpeak`) are the strongest predictors of heart disease.
- Asymptomatic chest pain (type 3) shows the highest correlation with disease presence.
- Disease cases tend to cluster at lower maximum heart rates and higher ST depression values.
- Random Forest and SVM both performed strongly after tuning; Naive Bayes lagged behind in recall.

---

## How to Run

1. Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn numpy
```

2. Place `heart.csv` in the same directory as the notebook.

3. Open and run the notebook:
```bash
jupyter notebook Heart.ipynb
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| pandas | Data loading and manipulation |
| matplotlib / seaborn | Data visualization |
| scikit-learn | Model training, evaluation, and tuning |
| numpy | Numerical operations |
