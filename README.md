# Machine Learning Assignment 2

## Problem Statement
Build and evaluate multiple classification models on a chosen dataset and deploy an interactive Streamlit app.

## Dataset Description
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Instances: 569
- Features: 30
- Target: Malignant (1) / Benign (0)

## Models Implemented
- Logistic Regression
- Decision Tree
- K-Nearest Neighbor
- Naive Bayes
- Random Forest
- XGBoost

## Evaluation Metrics Comparison
| Model                | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
|----------------------|----------|-------|-----------|--------|----------|-------|
| Logistic Regression  | 0.983    | 0.995 | 0.986     | 0.986  | 0.986    | 0.962 |
| Decision Tree        | 0.912    | 0.916 | 0.956     | 0.903  | 0.929    | 0.817 |
| KNN                  | 0.956    | 0.979 | 0.959     | 0.972  | 0.966    | 0.905 |
| Naive Bayes          | 0.939    | 0.988 | 0.945     | 0.958  | 0.952    | 0.868 |
| Random Forest        | 0.956    | 0.994 | 0.959     | 0.972  | 0.966    | 0.905 |
| XGBoost              | 0.956    | 0.990 | 0.947     | 0.986  | 0.966    | 0.906 |

## Observations
- Logistic Regression performed best overall.
- Decision Tree had the lowest performance.
- Ensemble models and KNN performed very well.
- Naive Bayes was decent but slightly behind others.

## How to Run Locall