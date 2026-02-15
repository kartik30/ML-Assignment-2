# Machine Learning Assignment 2

## Problem Statement
Build and evaluate multiple classification models on a chosen dataset and deploy an interactive Streamlit app.
Breast cancer is one of the most common cancers among women worldwide, accounting for approximately 25% of all cancer cases in women. Early and accurate detection of breast cancer is critical for effective treatment and improved patient outcomes. Traditional diagnostic methods rely heavily on manual examination and interpretation of cell nucleus characteristics from Fine Needle Aspirate (FNA) samples, which can be time-consuming and subject to human error.

This project addresses the need for automated, accurate, and reliable classification of breast tumors as malignant (cancerous) or benign (non-cancerous) using machine learning techniques. By implementing and comparing six different classification algorithms on the Breast Cancer Wisconsin (Diagnostic) Dataset, this project aims to:

   1. Identify the most effective machine learning model for breast cancer diagnosis based on cell nucleus features
   2. Provide a comparative analysis of traditional and ensemble learning methods
   3. Develop an interactive web application for real-time tumor classification
   4. Support medical professionals with an automated decision support tool

The ultimate goal is to create a robust, interpretable, and deployable solution that can assist healthcare professionals in making faster and more accurate diagnostic decisions, potentially saving lives through early detection.
## Dataset Description
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Instances: 569
- Features: 30
- Target: Malignant (1) / Benign (0)

    Problem Type: Binary Classification
    Total Instances: 569 samples
    Total Features: 32 columns
        - 1 ID column (non-informative, excluded from analysis)
        - 30 numeric feature columns
        - 1 target variable (diagnosis)
    Missing Values: None (0%)
    Data Type: Multivariate, Real-valued features
    Area: Life Sciences / Medical

Class Distribution
Class 	Label 	Count 	Percentage
Malignant 	M (1) 	212 	37.3%
Benign 	B (0) 	357 	62.7%

Class Imbalance: Mild imbalance (ratio ≈ 1.68:1), manageable without special techniques

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
# Create new environment
conda create -n ml_assignment python=3.9 -y

# Activate it
conda activate ml_assignment

# Install all required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit
