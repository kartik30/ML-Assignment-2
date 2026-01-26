import pandas as pd
import streamlit as st
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import os
import warnings
import sys

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
DATASET_PATH = st.file_uploader("Upload your CSV file (test data)", type=["csv"]) # IMPORTANT: Replace with the path to your dataset
TARGET_COLUMN = 'diagnosis'         # IMPORTANT: Replace with the name of your target column
MODEL_DIR = st.selectbox("Select a Model", [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
])
             # Directory to save trained models

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Load and Prepare Data ---
def load_and_preprocess_data(dataset_path, target_column):
    """
    Loads the dataset, identifies feature types, splits data, and creates a preprocessor.
    """
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please update DATASET_PATH to the correct location of your CSV file.")
        sys.exit(1) # <--- Changed from exit() to sys.exit(1)

    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Basic check for minimum requirements
    if df.shape[0] < 500:
        print(f"Warning: Dataset has fewer than 500 instances ({df.shape[0]}).")
    # For feature size, consider dropping non-feature columns first
    # Example of dropping 'id' and 'Unnamed: 32'
    df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore') # 'errors='ignore'' to prevent error if column not found
    if df.shape[1] < 13: # 12 features + 1 target
        print(f"Warning: Dataset has fewer than 12 features ({df.shape[1]-1}).")

    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        print("Please update TARGET_COLUMN to the correct target column name.")
        sys.exit(1) # <--- Changed from exit() to sys.exit(1)

    X = df.drop(columns=[target_column])
    y = df[target_column]


    # Identify categorical and numerical features
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    # Preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

    print(f"\nTraining data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"Numerical features: {numerical_cols}")
    print(f"Categorical features: {categorical_cols}")

    return X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols

# --- 2. Model Definitions and Evaluation ---
def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test, preprocessor):
    """
    Trains a given model within a pipeline, evaluates it, and saves it.
    """
    print(f"\n--- Training {name} ---")

    # Create a full pipeline including preprocessing
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) # Use weighted for multi-class
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)      # Use weighted for multi-class
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)              # Use weighted for multi-class
    mcc = matthews_corrcoef(y_test, y_pred)

    # AUC Score: Handle multi-class vs binary
    try:
        if len(np.unique(y_train)) > 2:
            # For multi-class, use predict_proba and specify multi_class='ovr' or 'ovo'
            y_proba = pipeline.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    except ValueError as e:
        auc = 0.0 # AUC might not be applicable or computable for certain conditions (e.g., single class in test set)
        print(f"Warning: Could not compute AUC for {name}. Error: {e}")


    metrics = {
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc
    }

    print(f"Metrics for {name}:")
    for metric, value in metrics.items():
        print(f"- {metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save the trained pipeline
    model_filename = os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}_pipeline.pkl")
    with open(model_filename, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"Model saved to {model_filename}")

    return metrics

# --- Main execution ---
if __name__ == "__main__":
    print("Starting ML Model Training Script...\n")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = load_and_preprocess_data(DATASET_PATH, TARGET_COLUMN)

    # Define the models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'), # 'liblinear' is good for small datasets
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor Classifier": KNeighborsClassifier(),
        "Naive Bayes Classifier (Gaussian)": GaussianNB(),
        # For MultinomialNB, features should be counts or frequencies.
        # It's less suitable for scaled numerical data. If your dataset has
        # categorical features that become one-hot encoded, it might work,
        # but GaussianNB is generally more robust for preprocessed continuous data.
        # "Naive Bayes Classifier (Multinomial)": MultinomialNB(),
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    all_metrics = {}

    # Train and evaluate each model
    for name, model in models.items():
        # Special handling for Naive Bayes if data types are complex
        if isinstance(model, MultinomialNB):
            # MultinomialNB expects non-negative integer counts.
            # If your preprocessed data has negative values (from StandardScaler),
            # or is not count-based, it won't work well.
            # For simplicity with the generic preprocessor, we will use GaussianNB.
            # If your dataset's features are counts (e.g., text data), you'd need
            # a different preprocessor or specific handling for MultinomialNB.
            # For this generic script, we'll stick to GaussianNB for 'Naive Bayes'.
            print(f"\nSkipping MultinomialNB as it's not suitable for standardized data. Using GaussianNB.")
            # If you specifically want to use MultinomialNB, you'd need to modify
            # the preprocessor (e.g., remove StandardScaler for numerical, or use a different scaling approach)
            # or choose a dataset where features are naturally counts.
            continue


        metrics = train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test, preprocessor)
        all_metrics[name] = metrics

    print("\n--- All Models Trained and Evaluated ---")
    print("\nSummary of Metrics:")
    metrics_df = pd.DataFrame(all_metrics).T
    print(metrics_df.round(4))

    print(f"\nAll trained pipelines saved to the '{MODEL_DIR}' directory.")
    print("You can now use these models in your Streamlit application.")


