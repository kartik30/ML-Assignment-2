# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
MODEL_DIR = 'models' # Directory where your saved models are located
TARGET_COLUMN = 'diagnosis' # The target column name from your dataset

# --- Function to load a specific model ---
@st.cache_resource # Cache the model loading for performance
def load_model(model_name_key):
    model_filename = os.path.join(MODEL_DIR, f"{model_name_key.lower().replace(' ', '_')}_pipeline.pkl")
    try:
        with open(model_filename, 'rb') as file:
            model_pipeline = pickle.load(file)
        return model_pipeline
    except FileNotFoundError:
        st.error(f"Model file not found for {model_name_key}. Please ensure '{model_filename}' exists in the '{MODEL_DIR}' directory.")
        return None

# --- Streamlit App Layout ---
st.title("ML Classification Model Demonstrator")
st.write("Upload a CSV file to make predictions and view model performance.")

# 1. Dataset Upload Option
uploaded_file = st.file_uploader("Upload your CSV file (test data)", type=["csv"])
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(input_df.head())

        # Check for target column in uploaded data
        if TARGET_COLUMN not in input_df.columns:
            st.warning(f"Warning: Target column '{TARGET_COLUMN}' not found in the uploaded file. Showing predictions only.")
            X_input = input_df
            y_true = None # No ground truth for evaluation
        else:
            X_input = input_df.drop(columns=[TARGET_COLUMN], errors='ignore')
            y_true = input_df[TARGET_COLUMN]

        # Drop 'id' and 'Unnamed: 32' if they exist in the uploaded file
        X_input = X_input.drop(columns=['id', 'Unnamed: 32'], errors='ignore')


        # 2. Model Selection Dropdown
        model_options = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree Classifier": "decision_tree_classifier",
            "K-Nearest Neighbor Classifier": "k-nearest_neighbor_classifier",
            "Naive Bayes Classifier (Gaussian)": "naive_bayes_classifier_(gaussian)", # Match saved filename
            "Random Forest Classifier": "random_forest_classifier",
            "XGBoost Classifier": "xgboost_classifier"
        }
        selected_model_display = st.selectbox("Select a Model", list(model_options.keys()))
        selected_model_key = model_options[selected_model_display]

        model_pipeline = load_model(selected_model_key)

        if model_pipeline:
            st.subheader(f"Results for: {selected_model_display}")

            # Make predictions
            y_pred = model_pipeline.predict(X_input)
            
            # Display predictions (optional)
            st.write("Predictions (first 10):")
            st.write(y_pred[:10])

            if y_true is not None: # Only evaluate if target column was in uploaded data
                # 3. Display Evaluation Metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_true, y_pred)

                try:
                    y_proba = model_pipeline.predict_proba(X_input)
                    if len(np.unique(y_true)) > 2: # Multi-class AUC
                        auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                    else: # Binary AUC
                        auc = roc_auc_score(y_true, y_proba[:, 1])
                except Exception as e:
                    auc = "N/A"
                    st.warning(f"Could not calculate AUC for this model/data. Error: {e}")

                st.write(f"**Accuracy:** {accuracy:.4f}")
                st.write(f"**Precision:** {precision:.4f}")
                st.write(f"**Recall:** {recall:.4f}")
                st.write(f"**F1 Score:** {f1:.4f}")
                st.write(f"**MCC:** {mcc:.4f}")
                st.write(f"**AUC:** {auc:.4f}" if isinstance(auc, float) else f"**AUC:** {auc}")

                # 4. Confusion Matrix / Classification Report
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                st.pyplot(fig) # Display Matplotlib figure in Streamlit

                st.subheader("Classification Report")
                st.text(classification_report(y_true, y_pred, zero_division=0))
            else:
                st.info("No ground truth (target column) found in uploaded data, so performance metrics cannot be calculated.")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.info("Please ensure your CSV file is correctly formatted and contains numerical features compatible with the trained models.")
else:
    st.info("Please upload a CSV file to get started.")
