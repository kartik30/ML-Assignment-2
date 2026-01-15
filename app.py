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

# --- Custom CSS for styling (optional, but good for colors/fonts) ---
st.markdown("""
<style>
.green-header {
    color: #4CAF50; /* Green color */
    font-size: 3em; /* Larger font size */
    font-weight: bold;
    text-align: center;
    margin-bottom: 0.2em;
}
.blue-subheader {
    color: #2196F3; /* Blue color */
    font-size: 1.5em; /* Medium font size */
    text-align: center;
    margin-bottom: 1em;
}
.created-by {
    font-size: 1em;
    color: #666666;
    text-align: center;
    margin-top: 2em;
    margin-bottom: 2em;
}
</style>
""", unsafe_allow_html=True)

# --- Function to load a specific model ---
@st.cache_resource # Cache the model loading for performance
def load_model(model_name_key):
    model_filename = os.path.join(MODEL_DIR, f"{model_name_key.lower().replace(' ', '_')}_pipeline.pkl")
    # Clean up name for file, e.g., 'naive_bayes_classifier__pipeline.pkl' becomes 'naive_bayes_classifier_pipeline.pkl'
    model_filename = model_filename.replace('__', '_')
    try:
        with open(model_filename, 'rb') as file:
            model_pipeline = pickle.load(file)
        return model_pipeline
    except FileNotFoundError:
        st.error(f"Model file not found for {model_name_key}. Please ensure '{model_filename}' exists in the '{MODEL_DIR}' directory.")
        return None

# --- Streamlit App Layout ---
# --- Streamlit App Layout ---

# 1. BITS Pilani Logo and Text
# You'll need to upload the logo image to your GitHub repo (e.g., in an 'images' folder)
# or use a publicly accessible URL for the image.
# For simplicity, let's assume the logo is at 'images/bits_pilani_logo.png'
# If you don't have the image file, you can comment this out or use a placeholder.

# First, create a column layout for the logo and text side-by-side
col1, col2 = st.columns([0.2, 0.8]) # Adjust ratios as needed

with col1:
    # Ensure you have the logo image in your GitHub repository
    # e.g., in a folder named 'images' in the root of your repo
    st.image("images/bits_pilani_logo.png", width=120) # Adjust path and width as needed

with col2:
    st.write(" ") # Add a little space
    st.markdown("<h3>BITS Pilani</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.8em; color: #555;'>Pilani | Dubai | Goa | Hyderabad | Mumbai</p>", unsafe_allow_html=True)
    st.markdown("<h4>WORK INTEGRATED</h4>", unsafe_allow_html=True)
    st.markdown("<h4>LEARNING PROGRAMMES</h4>", unsafe_allow_html=True)

st.markdown("---") # A horizontal line for separation

# 2. Main Title and Subtitle
st.markdown("<h1 class='green-header'>Machine Learning Assignment 2</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='blue-subheader'>Breast Cancer Wisconsin Dataset (Kaggle/UCI)</h2>", unsafe_allow_html=True)

# 3. Created By
st.markdown("<p class='created-by'>Created by: Karthik Moorthy</p>", unsafe_allow_html=True)

# Add some vertical space
st.write("---")
st.write(" ")
st.write(" ")


st.title("ML Classification Model Demonstrator") # This can be your functional title below the header
st.write("Upload a CSV file to make predictions and view model performance.")

# ... (rest of your app.py code for file uploader, model selection, metrics, etc.) ...

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
            y_true_raw = None # No ground truth for evaluation
        else:
            X_input = input_df.drop(columns=[TARGET_COLUMN], errors='ignore')
            y_true_raw = input_df[TARGET_COLUMN]

            # --- CRITICAL FIX: Convert y_true_raw to numerical labels ---
            # Ensure the mapping matches what was used in train_models.py
            # Map 'M' to 1 (malignant) and 'B' to 0 (benign)
            label_mapping = {'M': 1, 'B': 0}
            y_true = y_true_raw.map(label_mapping).astype(int) # Convert to int
            # Handle any potential NaN values if a label outside 'M'/'B' exists
            if y_true.isnull().any():
                st.warning("Warning: Some target labels in the uploaded file were not 'M' or 'B' and were converted to NaN. These rows might be dropped for evaluation.")
                # You might choose to drop these rows or handle them differently
                X_input = X_input[y_true.notnull()]
                y_true = y_true.dropna()
        
        # Drop 'id' and 'Unnamed: 32' if they exist in the uploaded file
        X_input = X_input.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

        # 2. Model Selection Dropdown
        model_options = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree Classifier": "decision_tree_classifier",
            "K-Nearest Neighbor Classifier": "k-nearest_neighbor_classifier",
            "Naive Bayes Classifier (Gaussian)": "naive_bayes_classifier_(gaussian)",
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

            # Check if y_true is defined and not None (meaning target column was present and converted)
            if y_true is not None and len(y_true) > 0: # Only evaluate if target column was in uploaded data AND it has valid entries
                # 3. Display Evaluation Metrics
                # Ensure metrics are calculated using numerical y_true and y_pred
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='binary', zero_division=0) # Use 'binary' for 0/1 target
                recall = recall_score(y_true, y_pred, average='binary', zero_division=0)      # Use 'binary' for 0/1 target
                f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)              # Use 'binary' for 0/1 target
                mcc = matthews_corrcoef(y_true, y_pred)

                try:
                    y_proba = model_pipeline.predict_proba(X_input)
                    # For binary classification, roc_auc_score expects probabilities for the positive class (1)
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
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['B', 'M'], yticklabels=['B', 'M']) # Add labels for clarity
                plt.xlabel("Predicted")
                plt.ylabel("True")
                st.pyplot(fig) # Display Matplotlib figure in Streamlit

                st.subheader("Classification Report")
                st.text(classification_report(y_true, y_pred, zero_division=0))
            else:
                st.info("No ground truth (target column) or valid labels found in uploaded data, so performance metrics cannot be calculated.")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.info("Please ensure your CSV file is correctly formatted and contains numerical features compatible with the trained models.")
else:
    st.info("Please upload a CSV file to get started.")
