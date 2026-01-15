
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Machine Learning Model Evaluation App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file (test data)", type=["csv"])

# Model selection dropdown
model_choice = st.selectbox("Select a Model", [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
])

# Load pre-trained model
model_path = f"model/{model_choice.replace(' ', '_').lower()}.pkl"
try:
    model = joblib.load(model_path)
except:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Process uploaded file
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(data.head())

    # Assume last column is target
    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC: {mcc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
else:
    st.info("Please upload a CSV file to evaluate the model.")
