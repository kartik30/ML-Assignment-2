"""
================================================================================
BREAST CANCER CLASSIFICATION - STREAMLIT WEB APPLICATION
BITS Pilani M.Tech (AIML/DSE) - Machine Learning Assignment 2
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS STYLING
# ==========================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e74c3c;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_DIR = 'models'
RESULTS_CSV = 'results/model_results.csv'
TARGET_COLUMN = 'diagnosis'

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'K-Nearest Neighbor': 'k_nearest_neighbor_model.pkl',
    'Naive Bayes': 'naive_bayes_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl'
}

SCALER_FILE = 'scaler.pkl'

# ==========================================
# HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def load_model(model_name):
    """Load trained model from disk"""
    try:
        model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.info("💡 Please run train_model.ipynb first to generate model files.")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load scaler from disk"""
    try:
        scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)
        
        if not os.path.exists(scaler_path):
            st.warning(f"⚠️ Scaler file not found: {scaler_path}")
            return None
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return scaler
    
    except Exception as e:
        st.error(f"❌ Error loading scaler: {e}")
        return None

def load_model_and_scaler(model_name):
    """Load both model and scaler"""
    model = load_model(model_name)
    scaler = load_scaler()
    return model, scaler

@st.cache_data
def load_results():
    """Load model results from CSV"""
    try:
        if os.path.exists(RESULTS_CSV):
            df = pd.read_csv(RESULTS_CSV)
            return df
        else:
            st.warning(f"⚠️ Results file not found: {RESULTS_CSV}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading results: {e}")
        return None

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {}
    
    try:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['F1 Score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        
        if y_proba is not None:
            metrics['AUC'] = roc_auc_score(y_true, y_proba)
        else:
            metrics['AUC'] = 0.0
            
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        # Return default values
        for key in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC', 'AUC']:
            metrics[key] = 0.0
    
    return metrics

def plot_confusion_matrix(cm, model_name):
    """Create confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='RdYlGn_r',
        xticklabels=['Benign (0)', 'Malignant (1)'],
        yticklabels=['Benign (0)', 'Malignant (1)'],
        ax=ax,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def encode_target(y):
    """Encode target variable (M/B to 1/0)"""
    if y.dtype == 'object':
        return y.map({'M': 1, 'B': 0, 'Malignant': 1, 'Benign': 0})
    return y

# ==========================================
# MAIN APP
# ==========================================

def main():
    
    # Header
    st.markdown('<p class="main-header">🎗️ Breast Cancer Classification Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Early Detection Saves Lives - AI-Powered Diagnostic Support Tool</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # ==========================================
    # SIDEBAR CONFIGURATION
    # ==========================================
    
    st.sidebar.header("⚙️ Model Selection")
    st.sidebar.markdown("Select a machine learning model:")
    
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        list(MODEL_FILES.keys()),
        index=5,  # Default to XGBoost
        help="Select a trained model for predictions"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Upload Test Data")
    st.sidebar.info("Upload CSV with same features as training data. File size < 200MB.")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload test data for predictions"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        """
        **BITS Pilani M.Tech (AIML/DSE)**
        
        Machine Learning Assignment 2
        
        Dataset: Breast Cancer Wisconsin
        
        Models: 6 Classification Algorithms
        """
    )
    
    # ==========================================
    # TAB NAVIGATION
    # ==========================================
    
    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🔮 Predictions", "📖 Documentation"])
    
    # ==========================================
    # TAB 1: MODEL COMPARISON
    # ==========================================
    
    with tab1:
        st.header("📊 Model Performance Comparison")
        
        results_df = load_results()
        
        if results_df is not None:
            
            # Display comparison table
            st.subheader("📋 Complete Results Table")
            
            styled_df = results_df.style.highlight_max(
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                color='lightgreen'
            ).highlight_min(
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                color='lightcoral'
            ).format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1': '{:.4f}',
                'MCC': '{:.4f}'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=280)
            
            # Best models by metric
            st.markdown("---")
            st.subheader("🏆 Best Models by Metric")
            
            metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            cols = st.columns(3)
            
            for idx, metric in enumerate(metrics):
                with cols[idx % 3]:
                    best_idx = results_df[metric].idxmax()
                    best_model = results_df.loc[best_idx, 'ML Model Name']
                    best_score = results_df.loc[best_idx, metric]
                    
                    st.metric(
                        label=f"🥇 Best {metric}",
                        value=best_model,
                        delta=f"{best_score:.4f}"
                    )
            
            # Selected model details
            st.markdown("---")
            st.subheader(f"🎯 Selected Model: {selected_model}")
            
            selected_row = results_df[results_df['ML Model Name'] == selected_model]
            
            if not selected_row.empty:
                selected_metrics = selected_row.iloc[0]
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("📈 Accuracy", f"{selected_metrics['Accuracy']:.4f}")
                with col2:
                    st.metric("📊 AUC", f"{selected_metrics['AUC']:.4f}")
                with col3:
                    st.metric("🎯 Precision", f"{selected_metrics['Precision']:.4f}")
                with col4:
                    st.metric("🔍 Recall", f"{selected_metrics['Recall']:.4f}")
                with col5:
                    st.metric("⚖️ F1 Score", f"{selected_metrics['F1']:.4f}")
                with col6:
                    st.metric("📉 MCC", f"{selected_metrics['MCC']:.4f}")
        
        else:
            st.warning("⚠️ Model results not found. Please run the training notebook first.")
    
    # ==========================================
    # TAB 2: PREDICTIONS
    # ==========================================
    
    with tab2:
        st.header("🔮 Make Predictions on Uploaded Data")
        
        if uploaded_file is not None:
            
            try:
                # Load uploaded data
                test_data = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Shape: {test_data.shape}")
                
                # Show data preview
                with st.expander("📋 View Uploaded Data (First 10 rows)"):
                    st.dataframe(test_data.head(10), use_container_width=True)
                
                # Load model and scaler
                model, scaler = load_model_and_scaler(selected_model)
                
                if model is not None and scaler is not None:
                    
                    # Identify features and target
                    target_col = None
                    possible_targets = ['diagnosis', 'target', 'class', 'label', 'output']
                    
                    for col in possible_targets:
                        if col in test_data.columns:
                            target_col = col
                            break
                    
                    # Prepare features
                    if target_col:
                        st.info(f"ℹ️ Target column '{target_col}' detected.")
                        
                        X_test = test_data.drop(columns=[target_col])
                        y_test = test_data[target_col].copy()
                        
                        # Display original target values
                        st.write(f"**Original target values:** {y_test.unique()}")
                        
                        # Encode if necessary (M/B -> 1/0)
                        if y_test.dtype == 'object':
                            # Clean values
                            y_test = y_test.str.strip().str.upper()
                            
                            # Map to 0/1
                            mapping = {'M': 1, 'MALIGNANT': 1, 'B': 0, 'BENIGN': 0}
                            y_test = y_test.map(mapping)
                            
                            # Check for NaN
                            if y_test.isna().any():
                                st.warning(f"⚠️ Warning: {y_test.isna().sum()} target values couldn't be mapped!")
                                st.write("**Unmapped values:**")
                                unmapped = test_data[target_col][y_test.isna()].unique()
                                st.write(unmapped)
                                
                                # Drop NaN rows
                                valid_mask = ~y_test.isna()
                                X_test = X_test[valid_mask]
                                y_test = y_test[valid_mask]
                                
                                st.info(f"✅ Kept {len(y_test)} valid samples, removed {(~valid_mask).sum()} invalid")
                        
                        # Ensure integer type
                        y_test = y_test.astype(int)
                        
                        st.success(f"✅ Target encoded: 0=Benign, 1=Malignant")
                        st.write(f"**Class distribution:** 0: {sum(y_test==0)}, 1: {sum(y_test==1)}")
                        
                        has_labels = True
                        
                    else:
                        X_test = test_data.copy()
                        has_labels = False
                        st.info("ℹ️ No target column found. Predictions only.")
                    
                    # Remove 'id' column if exists
                    if 'id' in X_test.columns:
                        X_test = X_test.drop(columns=['id'])
                    
                    # Check for NaN in features
                    if X_test.isna().any().any():
                        st.warning(f"⚠️ Warning: Features contain {X_test.isna().sum().sum()} NaN values")
                        
                        # Option 1: Drop rows with NaN
                        st.write("**Handling missing values...**")
                        X_test = X_test.dropna()
                        
                        if has_labels:
                            # Align y_test with X_test
                            y_test = y_test.loc[X_test.index]
                        
                        st.info(f"✅ Removed rows with missing values. Remaining: {len(X_test)}")
                    
                    # Scale features
                    try:
                        X_test_scaled = scaler.transform(X_test)
                    except Exception as e:
                        st.error(f"❌ Error scaling features: {e}")
                        st.error("Ensure uploaded data has the same features as training data.")
                        st.stop()
                    
                    # Make predictions
                    predictions = model.predict(X_test_scaled)
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        predictions_proba = model.predict_proba(X_test_scaled)
                    else:
                        predictions_proba = None
                    
                    # Display predictions
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    
                    # Create results dataframe
                    results_data = {
                        'Sample Index': range(len(predictions)),
                        'Prediction': ['Malignant' if p == 1 else 'Benign' for p in predictions],
                        'Prediction Code': predictions
                    }
                    
                    if predictions_proba is not None:
                        results_data['Prob (Benign)'] = predictions_proba[:, 0]
                        results_data['Prob (Malignant)'] = predictions_proba[:, 1]
                        results_data['Confidence'] = np.max(predictions_proba, axis=1)
                    
                    pred_df = pd.DataFrame(results_data)
                    
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📝 Total Samples", len(predictions))
                    with col2:
                        st.metric("🔴 Predicted Malignant", int(sum(predictions == 1)))
                    with col3:
                        st.metric("🟢 Predicted Benign", int(sum(predictions == 0)))
                    
                    # Download predictions
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"predictions_{selected_model.replace(' ', '_')}.csv",
                        mime='text/csv',
                        help="Download prediction results"
                    )
                    
                    # If actual labels exist, evaluate
                    if has_labels and len(y_test) > 0:
                        st.markdown("---")
                        st.subheader("✅ Model Evaluation on Uploaded Data")
                        
                        # Ensure y_test and predictions have same length
                        if len(y_test) != len(predictions):
                            st.error(f"❌ Length mismatch: y_test={len(y_test)}, predictions={len(predictions)}")
                            st.stop()
                        
                        # Check for NaN one more time
                        if pd.isna(y_test).any():
                            st.error(f"❌ y_test still contains NaN values!")
                            st.write(f"NaN count: {pd.isna(y_test).sum()}")
                            st.stop()
                        
                        # Calculate metrics
                        if predictions_proba is not None:
                            y_proba = predictions_proba[:, 1]
                        else:
                            y_proba = None
                        
                        try:
                            metrics = calculate_metrics(y_test, predictions, y_proba)
                            
                            # Display metrics
                            col1, col2, col3, col4, col5, col6 = st.columns(6)
                            
                            with col1:
                                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                            with col2:
                                st.metric("AUC", f"{metrics['AUC']:.4f}")
                            with col3:
                                st.metric("Precision", f"{metrics['Precision']:.4f}")
                            with col4:
                                st.metric("Recall", f"{metrics['Recall']:.4f}")
                            with col5:
                                st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                            with col6:
                                st.metric("MCC", f"{metrics['MCC']:.4f}")
                            
                            # Confusion Matrix
                            st.markdown("---")
                            st.subheader("🔢 Confusion Matrix")
                            
                            cm = confusion_matrix(y_test, predictions)
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # Display confusion matrix plot
                                fig = plot_confusion_matrix(cm, selected_model)
                                st.pyplot(fig)
                            
                            with col2:
                                # Display metrics breakdown
                                st.write("")
                                st.write("")
                                st.write("**Confusion Matrix Breakdown:**")
                                st.write(f"- **True Negatives (TN):** {cm[0,0]}")
                                st.write(f"- **False Positives (FP):** {cm[0,1]}")
                                st.write(f"- **False Negatives (FN):** {cm[1,0]}")
                                st.write(f"- **True Positives (TP):** {cm[1,1]}")
                                
                                st.write("")
                                st.write("**Clinical Interpretation:**")
                                
                                sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
                                specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
                                
                                st.write(f"- **Sensitivity:** {sensitivity:.4f}")
                                st.write(f"- **Specificity:** {specificity:.4f}")
                            
                            # Classification Report
                            st.markdown("---")
                            st.subheader("📝 Detailed Classification Report")
                            
                            report = classification_report(
                                y_test, 
                                predictions, 
                                target_names=['Benign', 'Malignant'],
                                output_dict=True,
                                zero_division=0
                            )
                            
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(
                                report_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', subset=['f1-score']),
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error calculating metrics: {str(e)}")
                            st.exception(e)
                    
            except pd.errors.EmptyDataError:
                st.error("❌ Uploaded file is empty!")
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("💡 Make sure your CSV has the same features as the training data.")
                
                with st.expander("🔍 See error details"):
                    st.exception(e)
        
        else:
            st.info("👈 Please upload a CSV file from the sidebar to make predictions")
            
            st.markdown("### 📌 Sample Data Format")
            
            st.code("""
Required features: 30 numeric features

Features include:
- radius_mean, texture_mean, perimeter_mean, area_mean, ...
- radius_se, texture_se, perimeter_se, area_se, ...
- radius_worst, texture_worst, perimeter_worst, area_worst, ...

Optional column:
- 'diagnosis' (M/B or Malignant/Benign) - for evaluation

Example CSV:
radius_mean,texture_mean,...,diagnosis
17.99,10.38,...,M
13.54,14.36,...,B
            """, language='text')
    
    # ==========================================
    # TAB 3: DOCUMENTATION
    # ==========================================
    
    with tab3:
        st.header("📖 Application Documentation")
        
        st.markdown("""
        ## 🎯 Purpose
        
        Interactive breast cancer classification using ML models trained on the 
        Breast Cancer Wisconsin (Diagnostic) Dataset.
        
        ## 📊 Dataset
        
        - **Source**: UCI Machine Learning Repository
        - **Samples**: 569 instances
        - **Features**: 30 numeric features
        - **Classes**: Benign (0), Malignant (1)
        
        ## 🤖 Models
        
        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbor
        4. Naive Bayes
        5. Random Forest
        6. XGBoost
        
        ## 📈 Metrics
        
        - **Accuracy**: Overall correctness
        - **AUC**: Area Under ROC Curve
        - **Precision**: Positive prediction accuracy
        - **Recall**: True positive rate
        - **F1 Score**: Harmonic mean of precision/recall
        - **MCC**: Matthews Correlation Coefficient
        
        ## ⚠️ Disclaimer
        
        **For educational purposes only. Not for medical diagnosis.**
        
        ## 🎓 Academic Information
        
        - **Course**: Machine Learning Assignment 2
        - **Program**: M.Tech (AIML/DSE)
        - **Institution**: BITS Pilani
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p><strong>🎓 BITS Pilani M.Tech (AIML/DSE) - ML Assignment 2</strong></p>
        <p>⚠️ <em>Educational purposes only. Not for medical diagnosis.</em></p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    main()
