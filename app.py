"""
================================================================================
BREAST CANCER CLASSIFICATION - STREAMLIT WEB APPLICATION
BITS Pilani M.Tech (AIML/DSE) - Machine Learning Assignment 2

Author: Karthik Moorthy
Date: 27/1/2026
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
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
def load_model_and_scaler(model_name):
    """Load trained model and scaler from disk"""
    try:
        model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
        scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: Model or scaler file not found!")
        st.error(f"Details: {e}")
        st.info("💡 Please ensure you have run train_model.ipynb first to generate model files.")
        return None, None
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

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

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics

# ==========================================
# MAIN APP
# ==========================================

def main():
    
    # Header
    st.markdown('<p class="main-header">🎗️ Breast Cancer Classification Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Early Detection Saves Lives - AI-Powered Diagnostic Support Tool</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("Configure your analysis settings below:")
    
    # Model Selection
    st.sidebar.subheader("🤖 Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose Machine Learning Model:",
        list(MODEL_FILES.keys()),
        index=5,  # Default to XGBoost (best model)
        help="Select a model to use for predictions"
    )
    
    # File Upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Upload Test Data")
    st.sidebar.info("Upload a CSV file with the same features as the training data. Due to Streamlit free tier limitations, keep file size < 200MB.")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="CSV file containing test samples with features"
    )
    
    # Additional Info
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        """
        **BITS Pilani M.Tech (AIML/DSE)**
        
        Machine Learning Assignment 2
        
        Dataset: Breast Cancer Wisconsin (Diagnostic)
        
        Models: 6 Classification Algorithms
        
        Metrics: Accuracy, AUC, Precision, Recall, F1, MCC
        """
    )
    
    # ==========================================
    # MAIN CONTENT
    # ==========================================
    
    # Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "🔮 Predictions", "📈 Visualizations", "📖 Documentation"])
    
    # ==========================================
    # TAB 1: MODEL COMPARISON
    # ==========================================
    
    with tab1:
        st.header("📊 Model Performance Comparison")
        
        # Load results
        results_df = load_results()
        
        if results_df is not None:
            
            # Display full comparison table
            st.subheader("📋 Complete Results Table")
            
            # Style the dataframe
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
            
            selected_metrics = results_df[results_df['ML Model Name'] == selected_model].iloc[0]
            
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
                        X_test = test_data.drop(columns=[target_col])
                        y_test = test_data[target_col]
                        
                        # Encode if necessary (M/B -> 1/0)
                        if y_test.dtype == 'object':
                            y_test = y_test.map({'M': 1, 'B': 0})
                        
                        has_labels = True
                        st.info(f"ℹ️ Target column '{target_col}' detected. Will evaluate predictions.")
                    else:
                        X_test = test_data.copy()
                        has_labels = False
                        st.info("ℹ️ No target column found. Predictions only (no evaluation).")
                    
                    # Remove 'id' column if exists
                    if 'id' in X_test.columns:
                        X_test = X_test.drop(columns=['id'])
                    
                    # Scale features
                    try:
                        X_test_scaled = scaler.transform(X_test)
                    except Exception as e:
                        st.error(f"❌ Error scaling features: {e}")
                        st.error("Ensure uploaded data has the same features as training data.")
                        return
                    
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
                    if has_labels:
                        st.markdown("---")
                        st.subheader("✅ Model Evaluation on Uploaded Data")
                        
                        # Calculate metrics
                        if predictions_proba is not None:
                            y_proba = predictions_proba[:, 1]
                        else:
                            y_proba = None
                        
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
                            
                            st.write(f"- **Sensitivity (True Positive Rate):** {sensitivity:.4f}")
                            st.write(f"- **Specificity (True Negative Rate):** {specificity:.4f}")
                            st.write(f"- **False Negative Rate:** {cm[1,0]/(cm[1,0]+cm[1,1]):.4f}")
                            st.write(f"- **False Positive Rate:** {cm[0,1]/(cm[0,1]+cm[0,0]):.4f}")
                        
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
Required features: 30 numeric features from breast cancer dataset

Features include:
- radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, ...
- radius_se, texture_se, perimeter_se, area_se, smoothness_se, ...
- radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, ...

Optional columns:
- 'diagnosis' (M/B) - for evaluation
- 'id' - will be ignored

Example CSV structure:
radius_mean,texture_mean,perimeter_mean,...,diagnosis
17.99,10.38,122.8,...,M
20.57,17.77,132.9,...,M
13.54,14.36,87.46,...,B
            """, language='text')
            
            # Sample data download
            st.markdown("### 📥 Download Sample Data")
            st.info("You can download a sample CSV file to test the application.")
    
    # ==========================================
    # TAB 3: VISUALIZATIONS
    # ==========================================
    
    with tab3:
        st.header("📈 Performance Visualizations")
        
        results_df = load_results()
        
        if results_df is not None:
            
            # Metrics comparison chart
            st.subheader("📊 Metrics Comparison Across All Models")
            
            metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(results_df))
            width = 0.12
            
            for i, metric in enumerate(metrics_to_plot):
                offset = (i - len(metrics_to_plot)/2) * width
                ax.bar(x + offset, results_df[metric], width, label=metric)
            
            ax.set_xlabel('Models', fontweight='bold', fontsize=12)
            ax.set_ylabel('Score', fontweight='bold', fontsize=12)
            ax.set_title('Model Performance Comparison - All Metrics', fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(results_df['ML Model Name'], rotation=45, ha='right')
            ax.legend(loc='lower right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.1])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Individual metric charts
            st.markdown("---")
            st.subheader("📈 Individual Metric Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metric_choice = st.selectbox(
                    "Select Metric to Visualize:",
                    metrics_to_plot,
                    index=0
                )
            
            # Create individual metric chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.barh(results_df['ML Model Name'], results_df[metric_choice])
            
            # Color the best performer
            best_idx = results_df[metric_choice].idxmax()
            bars[best_idx].set_color('#2ecc71')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
            
            ax.set_xlabel(metric_choice, fontweight='bold', fontsize=12)
            ax.set_ylabel('Models', fontweight='bold', fontsize=12)
            ax.set_title(f'{metric_choice} Comparison', fontweight='bold', fontsize=14)
            ax.set_xlim([0, 1.1])
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, results_df[metric_choice])):
                ax.text(value, bar.get_y() + bar.get_height()/2, 
                       f' {value:.4f}', 
                       va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Load images if they exist
            st.markdown("---")
            st.subheader("📸 Additional Visualizations")
            
            image_files = {
                "Class Distribution": "results/class_distribution.png",
                "Model Comparison": "results/model_comparison.png",
                "Confusion Matrices": "results/confusion_matrices.png",
                "ROC Curves": "results/roc_curves.png",
                "Feature Importance": "results/feature_importance.png"
            }
            
            available_images = {name: path for name, path in image_files.items() if os.path.exists(path)}
            
            if available_images:
                selected_image = st.selectbox(
                    "Select Visualization:",
                    list(available_images.keys())
                )
                
                st.image(available_images[selected_image], use_column_width=True)
            else:
                st.info("ℹ️ No additional visualizations found. Run the training notebook to generate them.")
        
        else:
            st.warning("⚠️ No results data available for visualization.")
    
    # ==========================================
    # TAB 4: DOCUMENTATION
    # ==========================================
    
    with tab4:
        st.header("📖 Application Documentation")
        
        st.markdown("""
        ## 🎯 Purpose
        
        This application provides an interactive interface for breast cancer classification using 
        machine learning models trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**.
        
        ## 📊 Dataset Information
        
        - **Source**: UCI Machine Learning Repository
        - **Samples**: 569 instances
        - **Features**: 30 numeric features computed from cell nuclei characteristics
        - **Classes**: 
          - **Benign (0)**: Non-cancerous
          - **Malignant (1)**: Cancerous
        
        ## 🤖 Models Implemented
        
        Six classification algorithms are available:
        
        1. **Logistic Regression** - Linear classification model
        2. **Decision Tree** - Tree-based model with interpretable rules
        3. **K-Nearest Neighbor** - Instance-based learning algorithm
        4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
        5. **Random Forest** - Ensemble of decision trees
        6. **XGBoost** - Gradient boosting framework (typically best performer)
        
        ## 📈 Evaluation Metrics
        
        Each model is evaluated using six metrics:
        
        - **Accuracy**: Overall correctness of predictions
        - **AUC**: Area Under ROC Curve - discrimination capability
        - **Precision**: Proportion of positive predictions that are correct
        - **Recall**: Proportion of actual positives correctly identified
        - **F1 Score**: Harmonic mean of precision and recall
        - **MCC**: Matthews Correlation Coefficient - balanced measure for imbalanced data
        
        ## 🔧 How to Use
        
        ### Step 1: Select Model
        - Choose a model from the sidebar dropdown
        - XGBoost is recommended for best performance
        
        ### Step 2: Upload Data
        - Prepare a CSV file with the same 30 features as training data
        - Optionally include 'diagnosis' column (M/B) for evaluation
        - Upload via sidebar file uploader
        
        ### Step 3: View Results
        - Navigate through tabs to see:
          - Model comparison metrics
          - Predictions on your data
          - Performance visualizations
          - This documentation
        
        ## 📁 Required Features
        
        Your CSV must include these 30 features:
        
        **Mean features (10):**
        - radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
        - compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean
        
        **Standard error features (10):**
        - radius_se, texture_se, perimeter_se, area_se, smoothness_se
        - compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se
        
        **Worst (largest) features (10):**
        - radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst
        - compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
        
        ## ⚠️ Important Disclaimers
        
        ### Medical Disclaimer
        
        This application is developed for **educational and research purposes only**. It should **NOT** be used as:
        - Primary diagnostic tool
        - Replacement for professional medical judgment
        - Sole basis for treatment decisions
        
        ### Limitations
        
        - Model trained on single institution data (University of Wisconsin)
        - May not generalize to different populations or imaging equipment
        - False negatives (missed cancers) can occur
        - Always confirm with additional medical testing
        
        ## 🎓 Academic Information
        
        **Course**: Machine Learning - Assignment 2  
        **Program**: M.Tech (AIML/DSE)  
        **Institution**: BITS Pilani - Work Integrated Learning  
        **Submission Date**: 15-Feb-2026  
        
        ## 📚 References
        
        1. Street, W.N., Wolberg, W.H., and Mangasarian, O.L. (1993). "Nuclear feature extraction for breast tumor diagnosis."
        2. UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Data Set
        
        ## 🔗 Links
        
        - [GitHub Repository](#) - Source code
        - [Dataset Source](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
        - [BITS Pilani](https://www.bits-pilani.ac.in/)
        
        ## 📞 Contact
        
        For questions or issues with this application:
        - GitHub Issues: [Create an issue](#)
        - Email: [Your Email]
        
        ---
        
        *This application is part of Machine Learning Assignment 2 for BITS Pilani M.Tech program.*
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p><strong>🎓 BITS Pilani M.Tech (AIML/DSE) - Machine Learning Assignment 2</strong></p>
        <p>⚠️ <em>Disclaimer: This tool is for educational purposes only. Not intended for actual medical diagnosis.</em></p>
        <p style='font-size: 0.8rem;'>Developed with ❤️ using Streamlit | Dataset: Breast Cancer Wisconsin (Diagnostic)</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    main()
