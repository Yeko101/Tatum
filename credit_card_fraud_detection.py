# -*- coding: utf-8 -*-
"""
Comprehensive Credit Card Fraud Detection Analysis
Created on Thu Jun  5 18:47:02 2025
@author: alfie's world
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    recall_score, precision_score, accuracy_score, confusion_matrix, 
    f1_score, roc_auc_score, average_precision_score, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Comprehensive model evaluation function
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if y_proba is not None:
        auc_roc = roc_auc_score(y_true, y_proba)
        auprc = average_precision_score(y_true, y_proba)
    else:
        auc_roc = 'N/A'
        auprc = 'N/A'
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"=== {model_name} Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc}")
    print(f"AUPRC: {auprc}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 50)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'AUPRC': auprc
    }, cm

def create_lstm_model(input_shape):
    """
    Create LSTM model for fraud detection
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def evaluate_anomaly_detector(y_true, y_pred, model_name):
    """
    Evaluate anomaly detection algorithms
    """
    # Convert anomaly predictions to binary (1 for anomaly/fraud, 0 for normal)
    if set(np.unique(y_pred)) == {-1, 1}:
        y_pred_binary = np.where(y_pred == -1, 1, 0)
    elif set(np.unique(y_pred)) == {0, 1}:
        y_pred_binary = y_pred
    else:
        y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Handle case where model predicts only one class
    if len(np.unique(y_pred_binary)) == 1:
        print(f"Warning: {model_name} predicted only one class!")
        if y_pred_binary[0] == 0:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    print(f"=== {model_name} Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 50)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': 'N/A',
        'AUPRC': 'N/A'
    }, cm

def plot_confusion_matrices(cms_dict, title="Confusion Matrices", figsize=(20, 10)):
    """
    Plot multiple confusion matrices
    """
    n_models = len(cms_dict)
    cols = 4
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, y=1.02)
    
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten() if n_models > 1 else axes
    
    for i, (model_name, cm) in enumerate(cms_dict.items()):
        if i < len(axes_flat):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_flat[i])
            axes_flat[i].set_title(model_name)
            axes_flat[i].set_xlabel('Predicted')
            axes_flat[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(results_dict, title="Model Performance Comparison"):
    """
    Plot comprehensive metrics comparison
    """
    df = pd.DataFrame(results_dict).T
    
    # Filter out non-numeric columns for plotting
    numeric_cols = []
    for col in df.columns:
        if col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            numeric_cols.append(col)
        elif col in ['AUC-ROC', 'AUPRC']:
            # Check if column has numeric values
            non_na_values = df[col][df[col] != 'N/A']
            if len(non_na_values) > 0:
                numeric_cols.append(col)
    
    if not numeric_cols:
        print("No numeric metrics to plot")
        return
    
    # Convert to numeric, replacing 'N/A' with NaN
    df_numeric = df[numeric_cols].copy()
    for col in numeric_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    n_metrics = len(numeric_cols)
    cols = 3
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    for i, metric in enumerate(numeric_cols):
        if i < len(axes_flat):
            # Remove NaN values for plotting
            plot_data = df_numeric[metric].dropna()
            if len(plot_data) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
                bars = axes_flat[i].bar(plot_data.index, plot_data.values, color=colors, alpha=0.8)
                axes_flat[i].set_title(f'{metric} Comparison')
                axes_flat[i].set_ylabel(metric)
                axes_flat[i].tick_params(axis='x', rotation=45)
                axes_flat[i].grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, plot_data.values):
                    axes_flat[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                axes_flat[i].text(0.5, 0.5, f'No data for {metric}', 
                                 ha='center', va='center', transform=axes_flat[i].transAxes)
                axes_flat[i].set_title(f'{metric} Comparison')
    
    # Hide unused subplots
    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==================== MAIN EXECUTION ====================

print("Loading dataset...")
df = pd.read_csv(r'C:\Users\YEKO\Documents\creditcard.csv')
print(f"Dataset loaded: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

# Prepare features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Training fraud rate: {y_train.mean():.4f}")

# ==================== PART 1: MODELS WITHOUT SMOTE ====================
print("\n" + "="*60)
print("     PART 1: MODELS WITHOUT SMOTE")
print("="*60)

results_no_smote = {}
cms_no_smote = {}
probabilities_no_smote = {}

# Scale data for neural networks
scaler_no_smote = StandardScaler()
X_train_scaled_no_smote = scaler_no_smote.fit_transform(X_train)
X_test_scaled_no_smote = scaler_no_smote.transform(X_test)

# 1. Logistic Regression (No SMOTE)
print("\nTraining Logistic Regression (No SMOTE)...")
lr_model_no_smote = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model_no_smote.fit(X_train, y_train)
y_pred_lr_no_smote = lr_model_no_smote.predict(X_test)
y_proba_lr_no_smote = lr_model_no_smote.predict_proba(X_test)[:, 1]
results_no_smote['Logistic Regression'], cms_no_smote['Logistic Regression'] = \
    evaluate_model(y_test, y_pred_lr_no_smote, y_proba_lr_no_smote, "Logistic Regression (No SMOTE)")
probabilities_no_smote['Logistic Regression'] = y_proba_lr_no_smote

# 2. XGBoost (No SMOTE)
print("\nTraining XGBoost (No SMOTE)...")
xgb_model_no_smote = xgb.XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
xgb_model_no_smote.fit(X_train, y_train)
y_pred_xgb_no_smote = xgb_model_no_smote.predict(X_test)
y_proba_xgb_no_smote = xgb_model_no_smote.predict_proba(X_test)[:, 1]
results_no_smote['XGBoost'], cms_no_smote['XGBoost'] = \
    evaluate_model(y_test, y_pred_xgb_no_smote, y_proba_xgb_no_smote, "XGBoost (No SMOTE)")
probabilities_no_smote['XGBoost'] = y_proba_xgb_no_smote

# 3. MLP Classifier (No SMOTE)
print("\nTraining MLP Classifier (No SMOTE)...")
mlp_model_no_smote = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp_model_no_smote.fit(X_train_scaled_no_smote, y_train)
y_pred_mlp_no_smote = mlp_model_no_smote.predict(X_test_scaled_no_smote)
y_proba_mlp_no_smote = mlp_model_no_smote.predict_proba(X_test_scaled_no_smote)[:, 1]
results_no_smote['MLP Classifier'], cms_no_smote['MLP Classifier'] = \
    evaluate_model(y_test, y_pred_mlp_no_smote, y_proba_mlp_no_smote, "MLP Classifier (No SMOTE)")
probabilities_no_smote['MLP Classifier'] = y_proba_mlp_no_smote

# 4. Decision Tree (No SMOTE)
print("\nTraining Decision Tree (No SMOTE)...")
dt_model_no_smote = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20, class_weight='balanced')
dt_model_no_smote.fit(X_train, y_train)
y_pred_dt_no_smote = dt_model_no_smote.predict(X_test)
y_proba_dt_no_smote = dt_model_no_smote.predict_proba(X_test)[:, 1]
results_no_smote['Decision Tree'], cms_no_smote['Decision Tree'] = \
    evaluate_model(y_test, y_pred_dt_no_smote, y_proba_dt_no_smote, "Decision Tree (No SMOTE)")
probabilities_no_smote['Decision Tree'] = y_proba_dt_no_smote

# ==================== PART 2: MODELS WITH SMOTE ====================
print("\n" + "="*60)
print("     PART 2: MODELS WITH SMOTE")
print("="*60)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_res.shape}, Fraud rate: {y_train_res.mean():.4f}")

results_smote = {}
cms_smote = {}
probabilities_smote = {}

# Scale data for neural networks
scaler_smote = StandardScaler()
X_train_scaled_smote = scaler_smote.fit_transform(X_train_res)
X_test_scaled_smote = scaler_smote.transform(X_test)

# 1. Logistic Regression (SMOTE)
print("\nTraining Logistic Regression (SMOTE)...")
lr_model_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_model_smote.fit(X_train_res, y_train_res)
y_pred_lr_smote = lr_model_smote.predict(X_test)
y_proba_lr_smote = lr_model_smote.predict_proba(X_test)[:, 1]
results_smote['Logistic Regression'], cms_smote['Logistic Regression'] = \
    evaluate_model(y_test, y_pred_lr_smote, y_proba_lr_smote, "Logistic Regression (SMOTE)")
probabilities_smote['Logistic Regression'] = y_proba_lr_smote

# 2. XGBoost (SMOTE)
print("\nTraining XGBoost (SMOTE)...")
xgb_model_smote = xgb.XGBClassifier(random_state=42)
xgb_model_smote.fit(X_train_res, y_train_res)
y_pred_xgb_smote = xgb_model_smote.predict(X_test)
y_proba_xgb_smote = xgb_model_smote.predict_proba(X_test)[:, 1]
results_smote['XGBoost'], cms_smote['XGBoost'] = \
    evaluate_model(y_test, y_pred_xgb_smote, y_proba_xgb_smote, "XGBoost (SMOTE)")
probabilities_smote['XGBoost'] = y_proba_xgb_smote

# 3. MLP Classifier (SMOTE)
print("\nTraining MLP Classifier (SMOTE)...")
mlp_model_smote = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp_model_smote.fit(X_train_scaled_smote, y_train_res)
y_pred_mlp_smote = mlp_model_smote.predict(X_test_scaled_smote)
y_proba_mlp_smote = mlp_model_smote.predict_proba(X_test_scaled_smote)[:, 1]
results_smote['MLP Classifier'], cms_smote['MLP Classifier'] = \
    evaluate_model(y_test, y_pred_mlp_smote, y_proba_mlp_smote, "MLP Classifier (SMOTE)")
probabilities_smote['MLP Classifier'] = y_proba_mlp_smote

# 4. Decision Tree (SMOTE)
print("\nTraining Decision Tree (SMOTE)...")
dt_model_smote = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20)
dt_model_smote.fit(X_train_res, y_train_res)
y_pred_dt_smote = dt_model_smote.predict(X_test)
y_proba_dt_smote = dt_model_smote.predict_proba(X_test)[:, 1]
results_smote['Decision Tree'], cms_smote['Decision Tree'] = \
    evaluate_model(y_test, y_pred_dt_smote, y_proba_dt_smote, "Decision Tree (SMOTE)")
probabilities_smote['Decision Tree'] = y_proba_dt_smote

# ==================== PART 3: COMPARISON ====================
print("\n" + "="*60)
print("     PART 3: COMPARISON OF MODELS WITH AND WITHOUT SMOTE")
print("="*60)

# Create comparison DataFrames
df_no_smote = pd.DataFrame(results_no_smote).T
df_smote = pd.DataFrame(results_smote).T
df_no_smote['Type'] = 'Without SMOTE'
df_smote['Type'] = 'With SMOTE'

print("\nPerformance WITHOUT SMOTE:")
print(df_no_smote.drop('Type', axis=1).round(4))

print("\nPerformance WITH SMOTE:")
print(df_smote.drop('Type', axis=1).round(4))

# Combined comparison
comparison_df = pd.concat([df_no_smote, df_smote])
print("\nCOMBINED COMPARISON:")
print(comparison_df.round(4))

# Find best models
best_no_smote = df_no_smote.loc[df_no_smote['F1-Score'].idxmax()]
best_smote = df_smote.loc[df_smote['F1-Score'].idxmax()]

print(f"\n🏆 Best model WITHOUT SMOTE: {best_no_smote.name} (F1-Score: {best_no_smote['F1-Score']:.4f})")
print(f"🏆 Best model WITH SMOTE: {best_smote.name} (F1-Score: {best_smote['F1-Score']:.4f})")

# ==================== PART 4: VISUALIZATIONS ====================
print("\n" + "="*60)
print("     PART 4: VISUALIZATIONS")
print("="*60)

# Plot confusion matrices
plot_confusion_matrices(cms_no_smote, "Confusion Matrices - Models WITHOUT SMOTE")
plot_confusion_matrices(cms_smote, "Confusion Matrices - Models WITH SMOTE")

# Plot metrics comparison
plot_metrics_comparison(results_no_smote, "Model Performance - WITHOUT SMOTE")
plot_metrics_comparison(results_smote, "Model Performance - WITH SMOTE")

# Combined comparison plot
comparison_df_reset = comparison_df.reset_index().rename(columns={'index': 'Model'})
metrics_to_plot = ['F1-Score', 'Recall', 'Precision', 'Accuracy']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('SMOTE vs No SMOTE Comparison', fontsize=16)
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    sns.barplot(x='Model', y=metric, hue='Type', data=comparison_df_reset, ax=axes[i])
    axes[i].set_title(f'{metric} Comparison')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ROC Curves
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.title('ROC Curves - Without SMOTE')
for name, proba in probabilities_no_smote.items():
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=plt.gca())
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.title('ROC Curves - With SMOTE')
for name, proba in probabilities_smote.items():
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=plt.gca())
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)

plt.tight_layout()
plt.show()

# Precision-Recall Curves
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.title('Precision-Recall Curves - Without SMOTE')
for name, proba in probabilities_no_smote.items():
    PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=plt.gca())

plt.subplot(1, 2, 2)
plt.title('Precision-Recall Curves - With SMOTE')
for name, proba in probabilities_smote.items():
    PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=plt.gca())

plt.tight_layout()
plt.show()

# ==================== PART 5: ANOMALY DETECTION ====================
print("\n" + "="*60)
print("     PART 5: ANOMALY DETECTION ALGORITHMS")
print("="*60)

anomaly_results = {}
anomaly_cms = {}

# Prepare data for anomaly detection
scaler_anomaly = StandardScaler()
X_scaled = scaler_anomaly.fit_transform(X)
contamination_rate = y.mean()

print(f"Contamination rate: {contamination_rate:.4f}")

# 1. Isolation Forest
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=100)
y_pred_iso = iso_forest.fit_predict(X_scaled)
anomaly_results['Isolation Forest'], anomaly_cms['Isolation Forest'] = \
    evaluate_anomaly_detector(y, y_pred_iso, "Isolation Forest")

# 2. Local Outlier Factor
print("\nTraining Local Outlier Factor...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination_rate)
y_pred_lof = lof.fit_predict(X_scaled)
anomaly_results['Local Outlier Factor'], anomaly_cms['Local Outlier Factor'] = \
    evaluate_anomaly_detector(y, y_pred_lof, "Local Outlier Factor")

# 3. One-Class SVM
print("\nTraining One-Class SVM...")
# Sample data for faster training (One-Class SVM can be slow on large datasets)
sample_size = min(10000, len(X_scaled))
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]
y_sample = y.iloc[sample_indices]

oc_svm = OneClassSVM(nu=contamination_rate, kernel='rbf', gamma='scale')
oc_svm.fit(X_sample)
y_pred_ocsvm = oc_svm.predict(X_sample)
anomaly_results['One-Class SVM'], anomaly_cms['One-Class SVM'] = \
    evaluate_anomaly_detector(y_sample, y_pred_ocsvm, "One-Class SVM")

# 4. LSTM Autoencoder
print("\nTraining LSTM...")
# Prepare data for LSTM (reshape for time series)
sequence_length = 10
X_train_lstm = X_train_scaled_smote[:-(len(X_train_scaled_smote) % sequence_length)]
X_train_lstm = X_train_lstm.reshape(-1, sequence_length, X_train_scaled_smote.shape[1])
y_train_lstm = y_train_res[:len(X_train_lstm)]

X_test_lstm = X_test_scaled_smote[:-(len(X_test_scaled_smote) % sequence_length)]
X_test_lstm = X_test_lstm.reshape(-1, sequence_length, X_test_scaled_smote.shape[1])
y_test_lstm = y_test[:len(X_test_lstm)]

# Create and train LSTM model
lstm_model = create_lstm_model((sequence_length, X_train_scaled_smote.shape[1]))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Make predictions
y_proba_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
y_pred_lstm = (y_proba_lstm > 0.5).astype(int)

anomaly_results['LSTM'], anomaly_cms['LSTM'] = \
    evaluate_model(y_test_lstm, y_pred_lstm, y_proba_lstm, "LSTM")

# ==================== PART 6: ANOMALY DETECTION RESULTS ====================
print("\n" + "="*60)
print("     PART 6: ANOMALY DETECTION RESULTS")
print("="*60)

# Display results
df_anomaly = pd.DataFrame(anomaly_results).T
print("\nAnomaly Detection Performance:")
print(df_anomaly.round(4))

# Find best anomaly detection method
if len(df_anomaly) > 0:
    best_anomaly = df_anomaly['F1-Score'].idxmax()
    print(f"\n🏆 Best anomaly detection method: {best_anomaly} (F1-Score: {df_anomaly.loc[best_anomaly, 'F1-Score']:.4f})")

# Plot anomaly detection results
plot_confusion_matrices(anomaly_cms, "Confusion Matrices - Anomaly Detection")
plot_metrics_comparison(anomaly_results, "Anomaly Detection Performance")

# ==================== PART 7: FINAL SUMMARY ====================
print("\n" + "="*60)
print("     FINAL SUMMARY")
print("="*60)

print("\nSUPERVISED LEARNING SUMMARY:")
print("="*40)
print(f"Best WITHOUT SMOTE: {best_no_smote.name} (F1: {best_no_smote['F1-Score']:.4f})")
print(f"Best WITH SMOTE: {best_smote.name} (F1: {best_smote['F1-Score']:.4f})")

if len(df_anomaly) > 0:
    print(f"\nANOMALY DETECTION SUMMARY:")
    print("="*40)
    print(f"Best method: {best_anomaly} (F1: {df_anomaly.loc[best_anomaly, 'F1-Score']:.4f})")

print(f"\nKEY INSIGHTS:")
print("="*40)
print("1. SMOTE Impact:")
if best_smote['F1-Score'] > best_no_smote['F1-Score']:
    print("   ✓ SMOTE improved overall performance")
else:
    print("   ✗ SMOTE did not improve overall performance")

print("2. Model Rankings (by F1-Score):")
print("   Without SMOTE:")
for i, (model, score) in enumerate(df_no_smote['F1-Score'].sort_values(ascending=False).items(), 1):
    print(f"   {i}. {model}: {score:.4f}")

print("   With SMOTE:")
for i, (model, score) in enumerate(df_smote['F1-Score'].sort_values(ascending=False).items(), 1):
    print(f"   {i}. {model}: {score:.4f}")

if len(df_anomaly) > 0:
    print("   Anomaly Detection:")
    for i, (model, score) in enumerate(df_anomaly['F1-Score'].sort_values(ascending=False).items(), 1):
        print(f"   {i}. {model}: {score:.4f}")

print("\n3. Business Recommendations:")
print("   - Consider precision-recall trade-offs based on business costs")
print("   - Fraud detection: Higher recall prioritizes catching all fraud")
print("   - Customer experience: Higher precision reduces false alarms")
print("   - Consider ensemble methods combining multiple approaches")

# Feature importance analysis
print(f"\n4. Feature Importance (XGBoost with SMOTE):")
if hasattr(xgb_model_smote, 'feature_importances_'):
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model_smote.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("   Top 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")

# Additional visualizations
print(f"\nGenerating additional performance visualizations...")

# Create a comprehensive summary plot
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Comprehensive Fraud Detection Analysis Summary', fontsize=16, y=1.02)

# Plot 1: F1-Score comparison across all methods
all_f1_scores = {}
for model, metrics in results_no_smote.items():
    all_f1_scores[f"{model} (No SMOTE)"] = metrics['F1-Score']
for model, metrics in results_smote.items():
    all_f1_scores[f"{model} (SMOTE)"] = metrics['F1-Score']
for model, metrics in anomaly_results.items():
    all_f1_scores[f"{model} (Anomaly)"] = metrics['F1-Score']

ax1 = axes[0, 0]
models = list(all_f1_scores.keys())
scores = list(all_f1_scores.values())
colors = ['lightblue' if 'No SMOTE' in m else 'lightcoral' if 'SMOTE' in m else 'lightgreen' for m in models]
bars = ax1.bar(range(len(models)), scores, color=colors, alpha=0.8)
ax1.set_title('F1-Score Comparison - All Methods')
ax1.set_ylabel('F1-Score')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, score in zip(bars, scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 2: Precision vs Recall scatter
ax2 = axes[0, 1]
for category, results, color, marker in [
    ('No SMOTE', results_no_smote, 'blue', 'o'),
    ('SMOTE', results_smote, 'red', 's'),
    ('Anomaly', anomaly_results, 'green', '^')
]:
    precisions = [metrics['Precision'] for metrics in results.values()]
    recalls = [metrics['Recall'] for metrics in results.values()]
    models = list(results.keys())
    
    scatter = ax2.scatter(recalls, precisions, c=color, marker=marker, s=100, alpha=0.7, label=category)
    
    for i, model in enumerate(models):
        ax2.annotate(f"{model[:3]}", (recalls[i], precisions[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision vs Recall Trade-off')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: ROC AUC comparison (only for models with probabilities)
ax3 = axes[0, 2]
auc_scores = {}
for model, metrics in results_no_smote.items():
    if metrics['AUC-ROC'] != 'N/A':
        auc_scores[f"{model} (No SMOTE)"] = metrics['AUC-ROC']
for model, metrics in results_smote.items():
    if metrics['AUC-ROC'] != 'N/A':
        auc_scores[f"{model} (SMOTE)"] = metrics['AUC-ROC']
if 'LSTM' in anomaly_results and anomaly_results['LSTM']['AUC-ROC'] != 'N/A':
    auc_scores['LSTM (Anomaly)'] = anomaly_results['LSTM']['AUC-ROC']

if auc_scores:
    models_auc = list(auc_scores.keys())
    scores_auc = list(auc_scores.values())
    colors_auc = ['lightblue' if 'No SMOTE' in m else 'lightcoral' if 'SMOTE' in m else 'lightgreen' for m in models_auc]
    
    bars = ax3.bar(range(len(models_auc)), scores_auc, color=colors_auc, alpha=0.8)
    ax3.set_title('AUC-ROC Comparison')
    ax3.set_ylabel('AUC-ROC')
    ax3.set_xticks(range(len(models_auc)))
    ax3.set_xticklabels(models_auc, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, scores_auc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=8)
else:
    ax3.text(0.5, 0.5, 'No AUC-ROC data available', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('AUC-ROC Comparison')

# Plot 4: Model accuracy comparison
ax4 = axes[1, 0]
all_accuracies = {}
for model, metrics in results_no_smote.items():
    all_accuracies[f"{model} (No SMOTE)"] = metrics['Accuracy']
for model, metrics in results_smote.items():
    all_accuracies[f"{model} (SMOTE)"] = metrics['Accuracy']
for model, metrics in anomaly_results.items():
    all_accuracies[f"{model} (Anomaly)"] = metrics['Accuracy']

models_acc = list(all_accuracies.keys())
scores_acc = list(all_accuracies.values())
colors_acc = ['lightblue' if 'No SMOTE' in m else 'lightcoral' if 'SMOTE' in m else 'lightgreen' for m in models_acc]

bars = ax4.bar(range(len(models_acc)), scores_acc, color=colors_acc, alpha=0.8)
ax4.set_title('Accuracy Comparison - All Methods')
ax4.set_ylabel('Accuracy')
ax4.set_xticks(range(len(models_acc)))
ax4.set_xticklabels(models_acc, rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)

for bar, score in zip(bars, scores_acc):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{score:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 5: SMOTE Impact Analysis
ax5 = axes[1, 1]
smote_impact = []
model_names = []
for model in results_no_smote.keys():
    if model in results_smote:
        impact = results_smote[model]['F1-Score'] - results_no_smote[model]['F1-Score']
        smote_impact.append(impact)
        model_names.append(model)

colors_impact = ['green' if x > 0 else 'red' for x in smote_impact]
bars = ax5.bar(range(len(model_names)), smote_impact, color=colors_impact, alpha=0.8)
ax5.set_title('SMOTE Impact on F1-Score')
ax5.set_ylabel('F1-Score Change')
ax5.set_xticks(range(len(model_names)))
ax5.set_xticklabels(model_names, rotation=45, ha='right')
ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax5.grid(axis='y', alpha=0.3)

for bar, impact in zip(bars, smote_impact):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.005 if impact > 0 else -0.015),
             f'{impact:.3f}', ha='center', va='bottom' if impact > 0 else 'top', fontsize=8)

# Plot 6: Feature importance (if available)
ax6 = axes[1, 2]
if hasattr(xgb_model_smote, 'feature_importances_'):
    top_features = importance_df.head(10)
    bars = ax6.barh(range(len(top_features)), top_features['Importance'], alpha=0.8)
    ax6.set_title('Top 10 Feature Importance (XGBoost)')
    ax6.set_xlabel('Importance')
    ax6.set_yticks(range(len(top_features)))
    ax6.set_yticklabels(top_features['Feature'])
    ax6.grid(axis='x', alpha=0.3)
else:
    ax6.text(0.5, 0.5, 'Feature importance not available', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Feature Importance')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("     ANALYSIS COMPLETE")
print("="*60)
print("All models have been trained and evaluated successfully!")
print("Check the generated plots for comprehensive performance comparison.")
print("="*60)
