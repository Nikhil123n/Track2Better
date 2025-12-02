"""
===============================================================================
Final Project – Multimodal Time-Series Classification and Anomaly Detection Using the AI-READI Dataset

Iterative Relabeling with XGBoost
Aishwarya Sajjan

This script implements an iterative relabeling algorithm using an XGBoost
classifier. The goal is to identify participants misclassified as 'true_healthy'
and relabel them, progressively refining the dataset over multiple iterations.

The workflow includes:
1. Loading summary-level participant metrics
2. Feature extraction and preprocessing
3. Handling class imbalance using SMOTE
4. Training an XGBoost classifier
5. Evaluating the model using confusion matrix & metrics
6. Identifying participants repeatedly classified as 'true_healthy'
7. Relabeling those participants and repeating until convergence

Why this approach?
------------------
• Provides data-driven relabeling for noisy medical datasets  
• Uses XGBoost, which is strong for tabular data  
• Integrates SMOTE for imbalanced labels  
• Includes visualization and interpretable metrics  

Benefit Over Baseline Models:
-----------------------------
Baseline models like logistic regression or a single decision tree often fail to
capture:
• non-linear relationships  
• feature interactions  
• class imbalance  

XGBoost improves on these by:
• boosting → lowers bias & variance  
• capturing complex patterns in glucose feature space  
• providing feature importance → interpretability  
• robust performance on tabular physiological data
===============================================================================
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt

# === Logging setup ===
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# === PATH SETUP ===
PILOT_ROOT_DATA_PATH = os.getenv(
    'PILOT_ROOT_DATA_PATH', 
    "C:/Users/nikhi/Box/AI-READI/nikhil working dataset/dataset/"
)
SUMMARY_PATH = os.path.join(PILOT_ROOT_DATA_PATH, "cleaned_data2", "summarized_metrics_all_participants.csv")

# === CONFIGURATION ===
MAX_ITERATIONS = 10  # Prevent infinite loops
MIN_IMPROVEMENT_THRESHOLD = 0.01  # Minimum improvement to continue iterations

# === FEATURE CONFIG ===
FEATURE_COLS = [
    'mean_blood_glucose', 'std_dev', 'cv_percent', 'tir_percent', 'tar_percent',
    'tbr_percent', 'mage', 'j_index', 'lability_index', 'conga_1h', 'conga_2h', 'age'
]

# === DATA LOADING ===
def load_data():
    """
    Loads the summarized participant metrics dataset.

    Returns:
        df (pd.DataFrame): Summary-level metrics.
    Raises:
        FileNotFoundError: If CSV is missing.
        ValueError: If dataset is empty.
    """
    
    try:
        logging.info("Loading summarized metrics CSV...")
        if not os.path.exists(SUMMARY_PATH):
            raise FileNotFoundError(f"Data file not found: {SUMMARY_PATH}")
        
        df = pd.read_csv(SUMMARY_PATH)
        if df.empty:
            raise ValueError("Dataset is empty")
    
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise
    
    return df

# === FEATURE PREPROCESSING ===
def preprocess_features(df):
    """
    Extracts model features, encodes labels, and validates integrity.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        X (DataFrame): Feature matrix.
        y_encoded (array): Encoded labels.
        participant_ids (array): Participant identifiers.
        le (LabelEncoder): Fitted encoder.
    """
    
    logging.info("Preprocessing features...")
    
    try:
        X = df[FEATURE_COLS].copy()
        y = df['study_group_cleaned']
        participant_ids = df['participant_id'].values

        # Validate target variable
        if y.isnull().any():
            raise ValueError("Target variable contains missing values")
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        logging.info(f"Features processed - Shape: {X.shape}, Classes: {le.classes_}")
        return X, y_encoded, participant_ids, le
    
    except KeyError as e:
        logging.error(f"Error preprocessing features: {str(e)}")
        raise

# === CLASS BALANCING WITH SMOTE ===
# Why SMOTE?
# SMOTE is required because the dataset has **class imbalance** — the number of true_healthy participants is much smaller. Without SMOTE:
# • The model would predict the majority class most of the time.
# • Minority classes would be ignored.
# SMOTE solves this by synthetically oversampling minority examples, improving the model's sensitivity and fairness.
def apply_smote_to_training(X_train, y_train, participant_ids=None):
    """
    Applies SMOTE oversampling to address class imbalance.
    Only applied to training data to avoid data leakage.

    Returns:
        X_resampled, y_resampled
    """
    
    try:
        logging.info("Applying SMOTE to training data...")

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    except Exception as e:
        logging.error(f"Error applying SMOTE: {str(e)}")
        # Return original data if SMOTE fails
        return X_train, y_train

    # if participant_ids is not None:
    #     X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
    #     participant_ids_resampled, _ = smote.fit_resample(participant_ids.reshape(-1, 1), y_encoded)
    #     return X_resampled, y_resampled, participant_ids_resampled.flatten()
    # else:
    #     return smote.fit_resample(X, y_encoded)

# === MODEL TRAINING ===
# Why XGBoost?
# XGBoost is chosen because:
# • It excels on structured/tabular medical datasets
# • Captures non-linear relationships
# • Handles feature interactions
# • Provides feature importance
# • Robust against noisy labels
# • Outperforms simple baselines (logistic regression, decision tree)

def train_model(X_train, y_train, num_classes):
    """
    Trains an XGBoost classifier on the resampled training set.

    Args:
        X_train (array)
        y_train (array)
        num_classes (int)

    Returns:
        model (xgb.XGBClassifier)
    """
    
    try:
        logging.info("Training XGBoost classifier...")
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            use_label_encoder=False,
            learning_rate=0.05,
            max_depth=8,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        logging.info("Model training completed")
        return model
    
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

# === MODEL EVALUATION ===
def evaluate_model(model, X_test, y_test, le):
    """
    Evaluates the model using classification report, confusion matrix, ROC-AUC.

    Returns:
        y_pred (array): Predicted labels.
    """
    
    try: 
        logging.info("Evaluating model...")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Classification report
        logging.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # ROC-AUC score
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
            logging.info(f"ROC-AUC Score: {roc_auc:.4f}")
        except Exception as e:
            logging.warning(f"Could not calculate ROC-AUC: {str(e)}")

        # Confusion Matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix")
            plt.show()
        except Exception as e:
            logging.warning(f"Could not display confusion matrix: {str(e)}")
        
        return y_pred
    
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        raise

# === IDENTIFY MISCLASSIFIED PARTICIPANTS ===
def identify_misclassified(y_test, y_pred, pid_test, le):
    """
    Finds participants misclassified as 'true_healthy'.

    Returns:
        misclassified_as_true_healthy (list)
    """
    
    try:
        logging.info("Identifying misclassified participants...")
        true_labels = le.inverse_transform(y_test)
        pred_labels = le.inverse_transform(y_pred)
        
        # Find misclassified as 'true_healthy'
        misclassified_as_true_healthy = [
            pid for pid, true_lbl, pred_lbl in zip(pid_test, true_labels, pred_labels)
            if pred_lbl == 'true_healthy' and true_lbl != 'true_healthy'
        ]
        logging.info(f"Misclassified as 'true_healthy': {len(misclassified_as_true_healthy)} participants")
        if misclassified_as_true_healthy:
            logging.info(f"IDs: {sorted([int(pid) for pid in misclassified_as_true_healthy])}")

        # Find healthy misclassified as 'true_healthy'
        misclassified_healthy_as_true_healthy = [
            pid for pid, true_lbl, pred_lbl in zip(pid_test, true_labels, pred_labels)
            if true_lbl == 'healthy' and pred_lbl == 'true_healthy'
        ]
        logging.info(f"Healthy misclassified as 'true_healthy': {len(misclassified_healthy_as_true_healthy)} participants")
        if misclassified_healthy_as_true_healthy:
            logging.info(f"IDs: {sorted([int(pid) for pid in misclassified_healthy_as_true_healthy])}")

        return misclassified_as_true_healthy
    
    except Exception as e:
        logging.error(f"Error identifying misclassified participants: {str(e)}")
        return []

# === RELABELING FUNCTION ===
def relabel_and_save(df, ids_to_relabel, iteration):
    """
    Updates the dataset by relabeling specific participants as 'true_healthy'.

    Args:
        df (DataFrame)
        ids_to_relabel (list)
        iteration (int)
    """
    
    try:
        if not ids_to_relabel:
            return df
        
        # Exclude specific IDs as in original code
        ids_to_relabel = [pid for pid in ids_to_relabel if int(pid) not in (4157, 7333)]

        if not ids_to_relabel:
            logging.info("No valid IDs to relabel after exclusions")
            return df

        # Relabel participants
        df.loc[df['participant_id'].isin(ids_to_relabel), 'study_group_cleaned'] = 'true_healthy'
        
        # Save updated dataset
        df.to_csv(SUMMARY_PATH, index=False)
        logging.info(f"Relabeled {len(ids_to_relabel)} participants as 'true_healthy' and saved to '{SUMMARY_PATH}'")
        return df
    
    except Exception as e:
        logging.error(f"Error relabeling and saving: {str(e)}")
        raise

# === MAIN ITERATION LOOP ===
# Purpose of Iterative Relabeling
# This algorithm attempts to clean label noise.
#
# Goal:
# Identify participants whose physiological metrics strongly resemble "true_healthy" even though their labels say otherwise.
#
# Why?
# Medical datasets often contain mislabels due to subjective or incomplete diagnoses. 
# 
# Iterative relabeling:
# • gradually refines the dataset
# • enables the classifier to learn a cleaner boundary
# • reveals hidden "true_healthy" patterns
# • ensures convergence as misclassified counts reduce over time
#
# Process:
# 1. Train model
# 2. Predict
# 3. Identify misclassified-as-true_healthy
# 4. Relabel them
# 5. Repeat until performance stabilizes
def main():
    """
    Runs the full iterative relabeling process until:
    - No misclassifications remain
    - Convergence detected
    - Maximum iterations reached
    """
    
    logging.info("Starting iterative relabeling process...")
    iteration = 0
    previous_misclassified = set()
    convergence_count = 0

    try:
        while iteration < MAX_ITERATIONS:
            logging.info(f"\n=== ITERATION {iteration + 1}/{MAX_ITERATIONS} ===")

            try:
                # Load and preprocess data
                df = load_data()
                X, y_encoded, participant_ids, le = preprocess_features(df)
                
                # Split data BEFORE applying SMOTE (prevents data leakage)
                # Why stratified split?
                # Ensures the training and test sets preserve the same class proportions.
                # Crucial for imbalanced datasets so every class appears in both splits.
                X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
                    X, y_encoded, participant_ids,
                    test_size=0.3, random_state=42, stratify=y_encoded
                )

                # Apply SMOTE only to training data
                X_train_resampled, y_train_resampled = apply_smote_to_training(X_train, y_train)
                
                # Train model
                model = train_model(X_train_resampled, y_train_resampled, len(le.classes_))
                
                # Evaluate model
                y_pred = evaluate_model(model, X_test, y_test, le)
                
                # Identify misclassified participants
                ids_to_relabel = identify_misclassified(y_test, y_pred, pid_test, le)

                # Check for convergence
                current_misclassified = set(ids_to_relabel)
                if not ids_to_relabel:
                    logging.info("No more participants to relabel - convergence achieved!")
                    break

                # Check if we're stuck in a loop (same IDs being misclassified)
                if current_misclassified == previous_misclassified:
                    convergence_count += 1
                    logging.warning(f"Same IDs misclassified for {convergence_count} consecutive iterations")
                    if convergence_count >= 3:
                        logging.warning("Stopping due to lack of progress (same IDs repeatedly misclassified)")
                        break
                else:
                    convergence_count = 0
                    
                # Check for improvement
                improvement = len(previous_misclassified - current_misclassified)
                if previous_misclassified and improvement < len(previous_misclassified) * MIN_IMPROVEMENT_THRESHOLD:
                    logging.warning(f"Minimal improvement detected (improved: {improvement})")

                # Update data with new labels
                df = relabel_and_save(df, ids_to_relabel, iteration + 1)
                
                # Update tracking variables
                previous_misclassified = current_misclassified
                iteration += 1

            except Exception as e:
                logging.error(f"Error in iteration {iteration + 1}: {str(e)}")
                break

        if iteration >= MAX_ITERATIONS:
            logging.warning(f"Reached maximum iterations ({MAX_ITERATIONS})")

        logging.info(f"Process completed after {iteration + 1} iterations")
    
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    
    finally:
        logging.info("Cleanup completed")

if __name__ == '__main__':
    main()