__CURR_FILE__ = "xgboost_relabel_truehealthy v2"

"""
===============================================================================
Iterative Relabeling with XGBoost

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
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt

# === PATH & LOGGING SETUP ===
from paths import SUMMARY_PATH, LOG_DIR

# Create logs directory and Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG to see more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, __CURR_FILE__ + ".log")),  # Save logs to file
        logging.StreamHandler()  # Also print to console
    ]
)
logging.info(f"Logs will be saved to: {LOG_DIR}")

# === CONFIGURATION ===
MAX_ITERATIONS = 10  # Prevent infinite loops
MIN_IMPROVEMENT_THRESHOLD = 0.01  # Minimum improvement to continue iterations
EARLY_STOPPING_ROUNDS = 30

# Freeze/test split & CV settings (new)
TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5  # CV folds for OOF predictions (new)
N_REPEATS = 3  # NEW: repeated CV rounds so "votes" are meaningful

# Relabeling strictness (new)
RELABEL_PROBA_THRESHOLD = 0.80     # Only relabel if model is confident
RELABEL_MIN_OOF_VOTES = 3          # Must be predicted true_healthy in >= this many folds

# === FEATURE CONFIG ===
FEATURE_COLS = [
    'mean_blood_glucose', 'std_dev', 'cv_percent', 'tir_percent', 'tar_percent',
    'tbr_percent', 'mage', 'j_index', 'lability_index', 'conga_1h', 'conga_2h', 'age'
]
# Optional interpretability toggle (new)
USE_INTERPRETABLE_FEATURES = False
INTERPRETABLE_FEATURES = [
    f for f in FEATURE_COLS if f not in ('cv_percent', 'tar_percent', 'j_index')
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
    # WHY THIS EXISTS:
    # We load a single participant-level summary file (1 row per participant_id).
    # We keep I/O isolated here so the main loop stays clean and failures are logged clearly.
    #
    # WHY we create study_group_original:
    # This preserves the initial labels for traceability so we can audit what changed over iterations.

    
    try:
        logging.info("Loading summarized metrics CSV...")
        if not os.path.exists(SUMMARY_PATH):
            raise FileNotFoundError(f"Data file not found: {SUMMARY_PATH}")
        
        df = pd.read_csv(SUMMARY_PATH)
        logging.info(f"rows={len(df)}, unique_pids={df['participant_id'].nunique()}")

        if df.empty:
            raise ValueError("Dataset is empty")
        
        # NEW: preserve original labels for research traceability
        if 'study_group_original' not in df.columns:
            df['study_group_original'] = df['study_group_cleaned']
            logging.info("Created 'study_group_original' column to preserve original labels.")
    
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
    # WHY LabelEncoder:
    # XGBoost expects numeric class labels for multi-class classification.
    #
    # WHY we freeze encoder later in main():
    # To keep class-to-index mapping stable across iterations (prevents label drift bugs).

    
    logging.info("Preprocessing features...")
    
    try:
        cols = INTERPRETABLE_FEATURES if USE_INTERPRETABLE_FEATURES else FEATURE_COLS
        X = df[cols].copy()        
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
    # WHY SMOTE (and only on training folds):
    # Applying SMOTE on validation/test would leak synthetic samples into evaluation.
    # Therefore we apply it only to the training portion of each split.
    #
    # WHY we log class counts:
    # SMOTE can fail for very small classes; counts provide debugging visibility.

    
    try:
        logging.info("Applying SMOTE to training data...")

        # SMOTE can fail when a class is tiny: SMOTE requires enough samples per class, so log class counts before SMOTE in each fold
        unique, counts = np.unique(y_train, return_counts=True)
        logging.info(f"Train class counts pre-SMOTE: {dict(zip(unique.tolist(), counts.tolist()))}")


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

def train_model(X_train, y_train, num_classes, X_val=None, y_val=None):
    """
    Trains an XGBoost classifier on the resampled training set.

    Args:
        X_train (array)
        y_train (array)
        num_classes (int)

    Returns:
        model (xgb.XGBClassifier)
    """
    # WHY NO early stopping:
    # Your current xgboost installation raised:
    #   "XGBClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'"
    # So we removed early stopping to keep training stable across environments.
    # Instead, we keep n_estimators moderate (300) to avoid excessive overfitting/time.
    
    try:
        logging.info("\nTraining XGBoost classifier...")

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            learning_rate=0.05,
            max_depth=8,
            n_estimators=300,          # keep reasonable since no early stopping
            subsample=0.8,
            colsample_bytree=0.8,
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
def evaluate_model(model, X_test, y_test_eval_encoded, le, y_test_original_str=None):
    """
    Evaluates the model using classification report, confusion matrix, ROC-AUC.

    Args:
        y_test_eval_encoded: encoded ground-truth labels to evaluate against (should be frozen holdout labels)
        y_test_original_str: optional original (string) labels to print report against (same frozen labels)

    Returns:
        y_pred (array): Predicted encoded labels.
    """

    try:
        logging.info("Evaluating model...")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Classification report
        if y_test_original_str is not None:
            y_true_eval = y_test_original_str
            y_pred_eval = le.inverse_transform(y_pred)
            print(classification_report(y_true_eval, y_pred_eval, labels=le.classes_))
        else:
            print(classification_report(y_test_eval_encoded, y_pred, target_names=le.classes_))

        # ROC-AUC (evaluate against frozen encoded ground truth)
        try:
            roc_auc = roc_auc_score(y_test_eval_encoded, y_pred_proba, multi_class='ovo')
            logging.info(f"ROC-AUC Score: {roc_auc:.4f}")
        except Exception as e:
            logging.warning(f"Could not calculate ROC-AUC: {str(e)}")

        # Confusion Matrix (evaluate against frozen encoded ground truth)
        try:
            cm = confusion_matrix(y_test_eval_encoded, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

            plt.figure(figsize=(10, 8))
            disp.plot(cmap="Blues", values_format='d')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.title("Confusion Matrix", fontsize=16)
            plt.tight_layout()
            # plt.show()
        except Exception as e:
            logging.warning(f"Could not display confusion matrix: {str(e)}")

        return y_pred

    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        raise


# === IDENTIFY MISCLASSIFIED PARTICIPANTS ===
# def identify_misclassified(y_test, y_pred, pid_test, le):
#     """
#     Finds participants misclassified as 'true_healthy'.

#     Returns:
#         misclassified_as_true_healthy (list)
#     """
    
#     try:
#         logging.info("Identifying misclassified participants...")
#         true_labels = le.inverse_transform(y_test)
#         pred_labels = le.inverse_transform(y_pred)
        
#         # Find misclassified as 'true_healthy'
#         misclassified_as_true_healthy = [
#             pid for pid, true_lbl, pred_lbl in zip(pid_test, true_labels, pred_labels)
#             if pred_lbl == 'true_healthy' and true_lbl != 'true_healthy'
#         ]
#         logging.info(f"Misclassified as 'true_healthy': {len(misclassified_as_true_healthy)} participants")
#         if misclassified_as_true_healthy:
#             logging.info(f"IDs: {sorted([int(pid) for pid in misclassified_as_true_healthy])}")

#         # Find healthy misclassified as 'true_healthy'
#         misclassified_healthy_as_true_healthy = [
#             pid for pid, true_lbl, pred_lbl in zip(pid_test, true_labels, pred_labels)
#             if true_lbl == 'healthy' and pred_lbl == 'true_healthy'
#         ]
#         logging.info(f"Healthy misclassified as 'true_healthy': {len(misclassified_healthy_as_true_healthy)} participants")
#         if misclassified_healthy_as_true_healthy:
#             logging.info(f"IDs: {sorted([int(pid) for pid in misclassified_healthy_as_true_healthy])}")

#         return misclassified_as_true_healthy
    
#     except Exception as e:
#         logging.error(f"Error identifying misclassified participants: {str(e)}")
#         return []

# === NEW: OOF RELABEL CANDIDATES (CV-based, avoids test leakage) ===
def get_oof_relabel_candidates(df_train_pool, le):
    """
    Uses RepeatedStratifiedKFold OOF predictions on TRAIN pool to select relabel candidates.

    Relabel rule:
      - predicted label == 'true_healthy' in >= RELABEL_MIN_OOF_VOTES OOF rounds
      - AND mean predicted probability for true_healthy >= RELABEL_PROBA_THRESHOLD
      - AND current true label != 'true_healthy'

    Returns:
      candidate_ids (list)
      diagnostics_df (DataFrame): per-participant votes/proba for transparency
    """
    # WHY RepeatedStratifiedKFold:
    # We want "votes" across multiple independent splits to reduce randomness.
    # A participant must be predicted 'true_healthy' consistently (votes) and confidently (prob threshold).
    #
    # WHY votes + probability threshold:
    # Votes ensure stability across repeats; probability ensures confidence in the prediction.

    cols = INTERPRETABLE_FEATURES if USE_INTERPRETABLE_FEATURES else FEATURE_COLS
    X_pool = df_train_pool[cols].copy()
    y_pool_str = df_train_pool['study_group_cleaned'].astype(str)
    pid_pool = df_train_pool['participant_id'].values

    # Encode using provided encoder classes (must be consistent across iterations)
    y_pool = le.transform(y_pool_str)

    if 'true_healthy' not in le.classes_:
        logging.warning("'true_healthy' not found in label encoder classes. No relabeling possible.")
        return [], pd.DataFrame()

    th_class_idx = int(np.where(le.classes_ == 'true_healthy')[0][0])

    # NEW: repeated CV so votes are meaningful
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )
    total_rounds = N_SPLITS * N_REPEATS

    votes_truehealthy = np.zeros(len(df_train_pool), dtype=int)
    proba_sum_truehealthy = np.zeros(len(df_train_pool), dtype=float)
    proba_count = np.zeros(len(df_train_pool), dtype=int)

    # OOF loop (each sample gets exactly N_REPEATS OOF predictions)
    for round_idx, (tr_idx, va_idx) in enumerate(rskf.split(X_pool, y_pool), start=1):
        X_tr, y_tr = X_pool.iloc[tr_idx], y_pool[tr_idx]
        X_va, y_va = X_pool.iloc[va_idx], y_pool[va_idx]

        # SMOTE only on fold-training set (prevents leakage)
        X_tr_res, y_tr_res = apply_smote_to_training(X_tr, y_tr)

        # Early stopping uses the fold's validation set (NO SMOTE on validation)
        model = train_model(X_tr_res, y_tr_res, num_classes=len(le.classes_))

        pred_va = model.predict(X_va)
        pred_va_proba = model.predict_proba(X_va)[:, th_class_idx]

        votes_truehealthy[va_idx] += (pred_va == th_class_idx).astype(int)
        proba_sum_truehealthy[va_idx] += pred_va_proba
        proba_count[va_idx] += 1

        logging.info(f"OOF split {round_idx}/{total_rounds} completed (each sample gets {N_REPEATS} OOF predictions total).")

    # Mean proba across repeats
    proba_mean_truehealthy = np.divide(
        proba_sum_truehealthy,
        np.maximum(proba_count, 1)
    )

    # Candidate selection
    current_labels = y_pool_str.values
    is_not_truehealthy = current_labels != 'true_healthy'
    high_votes = votes_truehealthy >= RELABEL_MIN_OOF_VOTES
    high_proba = proba_mean_truehealthy >= RELABEL_PROBA_THRESHOLD

    candidate_mask = is_not_truehealthy & high_votes & high_proba
    candidate_ids = pid_pool[candidate_mask].tolist()

    diagnostics_df = pd.DataFrame({
        'participant_id': pid_pool,
        'current_label': current_labels,
        'votes_truehealthy': votes_truehealthy,
        'mean_proba_truehealthy': proba_mean_truehealthy
    }).sort_values(['votes_truehealthy', 'mean_proba_truehealthy'], ascending=False)

    logging.info(
        f"OOF relabel candidates: {len(candidate_ids)} "
        f"(votes>={RELABEL_MIN_OOF_VOTES} out of {N_REPEATS} repeats, "
        f"mean_proba>={RELABEL_PROBA_THRESHOLD}). Max votes possible={N_REPEATS}."
    )
    if candidate_ids:
        logging.info(f"Candidate IDs (first 50): {sorted([int(pid) for pid in candidate_ids])[:50]}")

    return candidate_ids, diagnostics_df

# === RELABELING FUNCTION ===
def relabel_and_save(df, ids_to_relabel, iteration):
    """
    Updates the dataset by relabeling specific participants as 'true_healthy'.
    Saves back to SUMMARY_PATH (as requested).

    Args:
        df (DataFrame)
        ids_to_relabel (list)
        iteration (int)
    """
    # WHY THIS EXISTS:
    # Applies relabels only for selected participant_ids and persists them to disk,
    # so each iteration starts from the updated label state.
    #
    # WHY we log before/after:
    # Provides auditability: which participants changed, and from what label to what label.

    try:
        if not ids_to_relabel:
            return df

        # Exclude specific IDs as in original code
        ids_to_relabel = [pid for pid in ids_to_relabel if int(pid) not in (4157, 7333)]

        if not ids_to_relabel:
            logging.info("No valid IDs to relabel after exclusions")
            return df

        # Ensure original label column exists
        if 'study_group_original' not in df.columns:
            df['study_group_original'] = df['study_group_cleaned']

        # Log before/after for traceability
        before = df.loc[df['participant_id'].isin(ids_to_relabel), ['participant_id', 'study_group_cleaned']].copy()

        # Relabel participants
        df.loc[df['participant_id'].isin(ids_to_relabel), 'study_group_cleaned'] = 'true_healthy'

        after = df.loc[df['participant_id'].isin(ids_to_relabel), ['participant_id', 'study_group_cleaned']].copy()

        logging.info(f"Relabeled {len(ids_to_relabel)} participants as 'true_healthy' (iteration={iteration}).")
        logging.info(f"Sample relabels (first 20): {before.head(20).to_dict(orient='records')}")

        # Save updated dataset (you said you WANT to overwrite original)
        df.to_csv(SUMMARY_PATH, index=False)
        logging.info(f"Saved updated labels back to: {SUMMARY_PATH}")

        return df

    except Exception as e:
        logging.error(f"Error relabeling and saving: {str(e)}")
        raise


# Add this function before main()
def evaluate_with_cv(df_train_pool, le, n_splits=5):
    """
    Performs stratified k-fold CV on training pool to get robust metrics.
    
    Returns:
        cv_scores (dict): Mean and std of accuracy, ROC-AUC, precision, recall, F1 per fold
    """
    logging.info(f"\n=== Cross-Validation Evaluation ({n_splits} folds) ===")
    
    cols = INTERPRETABLE_FEATURES if USE_INTERPRETABLE_FEATURES else FEATURE_COLS
    X = df_train_pool[cols].copy()
    y = le.transform(df_train_pool['study_group_cleaned'].astype(str))
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    accuracies = []
    roc_aucs = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Apply SMOTE only to training fold
        X_tr_res, y_tr_res = apply_smote_to_training(X_tr, y_tr)
        
        # Train model
        model = train_model(X_tr_res, y_tr_res, len(le.classes_))
        
        # Evaluate on validation fold
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)
        
        # Accuracy
        acc = np.mean(y_pred == y_val)
        accuracies.append(acc)
        
        # Precision, Recall, F1 (macro-averaged)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='macro', zero_division=0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovo')
            roc_aucs.append(roc_auc)
        except:
            roc_auc = None
        
        # Log detailed fold metrics
        logging.info(
            f"Fold {fold}: Accuracy={acc:.4f}, Precision={precision:.4f}, "
            f"Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc if roc_auc else 'N/A'}"
        )
    
    cv_scores = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'roc_auc_mean': np.mean(roc_aucs) if roc_aucs else None,
        'roc_auc_std': np.std(roc_aucs) if roc_aucs else None
    }
    
    # Summary logging
    logging.info(f"\n=== CV Summary (Macro-Averaged) ===")
    logging.info(f"Accuracy:  {cv_scores['accuracy_mean']:.4f} ± {cv_scores['accuracy_std']:.4f}")
    logging.info(f"Precision: {cv_scores['precision_mean']:.4f} ± {cv_scores['precision_std']:.4f}")
    logging.info(f"Recall:    {cv_scores['recall_mean']:.4f} ± {cv_scores['recall_std']:.4f}")
    logging.info(f"F1-Score:  {cv_scores['f1_mean']:.4f} ± {cv_scores['f1_std']:.4f}")
    if cv_scores['roc_auc_mean']:
        logging.info(f"ROC-AUC:   {cv_scores['roc_auc_mean']:.4f} ± {cv_scores['roc_auc_std']:.4f}")
    
    return cv_scores

    
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
# === MAIN ITERATION LOOP ===
def main():
    """
    Runs the full iterative relabeling process until:
    - No misclassifications remain
    - Convergence detected
    - Maximum iterations reached

    Before executing, it requires that `cluster_true_healthy v2.py` has already run and
    """
    # WHY THIS EXISTS:
    # Orchestrates the full iterative relabeling loop:
    # 1) Freeze test participants once (never relabel them)
    # 2) Use OOF-only relabeling on train pool (prevents leakage)
    # 3) Relabel train pool only
    # 4) Evaluate each iteration on the original frozen test labels
    #
    # WHY freeze test set once:
    # If test changes each iteration, metrics become incomparable and can bias relabeling decisions.
    
    logging.info("Starting iterative relabeling process...")
    iteration = 0
    previous_relabels = set()
    convergence_count = 0

    # NEW: Freeze holdout test participants once (no relabeling from test)
    frozen = False
    test_pid_set = None
    y_test_original_str = None
    le_frozen = None

    try:
        while iteration < MAX_ITERATIONS:
            logging.info(f"\n=== ITERATION {iteration + 1}/{MAX_ITERATIONS} ===")

            try:
                # Load and preprocess data
                df = load_data()

                # Freeze encoder on first iteration so class mapping stays stable
                if le_frozen is None:
                    _, _, _, le_init = preprocess_features(df)
                    le_frozen = le_init
                le = le_frozen

                # Ensure all labels are known to encoder (if new labels appear, this is a problem)
                unseen = set(df['study_group_cleaned'].astype(str).unique()) - set(le.classes_)
                if unseen:
                    logging.error(f"Found unseen labels not in initial encoder classes: {sorted(unseen)}")
                    raise ValueError("Unseen labels encountered across iterations. Keep label set stable.")

                # Freeze holdout split once by participant IDs (NEW)
                if not frozen:
                    cols = INTERPRETABLE_FEATURES if USE_INTERPRETABLE_FEATURES else FEATURE_COLS
                    X_all = df[cols].copy()
                    y_all_str = df['study_group_cleaned'].astype(str)
                    pid_all = df['participant_id'].values

                    y_all_enc = le.transform(y_all_str)

                    X_train_pool, X_test_holdout, y_train_pool, y_test_holdout, pid_train_pool, pid_test_holdout = train_test_split(
                        X_all, y_all_enc, pid_all,
                        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all_enc
                    )

                    test_pid_set = set(pid_test_holdout.tolist())
                    y_test_original_str = le.inverse_transform(y_test_holdout)  # frozen ORIGINAL labels
                    global test_pid_to_label
                    test_pid_to_label = dict(zip(pid_test_holdout.tolist(), y_test_original_str.tolist()))
                    frozen = True

                    logging.info(f"Frozen TEST holdout created once: {len(test_pid_set)} participants.")

                # Build train pool DF and test holdout DF from current df (but using frozen PID sets)
                df_train_pool = df.loc[~df['participant_id'].isin(test_pid_set)].copy()
                df_test_holdout = df.loc[df['participant_id'].isin(test_pid_set)].copy()

                # Sanity check: ensure no PID leakage between train and test
                train_pids = set(df_train_pool['participant_id'].unique())
                test_pids  = set(df_test_holdout['participant_id'].unique())
                overlap = len(train_pids & test_pids)
                logging.info(f"[PID SPLIT CHECK] train_pids={len(train_pids)} test_pids={len(test_pids)} overlap={overlap}")
                if overlap > 0:
                    raise ValueError("PID leakage detected: overlap between train and test.")

                # Ensure deterministic row order in test (align with frozen y_test_original_str)
                # We sort by participant_id to keep consistent mapping.
                df_test_holdout = df_test_holdout.sort_values('participant_id').reset_index(drop=True)
                df_train_pool = df_train_pool.sort_values('participant_id').reset_index(drop=True)

                # --- OOF-based relabeling on TRAIN pool only (NEW) ---
                ids_to_relabel, diagnostics_df = get_oof_relabel_candidates(df_train_pool, le)

                # Defensive check: ensure no test PIDs are proposed for relabeling (shouldn't happen with OOF candidates, but just in case)
                ids_to_relabel = [pid for pid in ids_to_relabel if pid not in test_pid_set]
                current_relabels = set(ids_to_relabel)
                if not ids_to_relabel:
                    logging.info("No more participants to relabel - convergence achieved!")
                    break

                # --- Convergence check: stop if we are not adding any NEW relabels ---
                new_relabels = len(current_relabels - previous_relabels)

                if new_relabels == 0:
                    convergence_count += 1
                    logging.warning(f"No new relabels proposed. convergence_count={convergence_count}")
                    if convergence_count >= 3:
                        logging.warning("Stopping due to lack of progress (no new relabels for 3 iterations).")
                        break
                else:
                    convergence_count = 0

                # Apply relabeling ONLY to TRAIN participants (NEW safeguard)
                ids_to_relabel = [pid for pid in ids_to_relabel if pid not in test_pid_set]

                # Update data with new labels
                df = relabel_and_save(df, ids_to_relabel, iteration + 1)

                # --- Train final model on relabeled TRAIN pool and evaluate on frozen TEST labels (NEW) ---
                # Recompute pool/test features from updated df
                cols = INTERPRETABLE_FEATURES if USE_INTERPRETABLE_FEATURES else FEATURE_COLS

                df_train_pool2 = df.loc[~df['participant_id'].isin(test_pid_set)].copy().sort_values('participant_id')
                df_test_holdout2 = df.loc[df['participant_id'].isin(test_pid_set)].copy().sort_values('participant_id')

                X_train = df_train_pool2[cols].copy()
                y_train = le.transform(df_train_pool2['study_group_cleaned'].astype(str))

                X_test = df_test_holdout2[cols].copy()
                # Build frozen original labels in the SAME ORDER as df_test_holdout2
                y_test_original_eval_str = np.array(
                    [test_pid_to_label[pid] for pid in df_test_holdout2['participant_id'].values]
                )

                # SMOTE on full training pool only
                X_train_resampled, y_train_resampled = apply_smote_to_training(X_train, y_train)

                model = train_model(X_train_resampled, y_train_resampled, len(le.classes_))

                # Evaluate against ORIGINAL frozen labels for test (not updated)
                # Need to align y_test_original_str to current sorted pid order:
                # We stored y_test_original_str from initial split, but the order might differ.
                # Safer: build mapping pid -> original_str once and then map in sorted order.
                # We'll create mapping at freeze time by using pid_test_holdout. Reconstruct now:                

                # Minimal, correct approach: store pid->original label mapping at freeze time
                # (implemented below by storing test_pid_to_label on freeze)
                # Evaluate using that mapping:
                # Encode frozen original labels (this is what ROC-AUC + confusion matrix must use)
                y_test_original_eval_enc = le.transform(y_test_original_eval_str)

                _ = evaluate_model(
                    model,
                    X_test,
                    y_test_original_eval_enc,
                    le,
                    y_test_original_str=y_test_original_eval_str
                )

                # Update tracking variables
                previous_relabels = current_relabels
                iteration += 1

            except Exception as e:
                logging.error(f"Error in iteration {iteration + 1}: {str(e)}")
                break

        if iteration >= MAX_ITERATIONS:
            logging.warning(f"Reached maximum iterations ({MAX_ITERATIONS})")
        
        # After iteration loop completes:
        logging.info("\n=== Final Cross-Validation on Converged Labels ===")
        df_final = load_data()
        df_train_final = df_final.loc[~df_final['participant_id'].isin(test_pid_set)].copy()
        cv_results = evaluate_with_cv(df_train_final, le_frozen)

        logging.info(f"Process completed after {iteration + 1} iterations")
    
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    
    finally:
        logging.info("Cleanup completed")

# --- NEW: store frozen test PID -> original label mapping ---
test_pid_to_label = {}

if __name__ == '__main__':
    # Build mapping after first freeze occurs inside main (we need a global container for minimal edits)
    # We'll initialize it lazily in main when freezing. To do that, we slightly patch main by referencing this global.
    main()