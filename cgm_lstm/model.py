"""
Model creation, training, threshold selection, and evaluation for the CGM LSTM pipeline.
"""

import os
import json
import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    brier_score_loss,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input,
    Conv1D, MaxPooling1D, Bidirectional,
)
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from .data import Config

logger = logging.getLogger(__name__)

class TemperatureScaling:
    """Post-hoc calibration via temperature scaling.

    Rescales logits by a learned temperature parameter T:
        calibrated_prob = sigmoid(logit / T)

    Fitted on validation set after training to improve calibration.
    """
    def __init__(self):
        self.temperature = 1.0  # Default: no scaling

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, max_iter: int = 50) -> float:
        """Fit temperature on validation predictions using negative log-likelihood.

        Parameters
        ----------
        y_true : Binary labels
        y_prob : Model predicted probabilities (BEFORE temperature scaling)
        max_iter : Maximum optimization iterations

        Returns
        -------
        Optimal temperature T
        """
        from scipy.optimize import minimize

        # Convert probabilities back to logits
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)  # Avoid log(0)
        logits = np.log(y_prob / (1 - y_prob))

        def nll(T):
            """Negative log-likelihood loss."""
            T = float(T[0])
            scaled_probs = 1 / (1 + np.exp(-logits / T))
            scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
            return -np.mean(y_true * np.log(scaled_probs) + (1 - y_true) * np.log(1 - scaled_probs))

        result = minimize(nll, x0=[1.0], method='BFGS', options={'maxiter': max_iter})
        self.temperature = float(result.x[0])
        return self.temperature

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        logits = np.log(y_prob / (1 - y_prob))
        scaled_probs = 1 / (1 + np.exp(-logits / self.temperature))
        return scaled_probs

class LSTMTrainer:
    """Encapsulate model creation, training, threshold selection, and evaluation.
        This class owns the Keras model instance, tracks training history, manages
        class weights, and handles how thresholds are chosen from the validation set.
    """
    def __init__(self, config: Config):
        """Initialize trainer state for a given configuration."""
        self.config = config
        self.model = None
        self.history = None
        self.class_weights: Optional[Dict[int, float]] = None
        self.best_threshold: float = 0.5   # will be set from VAL
        self.val_auc: Optional[float] = None
        self.fold_tag: str = ""   # e.g., "[CV:FOLD 1/5] "
        self.fold_dir: Optional[str] = None  # where to save fold-specific artifacts
        self.temperature_scaler: Optional[TemperatureScaling] = None
        self.feature_scaler_mean: Optional[np.ndarray] = None  # Feature scaling mean
        self.feature_scaler_scale: Optional[np.ndarray] = None  # Feature scaling std

    def create_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build and compile the model for binary classification.

        If config.use_conv_frontend is True (default), builds:
            Input → Conv1D → MaxPool → Conv1D → MaxPool
                  → BiLSTM → Dropout → BiLSTM
                  → Dense → Dropout → Dense(1, sigmoid)

        The Conv layers extract local temporal patterns and reduce the
        sequence length (e.g. 2138 → 133) so the LSTM operates within its
        effective gradient-propagation range (~200 steps). L2 regularization
        is applied to LSTM kernels for generalization.

        If config.use_conv_frontend is False, builds the original:
            Input → LSTM(return_sequences) → Dropout → LSTM → Dense → Dense(1)
        """
        inp = Input(shape=input_shape)

        if self.config.use_conv_frontend:
            # --- Conv1D frontend: extract local features & reduce sequence ---
            x = Conv1D(
                filters=self.config.conv_filters_1,
                kernel_size=self.config.conv_kernel_1,
                padding='same',
                activation='relu',
            )(inp)
            x = MaxPooling1D(pool_size=self.config.conv_pool_1)(x)
            x = Conv1D(
                filters=self.config.conv_filters_2,
                kernel_size=self.config.conv_kernel_2,
                padding='same',
                activation='relu',
            )(x)
            x = MaxPooling1D(pool_size=self.config.conv_pool_2)(x)

            # --- Bidirectional LSTM on reduced sequence ---
            l2 = regularizers.l2(self.config.l2_reg)
            if self.config.lstm_bidirectional:
                x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2))(x)
                x = Dropout(self.config.dropout_rate)(x)
                x = Bidirectional(LSTM(32, kernel_regularizer=l2))(x)
            else:
                x = LSTM(64, return_sequences=True, kernel_regularizer=l2)(x)
                x = Dropout(self.config.dropout_rate)(x)
                x = LSTM(32, kernel_regularizer=l2)(x)

            x = Dense(32, activation='relu')(x)
            x = Dropout(self.config.dropout_rate)(x)
            arch_name = f"Conv+{'Bi' if self.config.lstm_bidirectional else ''}LSTM"
        else:
            # --- Original LSTM-only architecture ---
            x = LSTM(self.config.lstm_units_1, return_sequences=True)(inp)
            x = Dropout(self.config.dropout_rate)(x)
            x = LSTM(self.config.lstm_units_2)(x)
            x = Dense(self.config.dense_units, activation='relu')(x)
            arch_name = "LSTM-only"

        out = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=out)

        opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=self.config.clipnorm)
        metrics = [
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)

        logger.info(f"\n[MODEL] Architecture: {arch_name}")
        logger.info(f"[MODEL]   Input shape: {input_shape}")
        logger.info(f"[MODEL]   Total parameters: {model.count_params():,}")
        return model

    def prepare_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create Train/Val/Test splits and apply scaling / imbalance handling.

        Uses a random stratified split for the initial Train/Test partition,
        then delegates val split, class weights, SMOTE, and scaling to
        _finalize_splits().

        Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"[DATA] Dataset: {len(y)} samples, classes: {dict(zip(unique, counts))}")
        if len(unique) < 2:
            raise ValueError("Need at least 2 classes for binary classification.")

        idx = np.arange(len(y))
        X_tr_all, X_test, y_tr_all, y_test, idx_tr_all, idx_test = train_test_split(
            X, y, idx,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=y
        )

        # Delegate shared processing
        X_train, X_val, X_test, y_train, y_val, y_test = self._finalize_splits(
            X_tr_all, y_tr_all, X_test, y_test
        )

        # Re-derive the val sub-split indices for saving (deterministic with same seed)
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.config.val_size_from_train,
            random_state=self.config.random_seed
        )
        for tr_idx, val_idx in sss.split(X_tr_all, y_tr_all):
            pass
        np.save(os.path.join(self.config.models_dir, 'idx_train.npy'), idx_tr_all[tr_idx])
        np.save(os.path.join(self.config.models_dir, 'idx_val.npy'),   idx_tr_all[val_idx])
        np.save(os.path.join(self.config.models_dir, 'idx_test.npy'),  idx_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_training_data_from_indices(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Same as prepare_training_data(), but uses externally provided indices.

        Required for K-fold cross-validation so each fold controls its own
        holdout set. Delegates all post-split processing to _finalize_splits().
        """
        return self._finalize_splits(
            X[train_idx], y[train_idx],
            X[test_idx], y[test_idx],
        )

    def _finalize_splits(
        self,
        X_tr_all: np.ndarray,
        y_tr_all: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Shared post-split processing: val split, class weights, SMOTE, standardization.

        Called by both prepare_training_data() and prepare_training_data_from_indices()
        after they have established the initial train/test partition.

        Parameters
        ----------
        X_tr_all, y_tr_all : Training features (3D) and labels (1D).
        X_test, y_test     : Test features (3D) and labels (1D).
        save_dir           : Directory for feature_scaler.npz. Defaults to models_dir.

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        if save_dir is None:
            save_dir = self.config.models_dir

        # --- Val from Train (stratified) ---
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.config.val_size_from_train,
            random_state=self.config.random_seed,
        )
        for tr_idx, val_idx in sss.split(X_tr_all, y_tr_all):
            X_train, X_val = X_tr_all[tr_idx], X_tr_all[val_idx]
            y_train, y_val = y_tr_all[tr_idx], y_tr_all[val_idx]

        logger.info(f"[SPLIT] Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        logger.info(f"[DATA]  Train dist: {np.bincount(y_train)} | Val dist: {np.bincount(y_val)} | Test dist: {np.bincount(y_test)}")

        # --- Class weights (TRAIN only) ---
        if self.config.use_class_weights:
            classes = np.unique(y_train)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            self.class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
            logger.info(f"[CLASS WEIGHTS] {self.class_weights}")
        else:
            self.class_weights = None

        # --- Optional SMOTE (TRAIN only; discouraged for sequences) ---
        if self.config.use_smote:
            if len(np.unique(y_train)) >= 2:
                logger.info("[SMOTE] Applying SMOTE to TRAIN (flatten→resample→reshape)...")
                X_flat = X_train.reshape(X_train.shape[0], -1)
                sm = SMOTE(random_state=self.config.random_seed)
                X_res, y_res = sm.fit_resample(X_flat, y_train)
                X_train = X_res.reshape(-1, X_train.shape[1], X_train.shape[2])
                y_train = y_res
                logger.info(f"[SMOTE] TRAIN resampled: {X_train.shape[0]} | dist: {np.bincount(y_train)}")
            else:
                logger.warning("[SMOTE] TRAIN single-class; skipping SMOTE.")
        else:
            logger.info("[SMOTE] Disabled (use_smote=False). Using class weights if enabled.")

        # --- Standardize continuous features; keep binary flags as-is ---
        binary = set(self.config.binary_features)
        cont_idx = [i for i, n in enumerate(self.config.selected_features) if n not in binary]

        train_2d = X_train[:, :, cont_idx].reshape(-1, len(cont_idx))
        mu = train_2d.mean(0).astype(np.float32)
        sigma = train_2d.std(0).astype(np.float32)
        sigma[sigma < 1e-8] = 1.0

        def _apply(z):
            z = z.copy()
            z[:, :, cont_idx] = (z[:, :, cont_idx] - mu) / sigma
            return z

        X_train, X_val, X_test = _apply(X_train), _apply(X_val), _apply(X_test)

        # Store scaler parameters as instance variables
        self.feature_scaler_mean = mu
        self.feature_scaler_scale = sigma

        # Persist scaler for inference / analysis
        np.savez(
            os.path.join(save_dir, 'feature_scaler.npz'),
            mean=mu,
            std=sigma,
            cont_features=np.array(
                [self.config.selected_features[i] for i in cont_idx], dtype=object
            ),
        )
        logger.info("[SCALE] Train-only standardization applied (continuous features only).")

        # --- Multicollinearity diagnostics ---
        if getattr(self.config, "run_multicollinearity_check", False):
            _ = self.multicollinearity_diagnostics(X_train)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def compute_val_threshold(self, X_val: np.ndarray, y_val: np.ndarray, method: str = "youden") -> Tuple[float, np.ndarray]:
        """Select a probability threshold using only the validation set.
            The model outputs probabilities in [0,1]. This method chooses a fixed
            decision threshold based on validation data only so that the test set
            remains completely untouched.

            Supported methods:
            - "youden": maximize Youden's J statistic (TPR - FPR).
            - "fpr_le_5": choose the largest threshold with FPR <= 5%.

            Returns
            -------
            threshold : float
                Selected threshold
            y_prob : np.ndarray
                Validation predictions (for OOF pooling in CV)
        """
        y_prob = self.model.predict(X_val, verbose=0).ravel()

        # ROC & AUC on VAL
        fpr, tpr, thr = roc_curve(y_val, y_prob)
        self.val_auc = float(auc(fpr, tpr))

        if method == "youden":
            j = tpr - fpr
            best_idx = int(np.argmax(j))
            self.best_threshold = float(thr[best_idx])
            crit = f"Youden's J @ idx={best_idx}"
        elif method == "fpr_le_5":
            # Largest threshold with FPR <= 0.05
            ok = np.where(fpr <= 0.05)[0]
            idx = int(ok[-1]) if len(ok) else int(np.argmin(fpr))
            self.best_threshold = float(thr[idx])
            crit = f"FPR≤5% @ idx={idx}"
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"[THRESH] Selected threshold on VAL ({crit}): {self.best_threshold:.4f} | Val AUC: {self.val_auc:.4f}")
        return self.best_threshold, y_prob

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train the LSTM model with early stopping and LR scheduling.
            Uses the validation AUC as the primary monitor for:
            - Early stopping (restore best weights).
            - Model checkpointing (saving the best epoch to disk).
            - Learning rate reduction on plateau.
            After training, a validation-based threshold is computed and frozen.
        """
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.create_model(input_shape)

        callbacks = [
            EarlyStopping(monitor='val_auc', mode='max', patience=self.config.early_stopping_patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(os.path.join(self.config.models_dir, 'best_model.keras'), monitor='val_auc', mode='max', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_auc', mode='max', patience=3, factor=0.5, min_lr=1e-7, verbose=1)
        ]

        # Keras expects labels as column vectors for binary_crossentropy
        y_train_r = y_train.reshape(-1, 1)
        y_val_r   = y_val.reshape(-1, 1)

        logger.info("\n[TRAIN] Starting model training...")
        self.history = self.model.fit(
            X_train, y_train_r,
            validation_data=(X_val, y_val_r),  # validate on VAL, not TEST
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1,
            class_weight=self.class_weights,
            shuffle=True                     # reproducible batching
        )
        # Choose and freeze a threshold from VALIDATION
        _, _ = self.compute_val_threshold(X_val, y_val, method="youden")  # Discard y_prob in non-CV mode

        # Temperature scaling calibration
        if getattr(self.config, 'use_temperature_scaling', True):
            self.temperature_scaler = TemperatureScaling()
            y_val_prob = self.model.predict(X_val, verbose=0).ravel()
            temp = self.temperature_scaler.fit(y_val, y_val_prob)
            logger.info(f"[CALIB] Temperature scaling fitted: T={temp:.4f}")
        else:
            self.temperature_scaler = None

        logger.info("\n[OK] Model training completed successfully")

    def calibration_assessment(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calibration assessment (no correction).
        Computes:
        - Brier score
        - Expected Calibration Error (ECE)
        - Reliability curve points (prob_pred, prob_true)

        NOTE:
        - This is an evaluation-only diagnostic. It does not change model outputs.
        """
        y_true = y_true.astype(int).ravel()
        y_prob = y_prob.astype(float).ravel()

        # Brier score (lower is better)
        brier = float(brier_score_loss(y_true, y_prob))

        # Calibration curve (reliability diagram data) — quantile binning
        # ensures roughly equal samples per bin, more robust when predictions
        # cluster near 0 or 1 (common for well-trained binary classifiers)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

        # ECE (Expected Calibration Error) using quantile bins
        bins = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
        bins[0] = 0.0
        bins[-1] = 1.0
        bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
        ece = 0.0
        total = len(y_prob)

        for b in range(n_bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            bin_prob_mean = float(np.mean(y_prob[mask]))
            bin_acc = float(np.mean(y_true[mask]))
            weight = float(np.sum(mask)) / total
            ece += weight * abs(bin_acc - bin_prob_mean)

        return {
            "brier_score": brier,
            "ece": float(ece),
            "calibration_curve": {
                "prob_pred": prob_pred.tolist(),
                "prob_true": prob_true.tolist(),
                "n_bins": int(n_bins)
            }
        }

    def _evaluate_confidence_based(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
        """
        Evaluate using 3-tier confidence-based prediction rules.

        IMPORTANT: y_prob is the model's predicted probability of Class 1 (Healthy).
        - HIGH y_prob (e.g., 0.9) = High confidence patient is HEALTHY
        - LOW y_prob (e.g., 0.1) = Low confidence patient is healthy = High confidence PRE-DIABETES

        Zones (correctly mapped to model output):
        - High Confidence Pre-Diabetes: prob < confidence_low_threshold (low prob of healthy = high risk)
        - Uncertain (needs OGTT): confidence_low_threshold <= prob < confidence_high_threshold (middle range)
        - High Confidence Healthy: prob >= confidence_high_threshold (high prob of healthy = low risk)

        Returns metrics for each zone and overall detection/false positive rates.
        """
        high_thr = self.config.confidence_high_threshold
        low_thr = self.config.confidence_low_threshold

        # Classify into 3 zones (CORRECTED: swapped to match model output semantics)
        high_conf_prediabetes_mask = (y_prob < low_thr)           # Low prob of healthy = Pre-Diabetes
        uncertain_mask = (y_prob >= low_thr) & (y_prob < high_thr)  # Middle range
        high_conf_healthy_mask = (y_prob >= high_thr)             # High prob of healthy = Healthy

        # Count samples in each zone
        n_high_conf_prediabetes = int(np.sum(high_conf_prediabetes_mask))
        n_uncertain = int(np.sum(uncertain_mask))
        n_high_conf_healthy = int(np.sum(high_conf_healthy_mask))
        n_total = len(y_true)

        # Ground truth: 0 = Pre-Diabetes, 1 = Healthy
        n_true_prediabetes = int(np.sum(y_true == 0))
        n_true_healthy = int(np.sum(y_true == 1))

        # Performance metrics for each zone
        # Zone 1: High Confidence Pre-Diabetes
        if n_high_conf_prediabetes > 0:
            zone1_true_prediabetes = int(np.sum((y_true == 0) & high_conf_prediabetes_mask))
            zone1_precision = zone1_true_prediabetes / n_high_conf_prediabetes
            zone1_recall = zone1_true_prediabetes / n_true_prediabetes if n_true_prediabetes > 0 else 0.0
        else:
            zone1_precision = 0.0
            zone1_recall = 0.0
            zone1_true_prediabetes = 0

        # Zone 2: Uncertain (we'll send to OGTT, so measure what % are actually pre-diabetes)
        if n_uncertain > 0:
            zone2_true_prediabetes = int(np.sum((y_true == 0) & uncertain_mask))
            zone2_prediabetes_rate = zone2_true_prediabetes / n_uncertain
            zone2_recall_contribution = zone2_true_prediabetes / n_true_prediabetes if n_true_prediabetes > 0 else 0.0
        else:
            zone2_true_prediabetes = 0
            zone2_prediabetes_rate = 0.0
            zone2_recall_contribution = 0.0

        # Zone 3: High Confidence Healthy
        if n_high_conf_healthy > 0:
            zone3_true_healthy = int(np.sum((y_true == 1) & high_conf_healthy_mask))
            zone3_precision = zone3_true_healthy / n_high_conf_healthy
            zone3_specificity = zone3_true_healthy / n_true_healthy if n_true_healthy > 0 else 0.0
            # False negatives in zone 3 (pre-diabetes classified as healthy)
            zone3_false_negatives = int(np.sum((y_true == 0) & high_conf_healthy_mask))
        else:
            zone3_precision = 0.0
            zone3_specificity = 0.0
            zone3_true_healthy = 0
            zone3_false_negatives = 0

        # Overall detection metrics
        # Total detected pre-diabetes (Zone 1 + Zone 2 OGTT would catch)
        total_detected_prediabetes = zone1_true_prediabetes + zone2_true_prediabetes
        prediabetes_detection_rate = (total_detected_prediabetes / n_true_prediabetes * 100) if n_true_prediabetes > 0 else 0.0

        # False positive rate (healthy classified as pre-diabetes in Zone 1)
        zone1_false_positives = int(np.sum((y_true == 1) & high_conf_prediabetes_mask))
        false_positive_rate = (zone1_false_positives / n_true_healthy * 100) if n_true_healthy > 0 else 0.0

        # OGTT burden (% of patients needing confirmatory testing)
        ogtt_burden = (n_uncertain / n_total * 100)

        return {
            # Overall counts
            'n_total': n_total,
            'n_true_prediabetes': n_true_prediabetes,
            'n_true_healthy': n_true_healthy,

            # Zone counts
            'n_high_conf_prediabetes': n_high_conf_prediabetes,
            'n_uncertain': n_uncertain,
            'n_high_conf_healthy': n_high_conf_healthy,
            'pct_high_conf_prediabetes': n_high_conf_prediabetes / n_total * 100,
            'pct_uncertain': n_uncertain / n_total * 100,
            'pct_high_conf_healthy': n_high_conf_healthy / n_total * 100,

            # Zone 1 metrics (High Conf Pre-Diabetes)
            'zone1_precision': zone1_precision * 100,
            'zone1_recall': zone1_recall * 100,
            'zone1_true_prediabetes': zone1_true_prediabetes,
            'zone1_false_positives': zone1_false_positives,

            # Zone 2 metrics (Uncertain - OGTT)
            'zone2_prediabetes_rate': zone2_prediabetes_rate * 100,
            'zone2_true_prediabetes': zone2_true_prediabetes,
            'zone2_recall_contribution': zone2_recall_contribution * 100,
            'ogtt_burden': ogtt_burden,

            # Zone 3 metrics (High Conf Healthy)
            'zone3_precision': zone3_precision * 100,
            'zone3_specificity': zone3_specificity * 100,
            'zone3_true_healthy': zone3_true_healthy,
            'zone3_false_negatives': zone3_false_negatives,

            # Overall metrics
            'prediabetes_detection_rate': prediabetes_detection_rate,
            'false_positive_rate': false_positive_rate,
            'immediate_decision_rate': (n_high_conf_prediabetes + n_high_conf_healthy) / n_total * 100,
        }

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Evaluate the trained model on the held-out test set.
            Applies the chosen decision threshold (validation-based by default)
            to convert probabilities into class predictions and computes:
            - Classification report (per-class precision/recall/F1).
            - Confusion matrix.
            - ROC AUC and ROC curve (FPR, TPR).
            - Precision-Recall curve.

            Returns a dictionary with all metrics and arrays.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Predict probabilities on TEST
        y_prob = self.model.predict(X_test, verbose=0).flatten()

        # Apply temperature scaling if available
        if self.temperature_scaler is not None:
            y_prob = self.temperature_scaler.transform(y_prob)
            logger.info(f"[CALIB] Applied temperature scaling (T={self.temperature_scaler.temperature:.4f})")

        thr = self.best_threshold if threshold is None else float(threshold)

        # --- Threshold sensitivity analysis (TEST-only diagnostic) ---
        thr_grid = np.linspace(
            float(getattr(self.config, "thr_sweep_min", 0.05)),
            float(getattr(self.config, "thr_sweep_max", 0.95)),
            int(getattr(self.config, "thr_sweep_points", 19))
        )
        accs, precs, recs, f1s, specs = [], [], [], [], []
        for t in thr_grid:
            yp = (y_prob >= t).astype(int)

            accs.append(float(accuracy_score(y_test, yp)))
            precs.append(float(precision_score(y_test, yp, zero_division=0)))
            recs.append(float(recall_score(y_test, yp, zero_division=0)))
            f1s.append(float(f1_score(y_test, yp, zero_division=0)))

            # Specificity = TN / (TN + FP)
            cm_t = confusion_matrix(y_test, yp)
            if cm_t.shape == (2, 2):
                tn, fp, fn, tp = cm_t.ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                spec = 0.0
            specs.append(float(spec))
        threshold_sensitivity = {
            "thresholds": [float(x) for x in thr_grid],
            "accuracy": accs,
            "precision": precs,
            "recall": recs,
            "f1": f1s,
            "specificity": specs
        }
        nearest_idx = int(np.argmin(np.abs(thr_grid - thr)))
        logger.info(
            f"[THR-SWEEP] Around chosen thr={thr:.3f} | "
            f"acc={accs[nearest_idx]:.3f}, prec={precs[nearest_idx]:.3f}, "
            f"rec={recs[nearest_idx]:.3f}, f1={f1s[nearest_idx]:.3f}, spec={specs[nearest_idx]:.3f}"
        )
        # end of threshold sensitivity analysis -------------------------------
        y_pred = (y_prob >= thr).astype(int)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        pr_auc = float(average_precision_score(y_test, y_prob))
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        logger.info("\n[EVAL] Model evaluation completed")
        logger.info(f"[METRICS] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Accuracy: {report['accuracy']:.4f} | Thr={thr:.4f}")

        # Calibration assessment
        calib = self.calibration_assessment(y_test, y_prob, n_bins=getattr(self.config, "calibration_bins", 10))
        logger.info(f"[CALIB] Brier={calib['brier_score']:.4f} | ECE={calib['ece']:.4f} | bins={calib['calibration_curve']['n_bins']}")

        # Confidence-based evaluation (if enabled)
        confidence_metrics = None
        if self.config.use_confidence_based_prediction:
            confidence_metrics = self._evaluate_confidence_based(y_test, y_prob, thr)
            logger.info("\n[CONFIDENCE] 3-Tier Confidence-Based Evaluation:")
            logger.info(f"  High Confidence Pre-Diabetes (≥{self.config.confidence_high_threshold:.2f}): {confidence_metrics['n_high_conf_prediabetes']} samples ({confidence_metrics['pct_high_conf_prediabetes']:.1f}%)")
            logger.info(f"  Uncertain (needs OGTT) [{self.config.confidence_low_threshold:.2f}-{self.config.confidence_high_threshold:.2f}): {confidence_metrics['n_uncertain']} samples ({confidence_metrics['pct_uncertain']:.1f}%)")
            logger.info(f"  High Confidence Healthy (<{self.config.confidence_low_threshold:.2f}): {confidence_metrics['n_high_conf_healthy']} samples ({confidence_metrics['pct_high_conf_healthy']:.1f}%)")
            logger.info(f"[CONFIDENCE] Pre-Diabetes Detection Rate: {confidence_metrics['prediabetes_detection_rate']:.1f}% (across all zones)")
            logger.info(f"[CONFIDENCE] False Positive Rate: {confidence_metrics['false_positive_rate']:.1f}%")

        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'y_pred_proba': y_prob,
            'y_pred': y_pred,
            'threshold_used': thr,
            "calibration": calib,
            'threshold_sensitivity': threshold_sensitivity,
            'confidence_metrics': confidence_metrics,
        }

    def save_model_and_history(self) -> None:
        """Persist trained model, training history, and configuration to disk.
            Saves:
            - training_history.json: per-epoch metrics.
            - model_config.json: configuration snapshot (features, params, etc.).
            - best_model.keras: Keras model file monitored by val_auc.
            - lstm_binary_model.keras: human-friendly alias copy of the best model.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Save training history if available
        if self.history is not None:
            hist_path = os.path.join(self.config.models_dir, 'training_history.json')
            hist_json = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
            with open(hist_path, 'w') as f:
                json.dump(hist_json, f, indent=4)
            
        # Serialize configuration for reproducibility
        cfg_path = os.path.join(self.config.models_dir, 'model_config.json')
        # Build architecture dict depending on which mode was used
        arch_cfg = {
            'use_conv_frontend': self.config.use_conv_frontend,
            'dropout_rate': self.config.dropout_rate,
        }
        if self.config.use_conv_frontend:
            arch_cfg.update({
                'conv_filters_1': self.config.conv_filters_1,
                'conv_kernel_1': self.config.conv_kernel_1,
                'conv_pool_1': self.config.conv_pool_1,
                'conv_filters_2': self.config.conv_filters_2,
                'conv_kernel_2': self.config.conv_kernel_2,
                'conv_pool_2': self.config.conv_pool_2,
                'lstm_bidirectional': self.config.lstm_bidirectional,
                'l2_reg': self.config.l2_reg,
            })
        else:
            arch_cfg.update({
                'lstm_units_1': self.config.lstm_units_1,
                'lstm_units_2': self.config.lstm_units_2,
                'dense_units': self.config.dense_units,
            })

        cfg = {
            'version_folder': self.config.version_folder,
            'timestamp': datetime.now().isoformat(),
            'features': self.config.selected_features,
            'binary_features': self.config.binary_features,
            'model_architecture': arch_cfg,
            'training_params': {
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs,
                'test_size': self.config.test_size,
                'val_size_from_train': self.config.val_size_from_train,
                'sequence_length': self.config.expected_sequence_length,
                'use_smote': self.config.use_smote,
                'use_class_weights': self.config.use_class_weights,
                'clipnorm': self.config.clipnorm,
                'val_auc': self.val_auc,
                'val_threshold': self.best_threshold,
            },
            'random_seed': self.config.random_seed,
        }
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=4)
        # Alias copy
        from shutil import copyfile
        best = os.path.join(self.config.models_dir, 'best_model.keras')
        alias = os.path.join(self.config.models_dir, 'lstm_binary_model.keras')
        if os.path.exists(best):
            copyfile(best, alias)
            logger.info(f"[SAVE] Copied best model → alias: {best} → {alias}")
        logger.info(f"[SAVE] Artifacts saved to {self.config.models_dir}")

    def multicollinearity_diagnostics(self, X_train: np.ndarray) -> Dict[str, Any]:
        """
        Train-only diagnostic: detects redundant features using correlation + VIF.
        We flatten (participants × time) to compute feature-wise relationships.
        This is NOT used to modify training automatically; it only logs findings.

        Why here (and not before splits)?
        - Using TRAIN only avoids test leakage.
        - After scaling, correlation comparisons are stable across features.
        """
        binary = set(self.config.binary_features)
        feat_names = self.config.selected_features
        cont_idx = [i for i, n in enumerate(feat_names) if n not in binary]

        if len(cont_idx) < 2:
            logger.info("[COLLINEAR] Not enough continuous features to analyze.")
            return {}

        # Flatten sequences across time (train only)
        X2d = X_train[:, :, cont_idx].reshape(-1, len(cont_idx))

        # Remove any rows with NaN/inf to avoid breaking stats
        mask = np.isfinite(X2d).all(axis=1)
        X2d = X2d[mask]
        cont_names = [feat_names[i] for i in cont_idx]

        if X2d.shape[0] < 50:
            logger.warning("[COLLINEAR] Too few flattened rows for stable diagnostics.")
            return {}

        # Correlation matrix
        corr = np.corrcoef(X2d, rowvar=False)
        corr_pairs = []
        thr = float(self.config.corr_threshold)

        for i in range(len(cont_names)):
            for j in range(i + 1, len(cont_names)):
                r = corr[i, j]
                if abs(r) >= thr:
                    corr_pairs.append((cont_names[i], cont_names[j], float(r)))

        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        # VIF (optional but useful)
        vif_rows = []
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            for k in range(len(cont_names)):
                vif = float(variance_inflation_factor(X2d, k))
                vif_rows.append((cont_names[k], vif))
            vif_rows.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.warning(f"[COLLINEAR] VIF skipped (statsmodels missing or error): {e}")

        # Logging summary
        tag = getattr(self, "fold_tag", "")
        logger.info(f"\n{tag}[COLLINEAR] Train-only multicollinearity diagnostics")
        logger.info(f"{tag}[COLLINEAR] Continuous features analyzed: {cont_names}")

        if corr_pairs:
            logger.info(f"{tag}[COLLINEAR] High-correlation pairs (|r| ≥ {thr}): {min(len(corr_pairs), self.config.max_corr_pairs_to_log)} shown")
            for a, b, r in corr_pairs[: self.config.max_corr_pairs_to_log]:
                logger.info(f"{tag}[COLLINEAR]   {a} vs {b}: r={r:+.4f}")
        else:
            logger.info(f"{tag}[COLLINEAR] No pairs above |r| ≥ {thr}")

        if vif_rows:
            vt = float(self.config.vif_threshold)
            logger.info(f"{tag}[COLLINEAR] VIF values (threshold ≥ {vt} flagged)")
            for name, vif in vif_rows:
                flag = "  <-- HIGH" if vif >= vt else ""
                logger.info(f"{tag}[COLLINEAR]   {name}: VIF={vif:.3f}{flag}")
        
        # -------------------- SAVE ARTIFACTS (JSON + HEATMAP PNG) --------------------
        try:
            # Choose output directory:
            # - If fold_dir is provided (CV), save inside that fold folder.
            # - Else save to the main model directory.
            out_dir = getattr(self, "fold_dir", None) or self.config.models_dir
            os.makedirs(out_dir, exist_ok=True)

            # Decide filename suffix based on fold_tag (fallback to generic)
            fold_num = None
            tag_str = getattr(self, "fold_tag", "")
            # Extract fold number from strings like "[CV:FOLD 3/5] "
            if "FOLD" in tag_str:
                # very light parsing; safe if it fails
                try:
                    # "[CV:FOLD 3/5]" -> "3"
                    fold_num = tag_str.split("FOLD")[1].strip().split("/")[0].strip(" ]")
                except Exception:
                    fold_num = None

            suffix = f"_fold_{fold_num}" if fold_num is not None else ""
            json_path = os.path.join(out_dir, f"multicollinearity{suffix}.json")

            payload = {
                "continuous_features": cont_names,
                "corr_threshold": float(self.config.corr_threshold),
                "vif_threshold": float(self.config.vif_threshold),
                "corr_pairs": [{"a": a, "b": b, "r": float(r)} for a, b, r in corr_pairs],
                "vif": [{"feature": n, "vif": float(v)} for n, v in vif_rows],
            }
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"{tag}[COLLINEAR] Saved diagnostics JSON → {json_path}")

            # Optional: correlation heatmap (no seaborn)
            if bool(getattr(self.config, "save_collinearity_heatmap", True)):
                import matplotlib.pyplot as plt

                png_path = os.path.join(out_dir, f"multicollinearity_corr_heatmap{suffix}.png")
                plt.figure(figsize=(10, 8))
                plt.imshow(corr, aspect="auto")  # corr is the correlation matrix you already computed
                plt.colorbar()
                plt.xticks(range(len(cont_names)), cont_names, rotation=45, ha="right")
                plt.yticks(range(len(cont_names)), cont_names)
                plt.title("Train-only feature correlation (continuous features)")
                plt.tight_layout()
                plt.savefig(png_path, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info(f"{tag}[COLLINEAR] Saved correlation heatmap PNG → {png_path}")

        except Exception as e:
            logger.warning(f"{tag}[COLLINEAR] Failed to save artifacts: {e}")
        # ---------------------------------------------------------------------------
        
        return {
            "corr_pairs": corr_pairs,
            "vif": vif_rows,
            "cont_features": cont_names
        }
