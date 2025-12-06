"""
Model creation, training, threshold selection, and evaluation for the CGM LSTM pipeline.
"""

import os
import json
import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from .data import Config

logger = logging.getLogger(__name__)

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

    def create_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build and compile the LSTM model for binary classification.

            The architecture is:
            Input → LSTM(32, return_sequences=True) → Dropout
                → LSTM(16) → Dense(16, ReLU) → Dense(1, Sigmoid)
            Gradient clipping is enabled on the Adam optimizer for stability.
        """
        model = Sequential([
            Input(shape=input_shape),
            LSTM(self.config.lstm_units_1, return_sequences=True),
            Dropout(self.config.dropout_rate),
            LSTM(self.config.lstm_units_2),
            Dense(self.config.dense_units, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        # Minimal stability tweak: gradient clipping
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=self.config.clipnorm)
        metrics = [
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        model.compile(loss='binary_crossentropy', optimizer=opt,
                        metrics=metrics)
        logger.info("\n[MODEL] Model architecture created successfully")
        logger.info(f"[DATA]   Input shape: {input_shape}")
        logger.info(f"[DATA]   Total parameters: {model.count_params():,}")
        return model

    def prepare_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create Train/Val/Test splits and apply scaling / imbalance handling.

            Steps:
            1. Split into Train and Test using stratified sampling.
            2. Split a Validation set from the Train portion (also stratified).
            3. Optionally compute class weights from Train labels.
            4. Optionally apply SMOTE only on Train (flattened + reshaped).
            5. Fit a standardization transform on Train continuous features only,
            and apply it to Train, Val, and Test.

            Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"[DATA] Dataset: {len(y)} samples, classes: {dict(zip(unique, counts))}")
        if len(unique) < 2:
            raise ValueError("Need at least 2 classes for binary classification.")

        idx = np.arange(len(y))
        # Stage 1: Train/Test
        X_tr_all, X_test, y_tr_all, y_test, idx_tr_all, idx_test = train_test_split(
            X, y, idx,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=y
        )
        # Stage 2: Val from Train only (preserves test as untouched hold-out)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.config.val_size_from_train, random_state=self.config.random_seed)
        for tr_idx, val_idx in sss.split(X_tr_all, y_tr_all):
            X_train, X_val = X_tr_all[tr_idx], X_tr_all[val_idx]
            y_train, y_val = y_tr_all[tr_idx], y_tr_all[val_idx]
        
        # Save indices for reproducible mapping back to original participants
        np.save(os.path.join(self.config.models_dir, 'idx_train.npy'), idx_tr_all[tr_idx])
        np.save(os.path.join(self.config.models_dir, 'idx_val.npy'),   idx_tr_all[val_idx])
        np.save(os.path.join(self.config.models_dir, 'idx_test.npy'),  idx_test)
        logger.info(f"[SPLIT] Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
        logger.info(f"[DATA]   Train dist: {np.bincount(y_train)} | Val dist: {np.bincount(y_val)} | Test dist: {np.bincount(y_test)}")

        # Class weights (computed on TRAIN labels only)
        if self.config.use_class_weights:
            classes = np.unique(y_train)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            self.class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
            logger.info(f"[CLASS WEIGHTS] {self.class_weights}")
        else:
            self.class_weights = None

        # Optional SMOTE (TRAIN only; discouraged for true sequence modeling)
        if self.config.use_smote:
            if len(np.unique(y_train)) >= 2:
                logger.info("\n[SMOTE] Applying SMOTE to TRAIN (flatten→resample→reshape)...")
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

        # Standardize continuous features; keep binary flags (0/1) as is
        binary = {'is_meal_time', 'is_night'}
        cont_idx = [i for i,n in enumerate(self.config.selected_features) if n not in binary]

        # Fit scaler using TRAIN only
        train_2d = X_train[:, :, cont_idx].reshape(-1, len(cont_idx))
        mu = train_2d.mean(0).astype(np.float32)
        sigma = train_2d.std(0).astype(np.float32)
        sigma[sigma < 1e-8] = 1.0       # avoid division by very small numbers

        def _apply(z):
            """Apply train-fitted standardization to a 3D array."""
            z = z.copy()
            z[:, :, cont_idx] = (z[:, :, cont_idx] - mu) / sigma
            return z

        # Apply scaling to all splits using train-based statistics
        X_train, X_val, X_test = _apply(X_train), _apply(X_val), _apply(X_test)

        # Persist scaler parameters for future inference / analysis
        np.savez(
                    os.path.join(self.config.models_dir, 'feature_scaler.npz'),
                    mean=mu, 
                    std=sigma, 
                    cont_features=np.array([self.config.selected_features[i] for i in cont_idx], dtype=object)
                )
        logger.info("[SCALE] Train-only standardization applied (continuous features only).")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def compute_val_threshold(self, X_val: np.ndarray, y_val: np.ndarray, method: str = "youden") -> float:        
        """Select a probability threshold using only the validation set.
            The model outputs probabilities in [0,1]. This method chooses a fixed
            decision threshold based on validation data only so that the test set
            remains completely untouched.

            Supported methods:
            - "youden": maximize Youden's J statistic (TPR - FPR).
            - "fpr_le_5": choose the largest threshold with FPR <= 5%.

            Returns the chosen threshold and stores it in `self.best_threshold`.
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
        return self.best_threshold

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
        self.compute_val_threshold(X_val, y_val, method="youden")
        logger.info("\n[OK] Model training completed successfully")

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
        thr = self.best_threshold if threshold is None else float(threshold)
        y_pred = (y_prob >= thr).astype(int)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        logger.info("\n[EVAL] Model evaluation completed")
        logger.info(f"[METRICS] ROC AUC: {roc_auc:.4f} | Accuracy: {report['accuracy']:.4f} | Thr(Test)={thr:.4f}")

        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'y_pred_proba': y_prob,
            'y_pred': y_pred,
            'threshold_used': thr
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
        cfg = {
            'version_folder': self.config.version_folder,
            'timestamp': datetime.now().isoformat(),
            'features': self.config.selected_features,
            'model_architecture': {
                'lstm_units_1': self.config.lstm_units_1,
                'lstm_units_2': self.config.lstm_units_2,
                'dense_units': self.config.dense_units,
                'dropout_rate': self.config.dropout_rate
            },
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
                'val_threshold': self.best_threshold
            },
            'random_seed': self.config.random_seed
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
