"""
Production Model Analyzer v3
=============================

Post-hoc analysis of trained CGM LSTM production models.
Aligned with the cgm_lstm pipeline (Conv+BiLSTM, production_model/ artifacts,
held-out test set, temperature scaling, confidence-based 3-tier zones).

Capabilities:
  - Loads production model artifacts (best_model.keras, scaler, threshold, temperature, confidence)
  - Comprehensive evaluation (ROC-AUC, PR-AUC, MCC, F1, balanced accuracy, clinical metrics)
  - Confidence-based 3-tier evaluation (reuses LSTMTrainer._evaluate_confidence_based)
  - Calibration assessment (Brier score, ECE, reliability curve)
  - 5 feature importance methods (Permutation, Gradient, Variance, SHAP, LIME)
  - Group analysis by confidence zones
  - Temporal pattern analysis (reconstructs hour from sin/cos)
  - 12 publication-ready plots (saved to analysis/ subfolder, no plt.show())
  - 4 research CSV tables
  - JSON analysis summary

Usage:
  python "time_series_lstm_analyze_model v3.py"                        # auto-detect latest model
  python "time_series_lstm_analyze_model v3.py" model_YYYYMMDD_HHMMSS  # specific model

Author: CGM LSTM Pipeline
Date: 2026-02-08
"""

# Force non-interactive matplotlib backend BEFORE any pyplot import
import matplotlib
matplotlib.use('Agg')

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # match training numerics

import sys
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc, classification_report, confusion_matrix,
    precision_recall_curve, matthews_corrcoef, f1_score,
    balanced_accuracy_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

import tensorflow as tf
from tensorflow.keras.models import load_model

# Reuse pipeline components
from cgm_lstm.data import Config
from cgm_lstm.model import TemperatureScaling, LSTMTrainer
from cgm_lstm.viz import Visualizer

# Optional libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from paths import LOG_DIR

# ---------------------------------------------------------------------------
# Basic logging setup (will be enhanced per-analysis in ProductionModelAnalyzer)
# ---------------------------------------------------------------------------
__CURR_FILE__ = "time_series_lstm_analyze_model v3"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console only for now
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Helper Classes for Comprehensive Logging
# ===========================================================================
class TeeOutput:
    """Redirect stdout to both console and log file"""
    def __init__(self, file_path, original_stdout):
        self.file = open(file_path, 'a', encoding='utf-8')
        self.stdout = original_stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        if not self.file.closed:
            self.file.flush()


def setup_comprehensive_logging(analysis_dir: str) -> str:
    """Set up comprehensive logging to analysis directory (captures both logging and print).

    Returns:
        str: Path to the log file
    """
    log_file_path = os.path.join(analysis_dir, __CURR_FILE__ + ".log")

    # Clear the log file at the start
    with open(log_file_path, 'w', encoding='utf-8') as f:
        pass  # Just clear it

    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Redirect stdout to capture print() statements
    sys.stdout = TeeOutput(log_file_path, sys.stdout)

    logger.info(f"Logs and prints will be saved to: {log_file_path}")
    return log_file_path


def cleanup_comprehensive_logging():
    """Clean up logging by restoring original stdout and closing log file."""
    if hasattr(sys.stdout, 'file') and hasattr(sys.stdout, 'stdout'):
        original_stdout = sys.stdout.stdout
        sys.stdout.file.close()
        sys.stdout = original_stdout


# ===========================================================================
# ProductionModelAnalyzer
# ===========================================================================
class ProductionModelAnalyzer:
    """Load-once, analyze-many class for post-hoc production model analysis.

    All artifacts are loaded and validated in __init__. Analysis methods
    operate on preloaded state so they can be called in any order.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, models_base_dir: str = './models', specific_model_folder: Optional[str] = None):
        self.models_base_dir = models_base_dir

        # Discover model folder
        self.models_dir, self.model_folder_name = self._discover_model_folder(specific_model_folder)

        # Analysis output directory
        self.analysis_dir = os.path.join(self.models_dir, "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)

        # Set up comprehensive logging immediately so all subsequent logs are captured
        setup_comprehensive_logging(self.analysis_dir)

        # Feature configuration from Config defaults (no __post_init__ side effects)
        self.feature_names: List[str] = [
            'glucose_accel', 'cos_hour', 'blood_glucose_value',
            'glucose_rollmean_1h', 'glucose_rollstd_1h', 'glucose_diff',
            'sin_hour', 'is_meal_time', 'is_night',
        ]
        self.binary_features: List[str] = ['is_meal_time', 'is_night']

        # Load all production artifacts
        self._load_production_artifacts()
        self._load_data_and_splits()
        self._validate_artifacts()

        # Cached results (populated by evaluation methods)
        self.y_pred_proba: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None

        logger.info(f"[INIT] ProductionModelAnalyzer ready for: {self.model_folder_name}")
        logger.info(f"[INIT] Analysis outputs will be saved to: {self.analysis_dir}")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _discover_model_folder(self, specific_folder: Optional[str]) -> Tuple[str, str]:
        """Discover model folder (latest or specified)."""
        if specific_folder:
            model_dir = os.path.join(self.models_base_dir, specific_folder)
            if not os.path.exists(model_dir):
                raise ValueError(f"Specified model folder does not exist: {model_dir}")
            name = specific_folder
            logger.info(f"[DISCOVER] Using specified model folder: {name}")
        else:
            folders = sorted(
                [f for f in os.listdir(self.models_base_dir)
                 if f.startswith('model_') and os.path.isdir(os.path.join(self.models_base_dir, f))],
                reverse=True,
            )
            if not folders:
                raise ValueError(f"No model folders found in {self.models_base_dir}")
            name = folders[0]
            model_dir = os.path.join(self.models_base_dir, name)
            logger.info(f"[DISCOVER] Auto-detected latest model folder: {name}")

        # Verify production_model/ subfolder exists
        prod_dir = os.path.join(model_dir, "production_model")
        if not os.path.isdir(prod_dir):
            raise FileNotFoundError(
                f"production_model/ subfolder not found in {model_dir}. "
                "This model was not trained with the production model pipeline."
            )

        return model_dir, name

    # ------------------------------------------------------------------
    # Load production artifacts
    # ------------------------------------------------------------------
    def _load_production_artifacts(self) -> None:
        """Load all 6 production artifacts from production_model/ subfolder."""
        prod_dir = os.path.join(self.models_dir, "production_model")

        # 1. Keras model
        model_path = os.path.join(prod_dir, "best_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = load_model(model_path)
        logger.info(f"[LOAD] Keras model loaded: {model_path}")

        # 2. Global threshold
        thr_path = os.path.join(prod_dir, "global_threshold.json")
        if os.path.exists(thr_path):
            with open(thr_path) as f:
                self.threshold = float(json.load(f)['global_threshold'])
            logger.info(f"[LOAD] Global threshold: {self.threshold:.4f}")
        else:
            self.threshold = 0.5
            logger.warning("[LOAD] global_threshold.json not found, using default 0.5")

        # 3. Temperature scaler
        temp_path = os.path.join(prod_dir, "temperature_scaler.json")
        if os.path.exists(temp_path):
            with open(temp_path) as f:
                temp_val = float(json.load(f)['temperature'])
            self.temperature_scaler = TemperatureScaling()
            self.temperature_scaler.temperature = temp_val
            logger.info(f"[LOAD] Temperature scaler: T={temp_val:.4f}")
        else:
            self.temperature_scaler = None
            logger.warning("[LOAD] temperature_scaler.json not found, no temperature scaling")

        # 4. Confidence thresholds
        conf_path = os.path.join(prod_dir, "confidence_thresholds.json")
        if os.path.exists(conf_path):
            with open(conf_path) as f:
                conf_data = json.load(f)
            self.confidence_low_threshold = float(conf_data['confidence_low_threshold'])
            self.confidence_high_threshold = float(conf_data['confidence_high_threshold'])
            logger.info(f"[LOAD] Confidence thresholds: low={self.confidence_low_threshold:.4f}, "
                        f"high={self.confidence_high_threshold:.4f}")
        else:
            self.confidence_low_threshold = 0.35
            self.confidence_high_threshold = 0.65
            logger.warning("[LOAD] confidence_thresholds.json not found, using defaults")

        # 5. Feature scaler
        scaler_path = os.path.join(prod_dir, "feature_scaler.npz")
        if os.path.exists(scaler_path):
            pack = np.load(scaler_path, allow_pickle=True)
            self.feature_scaler_mean = pack['mean']
            # Production scaler uses 'scale' key (not 'std')
            self.feature_scaler_scale = pack['scale'] if 'scale' in pack else pack.get('std', None)
            if self.feature_scaler_scale is None:
                raise KeyError("Feature scaler missing 'scale' or 'std' key")
            logger.info(f"[LOAD] Feature scaler: mean shape={self.feature_scaler_mean.shape}, "
                        f"scale shape={self.feature_scaler_scale.shape}")
        else:
            raise FileNotFoundError(f"Feature scaler not found: {scaler_path}")

        # 6. Training config
        cfg_path = os.path.join(prod_dir, "training_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                self.training_config = json.load(f)
            logger.info(f"[LOAD] Training config loaded: {cfg_path}")
        else:
            self.training_config = {}
            logger.warning("[LOAD] training_config.json not found")

    # ------------------------------------------------------------------
    # Load data and splits
    # ------------------------------------------------------------------
    def _load_data_and_splits(self) -> None:
        """Load numpy arrays and held-out test indices."""
        # Full data arrays (from root of model folder)
        x_path = os.path.join(self.models_dir, "X_lstm.npy")
        y_path = os.path.join(self.models_dir, "y_lstm_binary.npy")
        pid_path = os.path.join(self.models_dir, "pid_lstm.npy")

        for p, name in [(x_path, 'X_lstm.npy'), (y_path, 'y_lstm_binary.npy')]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required data file not found: {p}")

        self.X_all = np.load(x_path)
        self.y_all = np.load(y_path)
        self.pid_all = np.load(pid_path) if os.path.exists(pid_path) else None
        self.timesteps = self.X_all.shape[1]
        self.n_features = self.X_all.shape[2]

        logger.info(f"[LOAD] Full data: X={self.X_all.shape}, y={self.y_all.shape}")

        # Held-out test indices
        held_out_idx_path = os.path.join(self.models_dir, "held_out_test", "held_out_test_indices.npy")
        if os.path.exists(held_out_idx_path):
            self.held_out_indices = np.load(held_out_idx_path)
            self.X_test = self.X_all[self.held_out_indices]
            self.y_test = self.y_all[self.held_out_indices]
            self.pid_test = self.pid_all[self.held_out_indices] if self.pid_all is not None else None
            logger.info(f"[LOAD] Held-out test set: {len(self.held_out_indices)} samples "
                        f"(Pre-D={np.sum(self.y_test == 0)}, CGM-Healthy={np.sum(self.y_test == 1)})")
        else:
            logger.warning("[LOAD] held_out_test_indices.npy not found; using full dataset as test")
            self.held_out_indices = np.arange(len(self.X_all))
            self.X_test = self.X_all
            self.y_test = self.y_all
            self.pid_test = self.pid_all

        # CV results
        cv_path = os.path.join(self.models_dir, "cross_validation", "cv_results.json")
        if os.path.exists(cv_path):
            with open(cv_path) as f:
                self.cv_results = json.load(f)
            logger.info(f"[LOAD] CV results loaded: {cv_path}")
        else:
            self.cv_results = None
            logger.warning("[LOAD] cv_results.json not found")

        # Held-out results (for comparison)
        ho_path = os.path.join(self.models_dir, "held_out_test", "held_out_test_results.json")
        if os.path.exists(ho_path):
            with open(ho_path) as f:
                self.held_out_results = json.load(f)
            logger.info(f"[LOAD] Held-out results loaded: {ho_path}")
        else:
            self.held_out_results = None
            logger.warning("[LOAD] held_out_test_results.json not found")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_artifacts(self) -> None:
        """Validate consistency across loaded artifacts."""
        errors = []

        # Feature count
        if self.n_features != len(self.feature_names):
            errors.append(
                f"Feature count mismatch: X has {self.n_features} features, "
                f"expected {len(self.feature_names)}"
            )

        # Scaler shape (should match number of continuous features)
        binary_set = set(self.binary_features)
        n_cont = sum(1 for f in self.feature_names if f not in binary_set)
        if self.feature_scaler_mean.shape[0] != n_cont:
            errors.append(
                f"Scaler shape mismatch: scaler has {self.feature_scaler_mean.shape[0]} values, "
                f"expected {n_cont} continuous features"
            )

        # Held-out indices within bounds
        if self.held_out_indices is not None and len(self.held_out_indices) > 0:
            if np.max(self.held_out_indices) >= len(self.X_all):
                errors.append(
                    f"Held-out indices out of bounds: max={np.max(self.held_out_indices)}, "
                    f"data size={len(self.X_all)}"
                )

        if errors:
            for e in errors:
                logger.error(f"[VALIDATE] {e}")
            raise ValueError(f"Artifact validation failed: {errors}")

        logger.info(f"[VALIDATE] All artifacts validated successfully")
        logger.info(f"[VALIDATE] Timesteps={self.timesteps}, Features={self.n_features}, "
                    f"Test samples={len(self.X_test)}")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _apply_feature_scaling(self, X: np.ndarray) -> np.ndarray:
        """Apply production feature scaler to data.

        Scales continuous features only; binary features are left untouched.
        Uses scaler from production_model/feature_scaler.npz.
        """
        binary_set = set(self.binary_features)
        cont_idx = [i for i, name in enumerate(self.feature_names) if name not in binary_set]

        X_scaled = X.copy()
        X_scaled[:, :, cont_idx] = (
            (X_scaled[:, :, cont_idx] - self.feature_scaler_mean) / self.feature_scaler_scale
        )
        return X_scaled

    def _apply_temperature_scaling(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to predicted probabilities."""
        if self.temperature_scaler is not None:
            return self.temperature_scaler.transform(y_prob)
        return y_prob

    def _get_predictions(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get calibrated predictions: raw predict -> temperature scale."""
        y_prob = self.model.predict(X_scaled, verbose=0).ravel()
        y_prob = self._apply_temperature_scaling(y_prob)
        return y_prob

    # ------------------------------------------------------------------
    # Evaluation methods
    # ------------------------------------------------------------------
    def evaluate_comprehensive(self) -> Dict[str, Any]:
        """Standard + research-grade evaluation on the held-out test set."""
        logger.info("[EVAL] Starting comprehensive evaluation...")

        # Scale and predict
        X_scaled = self._apply_feature_scaling(self.X_test)
        y_prob = self._get_predictions(X_scaled)
        y_pred = (y_prob >= self.threshold).astype(int)

        # Cache for downstream use
        self.y_pred_proba = y_prob
        self.y_pred = y_pred

        y_test = self.y_test.ravel()

        # Standard metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = float(auc(fpr, tpr))
        precision_c, recall_c, _ = precision_recall_curve(y_test, y_prob)
        pr_auc_val = float(average_precision_score(y_test, y_prob))

        # Research-grade metrics
        mcc = float(matthews_corrcoef(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        balanced_acc = float(balanced_accuracy_score(y_test, y_pred))

        # Clinical metrics from confusion matrix
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Threshold sensitivity sweep
        thr_grid = np.linspace(0.05, 0.95, 19)
        sweep = {'thresholds': [], 'accuracy': [], 'precision': [],
                 'recall': [], 'f1': [], 'specificity': []}
        for t in thr_grid:
            yp = (y_prob >= t).astype(int)
            sweep['thresholds'].append(float(t))
            sweep['accuracy'].append(float(accuracy_score(y_test, yp)))
            sweep['precision'].append(float(precision_score(y_test, yp, zero_division=0)))
            sweep['recall'].append(float(recall_score(y_test, yp, zero_division=0)))
            sweep['f1'].append(float(f1_score(y_test, yp, zero_division=0)))
            cm_t = confusion_matrix(y_test, yp)
            if cm_t.shape == (2, 2):
                tn_t, fp_t, _, _ = cm_t.ravel()
                sweep['specificity'].append(float(tn_t / (tn_t + fp_t)) if (tn_t + fp_t) > 0 else 0.0)
            else:
                sweep['specificity'].append(0.0)

        results = {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc_val,
            'pr_auc': pr_auc_val,
            'mcc': mcc,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_c,
            'recall_curve': recall_c,
            'y_pred_proba': y_prob,
            'y_pred': y_pred,
            'threshold_used': self.threshold,
            'threshold_sensitivity': sweep,
        }

        logger.info(f"[EVAL] ROC-AUC={roc_auc_val:.4f} | PR-AUC={pr_auc_val:.4f} | "
                    f"MCC={mcc:.4f} | Balanced Acc={balanced_acc:.4f} | Thr={self.threshold:.4f}")

        return results

    def evaluate_confidence_based(self) -> Optional[Dict[str, Any]]:
        """3-tier confidence-based evaluation reusing LSTMTrainer logic."""
        if self.y_pred_proba is None:
            logger.warning("[CONFIDENCE] Run evaluate_comprehensive() first")
            return None

        logger.info("[CONFIDENCE] Running 3-tier confidence-based evaluation...")

        # Lightweight reuse of LSTMTrainer._evaluate_confidence_based
        temp_config = object.__new__(Config)
        temp_config.confidence_low_threshold = self.confidence_low_threshold
        temp_config.confidence_high_threshold = self.confidence_high_threshold

        temp_trainer = object.__new__(LSTMTrainer)
        temp_trainer.config = temp_config

        result = temp_trainer._evaluate_confidence_based(
            self.y_test.ravel(), self.y_pred_proba, self.threshold
        )

        logger.info(f"[CONFIDENCE] Zone 1 (Pre-D): {result['n_high_conf_prediabetes']} samples "
                    f"({result['pct_high_conf_prediabetes']:.1f}%)")
        logger.info(f"[CONFIDENCE] Zone 2 (Uncertain): {result['n_uncertain']} samples "
                    f"({result['pct_uncertain']:.1f}%)")
        logger.info(f"[CONFIDENCE] Zone 3 (CGM-Healthy): {result['n_high_conf_healthy']} samples "
                    f"({result['pct_high_conf_healthy']:.1f}%)")
        logger.info(f"[CONFIDENCE] Detection Rate={result['prediabetes_detection_rate']:.1f}% | "
                    f"FPR={result['false_positive_rate']:.1f}% | "
                    f"OGTT Burden={result['ogtt_burden']:.1f}%")

        return result

    def evaluate_calibration(self) -> Optional[Dict[str, Any]]:
        """Calibration assessment reusing LSTMTrainer logic."""
        if self.y_pred_proba is None:
            logger.warning("[CALIB] Run evaluate_comprehensive() first")
            return None

        logger.info("[CALIB] Running calibration assessment...")

        temp_config = object.__new__(Config)
        temp_config.calibration_bins = 10

        temp_trainer = object.__new__(LSTMTrainer)
        temp_trainer.config = temp_config

        result = temp_trainer.calibration_assessment(
            self.y_test.ravel(), self.y_pred_proba, n_bins=10
        )

        logger.info(f"[CALIB] Brier={result['brier_score']:.4f} | ECE={result['ece']:.4f}")
        return result

    # ------------------------------------------------------------------
    # Feature importance methods
    # ------------------------------------------------------------------
    def compute_permutation_importance(self, n_repeats: int = 5, n_samples: int = 100) -> np.ndarray:
        """Permutation-based feature importance with statistical stability.

        Repeats permutation n_repeats times and reports mean +/- std.
        """
        logger.info(f"[IMPORTANCE] Computing permutation importance (n_repeats={n_repeats})...")

        X_scaled = self._apply_feature_scaling(self.X_test)
        X_subset = X_scaled[:n_samples]
        y_subset = self.y_test[:n_samples].ravel()

        baseline_prob = self._get_predictions(X_subset)
        baseline_auc = float(auc(*roc_curve(y_subset, baseline_prob)[:2]))

        all_importances = np.zeros((n_repeats, len(self.feature_names)))
        rng = np.random.RandomState(42)

        for rep in range(n_repeats):
            for i in range(len(self.feature_names)):
                X_perm = X_subset.copy()
                perm_idx = rng.permutation(X_perm.shape[0])
                X_perm[:, :, i] = X_perm[perm_idx, :, i]

                perm_prob = self.model.predict(X_perm, verbose=0).ravel()
                perm_prob = self._apply_temperature_scaling(perm_prob)
                perm_auc = float(auc(*roc_curve(y_subset, perm_prob)[:2]))

                all_importances[rep, i] = baseline_auc - perm_auc

        self._perm_importance_mean = all_importances.mean(axis=0)
        self._perm_importance_std = all_importances.std(axis=0)

        logger.info("[IMPORTANCE] Permutation importance completed")
        return self._perm_importance_mean

    def compute_gradient_importance(self, n_samples: int = 50) -> np.ndarray:
        """Gradient-based feature importance via GradientTape."""
        logger.info("[IMPORTANCE] Computing gradient-based importance...")

        X_scaled = self._apply_feature_scaling(self.X_test)
        X_subset = X_scaled[:n_samples]

        X_tensor = tf.convert_to_tensor(X_subset, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
            loss = tf.reduce_mean(predictions)

        gradients = tape.gradient(loss, X_tensor)
        feature_importance = tf.reduce_mean(tf.abs(gradients), axis=[0, 1]).numpy()

        logger.info("[IMPORTANCE] Gradient importance completed")
        return feature_importance

    def compute_variance_importance(self, n_samples: int = 100) -> np.ndarray:
        """Variance-based importance using prediction variance change."""
        logger.info("[IMPORTANCE] Computing variance-based importance...")

        X_scaled = self._apply_feature_scaling(self.X_test)
        X_subset = X_scaled[:n_samples]

        baseline_preds = self._get_predictions(X_subset)
        baseline_var = np.var(baseline_preds)

        importance_scores = []
        rng = np.random.RandomState(42)

        for i in range(len(self.feature_names)):
            X_noisy = X_subset.copy()
            noise = rng.normal(0, np.std(X_subset[:, :, i]) * 0.1, X_subset[:, :, i].shape)
            X_noisy[:, :, i] += noise

            noisy_preds = self.model.predict(X_noisy, verbose=0).ravel()
            noisy_preds = self._apply_temperature_scaling(noisy_preds)
            noisy_var = np.var(noisy_preds)

            importance_scores.append(abs(baseline_var - noisy_var))

        logger.info("[IMPORTANCE] Variance importance completed")
        return np.array(importance_scores)

    def compute_shap_importance(self, n_samples: int = 10) -> Optional[np.ndarray]:
        """SHAP analysis with KernelExplainer."""
        if not SHAP_AVAILABLE:
            logger.warning("[IMPORTANCE] SHAP not available (install shap package)")
            return None

        try:
            logger.info("[IMPORTANCE] Computing SHAP importance (KernelExplainer)...")

            X_scaled = self._apply_feature_scaling(self.X_test)
            X_subset = X_scaled[:n_samples]
            T = self.timesteps

            def model_predict(X):
                if len(X.shape) == 2:
                    X = X.reshape(-1, T, len(self.feature_names))
                preds = self.model.predict(X, verbose=0).ravel()
                return self._apply_temperature_scaling(preds)

            bg_size = min(3, len(X_subset))
            background = X_subset[:bg_size].reshape(bg_size, -1)

            explainer = shap.KernelExplainer(model_predict, background)

            test_size = min(3, len(X_subset))
            test_flat = X_subset[:test_size].reshape(test_size, -1)
            shap_values = explainer.shap_values(test_flat, nsamples=30)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_reshaped = shap_values.reshape(-1, T, len(self.feature_names))
            feature_importance = np.abs(shap_reshaped).mean(axis=0).mean(axis=0)

            logger.info("[IMPORTANCE] SHAP importance completed")
            return feature_importance

        except Exception as e:
            logger.error(f"[IMPORTANCE] SHAP failed: {e}")
            return None

    def compute_lime_importance(self, n_samples: int = 5) -> Optional[np.ndarray]:
        """LIME analysis for feature importance."""
        if not LIME_AVAILABLE:
            logger.warning("[IMPORTANCE] LIME not available (install lime package)")
            return None

        try:
            logger.info("[IMPORTANCE] Computing LIME importance...")

            X_scaled = self._apply_feature_scaling(self.X_test)
            X_subset = X_scaled[:n_samples]
            X_flat = X_subset.reshape(len(X_subset), -1)
            T = self.timesteps

            explainer = lime_tabular.LimeTabularExplainer(
                X_flat[:3],
                feature_names=[f"{feat}_{t}" for feat in self.feature_names for t in range(T)],
                class_names=['Pre-diabetic', 'CGM-Healthy'],
                mode='classification',
            )

            def model_predict_proba(X):
                X_reshaped = X.reshape(-1, T, len(self.feature_names))
                preds = self.model.predict(X_reshaped, verbose=0).ravel()
                preds = self._apply_temperature_scaling(preds)
                return np.column_stack([1 - preds, preds])

            importances = []
            for i in range(min(3, len(X_flat))):
                exp = explainer.explain_instance(
                    X_flat[i], model_predict_proba,
                    num_features=20, num_samples=50,
                )
                importances.append(dict(exp.as_list()))

            feature_avg = np.zeros(len(self.feature_names))
            for feat_idx, feat_name in enumerate(self.feature_names):
                feat_vals = []
                for imp_dict in importances:
                    for key, value in imp_dict.items():
                        if key.startswith(f"{feat_name}_"):
                            feat_vals.append(abs(value))
                if feat_vals:
                    feature_avg[feat_idx] = np.mean(feat_vals)

            logger.info("[IMPORTANCE] LIME importance completed")
            return feature_avg

        except Exception as e:
            logger.error(f"[IMPORTANCE] LIME failed: {e}")
            return None

    def compute_all_importances(self) -> pd.DataFrame:
        """Run all 5 importance methods and return unified DataFrame."""
        logger.info("[IMPORTANCE] Running all feature importance methods...")

        results = {}
        results['Permutation'] = self.compute_permutation_importance()
        results['Gradient'] = self.compute_gradient_importance()
        results['Variance'] = self.compute_variance_importance()

        shap_res = self.compute_shap_importance()
        if shap_res is not None:
            results['SHAP'] = shap_res

        lime_res = self.compute_lime_importance()
        if lime_res is not None:
            results['LIME'] = lime_res

        # Build DataFrame
        df = pd.DataFrame(results, index=self.feature_names)

        # Add normalized columns
        for col in df.columns:
            col_max = df[col].abs().max()
            df[f'{col}_normalized'] = df[col] / col_max if col_max > 0 else 0.0

        # Add rank columns
        for col in [c for c in df.columns if not c.endswith('_normalized')]:
            df[f'{col}_rank'] = df[col].abs().rank(ascending=False).astype(int)

        self._importance_df = df
        self._importance_raw = results
        logger.info(f"[IMPORTANCE] All methods complete: {list(results.keys())}")
        return df

    # ------------------------------------------------------------------
    # Group & temporal analysis
    # ------------------------------------------------------------------
    def analyze_by_prediction_confidence(self) -> Dict[str, Any]:
        """Analyze feature importance per confidence zone."""
        if self.y_pred_proba is None:
            logger.warning("[GROUP] Run evaluate_comprehensive() first")
            return {}

        logger.info("[GROUP] Analyzing feature importance by confidence zone...")

        y_prob = self.y_pred_proba
        X_scaled = self._apply_feature_scaling(self.X_test)

        # Define zone masks
        zone_masks = {
            'Zone1_HighConf_PreD': y_prob < self.confidence_low_threshold,
            'Zone2_Uncertain': (y_prob >= self.confidence_low_threshold) & (y_prob < self.confidence_high_threshold),
            'Zone3_HighConf_Healthy': y_prob >= self.confidence_high_threshold,
        }

        results = {}
        for zone_name, mask in zone_masks.items():
            n = int(mask.sum())
            logger.info(f"[GROUP] {zone_name}: {n} samples")

            if n < 10:
                logger.warning(f"[GROUP] Skipping {zone_name}: too few samples ({n})")
                continue

            X_zone = X_scaled[mask]
            y_zone = self.y_test[mask].ravel()

            # Quick permutation importance for this zone
            baseline_prob = self._get_predictions(X_zone)
            try:
                baseline_auc = float(auc(*roc_curve(y_zone, baseline_prob)[:2]))
            except ValueError:
                # Single class in zone
                baseline_auc = 0.5

            importance = np.zeros(len(self.feature_names))
            rng = np.random.RandomState(42)
            for i in range(len(self.feature_names)):
                X_perm = X_zone.copy()
                perm_idx = rng.permutation(X_perm.shape[0])
                X_perm[:, :, i] = X_perm[perm_idx, :, i]
                perm_prob = self.model.predict(X_perm, verbose=0).ravel()
                perm_prob = self._apply_temperature_scaling(perm_prob)
                try:
                    perm_auc = float(auc(*roc_curve(y_zone, perm_prob)[:2]))
                except ValueError:
                    perm_auc = 0.5
                importance[i] = baseline_auc - perm_auc

            results[zone_name] = {
                'importance': importance,
                'n_samples': n,
                'baseline_auc': baseline_auc,
            }

        logger.info("[GROUP] Confidence zone analysis completed")
        return results

    def analyze_by_study_group(self, n_repeats: int = 5) -> Dict[str, Any]:
        """Analyze feature importance by true study group (CGM-Healthy vs Pre-diabetes).

        Each true-label group contains only one class, so standard AUC-based
        permutation importance is undefined.  Instead we use the **signed mean
        prediction shift**: how much the average predicted probability changes
        when a feature is shuffled.

        For the CGM-Healthy group (y=1, model should predict HIGH):
            importance = baseline_mean - shuffled_mean
            Positive => feature was helping the model predict high (correct).

        For the Pre-Diabetes group (y=0, model should predict LOW):
            importance = shuffled_mean - baseline_mean
            Positive => feature was helping the model predict low (correct).

        This produces signed values analogous to "AUC Drop" and matches the
        research-paper plot format.
        """
        if self.y_pred_proba is None:
            logger.warning("[GROUP] Run evaluate_comprehensive() first")
            return {}

        logger.info("[GROUP] Analyzing feature importance by true study group...")

        X_scaled = self._apply_feature_scaling(self.X_test)
        y_test = self.y_test.ravel()

        group_masks = {
            'True_Healthy': y_test == 1,
            'True_PreDiabetes': y_test == 0,
        }

        results = {}
        rng = np.random.RandomState(42)

        for group_name, mask in group_masks.items():
            n = int(mask.sum())
            logger.info(f"[GROUP] {group_name}: {n} samples")

            if n < 10:
                logger.warning(f"[GROUP] Skipping {group_name}: too few samples ({n})")
                continue

            X_group = X_scaled[mask]

            # Baseline mean prediction for this group
            baseline_prob = self._get_predictions(X_group)
            baseline_mean = float(np.mean(baseline_prob))
            logger.info(f"[GROUP] {group_name} baseline mean prediction: {baseline_mean:.4f}")

            importance = np.zeros(len(self.feature_names))

            for rep in range(n_repeats):
                for i in range(len(self.feature_names)):
                    X_perm = X_group.copy()
                    perm_idx = rng.permutation(X_perm.shape[0])
                    X_perm[:, :, i] = X_perm[perm_idx, :, i]

                    perm_prob = self.model.predict(X_perm, verbose=0).ravel()
                    perm_prob = self._apply_temperature_scaling(perm_prob)
                    perm_mean = float(np.mean(perm_prob))

                    if group_name == 'True_Healthy':
                        # Model should predict HIGH for healthy
                        # Positive shift = feature was helping predict high (correct)
                        importance[i] += (baseline_mean - perm_mean)
                    else:
                        # Model should predict LOW for pre-diabetes
                        # Positive shift = feature was helping predict low (correct)
                        importance[i] += (perm_mean - baseline_mean)

            # Average over repeats
            importance /= n_repeats

            for feat_name, imp_val in zip(self.feature_names, importance):
                logger.info(f"[GROUP]   {feat_name:25s}: {imp_val:+.6f}")

            results[group_name] = {
                'importance': importance,
                'n_samples': n,
                'baseline_mean_prob': baseline_mean,
            }

        logger.info("[GROUP] Study group analysis completed")
        return results

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns by reconstructing hour from sin/cos features."""
        if self.y_pred_proba is None:
            logger.warning("[TEMPORAL] Run evaluate_comprehensive() first")
            return {}

        logger.info("[TEMPORAL] Analyzing temporal patterns...")

        # Reconstruct hour from sin_hour and cos_hour (using raw unscaled data)
        sin_idx = self.feature_names.index('sin_hour') if 'sin_hour' in self.feature_names else None
        cos_idx = self.feature_names.index('cos_hour') if 'cos_hour' in self.feature_names else None

        if sin_idx is None or cos_idx is None:
            logger.warning("[TEMPORAL] sin_hour/cos_hour features not found; skipping temporal analysis")
            return {}

        # Use raw (unscaled) data for hour reconstruction
        sin_vals = self.X_test[:, :, sin_idx].mean(axis=1)
        cos_vals = self.X_test[:, :, cos_idx].mean(axis=1)

        # atan2 → radians → hours (0-24)
        hours = np.arctan2(sin_vals, cos_vals) * 12 / np.pi
        hours = hours % 24  # wrap to [0, 24)

        result = {
            'hours': hours,
            'predictions': self.y_pred_proba,
            'true_labels': self.y_test.ravel(),
        }

        logger.info("[TEMPORAL] Temporal analysis completed")
        return result

    # ------------------------------------------------------------------
    # Plot methods (all saved to analysis/ subfolder, NO plt.show())
    # ------------------------------------------------------------------
    def _plot_publication_roc(self, eval_results: Dict) -> None:
        """Publication-quality ROC curve."""
        save_path = os.path.join(self.analysis_dir, 'publication_roc_curve.png')

        plt.figure(figsize=(8, 6))
        plt.plot(eval_results['fpr'], eval_results['tpr'], color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {eval_results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC) Curve\nCGM LSTM Binary Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[PLOT] Publication ROC curve saved to: {save_path}")

    def _plot_publication_pr(self, eval_results: Dict) -> None:
        """Publication-quality Precision-Recall curve."""
        save_path = os.path.join(self.analysis_dir, 'publication_pr_curve.png')

        plt.figure(figsize=(8, 6))
        plt.plot(eval_results['recall_curve'], eval_results['precision_curve'],
                 color='blue', lw=2, label=f'PR curve (AP = {eval_results["pr_auc"]:.3f})')
        if eval_results.get('ppv', 0) > 0:
            plt.axhline(y=eval_results['ppv'], color='red', linestyle='--',
                        label=f'Baseline PPV = {eval_results["ppv"]:.3f}')
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curve\nCGM LSTM Binary Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[PLOT] Publication PR curve saved to: {save_path}")

    def _plot_confusion_matrix(self, eval_results: Dict) -> None:
        """Standard confusion matrix."""
        save_path = os.path.join(self.analysis_dir, 'confusion_matrix.png')
        Visualizer.plot_confusion_matrix(eval_results['confusion_matrix'], save_path=save_path)
        logger.info(f"[PLOT] Confusion matrix saved to: {save_path}")

    def _plot_enhanced_confusion_matrix(self, eval_results: Dict) -> None:
        """Side-by-side raw + normalized confusion matrix."""
        save_path = os.path.join(self.analysis_dir, 'confusion_matrix_enhanced.png')

        cm = eval_results['confusion_matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Pre-diabetes', 'CGM-Healthy'],
                    yticklabels=['Pre-diabetes', 'CGM-Healthy'])
        ax1.set_title('Confusion Matrix (Raw Counts)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')

        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                    xticklabels=['Pre-diabetes', 'CGM-Healthy'],
                    yticklabels=['Pre-diabetes', 'CGM-Healthy'])
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[PLOT] Enhanced confusion matrix saved to: {save_path}")

    def _plot_reliability_curve(self, calib_results: Dict) -> None:
        """Reliability (calibration) curve."""
        save_path = os.path.join(self.analysis_dir, 'reliability_curve.png')
        curve = calib_results.get('calibration_curve', {})
        if curve:
            Visualizer.plot_reliability_curve(
                curve['prob_pred'], curve['prob_true'], save_path=save_path
            )
            logger.info(f"[PLOT] Reliability curve saved to: {save_path}")

    def _plot_threshold_sensitivity(self, eval_results: Dict) -> None:
        """Threshold sensitivity analysis plot."""
        save_path = os.path.join(self.analysis_dir, 'threshold_sensitivity.png')
        ts = eval_results.get('threshold_sensitivity', {})
        if ts:
            Visualizer.plot_threshold_sensitivity(
                np.array(ts['thresholds']),
                {k: ts[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'specificity']},
                save_path=save_path,
            )
            logger.info(f"[PLOT] Threshold sensitivity saved to: {save_path}")

    def _plot_feature_importance_comparison(self, importance_results: Dict[str, np.ndarray]) -> None:
        """Side-by-side bar charts for each importance method."""
        save_path = os.path.join(self.analysis_dir, 'feature_importance_comparison.png')

        n_methods = len(importance_results)
        if n_methods == 0:
            return

        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 8))
        if n_methods == 1:
            axes = [axes]

        for idx, (method_name, importance) in enumerate(importance_results.items()):
            ax = axes[idx]

            imp_max = np.abs(importance).max()
            imp_norm = importance / imp_max if imp_max > 0 else importance

            sorted_idx = np.argsort(np.abs(imp_norm))[::-1]
            bars = ax.bar(range(len(imp_norm)), [imp_norm[i] for i in sorted_idx])
            ax.set_xticks(range(len(imp_norm)))
            ax.set_xticklabels([self.feature_names[i] for i in sorted_idx], rotation=45, ha='right')
            ax.set_title(f'{method_name} Feature Importance')
            ax.set_ylabel('Normalized Importance')

            colors = plt.cm.Blues(np.abs(imp_norm[sorted_idx]))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[PLOT] Feature importance comparison saved to: {save_path}")

    def _plot_study_group_comparison(self, group_results: Dict) -> None:
        """Feature importance by confidence zone (grouped bar chart)."""
        save_path = os.path.join(self.analysis_dir, 'study_group_comparison.png')

        valid_groups = {k: v for k, v in group_results.items() if 'importance' in v}
        if not valid_groups:
            logger.warning("[PLOT] No valid group results to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(self.feature_names))
        n_groups = len(valid_groups)
        width = 0.8 / n_groups
        colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

        for i, (name, data) in enumerate(valid_groups.items()):
            offset = width * (i - n_groups / 2 + 0.5)
            ax.bar(x + offset, data['importance'], width,
                   label=f"{name} (n={data['n_samples']})",
                   color=colors[i % len(colors)])

        ax.set_xlabel('Features')
        ax.set_ylabel('Permutation Importance (AUC Drop)')
        ax.set_title('Feature Importance by Confidence Zone')
        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[PLOT] Study group comparison saved to: {save_path}")

    def _plot_study_group_by_true_label(self, study_group_results: Dict) -> None:
        """Feature importance (AUC Drop) grouped by true study group.

        Produces a single grouped bar chart matching the research-paper style:
        - X-axis : feature names (rotated 45 deg)
        - Y-axis : Importance (AUC Drop) -- supports negative values
        - Bars   : skyblue for CGM-Healthy, salmon for Pre-diabetic
        - Saves  : feature_importance_by_study_group.png
        """
        save_path = os.path.join(self.analysis_dir, 'feature_importance_by_study_group.png')

        valid_groups = {k: v for k, v in study_group_results.items() if 'importance' in v}
        if not valid_groups:
            logger.warning("[PLOT] No valid study group results to plot")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(self.feature_names))
        n_groups = len(valid_groups)
        total_width = 0.7
        width = total_width / max(n_groups, 1)

        # Map internal keys to display labels and colours matching the example
        display_config = {
            'True_Healthy':      {'label': 'Predicted CGM-Healthy',      'color': 'skyblue'},
            'True_PreDiabetes':  {'label': 'Predicted Pre-diabetic', 'color': 'salmon'},
        }
        fallback_colors = ['skyblue', 'salmon', '#2ecc71', '#f39c12']

        for i, (name, data) in enumerate(valid_groups.items()):
            offset = width * (i - n_groups / 2 + 0.5)
            cfg = display_config.get(name, {})
            label = cfg.get('label', name.replace('_', ' '))
            color = cfg.get('color', fallback_colors[i % len(fallback_colors)])

            ax.bar(
                x + offset,
                data['importance'],
                width,
                label=label,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.85,
            )

        # Axis labels and title
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Importance (AUC Drop)', fontsize=12)
        ax.set_title('Feature Importance by Predicted Study Group', fontsize=14, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right', fontsize=10)

        # Dynamic y-axis to include negative values with padding
        all_vals = np.concatenate([v['importance'] for v in valid_groups.values()])
        y_min = min(0.0, float(np.min(all_vals)))
        y_max = max(0.0, float(np.max(all_vals)))
        y_pad = max(abs(y_min), abs(y_max)) * 0.15   # 15 % padding
        if y_pad < 1e-6:
            y_pad = 0.05  # Fallback padding when all values are near zero
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # Reference line at zero
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)

        ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"[PLOT] Feature importance by study group saved to: {save_path}")

    def _plot_importance_method_correlation(self, importance_results: Dict[str, np.ndarray]) -> None:
        """Correlation heatmap between importance methods."""
        save_path = os.path.join(self.analysis_dir, 'importance_method_correlation.png')

        if len(importance_results) < 2:
            logger.warning("[PLOT] Need at least 2 methods for correlation matrix")
            return

        df = pd.DataFrame(importance_results, index=self.feature_names)
        corr = df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Between Feature Importance Methods')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[PLOT] Method correlation matrix saved to: {save_path}")

    def _plot_temporal_analysis(self, temporal_data: Dict) -> None:
        """4-panel temporal analysis plot."""
        save_path = os.path.join(self.analysis_dir, 'temporal_analysis.png')

        if not temporal_data:
            return

        hours = temporal_data['hours']
        predictions = temporal_data['predictions']
        y_test = temporal_data['true_labels']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Predictions by hour
        scatter = axes[0, 0].scatter(hours, predictions, alpha=0.6, c=y_test, cmap='RdYlBu', s=30)
        axes[0, 0].set_xlabel('Reconstructed Hour of Day')
        axes[0, 0].set_ylabel('Prediction Probability (CGM-Healthy)')
        axes[0, 0].set_title('Predictions by Time of Day')
        axes[0, 0].axhline(y=self.threshold, color='grey', linestyle='--', alpha=0.5,
                            label=f'Threshold={self.threshold:.3f}')
        axes[0, 0].legend()
        plt.colorbar(scatter, ax=axes[0, 0], label='True Label (0=Pre-D, 1=CGM-Healthy)')

        # Plot 2: Prediction distribution by class
        healthy_preds = predictions[y_test == 1]
        prediab_preds = predictions[y_test == 0]
        axes[0, 1].hist(healthy_preds, bins=20, alpha=0.7, label='True CGM-Healthy', color='green')
        axes[0, 1].hist(prediab_preds, bins=20, alpha=0.7, label='Pre-diabetes', color='red')
        axes[0, 1].axvline(x=self.threshold, color='black', linestyle='--',
                           label=f'Threshold={self.threshold:.3f}')
        axes[0, 1].set_xlabel('Prediction Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Prediction Distribution by True Class')
        axes[0, 1].legend()

        # Plot 3: Confidence boxplot by class
        confidence = np.abs(predictions - self.threshold)
        axes[1, 0].boxplot([confidence[y_test == 0], confidence[y_test == 1]])
        axes[1, 0].set_xticklabels(['Pre-diabetes', 'CGM-Healthy'])
        axes[1, 0].set_ylabel('Distance from Threshold')
        axes[1, 0].set_title('Model Confidence by True Class')

        # Plot 4: Calibration curve
        try:
            fraction_pos, mean_pred = calibration_curve(y_test, predictions, n_bins=10, strategy='quantile')
            axes[1, 1].plot(mean_pred, fraction_pos, marker='o', linewidth=1, label='Model')
            axes[1, 1].plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
            axes[1, 1].set_xlabel('Mean Predicted Probability')
            axes[1, 1].set_ylabel('Fraction of Positives')
            axes[1, 1].set_title('Reliability Diagram (Calibration)')
            axes[1, 1].legend()
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'Calibration plot\nunavailable:\n{e}',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[PLOT] Temporal analysis saved to: {save_path}")

    def _plot_confidence_zone_distribution(self, confidence_results: Dict) -> None:
        """Bar chart of zone sample counts and accuracy."""
        save_path = os.path.join(self.analysis_dir, 'confidence_zone_distribution.png')

        if confidence_results is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Zone sample distribution
        zones = ['Zone 1\n(Pre-D)', 'Zone 2\n(Uncertain)', 'Zone 3\n(CGM-Healthy)']
        counts = [
            confidence_results['n_high_conf_prediabetes'],
            confidence_results['n_uncertain'],
            confidence_results['n_high_conf_healthy'],
        ]
        pcts = [
            confidence_results['pct_high_conf_prediabetes'],
            confidence_results['pct_uncertain'],
            confidence_results['pct_high_conf_healthy'],
        ]
        colors_zones = ['#c44e52', '#dd8452', '#55a868']

        bars = ax1.bar(zones, counts, color=colors_zones)
        for bar, pct in zip(bars, pcts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=11)
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Confidence Zone Distribution')

        # Zone precision/specificity
        metrics_names = ['Zone 1\nPrecision', 'Zone 2\nPreD Rate', 'Zone 3\nPrecision']
        metrics_vals = [
            confidence_results['zone1_precision'],
            confidence_results['zone2_prediabetes_rate'],
            confidence_results['zone3_precision'],
        ]
        bars2 = ax2.bar(metrics_names, metrics_vals, color=colors_zones)
        for bar, val in zip(bars2, metrics_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Zone-Level Metrics')
        ax2.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[PLOT] Confidence zone distribution saved to: {save_path}")

    def _plot_prediction_distribution(self) -> None:
        """Histogram of calibrated probabilities by true class."""
        save_path = os.path.join(self.analysis_dir, 'prediction_distribution.png')

        if self.y_pred_proba is None:
            return

        y_test = self.y_test.ravel()
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(self.y_pred_proba[y_test == 0], bins=25, alpha=0.7,
                color='red', label='Pre-diabetes (Class 0)', density=True)
        ax.hist(self.y_pred_proba[y_test == 1], bins=25, alpha=0.7,
                color='green', label='CGM-Healthy (Class 1)', density=True)

        # Add zone boundaries
        ax.axvline(x=self.confidence_low_threshold, color='orange', linestyle='--',
                   label=f'Low threshold={self.confidence_low_threshold:.3f}')
        ax.axvline(x=self.confidence_high_threshold, color='blue', linestyle='--',
                   label=f'High threshold={self.confidence_high_threshold:.3f}')
        ax.axvline(x=self.threshold, color='black', linestyle='-', lw=2,
                   label=f'Global threshold={self.threshold:.3f}')

        # Shade zones
        ax.axvspan(0, self.confidence_low_threshold, alpha=0.05, color='red', label='_nolegend_')
        ax.axvspan(self.confidence_low_threshold, self.confidence_high_threshold,
                   alpha=0.05, color='orange', label='_nolegend_')
        ax.axvspan(self.confidence_high_threshold, 1.0, alpha=0.05, color='green', label='_nolegend_')

        ax.set_xlabel('Calibrated Probability (P(CGM-Healthy))')
        ax.set_ylabel('Density')
        ax.set_title('Prediction Distribution by True Class with Confidence Zones')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[PLOT] Prediction distribution saved to: {save_path}")

    # ------------------------------------------------------------------
    # Research tables (CSV)
    # ------------------------------------------------------------------
    def _generate_performance_table(self, eval_results: Dict, calib_results: Dict,
                                     confidence_results: Optional[Dict]) -> pd.DataFrame:
        """Comprehensive performance metrics CSV."""
        rows = [
            ('Accuracy', f"{eval_results['classification_report']['accuracy']:.4f}",
             'Overall classification accuracy'),
            ('Balanced Accuracy', f"{eval_results['balanced_accuracy']:.4f}",
             'Accuracy accounting for class imbalance'),
            ('ROC AUC', f"{eval_results['roc_auc']:.4f}",
             'Discriminative ability across all thresholds'),
            ('PR AUC', f"{eval_results['pr_auc']:.4f}",
             'Area under precision-recall curve'),
            ('F1 Score', f"{eval_results['f1_score']:.4f}",
             'Harmonic mean of precision and recall'),
            ('MCC', f"{eval_results['mcc']:.4f}",
             'Matthews correlation coefficient'),
            ('Sensitivity (Recall)', f"{eval_results['sensitivity']:.4f}",
             'Proportion of CGM-Healthy correctly identified'),
            ('Specificity', f"{eval_results['specificity']:.4f}",
             'Proportion of pre-diabetic correctly identified'),
            ('PPV', f"{eval_results['ppv']:.4f}",
             'Probability that positive prediction is correct'),
            ('NPV', f"{eval_results['npv']:.4f}",
             'Probability that negative prediction is correct'),
            ('Brier Score', f"{calib_results['brier_score']:.4f}",
             'Calibration quality (lower is better)'),
            ('ECE', f"{calib_results['ece']:.4f}",
             'Expected calibration error'),
            ('Threshold', f"{eval_results['threshold_used']:.4f}",
             'Global decision threshold (from OOF Youden J)'),
        ]

        if confidence_results:
            rows.extend([
                ('Detection Rate', f"{confidence_results['prediabetes_detection_rate']:.1f}%",
                 'Pre-diabetes detection across all zones'),
                ('False Positive Rate', f"{confidence_results['false_positive_rate']:.1f}%",
                 'CGM-Healthy misclassified as pre-diabetic (Zone 1)'),
                ('OGTT Burden', f"{confidence_results['ogtt_burden']:.1f}%",
                 'Patients needing confirmatory testing'),
            ])

        df = pd.DataFrame(rows, columns=['Metric', 'Value', 'Clinical_Interpretation'])
        save_path = os.path.join(self.analysis_dir, 'comprehensive_performance_metrics.csv')
        df.to_csv(save_path, index=False)
        logger.info(f"[TABLE] Performance metrics saved to: {save_path}")
        return df

    def _generate_importance_table(self) -> pd.DataFrame:
        """Feature importance comparison CSV."""
        if not hasattr(self, '_importance_df'):
            logger.warning("[TABLE] No importance data; run compute_all_importances() first")
            return pd.DataFrame()

        save_path = os.path.join(self.analysis_dir, 'comprehensive_feature_importance.csv')
        self._importance_df.to_csv(save_path)
        logger.info(f"[TABLE] Feature importance saved to: {save_path}")
        return self._importance_df

    def _generate_cv_fold_table(self) -> Optional[pd.DataFrame]:
        """CV fold-by-fold comparison CSV."""
        if self.cv_results is None:
            return None

        rows = []
        n_folds = self.cv_results.get('cv_n_splits', 5)
        fold_aucs = self.cv_results.get('fold_roc_auc', [])
        fold_pr_aucs = self.cv_results.get('fold_pr_auc', [])
        fold_accs = self.cv_results.get('fold_accuracy', [])
        fold_accs_global = self.cv_results.get('fold_accuracy_global', [])
        fold_thresholds = self.cv_results.get('fold_thresholds', [])

        for i in range(n_folds):
            rows.append({
                'Fold': i + 1,
                'ROC_AUC': fold_aucs[i] if i < len(fold_aucs) else None,
                'PR_AUC': fold_pr_aucs[i] if i < len(fold_pr_aucs) else None,
                'Accuracy_PerFold_Thr': fold_accs[i] if i < len(fold_accs) else None,
                'Accuracy_Global_Thr': fold_accs_global[i] if i < len(fold_accs_global) else None,
                'Threshold': fold_thresholds[i] if i < len(fold_thresholds) else None,
            })

        # Add mean +/- std row
        df = pd.DataFrame(rows)
        mean_row = {'Fold': 'Mean +/- Std'}
        for col in df.columns:
            if col == 'Fold':
                continue
            vals = df[col].dropna().values.astype(float)
            mean_row[col] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        save_path = os.path.join(self.analysis_dir, 'cv_fold_comparison.csv')
        df.to_csv(save_path, index=False)
        logger.info(f"[TABLE] CV fold comparison saved to: {save_path}")
        return df

    def _generate_held_out_vs_cv_table(self, eval_results: Dict) -> Optional[pd.DataFrame]:
        """Held-out vs CV comparison CSV."""
        if self.cv_results is None:
            return None

        rows = [
            {
                'Metric': 'ROC AUC',
                'CV_Mean': f"{self.cv_results.get('roc_auc_mean', 0):.4f}",
                'CV_Std': f"{self.cv_results.get('roc_auc_std', 0):.4f}",
                'Held_Out': f"{eval_results['roc_auc']:.4f}",
                'Difference': f"{eval_results['roc_auc'] - self.cv_results.get('roc_auc_mean', 0):.4f}",
            },
            {
                'Metric': 'PR AUC',
                'CV_Mean': f"{self.cv_results.get('pr_auc_mean', 0):.4f}",
                'CV_Std': f"{self.cv_results.get('pr_auc_std', 0):.4f}",
                'Held_Out': f"{eval_results['pr_auc']:.4f}",
                'Difference': f"{eval_results['pr_auc'] - self.cv_results.get('pr_auc_mean', 0):.4f}",
            },
            {
                'Metric': 'Accuracy (Global Thr)',
                'CV_Mean': f"{self.cv_results.get('accuracy_mean_global', 0):.4f}",
                'CV_Std': f"{self.cv_results.get('accuracy_std_global', 0):.4f}",
                'Held_Out': f"{eval_results['classification_report']['accuracy']:.4f}",
                'Difference': f"{eval_results['classification_report']['accuracy'] - self.cv_results.get('accuracy_mean_global', 0):.4f}",
            },
        ]

        df = pd.DataFrame(rows)
        save_path = os.path.join(self.analysis_dir, 'held_out_vs_cv_comparison.csv')
        df.to_csv(save_path, index=False)
        logger.info(f"[TABLE] Held-out vs CV comparison saved to: {save_path}")
        return df

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete post-hoc analysis pipeline."""
        logger.info("\n" + "=" * 70)
        logger.info("PRODUCTION MODEL ANALYSIS - START")
        logger.info(f"Model: {self.model_folder_name}")
        logger.info(f"Output: {self.analysis_dir}")
        logger.info("=" * 70)

        all_results = {}

        # Step 1: Comprehensive evaluation
        logger.info("\n[STEP 1/13] Comprehensive evaluation...")
        eval_results = self.evaluate_comprehensive()
        all_results['evaluation'] = eval_results

        # Step 2: Confidence-based evaluation
        logger.info("\n[STEP 2/13] Confidence-based 3-tier evaluation...")
        confidence_results = self.evaluate_confidence_based()
        all_results['confidence'] = confidence_results

        # Step 3: Calibration assessment
        logger.info("\n[STEP 3/13] Calibration assessment...")
        calib_results = self.evaluate_calibration()
        all_results['calibration'] = calib_results

        # Step 4: Feature importances (5 methods)
        logger.info("\n[STEP 4/13] Feature importance analysis (5 methods)...")
        importance_df = self.compute_all_importances()
        all_results['importance'] = self._importance_raw

        # Step 5: Group analysis by confidence zone
        logger.info("\n[STEP 5/13] Group analysis by confidence zone...")
        group_results = self.analyze_by_prediction_confidence()
        all_results['group_analysis'] = group_results

        # Step 6: Group analysis by true study group (CGM-Healthy vs Pre-Diabetes)
        logger.info("\n[STEP 6/13] Feature importance by study group (CGM-Healthy vs Pre-Diabetes)...")
        study_group_results = self.analyze_by_study_group()
        all_results['study_group_analysis'] = study_group_results

        # Step 7: Temporal pattern analysis
        logger.info("\n[STEP 7/13] Temporal pattern analysis...")
        temporal_data = self.analyze_temporal_patterns()

        # Step 8-13: Generate all plots
        logger.info("\n[STEP 8/13] Generating publication plots...")
        self._plot_publication_roc(eval_results)
        self._plot_publication_pr(eval_results)

        logger.info("\n[STEP 9/13] Generating confusion matrices...")
        self._plot_confusion_matrix(eval_results)
        self._plot_enhanced_confusion_matrix(eval_results)

        logger.info("\n[STEP 10/13] Generating calibration and threshold plots...")
        if calib_results:
            self._plot_reliability_curve(calib_results)
        self._plot_threshold_sensitivity(eval_results)

        logger.info("\n[STEP 11/13] Generating feature importance plots...")
        self._plot_feature_importance_comparison(self._importance_raw)
        self._plot_study_group_comparison(group_results)
        self._plot_study_group_by_true_label(study_group_results)
        self._plot_importance_method_correlation(self._importance_raw)

        logger.info("\n[STEP 12/13] Generating temporal and distribution plots...")
        self._plot_temporal_analysis(temporal_data)
        self._plot_confidence_zone_distribution(confidence_results)
        self._plot_prediction_distribution()

        # Step 13: Generate research tables
        logger.info("\n[STEP 13/13] Generating research tables...")
        perf_df = self._generate_performance_table(eval_results, calib_results, confidence_results)
        imp_df = self._generate_importance_table()
        cv_df = self._generate_cv_fold_table()
        ho_cv_df = self._generate_held_out_vs_cv_table(eval_results)

        # Save analysis summary JSON
        summary = {
            'model_folder': self.model_folder_name,
            'analysis_date': datetime.now().isoformat(),
            'test_set': {
                'n_samples': int(len(self.X_test)),
                'n_prediabetes': int(np.sum(self.y_test == 0)),
                'n_healthy': int(np.sum(self.y_test == 1)),
                'source': 'held_out_test' if os.path.exists(
                    os.path.join(self.models_dir, 'held_out_test', 'held_out_test_indices.npy')
                ) else 'full_dataset',
            },
            'threshold': self.threshold,
            'temperature': self.temperature_scaler.temperature if self.temperature_scaler else 1.0,
            'confidence_thresholds': {
                'low': self.confidence_low_threshold,
                'high': self.confidence_high_threshold,
            },
            'metrics': {
                'roc_auc': eval_results['roc_auc'],
                'pr_auc': eval_results['pr_auc'],
                'accuracy': eval_results['classification_report']['accuracy'],
                'mcc': eval_results['mcc'],
                'f1_score': eval_results['f1_score'],
                'balanced_accuracy': eval_results['balanced_accuracy'],
                'sensitivity': eval_results['sensitivity'],
                'specificity': eval_results['specificity'],
                'ppv': eval_results['ppv'],
                'npv': eval_results['npv'],
                'brier_score': calib_results['brier_score'] if calib_results else None,
                'ece': calib_results['ece'] if calib_results else None,
            },
            'confidence_metrics': {
                'detection_rate': confidence_results['prediabetes_detection_rate'] if confidence_results else None,
                'false_positive_rate': confidence_results['false_positive_rate'] if confidence_results else None,
                'ogtt_burden': confidence_results['ogtt_burden'] if confidence_results else None,
            },
            'importance_methods': list(self._importance_raw.keys()),
            'artifacts_generated': {
                'plots': 13,
                'csv_tables': 4,
                'json_summary': 1,
            },
        }

        summary_path = os.path.join(self.analysis_dir, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"[SUMMARY] Analysis summary saved to: {summary_path}")

        # Print console summary
        self._print_summary(eval_results, confidence_results, calib_results)

        return all_results

    def _print_summary(self, eval_results: Dict, confidence_results: Optional[Dict],
                       calib_results: Optional[Dict]) -> None:
        """Print formatted console summary."""
        print("\n" + "=" * 70)
        print("PRODUCTION MODEL ANALYSIS RESULTS")
        print("=" * 70)
        print(f"Model Version:  {self.model_folder_name}")
        print(f"Results Saved:  {self.analysis_dir}")
        print(f"Test Samples:   {len(self.X_test)} "
              f"(Pre-D={np.sum(self.y_test == 0)}, CGM-Healthy={np.sum(self.y_test == 1)})")
        print(f"Threshold:      {self.threshold:.4f}")
        print(f"Temperature:    {self.temperature_scaler.temperature if self.temperature_scaler else 1.0:.4f}")

        print("\n--- Performance Metrics ---")
        print(f"ROC-AUC:           {eval_results['roc_auc']:.4f}")
        print(f"PR-AUC:            {eval_results['pr_auc']:.4f}")
        print(f"Accuracy:          {eval_results['classification_report']['accuracy']:.4f}")
        print(f"Balanced Accuracy: {eval_results['balanced_accuracy']:.4f}")
        print(f"MCC:               {eval_results['mcc']:.4f}")
        print(f"F1 Score:          {eval_results['f1_score']:.4f}")
        print(f"Sensitivity:       {eval_results['sensitivity']:.4f}")
        print(f"Specificity:       {eval_results['specificity']:.4f}")
        print(f"PPV:               {eval_results['ppv']:.4f}")
        print(f"NPV:               {eval_results['npv']:.4f}")

        if calib_results:
            print(f"\n--- Calibration ---")
            print(f"Brier Score: {calib_results['brier_score']:.4f}")
            print(f"ECE:         {calib_results['ece']:.4f}")

        if confidence_results:
            print(f"\n--- Confidence-Based (3-Tier) ---")
            print(f"Zone 1 (High Conf Pre-D):  {confidence_results['n_high_conf_prediabetes']} samples "
                  f"({confidence_results['pct_high_conf_prediabetes']:.1f}%) | "
                  f"Precision={confidence_results['zone1_precision']:.1f}%")
            print(f"Zone 2 (Uncertain/OGTT):   {confidence_results['n_uncertain']} samples "
                  f"({confidence_results['pct_uncertain']:.1f}%)")
            print(f"Zone 3 (High Conf CGM-Healthy): {confidence_results['n_high_conf_healthy']} samples "
                  f"({confidence_results['pct_high_conf_healthy']:.1f}%) | "
                  f"Precision={confidence_results['zone3_precision']:.1f}%")
            print(f"Detection Rate:  {confidence_results['prediabetes_detection_rate']:.1f}%")
            print(f"False Positive:  {confidence_results['false_positive_rate']:.1f}%")
            print(f"OGTT Burden:     {confidence_results['ogtt_burden']:.1f}%")

        print(f"\n--- Feature Importance Methods ---")
        if hasattr(self, '_importance_raw'):
            print(f"Methods used: {', '.join(self._importance_raw.keys())}")

        print(f"\n--- Artifacts Generated ---")
        print(f"Plots:       13 PNGs in {self.analysis_dir}")
        print(f"Tables:      4 CSVs in {self.analysis_dir}")
        print(f"Summary:     analysis_summary.json")
        print("=" * 70)
        print("Analysis complete! All files ready for research paper.")
        print("=" * 70)


# ===========================================================================
# CLI entry point
# ===========================================================================
def main():
    """Main entry point with CLI argument support."""
    try:
        specific_folder = None
        if len(sys.argv) > 1:
            specific_folder = sys.argv[1]
            print(f"Using specified model folder: {specific_folder}")
        else:
            print("Auto-detecting latest model folder...")

        analyzer = ProductionModelAnalyzer(
            models_base_dir='./models',
            specific_model_folder=specific_folder,
        )

        results = analyzer.run_complete_analysis()

        print(f"\nAnalysis complete for: {analyzer.model_folder_name}")
        print(f"All results saved to: {analyzer.analysis_dir}")

        # Clean up logging
        cleanup_comprehensive_logging()

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\nAnalysis failed: {e}")
        print("\nPlease ensure:")
        print("1. Model folder exists with production_model/ subfolder containing:")
        print("   - best_model.keras, global_threshold.json, feature_scaler.npz")
        print("   - temperature_scaler.json, confidence_thresholds.json, training_config.json")
        print("2. Root model folder has: X_lstm.npy, y_lstm_binary.npy")
        print("3. held_out_test/ subfolder has: held_out_test_indices.npy")
        print("4. Required packages: tensorflow, sklearn, matplotlib, seaborn")
        print("5. Optional packages: shap, lime (for full feature importance)")
        print(f"\nUsage: python \"{os.path.basename(__file__)}\" [model_folder_name]")

        # Clean up logging even on error
        cleanup_comprehensive_logging()
        sys.exit(1)


if __name__ == "__main__":
    main()
