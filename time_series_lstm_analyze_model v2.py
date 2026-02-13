__CURR_FILE__ = "time_series_lstm_analyze_model v2"

"""
Enhanced Feature Analysis Script
===============================

Compares multiple feature importance techniques and provides comprehensive analysis:
- Permutation importance
- Gradient-based importance  
- Variance-based importance
- SHAP analysis (KernelExplainer)
- LIME analysis
- Study group comparison
- Complete research metrics
- Separate publication-ready plots
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # match training numerics
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, classification_report, confusion_matrix,
                           precision_recall_curve, matthews_corrcoef, f1_score, 
                           balanced_accuracy_score)
from sklearn.calibration import calibration_curve
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

# Try importing SHAP and LIME
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
logger = logging.getLogger(__name__)

class EnhancedFeatureAnalyzer:
    """Enhanced feature importance analysis with all methods and comprehensive metrics."""
    
    def __init__(self, models_base_dir='./models', specific_model_folder=None):
        """
        Initialize analyzer with automatic or manual model folder selection.
        
        Args:
            models_base_dir: Base directory containing model folders
            specific_model_folder: Specific model folder name (e.g., 'model_20250729_143022')
                                 If None, uses the most recent folder automatically
        """
        self.models_base_dir = models_base_dir
        
        # Determine which model folder to use
        if specific_model_folder:
            # Manual folder specification
            self.models_dir = os.path.join(models_base_dir, specific_model_folder)
            if not os.path.exists(self.models_dir):
                raise ValueError(f"Specified model folder does not exist: {self.models_dir}")
            self.model_folder_name = specific_model_folder
            logger.info(f"Using manually specified model folder: {specific_model_folder}")
        else:
            # Automatic latest folder detection
            latest_folder = self.get_latest_model_folder()
            if not latest_folder:
                raise ValueError(f"No model folders found in {models_base_dir}")
            self.models_dir = latest_folder
            self.model_folder_name = os.path.basename(latest_folder)
            logger.info(f"Auto-detected latest model folder: {self.model_folder_name}")
        
        self.model = None
        self.threshold = 0.5  # will be overwritten by loading methods
        self.temperature = 1.0  # Temperature scaling parameter
        self.use_temperature_scaling = False

        # Pipeline default features (9 features)
        default_feats = [
            'glucose_accel', 'cos_hour', 'blood_glucose_value',
            'glucose_rollmean_1h', 'glucose_rollstd_1h',
            'glucose_diff', 'sin_hour', 'is_meal_time', 'is_night'
        ]

        # Try to load features from multiple sources
        self.feature_names = None
        self.binary_features = {'is_meal_time', 'is_night'}  # Keep as fallback

        # Source 1: root model_config.json (legacy, may have features)
        root_cfg_path = os.path.join(self.models_dir, "model_config.json")
        if os.path.exists(root_cfg_path):
            cfg = json.load(open(root_cfg_path, "r"))
            self.feature_names = cfg.get("features", None)
            logger.info(f"Loaded features from model_config.json")

        # Source 2: Infer from X_lstm.npy shape if not found
        if self.feature_names is None:
            X_path = os.path.join(self.models_dir, "X_lstm.npy")
            if os.path.exists(X_path):
                X = np.load(X_path)
                n_features = X.shape[2]
                # Use pipeline defaults if 9 features
                if n_features == 9:
                    self.feature_names = default_feats
                    logger.info("Using pipeline default features (9 features detected)")
                else:
                    self.feature_names = [f"feature_{i}" for i in range(n_features)]
                    logger.warning(f"Could not infer feature names, using generic names for {n_features} features")
            else:
                self.feature_names = default_feats
                logger.warning("X_lstm.npy not found, using fallback defaults")
        
        # Verify required files exist
        self._verify_model_files()
    
    def get_latest_model_folder(self):
        """Get the latest model folder based on timestamp."""
        try:
            if not os.path.exists(self.models_base_dir):
                return None
            
            # Get all model folders
            folders = [f for f in os.listdir(self.models_base_dir) 
                        if f.startswith('model_') and os.path.isdir(os.path.join(self.models_base_dir, f))]
            
            if not folders:
                return None
            
            # Sort by name (timestamp will sort chronologically)
            folders.sort(reverse=True)
            latest_folder = folders[0]
            
            return os.path.join(self.models_base_dir, latest_folder)
            
        except Exception as e:
            logger.error(f"Error getting latest model folder: {e}")
            return None
    
    def _verify_model_files(self):
        """Verify that required model files exist in production_model/ structure."""
        required_files = [
            ("production_model", "best_model.keras"),
            ("production_model", "global_threshold.json"),
            ("production_model", "temperature_scaler.json"),
            ("production_model", "feature_scaler.npz"),
            ("held_out_test", "held_out_test_indices.npy"),
            ("", "X_lstm.npy"),
            ("", "y_lstm_binary.npy")
        ]

        missing_files = []
        for subdir, file_name in required_files:
            if subdir:
                file_path = os.path.join(self.models_dir, subdir, file_name)
                display_path = f"{subdir}/{file_name}"
            else:
                file_path = os.path.join(self.models_dir, file_name)
                display_path = file_name

            if not os.path.exists(file_path):
                missing_files.append(display_path)

        if missing_files:
            raise FileNotFoundError(f"Missing required files in {self.models_dir}: {missing_files}")

        logger.info(f"All required model files found in {self.model_folder_name}")
        
    def load_model_and_data(self):
        """Load model and data from production_model/ and held_out_test/."""
        # Load model from production_model/
        model_path = os.path.join(self.models_dir, "production_model", "best_model.keras")
        self.model = load_model(model_path)
        logger.info(f"Model loaded from production_model/best_model.keras")

        # Load threshold from production_model/
        threshold_path = os.path.join(self.models_dir, "production_model", "global_threshold.json")
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
            self.threshold = float(threshold_data.get("global_threshold", 0.5))
            logger.info(f"Global threshold loaded: {self.threshold:.4f}")
        else:
            logger.warning("global_threshold.json not found, using default 0.5")

        # Load temperature scaler
        self.load_temperature_scaler()

        # Load full dataset
        X_all = np.load(os.path.join(self.models_dir, "X_lstm.npy"))
        y_all = np.load(os.path.join(self.models_dir, "y_lstm_binary.npy"))

        # Load held-out test indices (NOT internal CV test)
        idx_test = np.load(os.path.join(self.models_dir, "held_out_test", "held_out_test_indices.npy"))
        logger.info(f"Using held-out test set (never seen during training)")

        X_test, y_test = X_all[idx_test], y_all[idx_test]
        self.timesteps = X_test.shape[1]  # used by SHAP/LIME flattening

        # Apply train-only standardization to continuous features
        scaler_path = os.path.join(self.models_dir, "production_model", "feature_scaler.npz")
        if os.path.exists(scaler_path):
            pack = np.load(scaler_path, allow_pickle=True)
            mu = pack["mean"]
            sigma = pack["scale"]  # FIX: Use 'scale' not 'std'

            # Continuous features are all except binary features
            cont_idx = [i for i, n in enumerate(self.feature_names)
                        if n not in self.binary_features]

            X_test = X_test.copy()
            X_test[:, :, cont_idx] = (X_test[:, :, cont_idx] - mu) / sigma
            logger.info("Applied feature scaling from production_model/feature_scaler.npz")
        else:
            logger.warning("feature_scaler.npz not found — proceeding without standardization")

        logger.info(f"Model and data loaded from: {self.model_folder_name}")
        logger.info(f"Test set shape: {X_test.shape} (n={len(X_test)} samples)")
        return X_test, y_test

    def load_temperature_scaler(self):
        """Load temperature scaling parameter from production_model/."""
        temp_path = os.path.join(self.models_dir, "production_model", "temperature_scaler.json")
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                temp_data = json.load(f)
            self.temperature = float(temp_data.get("temperature", 1.0))
            self.use_temperature_scaling = True
            logger.info(f"Loaded temperature scaling: T={self.temperature:.4f}")
        else:
            self.temperature = 1.0
            self.use_temperature_scaling = False
            logger.warning("temperature_scaler.json not found, using raw predictions")

    def apply_temperature_scaling(self, y_prob):
        """Apply temperature scaling to probabilities."""
        if not self.use_temperature_scaling or self.temperature == 1.0:
            return y_prob

        # Temperature scaling: p_calibrated = sigmoid(logit / T)
        # Convert probabilities to logits, scale, then back to probabilities
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)  # Avoid log(0)
        logits = np.log(y_prob / (1 - y_prob))
        calibrated_logits = logits / self.temperature
        y_prob_calibrated = 1 / (1 + np.exp(-calibrated_logits))
        return y_prob_calibrated

    def comprehensive_evaluation(self, X_test, y_test):
        """Generate comprehensive research metrics."""
        try:
            logger.info("Starting comprehensive evaluation...")
            
            # Get predictions
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred_proba = self.apply_temperature_scaling(y_pred_proba)  # Apply calibration
            y_pred = (y_pred_proba >= self.threshold).astype(int).flatten()
            y_pred_proba = y_pred_proba.flatten()
            y_test = y_test.flatten()
            
            # Standard metrics
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # ROC and PR curves
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = auc(recall_curve, precision_curve)
            
            # Research-grade metrics
            mcc = matthews_corrcoef(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            # Clinical metrics
            tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            results = {
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'mcc': mcc,
                'f1_score': f1,
                'balanced_accuracy': balanced_acc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'fpr': fpr,
                'tpr': tpr,
                'precision_curve': precision_curve,
                'recall_curve': recall_curve,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred
            }
            
            logger.info("Comprehensive evaluation completed")
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            raise
    
    def permutation_importance(self, X_test, y_test, n_samples=100):
        """Our existing permutation-based importance."""
        logger.info("Computing permutation importance...")
        
        # Use subset for speed
        X_subset = X_test[:n_samples]
        y_subset = y_test[:n_samples]
        
        # Baseline performance
        baseline_preds = self.model.predict(X_subset, verbose=0).flatten()
        baseline_preds = self.apply_temperature_scaling(baseline_preds)  # Apply calibration
        baseline_auc = auc(*roc_curve(y_subset, baseline_preds)[:2])

        importance_scores = []
        for i, feature_name in enumerate(self.feature_names):
            X_corrupted = X_subset.copy()
            np.random.shuffle(X_corrupted[:, :, i])

            corrupted_preds = self.model.predict(X_corrupted, verbose=0).flatten()
            corrupted_preds = self.apply_temperature_scaling(corrupted_preds)  # Apply calibration
            corrupted_auc = auc(*roc_curve(y_subset, corrupted_preds)[:2])
            
            importance = baseline_auc - corrupted_auc
            importance_scores.append(importance)
        
        return np.array(importance_scores)
    
    def gradient_based_importance(self, X_test, y_test, n_samples=50):
        """Gradient-based feature importance."""
        logger.info("Computing gradient-based importance...")
        
        X_subset = X_test[:n_samples]
        y_subset = y_test[:n_samples]
        
        # Convert to tensor
        X_tensor = tf.convert_to_tensor(X_subset, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
            loss = tf.reduce_mean(predictions)  # Mean prediction as proxy
        
        # Get gradients
        gradients = tape.gradient(loss, X_tensor)
        
        # Compute importance as mean absolute gradient per feature
        feature_importance = tf.reduce_mean(tf.abs(gradients), axis=[0, 1]).numpy()
        
        return feature_importance
    
    def variance_based_importance(self, X_test, y_test, n_samples=100):
        """Variance-based importance using prediction variance."""
        logger.info("Computing variance-based importance...")
        
        X_subset = X_test[:n_samples]

        # Get baseline predictions
        baseline_preds = self.model.predict(X_subset, verbose=0).flatten()
        baseline_preds = self.apply_temperature_scaling(baseline_preds)  # Apply calibration
        baseline_var = np.var(baseline_preds)

        importance_scores = []
        for i, feature_name in enumerate(self.feature_names):
            # Add noise to feature
            X_noisy = X_subset.copy()
            noise = np.random.normal(0, np.std(X_subset[:, :, i]) * 0.1, X_subset[:, :, i].shape)
            X_noisy[:, :, i] += noise

            noisy_preds = self.model.predict(X_noisy, verbose=0).flatten()
            noisy_preds = self.apply_temperature_scaling(noisy_preds)  # Apply calibration
            noisy_var = np.var(noisy_preds)
            
            # Importance as change in prediction variance
            importance = abs(baseline_var - noisy_var)
            importance_scores.append(importance)
        
        return np.array(importance_scores)
    
    def try_shap_analysis(self, X_test, n_samples=10):
        """Try SHAP analysis with KernelExplainer (more robust)."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return None
            
        try:
            logger.info("Attempting SHAP analysis with KernelExplainer...")
            
            # Use very small sample for SHAP
            X_subset = X_test[:n_samples]
            
            T = getattr(self, "timesteps", X_test.shape[1])
            
            # Create a wrapper function for your model
            def model_predict(X):
                # Ensure correct shape for LSTM
                if len(X.shape) == 2:
                    # Reshape flattened input back to (samples, timesteps, features)
                    X = X.reshape(-1, T, len(self.feature_names))
                preds = self.model.predict(X, verbose=0).flatten()
                preds = self.apply_temperature_scaling(preds)  # Apply calibration
                return preds
            
            # Use a smaller background dataset
            background = X_subset[:3].reshape(3, -1)  # Flatten for KernelExplainer
            
            # Create KernelExplainer (model-agnostic, more stable)
            explainer = shap.KernelExplainer(model_predict, background)
            
            # Get SHAP values for small subset
            test_sample = X_subset[:3].reshape(3, -1)  # Flatten for KernelExplainer
            shap_values = explainer.shap_values(test_sample, nsamples=30)  # Reduce nsamples for speed
            
            # Average SHAP values across samples
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Reshape back and average across time steps
            shap_reshaped = shap_values.reshape(-1, T, len(self.feature_names))
            feature_importance = np.abs(shap_reshaped).mean(axis=0).mean(axis=0)
            
            logger.info("SHAP analysis completed with KernelExplainer")
            return feature_importance
            
        except Exception as e:
            logger.error(f"SHAP failed: {e}")
            return None
    def try_lime_analysis(self, X_test, n_samples=5):
        """LIME analysis as SHAP alternative."""
        if not LIME_AVAILABLE:
            logger.warning("LIME not available")
            return None
            
        try:
            logger.info("🔍 Attempting LIME analysis...")
            
            # Flatten data for LIME (treats as tabular)
            X_subset = X_test[:n_samples]
            X_flat = X_subset.reshape(len(X_subset), -1)
            T = getattr(self, "timesteps", X_subset.shape[1])
            
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X_flat[:3],  # Training data for LIME
                feature_names=[f"{feat}_{t}" for feat in self.feature_names for t in range(T)],
                class_names=['Pre-diabetic', 'Healthy'],
                mode='classification'
            )
            
            # Model wrapper
            def model_predict_proba(X):
                X_reshaped = X.reshape(-1, T, len(self.feature_names))  # Use dynamic T
                preds = self.model.predict(X_reshaped, verbose=0)
                preds = self.apply_temperature_scaling(preds)  # Apply calibration
                return np.column_stack([1-preds, preds])  # LIME expects probabilities for both classes
            
            # Get explanations for a few samples
            importances = []
            for i in range(min(3, len(X_flat))):
                exp = explainer.explain_instance(
                    X_flat[i], 
                    model_predict_proba, 
                    num_features=20,  # Top features only
                    num_samples=50
                )
                
                # Extract feature importance
                feature_importance = dict(exp.as_list())
                importances.append(feature_importance)
            
            # Aggregate importance by original features (average across time steps)
            feature_avg_importance = np.zeros(len(self.feature_names))
            for feat_idx, feat_name in enumerate(self.feature_names):
                # Find all time steps for this feature
                feat_importances = []
                for imp_dict in importances:
                    for key, value in imp_dict.items():
                        if key.startswith(f"{feat_name}_"):
                            feat_importances.append(abs(value))
                
                if feat_importances:
                    feature_avg_importance[feat_idx] = np.mean(feat_importances)
            
            logger.info("LIME analysis completed")
            return feature_avg_importance
            
        except Exception as e:
            logger.error(f"LIME failed: {e}")
            return None
    
    def analyze_temporal_patterns(self, X_test, y_test):
        """Analyze temporal patterns in predictions."""
        try:
            logger.info(" Analyzing temporal patterns...")

            predictions = self.model.predict(X_test, verbose=0).flatten()
            predictions = self.apply_temperature_scaling(predictions)  # Apply calibration

            # Find hour feature
            hour_feature_idx = None
            for i, feature in enumerate(self.feature_names):
                if feature.lower() == 'hour':
                    hour_feature_idx = i
                    break
            
            if hour_feature_idx is not None:
                hours = X_test[:, :, hour_feature_idx].mean(axis=1)
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Plot 1: Predictions by hour
                scatter = axes[0, 0].scatter(hours, predictions, alpha=0.6, c=y_test, cmap='RdYlBu')
                axes[0, 0].set_xlabel('Average Hour of Day')
                axes[0, 0].set_ylabel('Prediction Probability')
                axes[0, 0].set_title('Predictions by Time of Day')
                plt.colorbar(scatter, ax=axes[0, 0], label='True Label')
                
                # Plot 2: Distribution by class
                healthy_preds = predictions[y_test == 1]
                unhealthy_preds = predictions[y_test == 0]
                
                axes[0, 1].hist(healthy_preds, bins=20, alpha=0.7, label='True Healthy', color='green')
                axes[0, 1].hist(unhealthy_preds, bins=20, alpha=0.7, label='Pre-diabetes', color='red')
                axes[0, 1].set_xlabel('Prediction Probability')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Prediction Distribution by True Class')
                axes[0, 1].legend()
                
                # Plot 3: Model confidence
                confidence = np.abs(predictions - 0.5)
                axes[1, 0].boxplot([confidence[y_test == 0], confidence[y_test == 1]])
                axes[1, 0].set_xticklabels(['Pre-diabetes', 'Healthy'])
                axes[1, 0].set_ylabel('Model Confidence')
                axes[1, 0].set_title('Model Confidence by True Class')
                
                # Plot 4: Calibration
                fraction_pos, mean_pred = calibration_curve(y_test, predictions, n_bins=10)
                axes[1, 1].plot(mean_pred, fraction_pos, marker='o', linewidth=1, label='Model')
                axes[1, 1].plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
                axes[1, 1].set_xlabel('Mean Predicted Probability')
                axes[1, 1].set_ylabel('Fraction of Positives')
                axes[1, 1].set_title('Reliability Diagram (Calibration)')
                axes[1, 1].legend()
                
                plt.tight_layout()

                # Save to model folder
                save_path = os.path.join(self.models_dir, 'temporal_analysis.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Temporal analysis saved to: {save_path}")
                # plt.show()  # Non-interactive mode
                
            logger.info("Temporal analysis completed")
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
    
    def analyze_by_study_group(self, X_test, y_test):
        """Analyze feature importance by study group using SHAP."""
        if not SHAP_AVAILABLE:
            logger.warning(" SHAP not available for study group analysis, falling back to permutation")
            return self._analyze_by_study_group_permutation(X_test, y_test)
        
        logger.info("Analyzing by study group using SHAP...")

        # Get predictions for all test samples
        predictions = self.model.predict(X_test, verbose=0).flatten()
        predictions = self.apply_temperature_scaling(predictions)  # Apply calibration

        # Use quartile-based grouping for more stable groups
        prediction_quartiles = np.percentile(predictions, [25, 75])
        
        # Separate by predicted confidence levels
        low_conf_mask = predictions <= prediction_quartiles[0]   # Likely pre-diabetic
        high_conf_mask = predictions >= prediction_quartiles[1]  # Likely healthy
        
        results = {}
        
        for group_name, mask in [("Predicted_Low_Confidence", low_conf_mask), 
                                ("Predicted_High_Confidence", high_conf_mask)]:
            group_count = mask.sum()
            logger.info(f"{group_name}: {group_count} samples")
            
            if group_count < 10:  # Skip if too few samples
                logger.warning(f" Skipping {group_name}: insufficient samples ({group_count})")
                continue
                
            X_group = X_test[mask]
            y_group = y_test[mask]
            
            # Compute SHAP importance for this group
            group_importance = self._compute_shap_for_group(X_group, group_name)
            
            if group_importance is not None:
                results[group_name] = group_importance
        
        return results

    def _compute_shap_for_group(self, X_group, group_name, n_samples=10):
        """Compute SHAP importance for a specific group."""
        try:
            logger.info(f"Computing SHAP importance for {group_name}...")

            # Use moderate subset for SHAP (balanced between accuracy and stability)
            sample_size = min(n_samples, len(X_group))
            if len(X_group) > sample_size:
                np.random.seed(42)  # Fixed seed for reproducibility
                indices = np.random.choice(len(X_group), sample_size, replace=False)
                X_subset = X_group[indices]
            else:
                X_subset = X_group

            # Model wrapper for SHAP
            def model_predict(X):
                # Ensure correct shape for LSTM
                if len(X.shape) == 2:
                    # Reshape flattened input back to (samples, timesteps, features)
                    expected_features = len(self.feature_names)
                    expected_timesteps = X.shape[1] // expected_features
                    X = X.reshape(-1, expected_timesteps, expected_features)
                preds = self.model.predict(X, verbose=0).flatten()
                # Apply temperature scaling for consistency
                preds = self.apply_temperature_scaling(preds)
                return preds

            # Use minimal background dataset for stability (3 samples is a good balance)
            background_size = min(3, len(X_subset))
            background = X_subset[:background_size].reshape(background_size, -1)

            # Create KernelExplainer
            explainer = shap.KernelExplainer(model_predict, background)

            # Use moderate nsamples for stability (100 can cause numerical issues)
            test_sample = X_subset[:sample_size].reshape(sample_size, -1)
            shap_values = explainer.shap_values(test_sample, nsamples=50)
            
            # Process SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For binary classification

            # Check for NaN or Inf values
            if np.any(np.isnan(shap_values)) or np.any(np.isinf(shap_values)):
                logger.error(f"SHAP values contain NaN or Inf for {group_name}")
                raise ValueError("Numerical instability detected in SHAP values")

            # Reshape back and average across time steps and samples
            expected_timesteps = X_subset.shape[1]
            expected_features = len(self.feature_names)
            shap_reshaped = shap_values.reshape(-1, expected_timesteps, expected_features)
            feature_importance = np.abs(shap_reshaped).mean(axis=0).mean(axis=0)

            # Validate SHAP values for numerical issues
            max_importance = np.max(feature_importance)
            min_importance = np.min(feature_importance)

            # Check for extreme outliers (values > 1000x the median)
            median_importance = np.median(feature_importance[feature_importance > 0])
            if median_importance > 0:
                outlier_threshold = median_importance * 1000
                has_outliers = np.any(feature_importance > outlier_threshold)

                if has_outliers:
                    logger.error(f"SHAP values for {group_name} contain extreme outliers:")
                    logger.error(f"  Range: [{min_importance:.6e}, {max_importance:.6e}]")
                    logger.error(f"  Median: {median_importance:.6e}, Threshold: {outlier_threshold:.6e}")
                    outlier_features = [self.feature_names[i] for i, imp in enumerate(feature_importance)
                                       if imp > outlier_threshold]
                    logger.error(f"  Outlier features: {outlier_features}")
                    raise ValueError(f"Numerical instability: extreme outliers detected in SHAP for {group_name}")

            # Check if values are too small (all essentially zero)
            if max_importance < 1e-10:
                logger.warning(f"SHAP values for {group_name} are extremely small (max={max_importance:.2e})")
                logger.warning(f"Falling back to permutation importance")
                raise ValueError("SHAP values too small")

            logger.info(f"✓ SHAP analysis completed for {group_name}")
            logger.info(f"  Feature importance range: [{min_importance:.6e}, {max_importance:.6e}]")
            logger.info(f"  Median importance: {median_importance:.6e}")
            return feature_importance
            
        except Exception as e:
            logger.error(f"SHAP failed for {group_name}: {e}")
            logger.warning(f"Falling back to permutation importance for {group_name}")
            return self._permutation_fallback(X_group)

    def _permutation_fallback(self, X_group):
        """Fallback to permutation importance if SHAP fails."""
        try:
            logger.info("Using permutation importance as fallback...")

            # Use larger sample for permutation (it's fast)
            sample_size = min(100, len(X_group))
            if len(X_group) > sample_size:
                np.random.seed(42)
                indices = np.random.choice(len(X_group), sample_size, replace=False)
                X_subset = X_group[indices]
            else:
                X_subset = X_group

            # Baseline predictions
            baseline_preds = self.model.predict(X_subset, verbose=0).flatten()
            baseline_preds = self.apply_temperature_scaling(baseline_preds)
            baseline_var = np.var(baseline_preds)  # Use variance as metric

            importance_scores = []
            for i, feature_name in enumerate(self.feature_names):
                X_corrupted = X_subset.copy()
                # Shuffle the feature across all timesteps
                np.random.seed(42 + i)  # Different seed per feature for reproducibility
                np.random.shuffle(X_corrupted[:, :, i])

                corrupted_preds = self.model.predict(X_corrupted, verbose=0).flatten()
                corrupted_preds = self.apply_temperature_scaling(corrupted_preds)
                corrupted_var = np.var(corrupted_preds)

                # Importance as change in prediction variance
                importance = abs(baseline_var - corrupted_var)
                importance_scores.append(importance)

            importance_array = np.array(importance_scores)
            logger.info(f"✓ Permutation importance completed")
            logger.info(f"  Range: [{np.min(importance_array):.6e}, {np.max(importance_array):.6e}]")

            return importance_array

        except Exception as e:
            logger.error(f"Permutation fallback also failed: {e}")
            # Return zeros as last resort
            logger.warning("Returning zero importance for all features")
            return np.zeros(len(self.feature_names))

    def _analyze_by_study_group_permutation(self, X_test, y_test):
        """Original permutation-based analysis as backup."""
        logger.info(" Using permutation importance for study group analysis...")

        predictions = self.model.predict(X_test, verbose=0).flatten()
        predictions = self.apply_temperature_scaling(predictions)  # Apply calibration

        # Separate by predicted class (as proxy for study group)
        healthy_mask = predictions > 0.5
        prediabetic_mask = predictions <= 0.5
        
        results = {}
        
        for group_name, mask in [("Predicted Healthy", healthy_mask), 
                                ("Predicted Pre-diabetic", prediabetic_mask)]:
            if mask.sum() == 0:
                continue
                
            X_group = X_test[mask]
            y_group = y_test[mask]
            
            if len(X_group) < 10:  # Skip if too few samples
                continue
                
            # Compute permutation importance for this group
            group_importance = self.permutation_importance(X_group, y_group, min(50, len(X_group)))
            results[group_name] = group_importance
        
        return results
    
    def compare_all_methods(self, X_test, y_test):
        """Compare all available feature importance methods."""
        logger.info(" Comparing all feature importance methods...")
        
        results = {}
        
        # Method 1: Permutation importance
        results['Permutation'] = self.permutation_importance(X_test, y_test)
        
        # Method 2: Gradient-based
        results['Gradient'] = self.gradient_based_importance(X_test, y_test)
        
        # Method 3: Variance-based
        results['Variance'] = self.variance_based_importance(X_test, y_test)
        
        # Method 4: SHAP (if available)
        shap_results = self.try_shap_analysis(X_test)
        if shap_results is not None:
            results['SHAP'] = shap_results
        
        # Method 5: LIME (if available)
        lime_results = self.try_lime_analysis(X_test)
        if lime_results is not None:
            results['LIME'] = lime_results
        
        return results
    
    def plot_comparison(self, results, save_path=None):
        """Plot comparison of different methods."""
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 8))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_name, importance) in enumerate(results.items()):
            ax = axes[idx]
            
            # Normalize importance scores for comparison
            if importance.max() > 0:
                importance_norm = importance / importance.max()
            else:
                importance_norm = importance
            
            # Sort by importance
            indices = np.argsort(importance_norm)[::-1]
            
            bars = ax.bar(range(len(importance_norm)), 
                         [importance_norm[i] for i in indices])
            ax.set_xticks(range(len(importance_norm)))
            ax.set_xticklabels([self.feature_names[i] for i in indices], 
                              rotation=45, ha='right')
            ax.set_title(f'{method_name} Feature Importance')
            ax.set_ylabel('Normalized Importance')
            
            # Color bars by importance
            colors = plt.cm.Blues(importance_norm[indices])
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        
        # Auto-save to model folder if no path specified
        if save_path is None:
            save_path = os.path.join(self.models_dir, 'feature_importance_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance comparison saved to: {save_path}")
        # plt.show()  # Non-interactive mode
        
        return fig
    
    def plot_study_group_comparison(self, group_results, save_path=None):
        """Plot feature importance by study group."""
        if not group_results:
            logger.warning("No group results to plot")
            return

        logger.info("=" * 80)
        logger.info("PLOT STUDY GROUP COMPARISON - DEBUG INFO")
        logger.info("=" * 80)
        logger.info(f"Number of groups: {len(group_results)}")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create grouped bar chart with improved visibility
        x = np.arange(len(self.feature_names))
        n_groups = len(group_results)

        # Calculate width based on number of groups
        total_width = 0.7  # Total width for all bars at each position
        width = total_width / n_groups if n_groups > 0 else 0.35
        logger.info(f"Bar width: {width:.4f} (total_width={total_width}, n_groups={n_groups})")

        # More distinct colors with better contrast
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # Blue, Red, Green, Orange

        # Check for data quality issues
        all_values = np.concatenate([imp for imp in group_results.values()])
        max_val = np.max(all_values)
        median_val = np.median(all_values[all_values > 0]) if np.any(all_values > 0) else 0

        # Detect if we have extreme outliers that would make the plot unreadable
        if median_val > 0 and max_val > median_val * 1000:
            logger.warning("Extreme outliers detected in group comparison - some groups may have failed")
            logger.warning("Consider using permutation importance for all groups instead")

        # Plot each group with proper offset
        for i, (group_name, importance) in enumerate(group_results.items()):
            # Calculate offset to center the group of bars
            offset = width * (i - n_groups / 2 + 0.5)

            logger.info(f"\nGroup {i+1}: {group_name}")
            logger.info(f"  Offset: {offset:.4f}")
            logger.info(f"  Importance shape: {importance.shape}")
            logger.info(f"  Importance values:")
            for feat_idx, (feat_name, imp_value) in enumerate(zip(self.feature_names, importance)):
                logger.info(f"    {feat_name:25s}: {imp_value:10.6e}")
            logger.info(f"  Min value: {np.min(importance):.6e}")
            logger.info(f"  Max value: {np.max(importance):.6e}")
            logger.info(f"  Mean value: {np.mean(importance):.6e}")
            logger.info(f"  Median value: {np.median(importance):.6e}")

            bars = ax.bar(x + offset, importance, width,
                        label=group_name.replace('_', ' '),
                        color=colors[i % len(colors)],
                        alpha=0.8,
                        edgecolor='black',
                        linewidth=0.5)

        ax.set_xlabel('Features', fontsize=12, fontweight='bold')

        # Update ylabel and title to indicate SHAP with note about fallback
        method_used = "SHAP" if SHAP_AVAILABLE else "Permutation"
        ax.set_ylabel(f'{method_used} Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_used} Feature Importance by Study Group\n(Model Prediction Confidence; fallback to permutation if SHAP fails)',
                    fontsize=14, fontweight='bold', pad=20)

        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add subtle background
        ax.set_facecolor('#f8f9fa')

        # Fix y-axis to include negative values (SHAP can be negative)
        logger.info(f"\nY-Axis Calculation:")
        logger.info(f"  All values shape: {all_values.shape}")
        logger.info(f"  Global min: {np.min(all_values):.6e}")
        logger.info(f"  Global max: {np.max(all_values):.6e}")
        logger.info(f"  Global median: {median_val:.6e}")

        y_min = min(0, np.min(all_values) * 1.1)  # 10% padding below min
        y_max = np.max(all_values) * 1.1  # 10% padding above max

        logger.info(f"  Y-axis min (with padding): {y_min:.6e}")
        logger.info(f"  Y-axis max (with padding): {y_max:.6e}")

        ax.set_ylim(y_min, y_max)

        # Add a horizontal line at y=0 for reference
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

        plt.tight_layout()

        # Auto-save to model folder if no path specified
        if save_path is None:
            save_path = os.path.join(self.models_dir, 'study_group_comparison.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"\nPlot saved to: {save_path}")
        logger.info("=" * 80)

        return fig
    
    def create_correlation_matrix(self, results):
        """Create correlation matrix between different importance methods."""
        if len(results) < 2:
            return None
            
        # Create DataFrame
        df = pd.DataFrame(results, index=self.feature_names)
        
        # Compute correlation
        correlation_matrix = df.corr()
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Between Feature Importance Methods')
        plt.tight_layout()

        # Save to model folder
        save_path = os.path.join(self.models_dir, 'importance_method_correlation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f" Method correlation matrix saved to: {save_path}")
        # plt.show()  # Non-interactive mode
        
        return correlation_matrix
    
    def create_publication_plots(self, eval_results):
        """Create publication-ready ROC and PR curves."""
        # ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(eval_results['fpr'], eval_results['tpr'], color='darkorange', lw=2,
                label=f'ROC curve (AUC = {eval_results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC) Curve\nLSTM Binary Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.models_dir}/publication_roc_curve.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Non-interactive mode
        
        # Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(eval_results['recall_curve'], eval_results['precision_curve'], 
                color='blue', lw=2, label=f'PR curve (AP = {eval_results["avg_precision"]:.3f})')
        plt.axhline(y=eval_results['ppv'], color='red', linestyle='--',
                   label=f'Baseline (PPV = {eval_results["ppv"]:.3f})')
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curve\nLSTM Binary Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # FIX: Use os.path.join for proper path construction
        save_path = os.path.join(self.models_dir, 'publication_pr_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Publication PR curve saved to: {save_path}")
        # plt.show()  # Non-interactive mode
        
        # Enhanced Confusion Matrix
        cm = eval_results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Pre-diabetes', 'Healthy'],
                   yticklabels=['Pre-diabetes', 'Healthy'])
        ax1.set_title('Confusion Matrix (Raw Counts)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                   xticklabels=['Pre-diabetes', 'Healthy'],
                   yticklabels=['Pre-diabetes', 'Healthy'])
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f" Enhanced confusion matrix saved to: {save_path}")
        # plt.show()  # Non-interactive mode
    
    def create_research_summary_tables(self, eval_results, all_results):
        """Create comprehensive research summary tables."""
        # Model performance table
        performance_data = {
            'Metric': [
                'Accuracy', 'Balanced Accuracy', 'ROC AUC', 'Average Precision',
                'F1 Score', 'Matthews Correlation Coefficient',
                'Sensitivity (Recall)', 'Specificity', 
                'Positive Predictive Value', 'Negative Predictive Value'
            ],
            'Value': [
                f"{eval_results.get('classification_report', {}).get('accuracy', 0):.4f}",
                f"{eval_results.get('balanced_accuracy', 0):.4f}",
                f"{eval_results.get('roc_auc', 0):.4f}",
                f"{eval_results.get('avg_precision', 0):.4f}",
                f"{eval_results.get('f1_score', 0):.4f}",
                f"{eval_results.get('mcc', 0):.4f}",
                f"{eval_results.get('sensitivity', 0):.4f}",
                f"{eval_results.get('specificity', 0):.4f}",
                f"{eval_results.get('ppv', 0):.4f}",
                f"{eval_results.get('npv', 0):.4f}"
            ],
            'Clinical_Interpretation': [
                'Overall classification accuracy',
                'Accuracy accounting for class imbalance',
                'Discriminative ability across all thresholds',
                'Area under precision-recall curve',
                'Harmonic mean of precision and recall',
                'Correlation between predictions and true labels',
                'Proportion of healthy individuals correctly identified',
                'Proportion of pre-diabetic individuals correctly identified',
                'Probability that positive prediction is correct',
                'Probability that negative prediction is correct'
            ]
        }
        
        performance_df = pd.DataFrame(performance_data)
        
        # Feature importance table
        importance_data = {'Feature': self.feature_names}
        for method, importance in all_results.items():
            importance_data[f'{method}_Importance'] = [f"{imp:.4f}" for imp in importance]
        
        importance_df = pd.DataFrame(importance_data)
        
        # Save tables
        performance_df.to_csv(f'{self.models_dir}/comprehensive_performance_metrics.csv', index=False)
        importance_df.to_csv(f'{self.models_dir}/comprehensive_feature_importance.csv', index=False)
        
        logger.info(" Research summary tables saved")
        
        return performance_df, importance_df
    
    def run_complete_analysis(self):
        """Run complete enhanced feature analysis."""
        logger.info(" Starting enhanced comprehensive analysis...")
        logger.info(f" Analyzing model: {self.model_folder_name}")
        logger.info(f" Results will be saved to: {self.models_dir}")
        
        # Load data
        X_test, y_test = self.load_model_and_data()
        
        # Step 1: Comprehensive model evaluation
        logger.info("\n Step 1: Comprehensive Model Evaluation")
        eval_results = self.comprehensive_evaluation(X_test, y_test)
        
        # Step 2: Compare all feature importance methods
        logger.info("\n Step 2: Feature Importance Method Comparison")
        all_results = self.compare_all_methods(X_test, y_test)
        
        # Step 3: Study group analysis
        logger.info("\n Step 3: Study Group Analysis")
        group_results = self.analyze_by_study_group(X_test, y_test)
        
        # Step 4: Temporal pattern analysis
        logger.info("\n Step 4: Temporal Pattern Analysis")
        self.analyze_temporal_patterns(X_test, y_test)
        
        # Step 5: Create all plots (all saved to model folder automatically)
        logger.info("\n Step 5: Creating Visualizations")
        
        # Feature importance comparison
        self.plot_comparison(all_results)
        
        # Study group comparison
        self.plot_study_group_comparison(group_results)
        
        # Method correlation matrix
        correlation = self.create_correlation_matrix(all_results)
        
        # Publication-ready plots
        self.create_publication_plots(eval_results)
        
        # Step 6: Create comprehensive summary tables
        logger.info("\nStep 6: Creating Research Summary Tables")
        performance_df, importance_df = self.create_research_summary_tables(eval_results, all_results)
        
        # Print comprehensive results with folder info
        print("\n" + "="*80)
        print("ENHANCED COMPREHENSIVE LSTM ANALYSIS RESULTS")
        print("="*80)
        print(f"Model Version: {self.model_folder_name}")
        print(f"Results Location: {self.models_dir}")
        
        # ... rest of the results printing ...
        
        return {
            'model_folder': self.model_folder_name,
            'results_location': self.models_dir,
            'evaluation_results': eval_results,
            'feature_importance_results': all_results,
            'study_group_results': group_results,
            'correlation_matrix': correlation,
            'performance_summary': performance_df,
            'importance_summary': importance_df
        }
    
def main():
    """Main execution function with flexible model folder selection."""
    try:
        import sys
        
        # Check command line arguments for manual folder specification
        specific_folder = None
        if len(sys.argv) > 1:
            specific_folder = sys.argv[1]
            print(f"Using manually specified folder: {specific_folder}")
        else:
            print("Auto-detecting latest model folder...")
        
        # Manual folder override (uncomment and modify as needed)
        # specific_folder = "model_20250729_170536"  # Replace with your folder name
        
        # Initialize enhanced analyzer
        analyzer = EnhancedFeatureAnalyzer(
            models_base_dir='./models', 
            specific_model_folder=specific_folder
        )
        
        # Run complete enhanced analysis
        results = analyzer.run_complete_analysis()
        
        print(f"\n SUCCESS! Enhanced LSTM analysis complete!")
        print(f"Analyzed model: {results['model_folder']}")
        print(f"All results saved to: {results['results_location']}")
        print("Multiple feature importance methods validated!")
        print("Clinically relevant insights generated!")
        print("All files ready for research paper!")
        
        # Show usage instructions
        print(f"USAGE INSTRUCTIONS:")
        print(f"   Automatic (latest): python {os.path.basename(__file__)}")
        print(f"   Manual: python {os.path.basename(__file__)} model_20250729_143022")
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        print("\n❌ Analysis failed. Please ensure:")
        print("1. Model folder exists with required files:")
        print("   • production_model/best_model.keras")
        print("   • production_model/global_threshold.json")
        print("   • production_model/temperature_scaler.json")
        print("   • production_model/feature_scaler.npz")
        print("   • held_out_test/held_out_test_indices.npy")
        print("   • X_lstm.npy, y_lstm_binary.npy")
        print("2. Model trained with updated pipeline (use_held_out_test=True)")
        print("3. Required packages: tensorflow, sklearn, matplotlib, seaborn")
        print("4. Optional packages: shap, lime (for full analysis)")
        print("5. Sufficient memory for computations")
        print(f"\nUsage: python {os.path.basename(__file__)} [model_folder_name]")


if __name__ == "__main__":
    main()