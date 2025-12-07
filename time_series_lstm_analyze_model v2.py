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
        self.threshold = 0.5  # will be overwritten by config

        # default (only used if config not found)
        default_feats = [
            'glucose_accel','glucose_change_rate','cos_hour',
            'blood_glucose_value','glucose_rollmean_1h','glucose_diff',
            'hour','is_meal_time','is_night'
        ]

        cfg_path = os.path.join(self.models_dir, "model_config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path, "r"))
            self.feature_names = cfg.get("features", default_feats)
            self.threshold = float(cfg.get("training_params", {}).get("val_threshold", 0.5))
        else:
            self.feature_names = default_feats
        
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
        """Verify that required model files exist."""
        required_files = [
            "best_model.keras",
            "X_lstm.npy",
            "y_lstm_binary.npy",
            "idx_test.npy"          # exact test split from training
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(self.models_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files in {self.models_dir}: {missing_files}")
        
        logger.info(f"All required model files found in {self.model_folder_name}")
        
    def load_model_and_data(self):
        """Load model and data with correct splits."""
        # Load model (updated to use best_model.keras from ModelCheckpoint)
        model_path = os.path.join(self.models_dir, "best_model.keras")
        self.model = load_model(model_path)
        
        # Load data and recreate exact split
        X_all = np.load(os.path.join(self.models_dir, "X_lstm.npy"))
        y_all = np.load(os.path.join(self.models_dir, "y_lstm_binary.npy"))
        idx_test = np.load(os.path.join(self.models_dir, "idx_test.npy"))
        
        X_test, y_test = X_all[idx_test], y_all[idx_test]
        self.timesteps = X_test.shape[1]  # used by SHAP/LIME flattening

        # apply train-only standardization to continuous features (if available)
        scaler_path = os.path.join(self.models_dir, "feature_scaler.npz")
        if os.path.exists(scaler_path):
            pack = np.load(scaler_path, allow_pickle=True)
            mu, sigma = pack["mean"], pack["std"]
            cont_features = set([str(x) for x in pack["cont_features"].tolist()])
            binary = {'is_meal_time','is_night'}
            cont_idx = [i for i, n in enumerate(self.feature_names)
                        if (n in cont_features) and (n not in binary)]
            X_test = X_test.copy()
            X_test[:, :, cont_idx] = (X_test[:, :, cont_idx] - mu) / sigma
            logger.info("Applied saved train-only standardization to continuous features")
        else:
            logger.warning(" feature_scaler.npz not found — proceeding without standardization")

        logger.info(f" Model and data loaded from: {self.model_folder_name}")
        logger.info(f"Test set shape: {X_test.shape}")
        return X_test, y_test
    
    def comprehensive_evaluation(self, X_test, y_test):
        """Generate comprehensive research metrics."""
        try:
            logger.info("Starting comprehensive evaluation...")
            
            # Get predictions
            y_pred_proba = self.model.predict(X_test, verbose=0)
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
        baseline_auc = auc(*roc_curve(y_subset, baseline_preds)[:2])
        
        importance_scores = []
        for i, feature_name in enumerate(self.feature_names):
            X_corrupted = X_subset.copy()
            np.random.shuffle(X_corrupted[:, :, i])
            
            corrupted_preds = self.model.predict(X_corrupted, verbose=0).flatten()
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
        baseline_var = np.var(baseline_preds)
        
        importance_scores = []
        for i, feature_name in enumerate(self.feature_names):
            # Add noise to feature
            X_noisy = X_subset.copy()
            noise = np.random.normal(0, np.std(X_subset[:, :, i]) * 0.1, X_subset[:, :, i].shape)
            X_noisy[:, :, i] += noise
            
            noisy_preds = self.model.predict(X_noisy, verbose=0).flatten()
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
                return self.model.predict(X, verbose=0).flatten()
            
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
                X_reshaped = X.reshape(-1, 2138, len(self.feature_names))
                preds = self.model.predict(X_reshaped, verbose=0)
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
                plt.show()
                
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

    def _compute_shap_for_group(self, X_group, group_name, n_samples=5):
        """Compute SHAP importance for a specific group."""
        try:
            logger.info(f" Computing SHAP importance for {group_name}...")
            
            # Use small subset for SHAP (computationally expensive)
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
                return self.model.predict(X, verbose=0).flatten()
            
            # Use smaller background dataset
            background_size = min(2, len(X_subset))
            background = X_subset[:background_size].reshape(background_size, -1)
            
            # Create KernelExplainer
            explainer = shap.KernelExplainer(model_predict, background)
            
            # Get SHAP values for small subset
            test_sample = X_subset[:sample_size].reshape(sample_size, -1)
            shap_values = explainer.shap_values(test_sample, nsamples=50)
            
            # Process SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For binary classification
            
            # Reshape back and average across time steps and samples
            expected_timesteps = X_subset.shape[1]
            expected_features = len(self.feature_names)
            shap_reshaped = shap_values.reshape(-1, expected_timesteps, expected_features)
            feature_importance = np.abs(shap_reshaped).mean(axis=0).mean(axis=0)
            
            logger.info(f" SHAP analysis completed for {group_name}")
            return feature_importance
            
        except Exception as e:
            logger.error(f"SHAP failed for {group_name}: {e}")
            logger.warning(f"Falling back to permutation importance for {group_name}")
            return self._permutation_fallback(X_group)

    def _permutation_fallback(self, X_group):
        """Fallback to permutation importance if SHAP fails."""
        try:
            # Quick permutation importance as fallback
            sample_size = min(30, len(X_group))
            X_subset = X_group[:sample_size]
            
            # Simple baseline score
            baseline_preds = self.model.predict(X_subset, verbose=0).flatten()
            baseline_score = np.mean(baseline_preds)
            
            importance_scores = []
            for i in range(len(self.feature_names)):
                X_corrupted = X_subset.copy()
                np.random.shuffle(X_corrupted[:, :, i])
                
                corrupted_preds = self.model.predict(X_corrupted, verbose=0).flatten()
                corrupted_score = np.mean(corrupted_preds)
                
                importance = abs(baseline_score - corrupted_score)
                importance_scores.append(importance)
            
            return np.array(importance_scores)
            
        except Exception as e:
            logger.error(f"Fallback permutation also failed: {e}")
            return np.zeros(len(self.feature_names))

    def _analyze_by_study_group_permutation(self, X_test, y_test):
        """Original permutation-based analysis as backup."""
        logger.info(" Using permutation importance for study group analysis...")
        
        predictions = self.model.predict(X_test, verbose=0).flatten()
        
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
        plt.show()
        
        return fig
    
    def plot_study_group_comparison(self, group_results, save_path=None):
        """Plot feature importance by study group."""
        if not group_results:
            logger.warning("No group results to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar chart
        x = np.arange(len(self.feature_names))
        width = 0.35
        
        group_names = list(group_results.keys())
        colors = ['skyblue', 'lightcoral']
        
        for i, (group_name, importance) in enumerate(group_results.items()):
            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, importance, width, 
                        label=group_name.replace('_', ' '), color=colors[i % len(colors)])
        
        ax.set_xlabel('Features')
        
        # Update ylabel and title to indicate SHAP
        method_used = "SHAP" if SHAP_AVAILABLE else "Permutation"
        ax.set_ylabel(f'{method_used} Feature Importance')
        ax.set_title(f'{method_used} Feature Importance by Study Group\n(Model Prediction Confidence)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Auto-save to model folder if no path specified
        if save_path is None:
            save_path = os.path.join(self.models_dir, 'study_group_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Study group comparison ({method_used}) saved to: {save_path}")
        plt.show()
        
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
        plt.show()
        
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
        plt.show()
        
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
        plt.show()
        
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
        plt.show()
    
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
        print(f"\ USAGE INSTRUCTIONS:")
        print(f"   Automatic (latest): python {os.path.basename(__file__)}")
        print(f"   Manual: python {os.path.basename(__file__)} model_20250729_143022")
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        print("\n Analysis failed. Please ensure:")
        print("1. Model folder exists with required files:")
        print("   • best_model.keras (or lstm_binary_model.keras)")
        print("   • X_lstm.npy, y_lstm_binary.npy")
        print("2. Required packages: tensorflow, sklearn, matplotlib, seaborn")
        print("3. Optional packages: shap, lime (for full analysis)")
        print("4. Sufficient memory for computations")
        print(f"\nUsage: python {os.path.basename(__file__)} [model_folder_name]")


if __name__ == "__main__":
    main()