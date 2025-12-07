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