"""
High-level pipeline orchestration for the CGM LSTM binary classification task.
"""

import os
import logging
from typing import Dict, Any

import numpy as np

from .data import Config, DataLoader
from .model import LSTMTrainer
from .viz import Visualizer

logger = logging.getLogger(__name__)

class LSTMPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.trainer = LSTMTrainer(config)
        self.visualizer = Visualizer()

    def run_full_pipeline(self) -> Dict[str, Any]:
        try:
            logger.info("\n[START] Starting LSTM Binary Classification Pipeline")
            logger.info(f"[VERSION] Version folder: {self.config.version_folder}")

            # Step 1: Load & prepare
            logger.info("\n[STEP 1] Loading and preparing data...")
            df_raw = self.data_loader.load_augmented_data()
            df_prep = self.data_loader.prepare_binary_classification_data(df_raw)

            # Step 2: Sequences
            logger.info("\n[STEP 2] Creating LSTM sequences...")
            X, y, pid_array = self.data_loader.create_lstm_sequences(df_prep)

            # Step 3: Save arrays
            logger.info("\n[STEP 3] Saving processed data...")
            self.data_loader.save_processed_data(X, y, pid_array)

            # Step 4: Splits
            logger.info("\n[STEP 4] Preparing Train/Val/Test splits...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_training_data(X, y)
            self._sanity_check_splits()  # quick log-only check

            # Step 5: Train
            logger.info("\n[STEP 5] Training LSTM model...")
            self.trainer.train_model(X_train, y_train, X_val, y_val)

            # Step 6: Evaluate (TEST only)
            logger.info("\n[STEP 6] Evaluating model...")
            results = self.trainer.evaluate_model(X_test, y_test)

            # Step 7: Save
            logger.info("\n[STEP 7] Saving model and history...")
            self.trainer.save_model_and_history()

            # Step 8: Plots
            logger.info("\n[STEP 8] Generating visualizations...")
            
            if self.trainer.history:
                self.visualizer.plot_training_curves(self.trainer.history.history, os.path.join(self.config.models_dir, 'training_curves.png'))
            self.visualizer.plot_roc_curve(results['fpr'], results['tpr'], results['roc_auc'], os.path.join(self.config.models_dir, 'roc_curve.png'))
            self.visualizer.plot_confusion_matrix(results['confusion_matrix'], os.path.join(self.config.models_dir, 'confusion_matrix.png'))
            logger.info("\n[COMPLETE] Pipeline completed successfully!")
            self._print_version_results(results)
            return {'model': self.trainer.model, 'results': results, 'config': self.config, 'version_folder': self.config.version_folder}
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _print_version_results(self, results: Dict[str, Any]) -> None:
        print("\n" + "="*60)
        print("LSTM BINARY CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Version: {self.config.version_folder}")
        print(f"Location: {self.config.models_dir}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"Accuracy: {results['classification_report']['accuracy']:.4f}")
        print("="*60)
        print("\nFiles created:")
        print(f"   Model (best): best_model.keras")
        print(f"   Model (alias): lstm_binary_model.keras")
        print(f"   Config: model_config.json")
        print(f"   Data: X_lstm.npy, y_lstm_binary.npy, pid_lstm.npy")
        print(f"   Indices: idx_train.npy, idx_val.npy, idx_test.npy")
        print(f"   History: training_history.json")
        print(f"   Encoder: label_encoder.pkl")
        print(f"   Plots: training_curves.png, roc_curve.png, confusion_matrix.png")
        print("\nUse this version folder for analysis!")
        print("="*60)
    
    def _sanity_check_splits(self) -> None:
        """Log any overlap between Train/Val/Test indices (should be zero)."""
        import numpy as np, os
        d = self.config.models_dir
        idx_tr = set(np.load(os.path.join(d, 'idx_train.npy')).tolist())
        idx_val = set(np.load(os.path.join(d, 'idx_val.npy')).tolist())
        idx_te = set(np.load(os.path.join(d, 'idx_test.npy')).tolist())
        a = len(idx_tr & idx_val); b = len(idx_tr & idx_te); c = len(idx_val & idx_te)
        logger.info(f"[SPLIT-SANITY] train∩val={a} | train∩test={b} | val∩test={c}")
        if a or b or c:
            logger.warning("[SPLIT-SANITY] Overlap found (unexpected)!")

