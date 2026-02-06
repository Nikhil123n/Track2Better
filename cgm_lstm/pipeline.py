"""
High-level pipeline orchestration for the CGM LSTM binary classification task.
"""

import os
import json
import copy
import logging
from typing import Dict, Any

import numpy as np

from .data import Config, DataLoader
from .model import LSTMTrainer
from .viz import Visualizer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

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

            # sanity check: participant IDs should be unique in X to avoid leakage
            assert len(pid_array) == len(np.unique(pid_array)), "Participant IDs repeat in X; leakage risk."

            # Step 3: Save arrays
            logger.info("\n[STEP 3] Saving processed data...")
            self.data_loader.save_processed_data(X, y, pid_array)

            # Step 4+: Train/Eval
            if self.config.use_cross_validation:
                logger.info("\n[STEP 4] Running Cross-Validation...")
                cv_summary = self._run_cross_validation(X, y, pid_array)
                logger.info("\n[COMPLETE] CV pipeline completed successfully!")
                self._print_cv_summary(cv_summary)
                return {
                    'model': None,
                    'results': cv_summary,
                    'config': self.config,
                    'version_folder': self.config.version_folder
                }
            else:
                logger.info("\n[STEP 4] Preparing Train/Val/Test splits...")
                X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_training_data(X, y)
                self._sanity_check_splits()

                logger.info("\n[STEP 5] Training LSTM model...")
                self.trainer.train_model(X_train, y_train, X_val, y_val)

                logger.info("\n[STEP 6] Evaluating model...")
                results = self.trainer.evaluate_model(X_test, y_test)

                logger.info("\n[STEP 7] Saving model and history...")
                self.trainer.save_model_and_history()

                logger.info("\n[STEP 8] Generating visualizations...")            
            
            if self.trainer.history:
                self.visualizer.plot_training_curves(self.trainer.history.history, os.path.join(self.config.models_dir, 'training_curves.png'))
            self.visualizer.plot_roc_curve(results['fpr'], results['tpr'], results['roc_auc'], os.path.join(self.config.models_dir, 'roc_curve.png'))
            self.visualizer.plot_confusion_matrix(results['confusion_matrix'], os.path.join(self.config.models_dir, 'confusion_matrix.png'))
            
            # --- NEW: Reliability curve (calibration) ---
            cal = results.get("calibration", {})
            curve = cal.get("calibration_curve", {})
            if curve:
                self.visualizer.plot_reliability_curve(
                    curve["prob_pred"],
                    curve["prob_true"],
                    os.path.join(self.config.models_dir, "reliability_curve.png")
                )
            # END of reliability curve -----------------------------
                
            # --- NEW: Threshold sensitivity analysis ---
            ts = results.get("threshold_sensitivity", {})
            if ts:
                self.visualizer.plot_threshold_sensitivity(
                    np.array(ts["thresholds"]),
                    {
                        "accuracy": ts["accuracy"],
                        "precision": ts["precision"],
                        "recall": ts["recall"],
                        "f1": ts["f1"],
                        "specificity": ts["specificity"],
                    },
                    os.path.join(self.config.models_dir, "threshold_sensitivity.png")
                )
            # END of threshold sensitivity analysis ----------------

            logger.info("\n[COMPLETE] Pipeline completed successfully!")
            self._print_version_results(results)
            return {'model': self.trainer.model, 'results': results, 'config': self.config, 'version_folder': self.config.version_folder}
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _run_cross_validation(self, X: np.ndarray, y: np.ndarray, pid_array: np.ndarray) -> Dict[str, Any]:
        """
        Participant-level Stratified K-Fold CV.
        For each fold:
          - Train/Val is created from fold-train only.
          - Threshold is selected from fold-val only.
          - Evaluation is done on fold-test only.
        Saves fold metrics + summary to cv_results.json inside the version folder.
        """
        n_splits = int(self.config.cv_n_splits)
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=bool(self.config.cv_shuffle),
            random_state=self.config.random_seed if self.config.cv_shuffle else None
        )

        fold_reports = []
        fold_aucs = []
        fold_accs = []
        fold_thresholds = []

        # Save CV artifacts under the version directory
        cv_dir = os.path.join(self.config.models_dir, "cross_validation")
        os.makedirs(cv_dir, exist_ok=True)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            logger.info(f"\n[CV] Fold {fold}/{n_splits} starting...")

            # Always create fold dir so fold plots don't depend on cv_save_fold_models
            fold_dir = os.path.join(cv_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            # Fresh trainer per fold to avoid weight carryover
            # Copy config and set fold-specific models_dir, 
            # Reason: to save any fold-specific plots/models if needed
            fold_cfg = copy.copy(self.config)
            fold_cfg.models_dir = fold_dir   # IMPORTANT: fold-specific output folder
            fold_trainer = LSTMTrainer(fold_cfg)
            fold_trainer.fold_tag = f"[CV:FOLD {fold}/{n_splits}] "
            fold_trainer.fold_tag = f"[CV:FOLD {fold}/{n_splits}] "
            fold_trainer.fold_dir = fold_dir

            # --- participant leakage sanity check (proof for paper) ---
            train_pids = set(pid_array[train_idx])
            test_pids  = set(pid_array[test_idx])
            assert train_pids.isdisjoint(test_pids), (
                f"Participant leakage detected in fold {fold}! "
                f"Overlap={len(train_pids & test_pids)}"
            )
            # --------------------------------------------------------------

            X_train, X_val, X_test, y_train, y_val, y_test = fold_trainer.prepare_training_data_from_indices(
                X, y, train_idx=np.array(train_idx), test_idx=np.array(test_idx)
            )

            fold_trainer.train_model(X_train, y_train, X_val, y_val)
            results = fold_trainer.evaluate_model(X_test, y_test)

            # --- Threshold sensitivity analysis for this fold ---
            ts = results.get("threshold_sensitivity", {})
            if ts:
                self.visualizer.plot_threshold_sensitivity(
                    np.array(ts["thresholds"]),
                    {
                        "accuracy": ts["accuracy"],
                        "precision": ts["precision"],
                        "recall": ts["recall"],
                        "f1": ts["f1"],
                        "specificity": ts["specificity"],
                    },
                    os.path.join(fold_dir, "threshold_sensitivity.png")
                )
            # END of threshold sensitivity analysis ----------------

            # --- NEW: Reliability curve (calibration) for this fold ---
            curve = results.get("calibration", {}).get("calibration_curve", {})
            if curve:
                self.visualizer.plot_reliability_curve(
                    curve["prob_pred"],
                    curve["prob_true"],
                    os.path.join(fold_dir, "reliability_curve.png")
                )
            # END of reliability curve -----------------------------

            # fold metrics
            fold_reports.append(results['classification_report'])
            fold_aucs.append(float(results['roc_auc']))
            fold_accs.append(float(results['classification_report']['accuracy']))
            fold_thresholds.append(float(results['threshold_used']))

            logger.info(
                f"[CV] Fold {fold} done | AUC={results['roc_auc']:.4f} | "
                f"Acc={results['classification_report']['accuracy']:.4f} | Thr={results['threshold_used']:.4f}"
            )

            # Optional: save fold model
            if self.config.cv_save_fold_models:
                fold_dir = os.path.join(cv_dir, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                fold_trainer.model.save(os.path.join(fold_dir, "best_model_fold.keras"))
                with open(os.path.join(fold_dir, "fold_results.json"), "w") as f:
                    json.dump(results['classification_report'], f, indent=2)

        # Aggregate
        summary = {
            "cv_n_splits": n_splits,
            "roc_auc_mean": float(np.mean(fold_aucs)),
            "roc_auc_std": float(np.std(fold_aucs)),
            "accuracy_mean": float(np.mean(fold_accs)),
            "accuracy_std": float(np.std(fold_accs)),
            "threshold_mean": float(np.mean(fold_thresholds)),
            "threshold_std": float(np.std(fold_thresholds)),
            "fold_roc_auc": fold_aucs,
            "fold_accuracy": fold_accs,
            "fold_thresholds": fold_thresholds,
            "fold_reports": fold_reports,
        }

        out_path = os.path.join(cv_dir, "cv_results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[CV] Saved CV summary to: {out_path}")

        return summary
    
    def _print_cv_summary(self, cv_summary: Dict[str, Any]) -> None:
        print("\n" + "="*60)
        print("LSTM CROSS-VALIDATION SUMMARY")
        print("="*60)
        print(f"Version: {self.config.version_folder}")
        print(f"Location: {self.config.models_dir}")
        print(f"Folds: {cv_summary['cv_n_splits']}")
        print(f"ROC AUC: {cv_summary['roc_auc_mean']:.4f} ± {cv_summary['roc_auc_std']:.4f}")
        print(f"Accuracy: {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
        print(f"Threshold: {cv_summary['threshold_mean']:.4f} ± {cv_summary['threshold_std']:.4f}")
        print("="*60)


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

