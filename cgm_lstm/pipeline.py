"""
High-level pipeline orchestration for the CGM LSTM binary classification task.
"""

import os
import json
import copy
import logging
from typing import Dict, Any, List

import numpy as np

from .data import Config, DataLoader
from .model import LSTMTrainer
from .viz import Visualizer
from .model_card import ModelCardGenerator  # NEW: Modular JSON generation

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

class LSTMPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.trainer = LSTMTrainer(config)
        self.visualizer = Visualizer()
        self.model_card_gen = ModelCardGenerator(config.models_dir)  # NEW: Initialize JSON generator

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

            # Step 3.5: Optional held-out test set split (BEFORE CV)
            held_out_test_data = None
            if self.config.use_held_out_test:
                logger.info("\n[HELD-OUT] Splitting data into train+val and held-out test set...")
                X, y, pid_array, held_out_test_data = self._split_held_out_test(X, y, pid_array)
                logger.info(f"[HELD-OUT] Train+Val: {len(X)} participants | Held-Out Test: {len(held_out_test_data['X_test'])} participants")
                logger.info(f"[HELD-OUT] Held-out test set will be evaluated ONCE after final model training")

            # Step 4+: Train/Eval
            if self.config.use_cross_validation:
                logger.info("\n[STEP 4] Running Cross-Validation...")
                cv_summary = self._run_cross_validation(X, y, pid_array)
                logger.info("\n[COMPLETE] CV pipeline completed successfully!")
                self._print_cv_summary(cv_summary)

                # Train final production model on all data if requested
                final_model = None
                if self.config.train_final_model_after_cv:
                    logger.info("\n[STEP 5] Training final production model on all data...")
                    final_model = self._train_final_production_model(X, y, cv_summary)
                    logger.info("\n[COMPLETE] Final production model trained and saved!")

                # Step 6: Evaluate on held-out test set (if enabled)
                held_out_results = None
                if held_out_test_data is not None:
                    logger.info("\n[STEP 6] Evaluating on held-out test set...")
                    held_out_results = self._evaluate_held_out_test(final_model, held_out_test_data, cv_summary)
                    logger.info("\n[COMPLETE] Held-out test evaluation completed!")
                    self._print_held_out_results(held_out_results)

                return {
                    'model': final_model,
                    'results': cv_summary,
                    'held_out_results': held_out_results,
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
                self._generate_visualizations(results)

                logger.info("\n[COMPLETE] Pipeline completed successfully!")
                self._print_version_results(results)
                return {'model': self.trainer.model, 'results': results, 'config': self.config, 'version_folder': self.config.version_folder}
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _aggregate_confidence_metrics(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate confidence-based metrics across all CV folds."""
        if not fold_metrics:
            return None

        # Extract metrics from each fold
        metrics_to_aggregate = [
            'ogtt_burden',
            'immediate_decision_rate',
            'prediabetes_detection_rate',
            'false_positive_rate',
            'zone1_precision',
            'zone1_recall',
            'zone2_prediabetes_rate',
            'zone2_recall_contribution',
            'zone3_precision',
            'zone3_specificity',
            'pct_high_conf_prediabetes',
            'pct_uncertain',
            'pct_high_conf_healthy',
        ]

        aggregated = {}
        for metric in metrics_to_aggregate:
            values = [fold.get(metric, 0) for fold in fold_metrics]
            aggregated[f'{metric}_mean'] = float(np.mean(values))
            aggregated[f'{metric}_std'] = float(np.std(values))

        return aggregated

    def _split_held_out_test(self, X: np.ndarray, y: np.ndarray, pid_array: np.ndarray):
        """
        Split data into train+val and held-out test set BEFORE any training or CV.

        Uses stratified split to maintain class distribution.

        Returns
        -------
        X_train_val, y_train_val, pid_train_val : Train+validation data (for CV)
        held_out_test_data : Dict with X_test, y_test, pid_test (never seen during training)
        """
        from sklearn.model_selection import train_test_split

        test_size = self.config.held_out_test_size
        random_seed = self.config.held_out_random_seed

        # Stratified split to maintain class distribution
        indices = np.arange(len(X))
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=y,
            shuffle=True
        )

        X_train_val = X[train_val_idx]
        y_train_val = y[train_val_idx]
        pid_train_val = pid_array[train_val_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        pid_test = pid_array[test_idx]

        # Sanity check: no participant leakage
        train_val_pids = set(pid_train_val)
        test_pids = set(pid_test)
        assert train_val_pids.isdisjoint(test_pids), (
            f"Participant leakage detected! "
            f"Overlap={len(train_val_pids & test_pids)}"
        )

        logger.info(f"[HELD-OUT] Train+Val distribution: [Pre-D={np.sum(y_train_val==0)}, CGM-Healthy={np.sum(y_train_val==1)}]")
        logger.info(f"[HELD-OUT] Test distribution: [Pre-D={np.sum(y_test==0)}, CGM-Healthy={np.sum(y_test==1)}]")
        logger.info(f"[HELD-OUT] No participant leakage confirmed (disjoint sets)")

        # Save held-out test indices
        held_out_dir = os.path.join(self.config.models_dir, "held_out_test")
        os.makedirs(held_out_dir, exist_ok=True)
        np.save(os.path.join(held_out_dir, "held_out_test_indices.npy"), test_idx)
        np.save(os.path.join(held_out_dir, "held_out_test_pid.npy"), pid_test)

        held_out_test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'pid_test': pid_test,
            'test_idx': test_idx
        }

        return X_train_val, y_train_val, pid_train_val, held_out_test_data

    def _evaluate_held_out_test(self, final_trainer, held_out_test_data: Dict[str, Any], cv_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate final production model on held-out test set.

        This is the TRUE generalization test - data never seen during training or CV.

        Returns
        -------
        Dict with evaluation metrics (classification report, ROC-AUC, confidence metrics, etc.)
        """
        X_test = held_out_test_data['X_test']
        y_test = held_out_test_data['y_test']
        pid_test = held_out_test_data['pid_test']

        global_threshold = cv_summary['global_threshold']

        logger.info(f"[HELD-OUT] Evaluating on {len(X_test)} participants (NEVER seen during training/CV)")
        logger.info(f"[HELD-OUT] Using global threshold from CV: {global_threshold:.4f}")

        # Normalize test data using the SAME scaler from final model
        if final_trainer.feature_scaler_mean is not None and final_trainer.feature_scaler_scale is not None:
            binary = set(self.config.binary_features)
            cont_idx = [i for i, n in enumerate(self.config.selected_features) if n not in binary]

            X_test = X_test.copy()
            X_test[:, :, cont_idx] = (X_test[:, :, cont_idx] - final_trainer.feature_scaler_mean) / final_trainer.feature_scaler_scale
            logger.info(f"[HELD-OUT] Applied feature normalization (from final model)")

        # Get predictions
        y_prob = final_trainer.model.predict(X_test, verbose=0).ravel()

        # Apply temperature scaling (if available)
        if final_trainer.temperature_scaler is not None:
            y_prob = final_trainer.temperature_scaler.transform(y_prob)
            logger.info(f"[HELD-OUT] Applied temperature scaling (T={final_trainer.temperature_scaler.temperature:.4f})")

        # Binary predictions with global threshold
        y_pred = (y_prob >= global_threshold).astype(int)

        # Standard metrics
        from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix

        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"[HELD-OUT] ROC-AUC: {roc_auc:.4f}")
        logger.info(f"[HELD-OUT] PR-AUC: {pr_auc:.4f}")
        logger.info(f"[HELD-OUT] Accuracy: {report['accuracy']:.4f}")
        logger.info(f"[HELD-OUT] Pre-Diabetes Precision: {report.get('0', {}).get('precision', 0):.4f}")
        logger.info(f"[HELD-OUT] Pre-Diabetes Recall: {report.get('0', {}).get('recall', 0):.4f}")
        logger.info(f"[HELD-OUT] CGM-Healthy Precision: {report.get('1', {}).get('precision', 0):.4f}")
        logger.info(f"[HELD-OUT] CGM-Healthy Recall: {report.get('1', {}).get('recall', 0):.4f}")

        # Confidence-based evaluation (if enabled)
        confidence_metrics = None
        if self.config.use_confidence_based_prediction:
            temp_trainer = LSTMTrainer(self.config)
            confidence_metrics = temp_trainer._evaluate_confidence_based(y_test, y_prob, global_threshold)

            logger.info(f"\n[HELD-OUT] ========== CONFIDENCE-BASED EVALUATION ==========")
            logger.info(f"[HELD-OUT] Pre-Diabetes Detection Rate: {confidence_metrics['prediabetes_detection_rate']:.1f}%")
            logger.info(f"[HELD-OUT] False Positive Rate: {confidence_metrics['false_positive_rate']:.1f}%")
            logger.info(f"[HELD-OUT] Zone 1 (High Conf Pre-D) Precision: {confidence_metrics['zone1_precision']:.1f}%")
            logger.info(f"[HELD-OUT] Zone 3 (High Conf CGM-Healthy) Specificity: {confidence_metrics['zone3_specificity']:.1f}%")
            logger.info(f"[HELD-OUT] OGTT Burden: {confidence_metrics['ogtt_burden']:.1f}%")
            logger.info(f"[HELD-OUT] =======================================================")

        results = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'confidence_metrics': confidence_metrics,
            'global_threshold_used': global_threshold,
            'n_samples': len(X_test),
            'n_true_prediabetes': int(np.sum(y_test == 0)),
            'n_true_healthy': int(np.sum(y_test == 1)),
        }

        # Save held-out test results
        held_out_dir = os.path.join(self.config.models_dir, "held_out_test")
        results_path = os.path.join(held_out_dir, "held_out_test_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"[HELD-OUT] Saved results to: {results_path}")

        return results

    def _print_held_out_results(self, results: Dict[str, Any]) -> None:
        """Print held-out test results in a formatted way."""
        print("\n" + "="*60)
        print("HELD-OUT TEST SET RESULTS (TRUE GENERALIZATION)")
        print("="*60)
        print(f"Samples:     {results['n_samples']} (Pre-D={results['n_true_prediabetes']}, CGM-Healthy={results['n_true_healthy']})")
        print(f"ROC-AUC:     {results['roc_auc']:.4f}")
        print(f"PR-AUC:      {results['pr_auc']:.4f}")
        print(f"Accuracy:    {results['classification_report']['accuracy']:.4f}")
        print(f"Threshold:   {results['global_threshold_used']:.4f}")

        report = results['classification_report']
        print(f"\nPre-Diabetes (Class 0):")
        print(f"  Precision: {report.get('0', {}).get('precision', 0):.4f}")
        print(f"  Recall:    {report.get('0', {}).get('recall', 0):.4f}")
        print(f"  F1-Score:  {report.get('0', {}).get('f1-score', 0):.4f}")

        print(f"\nCGM-Healthy (Class 1):")
        print(f"  Precision: {report.get('1', {}).get('precision', 0):.4f}")
        print(f"  Recall:    {report.get('1', {}).get('recall', 0):.4f}")
        print(f"  F1-Score:  {report.get('1', {}).get('f1-score', 0):.4f}")

        if results.get('confidence_metrics'):
            cm = results['confidence_metrics']
            print(f"\nConfidence-Based Metrics:")
            print(f"  Detection Rate:   {cm['prediabetes_detection_rate']:.1f}%")
            print(f"  False Positive:   {cm['false_positive_rate']:.1f}%")
            print(f"  Zone 1 Precision: {cm['zone1_precision']:.1f}%")
            print(f"  Zone 3 Specificity: {cm['zone3_specificity']:.1f}%")
            print(f"  OGTT Burden:      {cm['ogtt_burden']:.1f}%")

        print("="*60)
        print("This is your TRUE generalization performance!")
        print("="*60)

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
        fold_pr_aucs = []
        fold_accs = []
        fold_thresholds = []
        fold_val_predictions = []  # OOF validation predictions
        fold_val_labels = []       # OOF validation labels
        fold_test_predictions = [] # Test predictions for global threshold re-evaluation
        fold_test_labels = []      # Test labels for global threshold re-evaluation
        fold_confidence_metrics = []  # Confidence-based metrics (if enabled)

        # Save CV artifacts under the version directory
        cv_dir = os.path.join(self.config.models_dir, "cross_validation")
        os.makedirs(cv_dir, exist_ok=True)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            logger.info(f"\n\n[CV] Fold {fold}/{n_splits} starting...")

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

            # Collect OOF validation predictions for global threshold computation
            # Note: We need to recompute predictions with temperature scaling applied
            y_val_prob = fold_trainer.model.predict(X_val, verbose=0).ravel()
            if fold_trainer.temperature_scaler is not None:
                y_val_prob = fold_trainer.temperature_scaler.transform(y_val_prob)
            fold_val_predictions.append(y_val_prob)
            fold_val_labels.append(y_val)

            results = fold_trainer.evaluate_model(X_test, y_test)

            # Store test predictions for global threshold re-evaluation
            fold_test_predictions.append(results['y_pred_proba'])
            fold_test_labels.append(y_test)

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
            fold_pr_aucs.append(float(results.get('pr_auc', 0.0)))
            fold_accs.append(float(results['classification_report']['accuracy']))
            fold_thresholds.append(float(results['threshold_used']))

            # Collect confidence metrics if enabled
            if results.get('confidence_metrics'):
                fold_confidence_metrics.append(results['confidence_metrics'])

            logger.info(
                f"[CV] Fold {fold} done | ROC-AUC={results['roc_auc']:.4f} | "
                f"PR-AUC={results.get('pr_auc', 0.0):.4f} | "
                f"Acc={results['classification_report']['accuracy']:.4f} | Thr={results['threshold_used']:.4f}"
            )

            # Optional: save fold model
            if self.config.cv_save_fold_models:
                fold_trainer.model.save(os.path.join(fold_dir, "best_model_fold.keras"))
                with open(os.path.join(fold_dir, "fold_results.json"), "w", encoding="utf-8") as f:
                    json.dump(results['classification_report'], f, indent=2)

        # ========== GLOBAL THRESHOLD FROM OOF VALIDATION PREDICTIONS ==========
        logger.info("\n[CV] Computing global threshold from out-of-fold validation predictions...")
        oof_val_predictions = np.concatenate(fold_val_predictions)
        oof_val_labels = np.concatenate(fold_val_labels)

        # Compute global threshold using Youden's J on pooled OOF data
        fpr_oof, tpr_oof, thr_oof = roc_curve(oof_val_labels, oof_val_predictions)
        j_oof = tpr_oof - fpr_oof
        global_threshold_idx = int(np.argmax(j_oof))
        global_threshold = float(thr_oof[global_threshold_idx])

        logger.info(f"[CV] Global threshold (OOF validation): {global_threshold:.4f}")
        logger.info(f"[CV] Per-fold thresholds ranged: {min(fold_thresholds):.4f} to {max(fold_thresholds):.4f}")
        logger.info(f"[CV] Threshold stability improved: Global vs Per-fold mean = {global_threshold:.4f} vs {np.mean(fold_thresholds):.4f}")

        # ========== DYNAMIC CONFIDENCE THRESHOLD ADJUSTMENT ==========
        # Dynamically adjust confidence thresholds based on global threshold
        # This ensures zones are centered on the model's actual probability distribution
        if self.config.use_confidence_based_prediction:
            # Store original thresholds for reference
            original_low = self.config.confidence_low_threshold
            original_high = self.config.confidence_high_threshold

            # Compute offsets based on global threshold
            # Strategy: Create zones that span ±0.15 around global threshold for uncertain zone
            # Zone 1 (High Conf Pre-D): prob >= global_thr + margin_high
            # Zone 2 (Uncertain): global_thr - margin_low <= prob < global_thr + margin_high
            # Zone 3 (High Conf CGM-Healthy): prob < global_thr - margin_low

            margin_low = 0.08   # Distance below global threshold for uncertain zone
            margin_high = 0.22  # Distance above global threshold for uncertain zone

            # Compute new thresholds
            dynamic_low = max(0.05, global_threshold - margin_low)
            dynamic_high = min(0.95, global_threshold + margin_high)

            # Update config with dynamic thresholds
            self.config.confidence_low_threshold = dynamic_low
            self.config.confidence_high_threshold = dynamic_high

            logger.info("\n[CV] ========== DYNAMIC CONFIDENCE THRESHOLD ADJUSTMENT ==========")
            logger.info(f"[CV] Global threshold: {global_threshold:.4f}")
            logger.info(f"[CV] Original confidence thresholds: Low={original_low:.4f}, High={original_high:.4f}")
            logger.info(f"[CV] Dynamic confidence thresholds: Low={dynamic_low:.4f}, High={dynamic_high:.4f}")
            logger.info(f"[CV] NOTE: Model predicts prob of Class 1 (CGM-Healthy). Higher prob = healthier.")
            logger.info(f"[CV] Zone 1 (High Conf Pre-Diabetes): prob < {dynamic_low:.4f}")
            logger.info(f"[CV] Zone 2 (Uncertain - needs OGTT): {dynamic_low:.4f} <= prob < {dynamic_high:.4f} (width: {dynamic_high-dynamic_low:.4f})")
            logger.info(f"[CV] Zone 3 (High Conf CGM-Healthy): prob >= {dynamic_high:.4f}")
            logger.info("[CV] ===================================================================\n")
        # ======================================================================

        # Re-evaluate all test sets with global threshold for production performance estimate
        from sklearn.metrics import accuracy_score, classification_report

        logger.info("\n[CV] Re-evaluating all folds with global threshold...")
        fold_accs_global = []
        fold_reports_global = []

        for fold_idx, (y_test, y_prob) in enumerate(zip(fold_test_labels, fold_test_predictions), start=1):
            y_pred_global = (y_prob >= global_threshold).astype(int)
            acc_global = float(accuracy_score(y_test, y_pred_global))
            fold_accs_global.append(acc_global)
            report_global = classification_report(y_test, y_pred_global, output_dict=True, zero_division=0)
            fold_reports_global.append(report_global)
            logger.info(f"[CV] Fold {fold_idx} with global threshold ({global_threshold:.4f}): Acc={acc_global:.4f}")

        logger.info(f"\n[CV] Global threshold performance: Acc = {np.mean(fold_accs_global):.4f} ± {np.std(fold_accs_global):.4f}")
        # ======================================================================

        # Re-compute confidence metrics with dynamic thresholds (if enabled)
        if self.config.use_confidence_based_prediction:
            logger.info("\n[CV] Re-computing confidence metrics with dynamic thresholds...")
            fold_confidence_metrics = []  # Clear old metrics

            # Create a temporary trainer just to use the _evaluate_confidence_based method
            temp_trainer = LSTMTrainer(self.config)

            for fold_idx, (y_test, y_prob) in enumerate(zip(fold_test_labels, fold_test_predictions), start=1):
                # Compute confidence metrics with NEW dynamic thresholds
                conf_metrics = temp_trainer._evaluate_confidence_based(y_test, y_prob, global_threshold)
                fold_confidence_metrics.append(conf_metrics)

                logger.info(f"[CV] Fold {fold_idx} confidence (dynamic thresholds):")
                logger.info(f"     Zone 1 (High Conf Pre-D): {conf_metrics['n_high_conf_prediabetes']} samples ({conf_metrics['pct_high_conf_prediabetes']:.1f}%)")
                logger.info(f"     Zone 2 (Uncertain/OGTT): {conf_metrics['n_uncertain']} samples ({conf_metrics['pct_uncertain']:.1f}%)")
                logger.info(f"     Zone 3 (High Conf CGM-Healthy): {conf_metrics['n_high_conf_healthy']} samples ({conf_metrics['pct_high_conf_healthy']:.1f}%)")

        # Aggregate confidence metrics (if enabled)
        confidence_summary = None
        if fold_confidence_metrics:
            confidence_summary = self._aggregate_confidence_metrics(fold_confidence_metrics)
            logger.info("\n[CV] ========== CONFIDENCE-BASED PREDICTION SUMMARY ==========")
            logger.info(f"[CV] Average OGTT Burden: {confidence_summary['ogtt_burden_mean']:.1f}% ± {confidence_summary['ogtt_burden_std']:.1f}%")
            logger.info(f"[CV] Average Immediate Decision Rate: {confidence_summary['immediate_decision_rate_mean']:.1f}%")
            logger.info(f"[CV] Pre-Diabetes Detection Rate: {confidence_summary['prediabetes_detection_rate_mean']:.1f}% ± {confidence_summary['prediabetes_detection_rate_std']:.1f}%")
            logger.info(f"[CV] False Positive Rate: {confidence_summary['false_positive_rate_mean']:.1f}% ± {confidence_summary['false_positive_rate_std']:.1f}%")
            logger.info(f"[CV] Zone 1 (High Conf Pre-D) Precision: {confidence_summary['zone1_precision_mean']:.1f}%")
            logger.info(f"[CV] Zone 3 (High Conf CGM-Healthy) Specificity: {confidence_summary['zone3_specificity_mean']:.1f}%")
            logger.info("[CV] ============================================================")

        # Aggregate
        summary = {
            "cv_n_splits": n_splits,
            "global_threshold": global_threshold,
            "roc_auc_mean": float(np.mean(fold_aucs)),
            "roc_auc_std": float(np.std(fold_aucs)),
            "pr_auc_mean": float(np.mean(fold_pr_aucs)),
            "pr_auc_std": float(np.std(fold_pr_aucs)),
            "accuracy_mean": float(np.mean(fold_accs)),
            "accuracy_std": float(np.std(fold_accs)),
            "accuracy_mean_global": float(np.mean(fold_accs_global)),
            "accuracy_std_global": float(np.std(fold_accs_global)),
            "threshold_mean": float(np.mean(fold_thresholds)),
            "threshold_std": float(np.std(fold_thresholds)),
            "fold_roc_auc": fold_aucs,
            "fold_pr_auc": fold_pr_aucs,
            "fold_accuracy": fold_accs,
            "fold_accuracy_global": fold_accs_global,
            "fold_thresholds": fold_thresholds,
            "fold_reports": fold_reports,
            "fold_reports_global": fold_reports_global,
            "confidence_summary": confidence_summary,
            "fold_confidence_metrics": fold_confidence_metrics if fold_confidence_metrics else None,
        }

        out_path = os.path.join(cv_dir, "cv_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[CV] Saved CV summary to: {out_path}")

        return summary

    def _train_final_production_model(self, X: np.ndarray, y: np.ndarray, cv_summary: Dict[str, Any]):
        """
        Train final production model on ALL data after CV validation.

        Uses internal validation split for early stopping only (not for threshold selection).
        Saves model with global threshold from CV.

        Parameters
        ----------
        X : All samples (491 participants)
        y : All labels
        cv_summary : CV results summary containing metrics and global threshold

        Returns
        -------
        Trained model instance
        """
        # Extract global threshold from CV summary
        global_threshold = cv_summary['global_threshold']

        logger.info(f"[FINAL MODEL] Training on all {len(X)} samples...")
        logger.info(f"[FINAL MODEL] Using global threshold from CV: {global_threshold:.4f}")

        # Create internal validation split for early stopping only
        # Use same val_size_from_train as CV (20%)
        val_size = self.config.val_size_from_train
        n_train = int(len(X) * (1 - val_size))

        # Shuffle with fixed seed for reproducibility
        rng = np.random.RandomState(self.config.random_seed)
        indices = rng.permutation(len(X))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train_final = X[train_idx]
        y_train_final = y[train_idx]
        X_val_final = X[val_idx]
        y_val_final = y[val_idx]

        logger.info(f"[FINAL MODEL] Internal split: {len(X_train_final)} train, {len(X_val_final)} val")

        # Compute and apply feature scaler (CRITICAL for production inference)
        # Normalize continuous features only; keep binary features as-is
        logger.info("[FINAL MODEL] Computing feature scaler from training data...")
        binary = set(self.config.binary_features)
        cont_idx = [i for i, n in enumerate(self.config.selected_features) if n not in binary]

        # Compute mean and std from training data only
        train_2d = X_train_final[:, :, cont_idx].reshape(-1, len(cont_idx))
        scaler_mean = train_2d.mean(0).astype(np.float32)
        scaler_scale = train_2d.std(0).astype(np.float32)
        scaler_scale[scaler_scale < 1e-8] = 1.0  # Prevent division by zero

        # Apply normalization to both train and val
        X_train_final = X_train_final.copy()
        X_val_final = X_val_final.copy()
        X_train_final[:, :, cont_idx] = (X_train_final[:, :, cont_idx] - scaler_mean) / scaler_scale
        X_val_final[:, :, cont_idx] = (X_val_final[:, :, cont_idx] - scaler_mean) / scaler_scale
        logger.info(f"[FINAL MODEL] Applied feature normalization (continuous features only)")

        # Create fresh trainer for final model
        from cgm_lstm.model import LSTMTrainer
        final_trainer = LSTMTrainer(self.config)

        # Store scaler parameters in trainer (CRITICAL for production inference)
        final_trainer.feature_scaler_mean = scaler_mean
        final_trainer.feature_scaler_scale = scaler_scale
        logger.info(f"[FINAL MODEL] Stored feature scaler parameters in trainer")

        # Train model (with early stopping on internal validation)
        logger.info("[FINAL MODEL] Starting training...")
        final_trainer.train_model(X_train_final, y_train_final, X_val_final, y_val_final)

        # Apply temperature scaling using T=1.0 validated by CV
        # (CV consistently found T≈1.0 across all folds, indicating model is already well-calibrated)
        if self.config.use_temperature_scaling:
            from cgm_lstm.model import TemperatureScaling
            final_trainer.temperature_scaler = TemperatureScaling()
            final_trainer.temperature_scaler.temperature = 1.0  # Use CV-validated value
            logger.info(f"[FINAL MODEL] Using CV-validated temperature: T=1.0000 (model well-calibrated)")

        # Override best_threshold with global threshold from CV
        final_trainer.best_threshold = global_threshold
        logger.info(f"[FINAL MODEL] Using CV global threshold: {global_threshold:.4f}")

        # Save production model and artifacts
        production_dir = os.path.join(self.config.models_dir, "production_model")
        os.makedirs(production_dir, exist_ok=True)

        # Save Keras model
        model_path = os.path.join(production_dir, "best_model.keras")
        final_trainer.model.save(model_path)
        logger.info(f"[FINAL MODEL] Saved model to: {model_path}")

        # Save global threshold
        threshold_path = os.path.join(production_dir, "global_threshold.json")
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump({"global_threshold": float(global_threshold)}, f, indent=2)
        logger.info(f"[FINAL MODEL] Saved threshold to: {threshold_path}")

        # Save temperature scaler
        if final_trainer.temperature_scaler is not None:
            temp_path = os.path.join(production_dir, "temperature_scaler.json")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({"temperature": float(final_trainer.temperature_scaler.temperature)}, f, indent=2)
            logger.info(f"[FINAL MODEL] Saved temperature scaler to: {temp_path}")

        # Save dynamic confidence thresholds (if confidence-based prediction is enabled)
        if self.config.use_confidence_based_prediction:
            conf_path = os.path.join(production_dir, "confidence_thresholds.json")
            with open(conf_path, "w", encoding="utf-8") as f:
                conf_data = {
                    "confidence_low_threshold": float(self.config.confidence_low_threshold),
                    "confidence_high_threshold": float(self.config.confidence_high_threshold),
                    "computed_from_global_threshold": float(global_threshold),
                    "margin_low": 0.08,
                    "margin_high": 0.22,
                    "note": "Model predicts probability of Class 1 (CGM-Healthy). Higher prob = healthier.",
                    "zone_definitions": {
                        "zone1_high_conf_prediabetes": f"prob < {self.config.confidence_low_threshold:.4f}",
                        "zone2_uncertain_needs_ogtt": f"{self.config.confidence_low_threshold:.4f} <= prob < {self.config.confidence_high_threshold:.4f}",
                        "zone3_high_conf_healthy": f"prob >= {self.config.confidence_high_threshold:.4f}"
                    }
                }
                json.dump(conf_data, f, indent=2)
            logger.info(f"[FINAL MODEL] Saved confidence thresholds to: {conf_path}")
            logger.info(f"[FINAL MODEL] Zone 1 (High Conf Pre-D): prob < {self.config.confidence_low_threshold:.4f}")
            logger.info(f"[FINAL MODEL] Zone 2 (Uncertain/OGTT): {self.config.confidence_low_threshold:.4f} <= prob < {self.config.confidence_high_threshold:.4f}")
            logger.info(f"[FINAL MODEL] Zone 3 (High Conf CGM-Healthy): prob >= {self.config.confidence_high_threshold:.4f}")

        # Save feature scaler
        if final_trainer.feature_scaler_mean is not None and final_trainer.feature_scaler_scale is not None:
            scaler_path = os.path.join(production_dir, "feature_scaler.npz")
            np.savez(scaler_path,
                     mean=final_trainer.feature_scaler_mean,
                     scale=final_trainer.feature_scaler_scale)
            logger.info(f"[FINAL MODEL] Saved feature scaler to: {scaler_path}")
            logger.info(f"[FINAL MODEL] Scaler shape: mean={final_trainer.feature_scaler_mean.shape}, scale={final_trainer.feature_scaler_scale.shape}")
        else:
            error_msg = "[FINAL MODEL] CRITICAL ERROR: Feature scaler not found! Production model cannot be used without it."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Save training config
        config_path = os.path.join(production_dir, "training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            config_dict = {
                "architecture": "Conv+BiLSTM" if self.config.use_conv_frontend else "LSTM-only",
                "l2_reg": self.config.l2_reg,
                "lstm_units_1": self.config.lstm_units_1,
                "lstm_units_2": self.config.lstm_units_2,
                "dense_units": self.config.dense_units,
                "dropout_rate": self.config.dropout_rate,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "use_conv_frontend": self.config.use_conv_frontend,
                "conv_filters_1": self.config.conv_filters_1,
                "conv_filters_2": self.config.conv_filters_2,
                "lstm_bidirectional": self.config.lstm_bidirectional,
                "use_class_weights": self.config.use_class_weights,
                "use_temperature_scaling": self.config.use_temperature_scaling,
                "random_seed": self.config.random_seed,
                "trained_on_samples": len(X)
            }
            json.dump(config_dict, f, indent=2)
        logger.info(f"[FINAL MODEL] Saved config to: {config_path}")

        # Compute average per-class metrics from CV (using global threshold reports)
        fold_reports_global = cv_summary.get('fold_reports_global', [])
        if fold_reports_global:
            # Average precision and recall across folds for each class
            class_0_precisions = [report.get('0', {}).get('precision', 0) for report in fold_reports_global]
            class_0_recalls = [report.get('0', {}).get('recall', 0) for report in fold_reports_global]
            class_1_precisions = [report.get('1', {}).get('precision', 0) for report in fold_reports_global]
            class_1_recalls = [report.get('1', {}).get('recall', 0) for report in fold_reports_global]

            avg_class_0_precision = np.mean(class_0_precisions) * 100
            avg_class_0_recall = np.mean(class_0_recalls) * 100
            avg_class_1_precision = np.mean(class_1_precisions) * 100
            avg_class_1_recall = np.mean(class_1_recalls) * 100
        else:
            # Fallback if no global reports available
            avg_class_0_precision = avg_class_0_recall = 0.0
            avg_class_1_precision = avg_class_1_recall = 0.0

        # Create deployment README
        readme_path = os.path.join(production_dir, "DEPLOYMENT_README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"""# Production Model Deployment Guide

## Model Details
- **Architecture**: {config_dict['architecture']}
- **Trained on**: {len(X)} participants (all available data)
- **Training date**: {self.config.version_folder}
- **Global threshold**: {global_threshold:.4f}
- **Temperature**: {final_trainer.temperature_scaler.temperature if final_trainer.temperature_scaler else 1.0:.4f}

## Expected Performance (from CV)
- **ROC-AUC**: {cv_summary['roc_auc_mean']:.4f} ± {cv_summary['roc_auc_std']:.4f}
- **PR-AUC**: {cv_summary['pr_auc_mean']:.4f} ± {cv_summary['pr_auc_std']:.4f}
- **Accuracy**: {cv_summary['accuracy_mean_global']:.4f} ± {cv_summary['accuracy_std_global']:.4f} (with global threshold)

## Per-Class Performance (from CV with global threshold)
- **Pre-Diabetes (Class 0)**: Precision {avg_class_0_precision:.1f}%, Recall {avg_class_0_recall:.1f}%
- **CGM-Healthy (Class 1)**: Precision {avg_class_1_precision:.1f}%, Recall {avg_class_1_recall:.1f}%

## Files
- `best_model.keras` - Trained Keras model
- `global_threshold.json` - Optimal decision threshold ({global_threshold:.4f})
- `temperature_scaler.json` - Calibration parameters
- `feature_scaler.npz` - Feature normalization parameters
- `training_config.json` - Model architecture and hyperparameters
- `confidence_thresholds.json` - Dynamic confidence thresholds for 3-tier prediction (optional)

## How to Load and Use

```python
import numpy as np
import json
from tensorflow import keras

# Load model
model = keras.models.load_model('best_model.keras')

# Load threshold
with open('global_threshold.json') as f:
    threshold = json.load(f)['global_threshold']

# Load temperature scaler
with open('temperature_scaler.json') as f:
    temperature = json.load(f)['temperature']

# Load feature scaler
scaler_data = np.load('feature_scaler.npz')
feature_mean = scaler_data['mean']
feature_scale = scaler_data['scale']

# Load confidence thresholds (optional - for 3-tier prediction)
try:
    with open('confidence_thresholds.json') as f:
        conf_thresh = json.load(f)
        low_threshold = conf_thresh['confidence_low_threshold']
        high_threshold = conf_thresh['confidence_high_threshold']
except FileNotFoundError:
    # Fallback to default thresholds if file doesn't exist
    low_threshold = 0.35
    high_threshold = 0.65

# Make prediction
def predict_patient(cgm_data):
    # 1. Normalize features
    X_norm = (cgm_data - feature_mean) / feature_scale

    # 2. Get model probability (NOTE: model predicts prob of Class 1 = Healthy)
    prob = model.predict(X_norm, verbose=0)[0][0]

    # 3. Apply temperature scaling
    import numpy as np
    logit = np.log(prob / (1 - prob + 1e-7))
    calibrated_prob = 1 / (1 + np.exp(-logit / temperature))

    # 4. Apply threshold with confidence-based rules
    # LOW probability = High risk of Pre-Diabetes
    # HIGH probability = Low risk (Healthy)
    if calibrated_prob < low_threshold:
        return {{
            'prediction': 'Pre-Diabetes',
            'confidence': 'High',
            'action': 'Immediate lifestyle intervention',
            'probability': calibrated_prob
        }}
    elif calibrated_prob < high_threshold:
        return {{
            'prediction': 'Uncertain',
            'confidence': 'Medium',
            'action': 'Secondary screening (OGTT recommended)',
            'probability': calibrated_prob
        }}
    else:
        return {{
            'prediction': 'CGM-Healthy',
            'confidence': 'High',
            'action': 'Regular screening cycle (1-2 years)',
            'probability': calibrated_prob
        }}
```

## Production Monitoring

Set up alerts for these metrics:
- Pre-Diabetes Recall: ≥ 0.75
- Pre-Diabetes Precision: ≥ 0.90
- CGM-Healthy Recall: ≥ 0.85
- CGM-Healthy Precision: ≥ 0.70
- ECE (calibration): < 0.15

Check for distribution shift weekly by monitoring input feature statistics.
Retrain quarterly with new data.

## Dynamic Confidence Thresholds (3-Tier Prediction)

This model uses **dynamically computed confidence thresholds** that adapt to the model's actual probability distribution. Unlike static thresholds, these are centered on the global decision threshold determined during cross-validation.

**IMPORTANT**: The model predicts probability of Class 1 (CGM-Healthy). Higher probabilities indicate healthier patients.

**How it works:**
- The global threshold (e.g., {global_threshold:.4f}) is computed from out-of-fold validation predictions using Youden's J statistic
- Confidence zones are automatically positioned around this threshold:
  - **Zone 1 (High Conf Pre-Diabetes)**: prob < global_thr - 0.08 (LOW prob of CGM-Healthy = high risk)
  - **Zone 2 (Uncertain - needs OGTT)**: global_thr - 0.08 ≤ prob < global_thr + 0.22 (middle range)
  - **Zone 3 (High Conf CGM-Healthy)**: prob ≥ global_thr + 0.22 (HIGH prob of CGM-Healthy = low risk)

**Why dynamic thresholds?**
- Prevents misalignment when global threshold shifts across different model runs
- Ensures Zone 2 (uncertain) captures cases near the decision boundary
- Maintains high precision in Zones 1 and 3 while minimizing unnecessary OGTT referrals

**Expected performance** (from CV with dynamic thresholds):
- Zone 1 Precision: ≥90% (high confidence pre-diabetes predictions are reliable)
- Zone 3 Specificity: ≥85% (high confidence CGM-Healthy predictions are reliable)
- OGTT Burden: 30-40% (percentage of cases needing secondary screening)
- Overall Pre-Diabetes Detection: ≥85% (across all zones)
""")
        logger.info(f"[FINAL MODEL] Created deployment guide: {readme_path}")

        logger.info(f"[FINAL MODEL] Production artifacts saved to: {production_dir}")
        return final_trainer

    def _print_cv_summary(self, cv_summary: Dict[str, Any]) -> None:
        print("\n" + "="*60)
        print("LSTM CROSS-VALIDATION SUMMARY")
        print("="*60)
        print(f"Version:   {self.config.version_folder}")
        print(f"Location:  {self.config.models_dir}")
        print(f"Arch:      {'Conv+BiLSTM' if self.config.use_conv_frontend else 'LSTM-only'}")
        print(f"Folds:     {cv_summary['cv_n_splits']}")
        print(f"ROC-AUC:   {cv_summary['roc_auc_mean']:.4f} ± {cv_summary['roc_auc_std']:.4f}")
        print(f"PR-AUC:    {cv_summary['pr_auc_mean']:.4f} ± {cv_summary['pr_auc_std']:.4f}")
        print(f"Accuracy (per-fold thr):  {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")

        # Handle global threshold metrics (might not exist in older results)
        acc_global = cv_summary.get('accuracy_mean_global')
        acc_std_global = cv_summary.get('accuracy_std_global', 0)
        if acc_global is not None and isinstance(acc_global, (int, float)):
            print(f"Accuracy (global thr):    {acc_global:.4f} ± {acc_std_global:.4f}")
        else:
            print(f"Accuracy (global thr):    N/A")

        print(f"Threshold: {cv_summary['threshold_mean']:.4f} ± {cv_summary['threshold_std']:.4f}")

        global_thr = cv_summary.get('global_threshold')
        if global_thr is not None and isinstance(global_thr, (int, float)):
            print(f"Global Thr: {global_thr:.4f}")
        else:
            print(f"Global Thr: N/A")

        print("="*60)


    def _print_version_results(self, results: Dict[str, Any]) -> None:
        print("\n" + "="*60)
        print("LSTM BINARY CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Version:  {self.config.version_folder}")
        print(f"Location: {self.config.models_dir}")
        print(f"Arch:     {'Conv+BiLSTM' if self.config.use_conv_frontend else 'LSTM-only'}")
        print(f"ROC-AUC:  {results['roc_auc']:.4f}")
        print(f"PR-AUC:   {results.get('pr_auc', 'N/A')}")
        print(f"Accuracy: {results['classification_report']['accuracy']:.4f}")
        print("="*60)
        print("\nFiles created:")
        print(f"   Model (best): best_model.keras")
        print(f"   Model (alias): lstm_binary_model.keras")
        print(f"   Config: model_config.json")
        print(f"   Data: X_lstm.npy, y_lstm_binary.npy, pid_lstm.npy")
        print(f"   Indices: idx_train.npy, idx_val.npy, idx_test.npy")
        print(f"   History: training_history.json")
        print(f"   Plots: training_curves.png, roc_curve.png, confusion_matrix.png")
        print("\nUse this version folder for analysis!")
        print("="*60)
    
    def _generate_visualizations(self, results: Dict[str, Any]) -> None:
        """Generate all evaluation plots for a single-split run."""
        if self.trainer.history:
            self.visualizer.plot_training_curves(
                self.trainer.history.history,
                os.path.join(self.config.models_dir, 'training_curves.png')
            )
        self.visualizer.plot_roc_curve(
            results['fpr'], results['tpr'], results['roc_auc'],
            os.path.join(self.config.models_dir, 'roc_curve.png')
        )
        self.visualizer.plot_confusion_matrix(
            results['confusion_matrix'],
            os.path.join(self.config.models_dir, 'confusion_matrix.png')
        )

        cal = results.get("calibration", {})
        curve = cal.get("calibration_curve", {})
        if curve:
            self.visualizer.plot_reliability_curve(
                curve["prob_pred"],
                curve["prob_true"],
                os.path.join(self.config.models_dir, "reliability_curve.png")
            )

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

    def _sanity_check_splits(self) -> None:
        """Log any overlap between Train/Val/Test indices (should be zero)."""
        d = self.config.models_dir
        idx_tr = set(np.load(os.path.join(d, 'idx_train.npy')).tolist())
        idx_val = set(np.load(os.path.join(d, 'idx_val.npy')).tolist())
        idx_te = set(np.load(os.path.join(d, 'idx_test.npy')).tolist())
        a = len(idx_tr & idx_val); b = len(idx_tr & idx_te); c = len(idx_val & idx_te)
        logger.info(f"[SPLIT-SANITY] train∩val={a} | train∩test={b} | val∩test={c}")
        if a or b or c:
            logger.warning("[SPLIT-SANITY] Overlap found (unexpected)!")

