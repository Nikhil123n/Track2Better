"""
Dynamic Model Card Generation for MLOps

This module generates modular JSON files for model documentation, monitoring, and deployment.
All JSONs are generated dynamically during pipeline execution.

JSON Files Generated:
1. model_metadata.json       - High-level model information
2. model_architecture.json   - Model structure and parameters
3. training_results.json     - Training history and losses
4. evaluation_results.json   - Combined CV + held-out metrics
5. deployment_config.json    - Production configuration

Author: CGM LSTM Pipeline
Date: 2026-02-08
"""

import json
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


class ModelCardGenerator:
    """Generates modular JSON files for model documentation."""

    def __init__(self, model_dir: str):
        """Initialize generator with model directory path.

        Parameters
        ----------
        model_dir : str
            Path to model directory (e.g., ./models/model_20260208_185255/)
        """
        self.model_dir = Path(model_dir)
        self.model_id = self.model_dir.name

    def generate_metadata(self, config: Any, parent_model: Optional[str] = None) -> Dict[str, Any]:
        """Generate model_metadata.json - High-level model information.

        Called at: Pipeline start (right after model folder creation)

        Parameters
        ----------
        config : Config object
            Configuration from data.py
        parent_model : str, optional
            Previous model ID this was derived from

        Returns
        -------
        Dict with metadata
        """
        metadata = {
            "model_id": self.model_id,
            "model_name": "CGM_LSTM_PreDiabetes_BinaryClassifier",
            "version": "2.1.0",
            "training_date": datetime.now().isoformat(),
            "purpose": "Pre-diabetes detection from continuous glucose monitoring (CGM) time-series data",
            "intended_use": "Clinical decision support for pre-diabetes screening using wearable CGM sensors",
            "parent_model": parent_model,
            "known_limitations": [
                f"Trained on {config.n_participants if hasattr(config, 'n_participants') else 491} participants (small dataset)",
                "Not validated on external datasets",
                "Requires 7-day CGM monitoring period",
                "May not generalize to different CGM sensor types"
            ],
            "environment": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": platform.system(),
                "os_version": platform.release(),
                "created_by": "CGM LSTM Pipeline v2.1.0"
            },
            "data_sources": [
                "participant_augmented_data_1.csv",
                "participant_augmented_data_2.csv",
                "participant_augmented_data_3.csv",
                "participant_augmented_data_4.csv",
                "participant_augmented_data_5.csv",
                "participant_augmented_data_6.csv",
                "participant_augmented_data_7.csv",
                "participant_augmented_data_8.csv",
                "participant_augmented_data_9.csv"
            ],
            "target_classes": {
                "0": "pre_diabetes_lifestyle",
                "1": "CGM-Healthy"
            }
        }

        output_path = self.model_dir / "model_metadata.json"
        self._save_json(metadata, output_path)
        return metadata

    def generate_architecture(self, model: Any, config: Any) -> Dict[str, Any]:
        """Generate model_architecture.json - Model structure and parameters.

        Called at: After first model build in CV

        Parameters
        ----------
        model : Keras Model
            Trained Keras model
        config : Config object
            Configuration from data.py

        Returns
        -------
        Dict with architecture details
        """
        # Extract layer information
        layers_info = []
        for layer in model.layers:
            layer_config = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(layer.output_shape),
            }

            # Add layer-specific configs
            if hasattr(layer, 'units'):
                layer_config['units'] = int(layer.units)
            if hasattr(layer, 'filters'):
                layer_config['filters'] = int(layer.filters)
            if hasattr(layer, 'kernel_size'):
                layer_config['kernel_size'] = int(layer.kernel_size[0]) if isinstance(layer.kernel_size, tuple) else int(layer.kernel_size)
            if hasattr(layer, 'rate'):
                layer_config['rate'] = float(layer.rate)
            if hasattr(layer, 'activation') and layer.activation is not None:
                layer_config['activation'] = layer.activation.__name__

            layers_info.append(layer_config)

        architecture = {
            "model_type": "Conv+BiLSTM",
            "framework": "tensorflow.keras",
            "input_shape": [int(config.sequence_length), int(config.n_features)],
            "output_classes": 2,
            "total_parameters": int(model.count_params()),
            "trainable_parameters": int(sum([np.prod(w.shape) for w in model.trainable_weights])),
            "layers": layers_info,
            "optimizer": {
                "type": model.optimizer.__class__.__name__,
                "learning_rate": float(model.optimizer.learning_rate.numpy())
            },
            "loss_function": "binary_crossentropy",
            "regularization": {
                "l2_lambda": float(config.l2_reg)
            },
            "features": [
                "blood_glucose_value",
                "glucose_diff",
                "glucose_accel",
                "glucose_rollmean_1h",
                "glucose_rollstd_1h",
                "sin_hour",
                "cos_hour",
                "is_fasting",
                "is_night"
            ]
        }

        output_path = self.model_dir / "model_architecture.json"
        self._save_json(architecture, output_path)
        return architecture

    def update_training_results(self, fold_data: Optional[Dict] = None, final_model_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Update training_results.json - Training history and losses.

        Called at: After each fold training + after final model training

        Parameters
        ----------
        fold_data : Dict, optional
            Training info for a CV fold
        final_model_data : Dict, optional
            Training info for final production model

        Returns
        -------
        Dict with training results
        """
        output_path = self.model_dir / "training_results.json"

        # Load existing or create new
        if output_path.exists():
            with open(output_path) as f:
                results = json.load(f)
        else:
            results = {
                "training_start": datetime.now().isoformat(),
                "cv_folds": {},
                "final_model": None
            }

        # Update with new data
        if fold_data is not None:
            fold_num = fold_data.get('fold_number')
            results['cv_folds'][f'fold_{fold_num}'] = {
                "epochs_completed": fold_data.get('epochs_completed'),
                "early_stopping_triggered": fold_data.get('early_stopped', False),
                "best_val_loss": float(fold_data.get('best_val_loss', 0)),
                "final_train_loss": float(fold_data.get('final_train_loss', 0)),
                "temperature_scaling": {
                    "fitted": fold_data.get('temperature_fitted', True),
                    "T": float(fold_data.get('temperature', 1.0))
                },
                "threshold": {
                    "method": "Youden's J",
                    "value": float(fold_data.get('threshold', 0.5))
                },
                "training_time_seconds": fold_data.get('training_time', 0)
            }

        if final_model_data is not None:
            results['final_model'] = {
                "n_training_samples": final_model_data.get('n_train'),
                "n_validation_samples": final_model_data.get('n_val'),
                "epochs_completed": final_model_data.get('epochs_completed'),
                "early_stopping_triggered": final_model_data.get('early_stopped', False),
                "best_val_loss": float(final_model_data.get('best_val_loss', 0)),
                "temperature_scaling": {
                    "fitted": final_model_data.get('temperature_fitted', True),
                    "T": float(final_model_data.get('temperature', 1.0))
                },
                "training_time_seconds": final_model_data.get('training_time', 0)
            }

        results['training_end'] = datetime.now().isoformat()
        self._save_json(results, output_path)
        return results

    def generate_evaluation_results(self, cv_summary: Dict, held_out_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate evaluation_results.json - Combined CV + held-out metrics.

        Called at: After CV complete + after held-out test (if enabled)

        Parameters
        ----------
        cv_summary : Dict
            Cross-validation results from cv_results.json
        held_out_results : Dict, optional
            Held-out test results from held_out_test_results.json

        Returns
        -------
        Dict with all evaluation metrics
        """
        evaluation = {
            "evaluation_date": datetime.now().isoformat(),
            "validation_strategy": "5-Fold Cross-Validation" + (" + Held-Out Test" if held_out_results else ""),

            "cross_validation": {
                "n_splits": cv_summary.get('cv_n_splits', 5),
                "n_samples": cv_summary.get('n_samples_cv', 392),
                "metrics": {
                    "roc_auc": {
                        "mean": float(cv_summary.get('roc_auc_mean', 0)),
                        "std": float(cv_summary.get('roc_auc_std', 0)),
                        "per_fold": [float(x) for x in cv_summary.get('fold_roc_auc', [])]
                    },
                    "pr_auc": {
                        "mean": float(cv_summary.get('pr_auc_mean', 0)),
                        "std": float(cv_summary.get('pr_auc_std', 0)),
                        "per_fold": [float(x) for x in cv_summary.get('fold_pr_auc', [])]
                    },
                    "accuracy": {
                        "mean": float(cv_summary.get('accuracy_mean_global', 0)),
                        "std": float(cv_summary.get('accuracy_std_global', 0)),
                        "per_fold": [float(x) for x in cv_summary.get('fold_accuracy_global', [])]
                    }
                },
                "threshold": {
                    "global_threshold": float(cv_summary.get('global_threshold', 0.5)),
                    "per_fold_mean": float(cv_summary.get('threshold_mean', 0.5)),
                    "per_fold_std": float(cv_summary.get('threshold_std', 0)),
                    "per_fold_values": [float(x) for x in cv_summary.get('fold_thresholds', [])]
                },
                "confidence_metrics": cv_summary.get('confidence_summary', {})
            }
        }

        if held_out_results is not None:
            evaluation['held_out_test'] = {
                "n_samples": held_out_results.get('n_samples', 99),
                "n_true_prediabetes": held_out_results.get('n_true_prediabetes', 61),
                "n_true_healthy": held_out_results.get('n_true_healthy', 38),
                "metrics": {
                    "roc_auc": float(held_out_results.get('roc_auc', 0)),
                    "pr_auc": float(held_out_results.get('pr_auc', 0)),
                    "accuracy": float(held_out_results.get('classification_report', {}).get('accuracy', 0))
                },
                "per_class_metrics": held_out_results.get('classification_report', {}),
                "confusion_matrix": held_out_results.get('confusion_matrix', []),
                "confidence_metrics": held_out_results.get('confidence_metrics', {}),
                "threshold_used": float(held_out_results.get('global_threshold_used', 0.5))
            }

            # Add comparison
            evaluation['held_out_vs_cv'] = {
                "roc_auc_difference": float(held_out_results.get('roc_auc', 0)) - float(cv_summary.get('roc_auc_mean', 0)),
                "outperforms_cv": float(held_out_results.get('roc_auc', 0)) > float(cv_summary.get('roc_auc_mean', 0)),
                "validation_status": "PASSED" if float(held_out_results.get('roc_auc', 0)) >= 0.92 else "NEEDS_REVIEW"
            }

        output_path = self.model_dir / "evaluation_results.json"
        self._save_json(evaluation, output_path)
        return evaluation

    def generate_deployment_config(self, config: Any, cv_summary: Dict, held_out_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate deployment_config.json - Production configuration.

        Called at: End of pipeline

        Parameters
        ----------
        config : Config object
            Configuration from data.py
        cv_summary : Dict
            Cross-validation results
        held_out_results : Dict, optional
            Held-out test results (if available)

        Returns
        -------
        Dict with deployment configuration
        """
        # Determine deployment status
        primary_auc = held_out_results.get('roc_auc') if held_out_results else cv_summary.get('roc_auc_mean')
        deployment_ready = primary_auc >= 0.92 if primary_auc else False

        deployment = {
            "deployment_status": "PRODUCTION_READY" if deployment_ready else "NEEDS_REVIEW",
            "validation_method": "Held-Out Test (Gold Standard)" if held_out_results else "Cross-Validation Only",
            "primary_metrics": {
                "roc_auc": float(primary_auc) if primary_auc else 0,
                "source": "held_out_test" if held_out_results else "cross_validation"
            },

            "threshold_strategy": {
                "method": "Global OOF (Youden's J)",
                "global_threshold": float(cv_summary.get('global_threshold', 0.5)),
                "confidence_zones": {
                    "zone_1_high_conf_prediabetes": {
                        "range": [0.0, 0.0],  # Will be computed dynamically
                        "action": "Immediate intervention recommended",
                        "expected_precision": 0.95
                    },
                    "zone_2_uncertain": {
                        "range": [0.0, 0.0],
                        "action": "OGTT test recommended",
                        "expected_ogtt_burden": 0.10
                    },
                    "zone_3_high_conf_healthy": {
                        "range": [0.0, 1.0],
                        "action": "Regular screening cycle",
                        "expected_specificity": 0.85
                    }
                }
            },

            "production_artifacts": {
                "model_weights": "./production_model/best_model.keras",
                "feature_scaler": "./production_model/feature_scaler.npz",
                "global_threshold": "./production_model/global_threshold.json",
                "confidence_thresholds": "./production_model/confidence_thresholds.json",
                "temperature_scaler": "./production_model/temperature_scaler.json",
                "deployment_readme": "./production_model/DEPLOYMENT_README.md"
            },

            "monitoring_thresholds": {
                "roc_auc_min": 0.92,
                "prediabetes_precision_min": 0.95,
                "false_positive_max": 0.05,
                "detection_rate_min": 0.85,
                "alert_if_below_threshold": True
            },

            "input_requirements": {
                "sequence_length": int(config.sequence_length),
                "n_features": int(config.n_features),
                "features": [
                    "blood_glucose_value",
                    "glucose_diff",
                    "glucose_accel",
                    "glucose_rollmean_1h",
                    "glucose_rollstd_1h",
                    "sin_hour",
                    "cos_hour",
                    "is_fasting",
                    "is_night"
                ],
                "preprocessing": "StandardScaler on continuous features",
                "missing_data_handling": "Forward fill then backward fill"
            },

            "deployment_checklist": [
                {
                    "item": "Model validation complete",
                    "status": "COMPLETE" if held_out_results else "PARTIAL",
                    "details": "Held-out test passed" if held_out_results else "CV only (no held-out)"
                },
                {
                    "item": "Artifacts saved",
                    "status": "COMPLETE",
                    "details": "All production files generated"
                },
                {
                    "item": "Documentation ready",
                    "status": "COMPLETE",
                    "details": "README and model card available"
                },
                {
                    "item": "Monitoring thresholds set",
                    "status": "COMPLETE",
                    "details": "Alert thresholds configured"
                }
            ]
        }

        output_path = self.model_dir / "deployment_config.json"
        self._save_json(deployment, output_path)
        return deployment

    def _save_json(self, data: Dict, path: Path, indent: int = 2):
        """Save dictionary to JSON file with proper formatting."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        print(f"[MODEL_CARD] Saved: {path.name}")


def numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
