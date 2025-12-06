"""
Visualization helpers for training curves, ROC, and confusion matrix.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class Visualizer:
    """Helper class for plotting training and evaluation diagnostics.
        This class groups together static methods that generate common
        visualizations for model development:
        - Training curves (loss, accuracy, AUC, precision, recall) across epochs.
        - ROC curve and AUC on the test set.
        - Confusion matrix heatmap for predicted vs actual labels.

        All methods are static so they can be called without instantiating
        the class, and each method optionally saves plots to disk.
    """
    @staticmethod
    def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
        try:
            metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            for i, metric in enumerate(metrics):
                if i < len(axes):
                    ax = axes[i]
                    if metric in history and f"val_{metric}" in history:
                        ax.plot(history[metric], label=f"Train {metric}")
                        ax.plot(history[f"val_{metric}"], label=f"Val {metric}")
                        ax.set_title(f"{metric.capitalize()} Curve")
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel(metric.capitalize())
                        ax.legend(); ax.grid(True)
            if len(metrics) < len(axes):
                fig.delaxes(axes[-1])
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"[PLOT] Training curves saved to {save_path}")
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting training curves: {e}")

    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, save_path: Optional[str] = None) -> None:
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
            plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"[PLOT] ROC curve saved to {save_path}")
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {e}")

    @staticmethod
    def plot_confusion_matrix(conf_matrix: np.ndarray, save_path: Optional[str] = None) -> None:
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Healthy', 'Healthy'],
                        yticklabels=['Not Healthy', 'Healthy'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted'); plt.ylabel('Actual')
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"[PLOT] Confusion matrix saved to {save_path}")
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
