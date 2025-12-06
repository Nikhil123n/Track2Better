"""
cgm_lstm package

Small package that implements an end-to-end LSTM binary classification
pipeline for CGM time-series data.
"""

import os

# Environment variables to encourage deterministic TensorFlow behavior
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"        # deterministic TF ops
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"      # harmless on CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"       # deterministic CPU kernels (slower)

from .data import Config, DataValidator, DataLoader
from .model import LSTMTrainer
from .viz import Visualizer
from .pipeline import LSTMPipeline

__all__ = [
    "Config",
    "DataValidator",
    "DataLoader",
    "LSTMTrainer",
    "Visualizer",
    "LSTMPipeline",
]
