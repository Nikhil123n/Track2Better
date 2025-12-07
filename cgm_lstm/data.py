"""
Data configuration, validation, and loading utilities for the CGM LSTM pipeline.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

logger = logging.getLogger(__name__)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

from paths import PILOT_ROOT_DATA_PATH, AUGMENTED_OUTPUT_DIR


@dataclass
class Config:
    """Configuration container for the LSTM pipeline.

    This class centralizes all configurable aspects of the pipeline, including:
    - Paths for input data and output models.
    - Sequence length and feature list for LSTM input.
    - Model architecture hyperparameters.
    - Training hyperparameters (batch size, epochs, split fractions).
    - Imbalance-handling strategy and optimizer tuning.

    A timestamped version folder is automatically created in `models_base_dir`
    so each run has its own isolated artifacts.
    """

    # Paths
    base_path: str = os.getenv('AI_READI_PATH', PILOT_ROOT_DATA_PATH)
    augmented_dir: str = os.getenv('AUGMENTED_DIR', AUGMENTED_OUTPUT_DIR)
    models_base_dir: str = './models'

    # Data
    expected_sequence_length: int = 2138
    selected_features: List[str] = None

    # Model
    lstm_units_1: int = 32
    lstm_units_2: int = 16
    dense_units: int = 16
    dropout_rate: float = 0.3

    # Training
    batch_size: int = 16
    epochs: int = 30
    test_size: float = 0.2           # fraction to hold out for TEST
    val_size_from_train: float = 0.2 # fraction of TRAIN to use as VAL
    early_stopping_patience: int = 8

    # Imbalance handling
    use_smote: bool = False            # OFF by default (sequence aware)
    use_class_weights: bool = True     # ON by default

    # Optimizer tweak
    clipnorm: float = 1.0              # gradient clipping for stability

    # Seed
    random_seed: int = RANDOM_SEED

    def __post_init__(self):
        """Finalize configuration after initialization.

        - Populate default feature list if none is provided.
        - Create a unique version folder based on timestamp.
        - Ensure model and augmented data directories exist.
        - Log resolved paths for traceability.
        """
        if self.selected_features is None:
            self.selected_features = [
                'glucose_accel',
                'glucose_change_rate',
                'cos_hour',
                'blood_glucose_value',
                'glucose_rollmean_1h',
                'glucose_diff',
                'sin_hour',                
                'is_meal_time',
                'is_night'
            ]

        # Timestamp-based version folder to keep runs separated
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.version_folder = f'model_{ts}'
        self.models_dir = os.path.join(self.models_base_dir, self.version_folder)

        # Ensure directories exist
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.augmented_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"[VERSION] Created new version folder: {self.version_folder}")
        logger.info(f"[PATH] Model directory: {self.models_dir}")


class DataValidator:
    """Utility class with static methods for validating input data.

    All checks are non-destructive and either raise clear exceptions or
    log warnings when something looks suspicious.
    """

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str], name: str = 'DataFrame') -> bool:
        """Validate that a DataFrame is non-empty and has required columns."""
        if df.empty:
            raise ValueError(f"{name} is empty")
        
        # Check for any missing required columns
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        logger.info(f"[OK] {name} validation passed: {df.shape[0]} rows, {df.shape[1]} cols")

        return True

    @staticmethod
    def validate_sequence_length(df: pd.DataFrame, expected_length: int, pid: str) -> bool:
        """Check if a participant's sequence has the expected length.
            Logs a warning if the sequence length does not match and returns False
            so the caller can decide to skip that participant.
        """
        if len(df) != expected_length:
            logger.warning(f"[WARNING] Participant {pid} has {len(df)} rows, expected {expected_length}")
            return False
        return True

    @staticmethod
    def validate_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Apply simple cleaning and diagnostics on feature columns.
            - Fills NaNs in `glucose_rollstd_1h` (if present) to avoid propagation.
            - Logs any NaNs found in the selected features for future inspection.
            Returns a copy of the DataFrame to avoid modifying the original.
        """
        df = df.copy()

        # Backward-compatibility / safety for older pipelines that included this feature
        if 'glucose_rollstd_1h' in df.columns:
            df['glucose_rollstd_1h'] = df['glucose_rollstd_1h'].fillna(0)
        nan_counts = df[features].isna().sum()
        if nan_counts.any():
            logger.warning(f"[WARNING] NaN values found: {nan_counts[nan_counts>0].to_dict()}")
        return df


class DataLoader:
    """Load and prepare data for the LSTM classifier.
        Responsibilities:
        - Read augmented, feature-engineered CSV batches.
        - Filter to the desired binary classification labels.
        - Build fixed-length sequences per participant suitable for LSTM input.
        - Save processed numpy arrays and metadata for later reuse.
    """
    def __init__(self, config: Config):
        """Initialize the DataLoader with configuration and validators."""
        self.config = config
        self.validator = DataValidator()
        # Kept for future compatibility if label encoding is needed later
        self.label_encoder = LabelEncoder()

    def load_augmented_data(self) -> pd.DataFrame:
        """Load all augmented CSV batches into a single DataFrame.
            Finds files matching '*_labeled_augmented.csv' inside `config.augmented_dir`,
            loads them, and concatenates them into one combined DataFrame.
            Raises:
                FileNotFoundError: If no matching files are found.
                ValueError: If matching files exist but none could be loaded.
        """
        pattern = os.path.join(self.config.augmented_dir, "*_labeled_augmented.csv")
        files = sorted(glob(pattern))
        if not files:
            raise FileNotFoundError(f"No augmented files found in {self.config.augmented_dir}")
        logger.info(f"Loading {len(files)} augmented files...")
        frames = []
        for fp in files:
            try:
                frames.append(pd.read_csv(fp))
            except Exception as e:
                logger.error(f"Error loading {fp}: {e}")
        if not frames:
            raise ValueError("No valid CSV files could be loaded")
        df = pd.concat(frames, ignore_index=True)
        logger.info(f"[OK] Loaded combined dataset: {df.shape}")
        return df

    def prepare_binary_classification_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter and relabel data for binary classification.
            Keeps only the `true_healthy` and `pre_diabetes_lifestyle` groups and
            constructs a `label_binary` column where:
            - 1 = true_healthy
            - 0 = pre_diabetes_lifestyle
            Also runs DataFrame and feature validations before returning.
        """
        required = self.config.selected_features + ['participant_id', 'time_index', 'study_group']
        self.validator.validate_dataframe(df, required, 'Input DataFrame')
        df = self.validator.validate_features(df, self.config.selected_features)

        targets = ['true_healthy', 'pre_diabetes_lifestyle']
        dff = df[df['study_group'].isin(targets)].copy()
        dff.reset_index(drop=True, inplace=True)
        logger.info(f"[DATA] Filtered for binary classification: {dff.shape[0]} rows")
        logger.info(f"[DATA] Class distribution: {dff['study_group'].value_counts().to_dict()}")

        # Binary target: 1 if true_healthy, else 0
        dff['label_binary'] = (dff['study_group'] == 'true_healthy').astype(int)
        return dff

    def create_lstm_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create fixed-length sequences per participant for LSTM input.

            Groups by `participant_id`, sorts by `time_index`, enforces a strict
            `expected_sequence_length`, and constructs:
            - X: 3D array of shape (n_participants, sequence_length, n_features)
            - y: 1D array of binary labels per participant
            - pid_array: 1D array of participant IDs

            Participants whose sequences do not match `expected_sequence_length`
            are skipped (with a warning logged).
        """
        Xs, ys, pids = [], [], []
        grouped = df.groupby('participant_id')
        total = len(grouped)
        valid = 0
        for pid, g in grouped:
            # Ensure time order within each participant
            g = g.sort_values('time_index')

            if not self.validator.validate_sequence_length(g, self.config.expected_sequence_length, str(pid)):
                # Skip participants with incomplete or mismatched sequences
                continue

            # Extract feature matrix for this participant and convert to float32
            X_seq = g[self.config.selected_features].values.astype(np.float32)
            
            # Use the first label for the participant (labels are constant within participant)
            y_bin = int(g['label_binary'].iloc[0])
            
            # Replace any remaining NaNs with zeros to keep shapes stable
            X_seq = np.nan_to_num(X_seq, nan=0.0)
            Xs.append(X_seq); ys.append(y_bin); pids.append(pid); valid += 1
        if not Xs:
            raise ValueError("No valid sequences found")
        X = np.stack(Xs)
        y = np.array(ys)
        pid_array = np.array(pids)
        logger.info(f"[OK] Created LSTM sequences | participants total={total}, valid={valid}")
        logger.info(f"[DATA] X shape: {X.shape} | y shape: {y.shape} | label dist: {np.bincount(y)}")
        return X, y, pid_array

    def save_processed_data(self, X: np.ndarray, y: np.ndarray, pid_array: np.ndarray) -> None:
        """Save processed numpy arrays and label encoder to disk.
            This writes:
            - X_lstm.npy: 3D feature array
            - y_lstm_binary.npy: 1D labels
            - pid_lstm.npy: participant IDs
            - label_encoder.pkl: placeholder encoder for future use
        """
        np.save(os.path.join(self.config.models_dir, 'X_lstm.npy'), X)
        np.save(os.path.join(self.config.models_dir, 'y_lstm_binary.npy'), y)
        np.save(os.path.join(self.config.models_dir, 'pid_lstm.npy'), pid_array)
        # Save label encoder if needed later
        import pickle
        with open(os.path.join(self.config.models_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Saved processed data to {self.config.models_dir}")