"""
CGM Pre-Diabetes Detection - Production Inference Script

Complete end-to-end inference pipeline supporting both:
1. Raw CGM data (with automatic feature engineering)
2. Pre-computed features (from training pipeline output)

Usage:
    # For pre-computed features (uses model files in current directory)
    python inference.py --csv test_set.csv
    
    # For raw CGM data (with feature engineering)
    python inference.py --csv patient.csv --raw
    
    # Use model from different directory
    python inference.py --csv test_set.csv --model ../models/model_YYYYMMDD_HHMMSS/production_model
    
    # Save results to JSON
    python inference.py --csv test_set.csv --output results.json
    
    # Run setup verification
    python inference.py --verify

Author: Track2Better Team
Date: 2026
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for raw CGM data preprocessing."""
    
    def __init__(self):
        self.expected_length = 2138
        
        # CRITICAL: Feature order must match training pipeline (cgm_lstm/data.py)
        # Training order: glucose_accel, cos_hour, blood_glucose_value, glucose_rollmean_1h,
        #                 glucose_rollstd_1h, glucose_diff, sin_hour, is_meal_time, is_night
        self.all_features = [
            'glucose_accel',           # 0 - continuous
            'cos_hour',                # 1 - continuous
            'blood_glucose_value',     # 2 - continuous
            'glucose_rollmean_1h',     # 3 - continuous
            'glucose_rollstd_1h',      # 4 - continuous
            'glucose_diff',            # 5 - continuous
            'sin_hour',                # 6 - continuous
            'is_meal_time',            # 7 - binary
            'is_night'                 # 8 - binary
        ]
        self.features_continuous = self.all_features[:7]
        self.features_binary = self.all_features[7:]
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Apply complete feature engineering pipeline for raw CGM data.
        
        Parameters
        ----------
        df : DataFrame with columns [participant_id, time_index, time, blood_glucose_value]
        
        Returns
        -------
        X : np.ndarray of shape (1, 2138, 9) ready for model
        metadata : dict with preprocessing info
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Validate input
        required_cols = ['participant_id', 'time_index', 'time', 'blood_glucose_value']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Get single participant
        participant_ids = df['participant_id'].unique()
        if len(participant_ids) > 1:
            logger.warning(f"Multiple participants found. Using first: {participant_ids[0]}")
        
        df = df[df['participant_id'] == participant_ids[0]].copy()
        
        # Sort by time index
        df = df.sort_values('time_index').reset_index(drop=True)
        
        # Check sequence length
        if len(df) != self.expected_length:
            raise ValueError(
                f"Invalid sequence length: expected {self.expected_length}, got {len(df)}. "
                f"Model requires exactly 2138 time points (~7.4 days at 5-min intervals)."
            )
        
        logger.info(f"Participant: {participant_ids[0]}, Length: {len(df)}")
        
        # Apply feature engineering steps
        df = self._extract_time_features(df)
        df = self._add_rolling_features(df)
        df = self._add_circadian_features(df)
        
        # Extract feature array
        X = self._prepare_model_input(df)
        
        # Metadata
        metadata = {
            'participant_id': str(participant_ids[0]),
            'sequence_length': len(df),
            'glucose_mean': float(df['blood_glucose_value'].mean()),
            'glucose_std': float(df['blood_glucose_value'].std()),
            'glucose_min': float(df['blood_glucose_value'].min()),
            'glucose_max': float(df['blood_glucose_value'].max()),
            'features_engineered': self.all_features
        }
        
        logger.info("Feature engineering complete")
        return X, metadata
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract hour from time string."""
        try:
            # Parse time string (format: HH:MM:SS)
            df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.hour
            
            if df['hour'].isnull().any():
                logger.warning(f"Failed to parse {df['hour'].isnull().sum()} time values")
                df['hour'] = df['time'].str.split(':').str[0].astype(int, errors='ignore')
            
            return df
        except Exception as e:
            logger.error(f"Error extracting time features: {e}")
            raise
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics and derivatives."""
        try:
            # Rolling mean (1 hour = 12 points at 5-min intervals)
            df['glucose_rollmean_1h'] = df['blood_glucose_value'].rolling(
                window=12, min_periods=1
            ).mean()
            
            # Rolling std (1 hour)
            df['glucose_rollstd_1h'] = df['blood_glucose_value'].rolling(
                window=12, min_periods=1
            ).std().fillna(0)
            
            # First derivative (glucose change)
            df['glucose_diff'] = df['blood_glucose_value'].diff().fillna(0)
            
            # Second derivative (acceleration)
            df['glucose_accel'] = df['glucose_diff'].diff().fillna(0)
            
            return df
        except Exception as e:
            logger.error(f"Error adding rolling features: {e}")
            raise
    
    def _add_circadian_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add circadian rhythm encoding."""
        try:
            # Cyclic encoding of hour
            df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Binary time flags
            df['is_meal_time'] = df['hour'].isin([7,8,9,12,13,14,18,19,20]).astype(int)
            df['is_night'] = df['hour'].isin([0,1,2,3,4,5,22,23]).astype(int)
            
            return df
        except Exception as e:
            logger.error(f"Error adding circadian features: {e}")
            raise
    
    def _prepare_model_input(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and reshape features for model input."""
        try:
            X = df[self.all_features].values
            X = X.reshape(1, self.expected_length, len(self.all_features))
            logger.info(f"Model input shape: {X.shape}")
            return X
        except KeyError as e:
            logger.error(f"Missing feature: {e}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise


class ProductionPredictor:
    """Production model predictor with confidence-based clinical decision rules."""
    
    def __init__(self, model_dir: str):
        """
        Load production model and all artifacts.
        
        Parameters
        ----------
        model_dir : Path to production_model directory containing:
                   - best_model.keras
                   - feature_scaler.npz
                   - global_threshold.json
                   - temperature_scaler.json
                   - training_config.json
        """
        self.model_dir = Path(model_dir)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Import tensorflow here to avoid slow startup for --verify
        from tensorflow import keras
        
        # Load Keras model
        model_path = self.model_dir / "best_model.keras"
        self.model = keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load global threshold
        threshold_path = self.model_dir / "global_threshold.json"
        with open(threshold_path) as f:
            self.threshold = json.load(f)['global_threshold']
        logger.info(f"Loaded threshold: {self.threshold:.4f}")
        
        # Load temperature scaler
        temp_path = self.model_dir / "temperature_scaler.json"
        with open(temp_path) as f:
            self.temperature = json.load(f)['temperature']
        logger.info(f"Loaded temperature: {self.temperature:.4f}")
        
        # Load feature scaler
        scaler_path = self.model_dir / "feature_scaler.npz"
        scaler_data = np.load(scaler_path)
        self.feature_mean = scaler_data['mean']
        self.feature_scale = scaler_data['scale']
        logger.info(f"Loaded feature scaler (features: {len(self.feature_mean)})")
        
        # Load config
        config_path = self.model_dir / "training_config.json"
        with open(config_path) as f:
            self.config = json.load(f)
        logger.info(f"Model architecture: {self.config['architecture']}")
        
        # Load confidence thresholds (dynamic, computed during training)
        conf_threshold_path = self.model_dir / "confidence_thresholds.json"
        if conf_threshold_path.exists():
            with open(conf_threshold_path) as f:
                conf_data = json.load(f)
                self.confidence_low = conf_data['confidence_low_threshold']
                self.confidence_high = conf_data['confidence_high_threshold']
            logger.info(f"Loaded dynamic confidence thresholds: Low={self.confidence_low:.4f}, High={self.confidence_high:.4f}")
        else:
            # Fallback to hardcoded thresholds if file not found
            self.confidence_low = 0.35
            self.confidence_high = 0.65
            logger.warning("confidence_thresholds.json not found, using default thresholds (0.35, 0.65)")
    
    def predict(self, X: np.ndarray, return_probability: bool = False):
        """
        Make prediction with temperature-scaled calibration.
        
        Parameters
        ----------
        X : Input features (shape: [n_samples, timesteps, n_features])
        return_probability : If True, return probability instead of class
        
        Returns
        -------
        predictions : Binary predictions (0=pre_diabetes, 1=healthy) or probabilities
        """
        # Scale only continuous features (first 7), keep binary features unchanged
        X_scaled = X.copy()
        X_scaled[:, :, :7] = (X[:, :, :7] - self.feature_mean) / self.feature_scale
        
        # Get model probability
        probs = self.model.predict(X_scaled, verbose=0).ravel()
        
        # Apply temperature scaling for calibration
        logits = np.log(probs / (1 - probs + 1e-7))
        calibrated_probs = 1 / (1 + np.exp(-logits / self.temperature))
        
        if return_probability:
            return calibrated_probs
        
        # Apply threshold
        predictions = (calibrated_probs >= self.threshold).astype(int)
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray) -> Dict:
        """
        Make prediction with 3-tier confidence-based decision rules.
        
        Returns structured output with prediction, confidence, action, and probabilities.
        
        Parameters
        ----------
        X : Input features (shape: [1, timesteps, n_features]) for single patient
        
        Returns
        -------
        result : dict with clinical decision information
        """
        assert X.shape[0] == 1, "This method expects single patient data"
        
        # Get calibrated probability (model predicts prob of Class 1 = Healthy)
        prob = self.predict(X, return_probability=True)[0]
        
        # Apply 3-tier confidence-based clinical decision rules using dynamic thresholds
        # Thresholds are computed during training based on global threshold ± margins
        if prob >= self.confidence_high:
            # High confidence healthy
            return {
                'prediction': 'Healthy',
                'confidence': 'High',
                'action': 'Continue regular screening cycle (1-2 years)',
                'probability': float(prob),
                'probability_healthy': float(prob),
                'probability_prediabetes': float(1 - prob),
                'class_label': 1,
                'risk_level': 'Low',
                'threshold_used': float(self.confidence_high)
            }
        elif prob >= self.confidence_low:
            # Uncertain zone - recommend secondary screening
            return {
                'prediction': 'Uncertain',
                'confidence': 'Medium',
                'action': 'Secondary screening (OGTT) recommended for confirmation',
                'probability': float(prob),
                'probability_healthy': float(prob),
                'probability_prediabetes': float(1 - prob),
                'class_label': None,
                'risk_level': 'Medium',
                'threshold_low': float(self.confidence_low),
                'threshold_high': float(self.confidence_high)
            }
        else:
            # High confidence pre-diabetes
            return {
                'prediction': 'Pre-Diabetes',
                'confidence': 'High',
                'action': 'Immediate lifestyle intervention recommended',
                'probability': float(prob),
                'probability_healthy': float(prob),
                'probability_prediabetes': float(1 - prob),
                'class_label': 0,
                'risk_level': 'High',
                'threshold_used': float(self.confidence_low)
            }


def load_precomputed_features(csv_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load CSV with pre-computed features from training pipeline.
    
    Expected columns: participant_id, blood_glucose_value, glucose_rollmean_1h,
                     glucose_rollstd_1h, glucose_diff, glucose_accel, sin_hour,
                     cos_hour, is_meal_time, is_night
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Expected features in correct order (9 features total)
    # CRITICAL: This order MUST match cgm_lstm/data.py Config.selected_features
    expected_features = [
        'glucose_accel',           # 0 - continuous
        'cos_hour',                # 1 - continuous
        'blood_glucose_value',     # 2 - continuous
        'glucose_rollmean_1h',     # 3 - continuous
        'glucose_rollstd_1h',      # 4 - continuous
        'glucose_diff',            # 5 - continuous
        'sin_hour',                # 6 - continuous
        'is_meal_time',            # 7  - binary (not scaled)
        'is_night'                 # 8 - binary (not scaled)
    ]
    
    # Validate columns
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Get participant info
    participant_id = df['participant_id'].iloc[0]
    
    # Validate sequence length
    if len(df) != 2138:
        raise ValueError(
            f"Invalid sequence length: expected 2138, got {len(df)}. "
            f"Model requires exactly 2138 time points."
        )
    
    logger.info(f"Participant: {participant_id}, Length: {len(df)}")
    logger.info(f"All features present")
    
    # Extract features
    X = df[expected_features].values
    X = X.reshape(1, 2138, 9)
    
    # Metadata
    metadata = {
        'participant_id': str(participant_id),
        'sequence_length': len(df),
        'glucose_mean': float(df['blood_glucose_value'].mean()),
        'glucose_std': float(df['blood_glucose_value'].std()),
        'glucose_min': float(df['blood_glucose_value'].min()),
        'glucose_max': float(df['blood_glucose_value'].max()),
    }
    
    # Check for ground truth label
    if 'label_binary' in df.columns:
        true_label = df['label_binary'].iloc[0]
        metadata['true_label'] = int(true_label)
        metadata['true_label_text'] = 'Healthy' if true_label == 1 else 'Pre-Diabetes'
        logger.info(f"Ground truth label: {metadata['true_label_text']}")
    
    if 'study_group' in df.columns:
        study_group = df['study_group'].iloc[0]
        metadata['study_group'] = str(study_group)
        logger.info(f"Study group: {study_group}")
    
    return X, metadata


def print_results(result: Dict, metadata: Dict):
    """Display prediction results in formatted output."""
    print("\n" + "="*70)
    print("CGM PRE-DIABETES DETECTION - PREDICTION RESULTS")
    print("="*70)
    
    print(f"\nPatient Information:")
    print(f"   Participant ID: {metadata['participant_id']}")
    if 'study_group' in metadata:
        print(f"   Study Group: {metadata['study_group']}")
    print(f"   Sequence Length: {metadata['sequence_length']} points")
    print(f"   Glucose Stats: {metadata['glucose_mean']:.1f} +/- {metadata['glucose_std']:.1f} mg/dL")
    print(f"   Glucose Range: [{metadata['glucose_min']:.1f}, {metadata['glucose_max']:.1f}] mg/dL")
    
    if 'true_label_text' in metadata:
        print(f"\nGround Truth:")
        print(f"   True Label: {metadata['true_label_text']}")
    
    print(f"\nPrediction:")
    print(f"   Status: {result['prediction']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Risk Level: {result['risk_level']}")
    
    print(f"\nProbabilities:")
    print(f"   Healthy: {result['probability_healthy']*100:.1f}%")
    print(f"   Pre-Diabetes: {result['probability_prediabetes']*100:.1f}%")
    
    # Check accuracy if ground truth available
    if 'true_label' in metadata:
        predicted_label = result['class_label']
        true_label = metadata['true_label']
        
        if predicted_label is not None:
            correct = (predicted_label == true_label)
            print(f"\nPrediction Accuracy:")
            if correct:
                print(f"   CORRECT - Prediction matches ground truth")
            else:
                print(f"   INCORRECT - Prediction differs from ground truth")
        else:
            print(f"\nUncertain Zone - Ground truth: {metadata['true_label_text']}")
    
    print(f"\nRecommended Action:")
    print(f"   {result['action']}")
    
    print("\n" + "="*70)
    
    # Clinical interpretation
    if result['risk_level'] == 'High':
        print("\nHIGH RISK: Patient shows patterns consistent with pre-diabetes.")
        print("Recommend: Lifestyle intervention, dietary counseling, follow-up OGTT")
    elif result['risk_level'] == 'Medium':
        print("\nUNCERTAIN: Pattern is near decision boundary.")
        print("Recommend: OGTT confirmation test within 2-4 weeks")
    else:
        print("\nLOW RISK: Patient shows healthy glucose control patterns.")
        print("Recommend: Continue annual screening")
    
    print("="*70 + "\n")


def verify_setup(csv_path: Optional[str] = None, model_dir: Optional[str] = None):
    """
    Verify that the inference environment is properly configured.
    
    Checks:
    - Python dependencies (numpy, pandas, tensorflow)
    - Model files exist and are complete
    - CSV format is valid (if provided)
    """
    print("="*70)
    print("INFERENCE SETUP VERIFICATION")
    print("="*70 + "\n")
    
    results = []
    
    # Check dependencies
    print("Checking dependencies...")
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tensorflow': 'tensorflow/keras'
    }
    
    missing = []
    for package, display_name in required_packages.items():
        try:
            __import__(package)
            print(f"  [OK] {display_name}")
        except ImportError:
            print(f"  [FAIL] {display_name} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        results.append(('Dependencies', False))
    else:
        print("All dependencies installed\n")
        results.append(('Dependencies', True))
    
    # Check model directory if provided
    if model_dir:
        print(f"Checking model directory: {model_dir}")
        model_path = Path(model_dir)
        
        if not model_path.exists():
            print(f"  [FAIL] Model directory not found")
            results.append(('Model Directory', False))
        else:
            required_files = [
                'best_model.keras',
                'feature_scaler.npz',
                'global_threshold.json',
                'temperature_scaler.json',
                'training_config.json'
            ]
            
            optional_files = [
                'confidence_thresholds.json'
            ]
            
            missing_files = []
            for filename in required_files:
                if (model_path / filename).exists():
                    print(f"  [OK] {filename}")
                else:
                    print(f"  [FAIL] {filename} - MISSING")
                    missing_files.append(filename)
            
            # Check optional files
            for filename in optional_files:
                if (model_path / filename).exists():
                    print(f"  [OK] {filename} (optional)")
                else:
                    print(f"  [WARN] {filename} - MISSING (will use default thresholds)")
            
            if missing_files:
                results.append(('Model Directory', False))
            else:
                print("Model directory complete\n")
                results.append(('Model Directory', True))
    
    # Check CSV if provided
    if csv_path:
        print(f"Checking CSV file: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"  [FAIL] CSV file not found")
            results.append(('CSV File', False))
        else:
            try:
                df = pd.read_csv(csv_path)
                print(f"  [OK] File loaded ({len(df)} rows, {len(df.columns)} columns)")
                
                if len(df) == 2138:
                    print(f"  [OK] Correct row count (2138)")
                else:
                    print(f"  [WARN] Expected 2138 rows, got {len(df)}")
                
                results.append(('CSV File', True))
            except Exception as e:
                print(f"  [FAIL] Error reading CSV: {e}")
                results.append(('CSV File', False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for check_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status:8} - {check_name}")
    
    all_passed = all(result for _, result in results)
    
    print("="*70)
    
    if all_passed:
        print("\nAll checks passed! Ready to run inference.")
        return 0
    else:
        print("\nSome checks failed. Please fix the issues above.")
        return 1


def main():
    """Main inference pipeline with argument parsing."""
    parser = argparse.ArgumentParser(
        description='CGM Pre-Diabetes Detection - Production Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference with pre-computed features (uses model files in current directory)
  python inference.py --csv test_set.csv
  
  # Inference with raw CGM data (automatic feature engineering)
  python inference.py --csv patient.csv --raw
  
  # Save results to JSON file
  python inference.py --csv test.csv --output results.json
  
  # Use model from different directory
  python inference.py --csv test.csv --model ../models/model_20260209_020732/production_model
  
  # Verify setup
  python inference.py --verify
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to patient CGM CSV file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to production model directory (default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional: Save results to JSON file'
    )
    parser.add_argument(
        '--raw',
        action='store_true',
        help='Process raw CGM data with automatic feature engineering'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify setup and dependencies'
    )
    
    args = parser.parse_args()
    
    # Set default model directory to current directory if not specified
    if args.model is None:
        args.model = str(Path(__file__).parent)
        logger.info(f"Using model files from current directory: {args.model}")
    
    # Verification mode
    if args.verify:
        return verify_setup(args.csv, args.model)
    
    # Normal inference mode - require CSV
    if not args.csv:
        parser.error("--csv is required for inference (use --verify for setup check)")
    
    try:
        # Load and prepare data
        if args.raw:
            # Raw CGM data - apply feature engineering
            logger.info("Mode: Raw CGM data with feature engineering")
            df = pd.read_csv(args.csv)
            engineer = FeatureEngineer()
            X, metadata = engineer.preprocess(df)
        else:
            # Pre-computed features
            logger.info("Mode: Pre-computed features")
            X, metadata = load_precomputed_features(args.csv)
        
        # Load model and predict
        predictor = ProductionPredictor(args.model)
        result = predictor.predict_with_confidence(X)
        
        # Display results
        print_results(result, metadata)
        
        # Save to file if requested
        if args.output:
            output_data = {
                'metadata': metadata,
                'prediction': result
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
