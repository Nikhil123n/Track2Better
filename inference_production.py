"""
Production Model Inference Script

Demonstrates how to load and use the production model for predictions.
Uses the confidence-based decision rules with three tiers:
- High confidence Pre-Diabetes (prob >= 0.65)
- Uncertain (0.35 <= prob < 0.65) → recommend secondary screening
- High confidence Healthy (prob < 0.35)
"""

import numpy as np
import json
import os
from tensorflow import keras
from pathlib import Path


class ProductionPredictor:
    """Production model predictor with confidence-based rules."""

    def __init__(self, model_dir: str):
        """
        Load production model and artifacts.

        Parameters
        ----------
        model_dir : Path to production_model directory
        """
        self.model_dir = Path(model_dir)

        # Load Keras model
        model_path = self.model_dir / "best_model.keras"
        self.model = keras.models.load_model(model_path)
        print(f"✓ Loaded model from {model_path}")

        # Load global threshold
        threshold_path = self.model_dir / "global_threshold.json"
        with open(threshold_path) as f:
            self.threshold = json.load(f)['global_threshold']
        print(f"✓ Loaded threshold: {self.threshold:.4f}")

        # Load temperature scaler
        temp_path = self.model_dir / "temperature_scaler.json"
        with open(temp_path) as f:
            self.temperature = json.load(f)['temperature']
        print(f"✓ Loaded temperature: {self.temperature:.4f}")

        # Load feature scaler
        scaler_path = self.model_dir / "feature_scaler.npz"
        scaler_data = np.load(scaler_path)
        self.feature_mean = scaler_data['mean']
        self.feature_scale = scaler_data['scale']
        print(f"✓ Loaded feature scaler (shape: {self.feature_mean.shape})")

        # Load config for reference
        config_path = self.model_dir / "training_config.json"
        with open(config_path) as f:
            self.config = json.load(f)
        print(f"✓ Model architecture: {self.config['architecture']}")
        print(f"✓ Trained on {self.config['trained_on_samples']} samples")

    def predict(self, X: np.ndarray, return_probability: bool = False):
        """
        Make prediction with confidence-based rules.

        Parameters
        ----------
        X : Input features (shape: [n_samples, timesteps, n_features])
        return_probability : If True, return probability instead of class

        Returns
        -------
        predictions : Binary predictions (0=pre_diabetes, 1=healthy) or probabilities
        """
        # 1. Normalize features
        X_norm = (X - self.feature_mean) / self.feature_scale

        # 2. Get model probability
        probs = self.model.predict(X_norm, verbose=0).ravel()

        # 3. Apply temperature scaling
        logits = np.log(probs / (1 - probs + 1e-7))
        calibrated_probs = 1 / (1 + np.exp(-logits / self.temperature))

        if return_probability:
            return calibrated_probs

        # 4. Apply threshold
        predictions = (calibrated_probs >= self.threshold).astype(int)
        return predictions

    def predict_with_confidence(self, X: np.ndarray):
        """
        Make prediction with confidence-based decision rules.

        Returns structured output with prediction, confidence level, and recommended action.

        Parameters
        ----------
        X : Input features (shape: [1, timesteps, n_features]) for single patient

        Returns
        -------
        result : dict with keys 'prediction', 'confidence', 'action', 'probability'
        """
        assert X.shape[0] == 1, "This method expects single patient data"

        # Get calibrated probability
        prob = self.predict(X, return_probability=True)[0]

        # Apply confidence-based rules
        if prob >= 0.65:
            return {
                'prediction': 'Pre-Diabetes',
                'confidence': 'High',
                'action': 'Immediate lifestyle intervention recommended',
                'probability': float(prob),
                'class_label': 0
            }
        elif prob >= 0.35:
            return {
                'prediction': 'Uncertain',
                'confidence': 'Medium',
                'action': 'Secondary screening (OGTT) recommended for confirmation',
                'probability': float(prob),
                'class_label': None  # Uncertain
            }
        else:
            return {
                'prediction': 'Healthy',
                'confidence': 'High',
                'action': 'Continue regular screening cycle (1-2 years)',
                'probability': float(prob),
                'class_label': 1
            }


def main():
    """Example usage of production predictor."""

    # Path to production model (adjust to your model location)
    # Example: models/model_20260208_141917/production_model
    model_dir = "models/model_20260208_141917/production_model"

    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        print("Please run the pipeline with CV to generate production model first.")
        print("Command: python 'lstm_pipeline v1.py'")
        return

    # Load predictor
    print("\n=== Loading Production Model ===")
    predictor = ProductionPredictor(model_dir)

    # Example: Load test data (replace with your actual data loading)
    print("\n=== Example Prediction ===")
    print("Note: This is a demo. Replace with actual patient CGM data.")

    # Dummy example (replace with real data)
    # Expected shape: [1, 2138, 8] for single patient
    # X_patient = np.load("patient_cgm_data.npy")
    # For demonstration, we'll skip the actual prediction

    print("\nTo use this predictor in production:")
    print("1. Load patient CGM data: X_patient = load_patient_data(patient_id)")
    print("2. Make prediction: result = predictor.predict_with_confidence(X_patient)")
    print("3. Use result['action'] to guide clinical decision")

    print("\n=== Predictor Ready for Production Use ===")


if __name__ == "__main__":
    main()
