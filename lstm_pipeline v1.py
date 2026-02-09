"""
Entry point to run the CGM LSTM binary classification pipeline.
"""

import os
import json
import logging

from cgm_lstm import Config, LSTMPipeline

# Initial logging configuration (will be reconfigured after model folder is created)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console only initially
    ],
)
logger = logging.getLogger(__name__)


def setup_logging_to_model_folder(model_dir: str):
    """Reconfigure logging to save log file in model folder.

    Parameters
    ----------
    model_dir : str
        Path to model directory (e.g., ./models/model_20260208_185255/)
    """
    log_file = os.path.join(model_dir, "lstm_pipeline.log")

    # Add file handler to root logger
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)

    logger.info(f"[LOG] Log file configured: {log_file}")


def list_model_versions(models_base_dir: str = "./models") -> None:
    """Print available model version folders and their timestamps."""
    try:
        if not os.path.exists(models_base_dir):
            print("No models directory found.")
            return

        folders = [
            f
            for f in os.listdir(models_base_dir)
            if f.startswith("model_")
            and os.path.isdir(os.path.join(models_base_dir, f))
        ]
        if not folders:
            print("No model versions found.")
            return

        folders.sort(reverse=True)
        print(f"\nAvailable Model Versions ({len(folders)} total):")
        print("=" * 50)
        for i, folder in enumerate(folders):
            folder_path = os.path.join(models_base_dir, folder)
            config_path = os.path.join(folder_path, "model_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                    ts = cfg.get("timestamp", "Unknown")
                    print(f"{i+1:2d}. {folder}")
                    print(f"     Created: {ts}")
                    if i == 0:
                        print("     Latest")
                except Exception:
                    print(f"{i+1:2d}. {folder} (config error)")
            else:
                print(f"{i+1:2d}. {folder} (no config)")
        print("=" * 50)
    except Exception as e:
        logger.error(f"Error listing model versions: {e}")


def main() -> None:
    """Construct config, run the pipeline, and report the created version."""
    try:
        print("Checking existing model versions...")
        list_model_versions()

        config = Config()

        # Setup logging to model folder (now that config has created the folder)
        setup_logging_to_model_folder(config.models_dir)

        config.use_cross_validation = True
        config.cv_n_splits = 5
        config.cv_save_fold_models = False  # keep off initially

        # ===== CONFIDENCE-BASED PREDICTION RULES =====
        # Turn ON to enable 3-tier prediction (High Conf Pre-D | Uncertain/OGTT | High Conf Healthy)
        # Turn OFF to use standard binary classification (default)
        config.use_confidence_based_prediction = True  # ← Turn ON/OFF here
        config.confidence_high_threshold = 0.65  # prob >= 0.65: High confidence Pre-Diabetes
        config.confidence_low_threshold = 0.35   # prob < 0.35: High confidence Healthy
        # ============================================

        # ===== HELD-OUT TEST SET (TRUE GENERALIZATION) =====
        # Turn ON to split data into train+val (for CV) and held-out test (never seen)
        # Turn OFF to use CV without held-out test (use all data for CV)
        config.use_held_out_test = True  # ← Turn ON/OFF here
        config.held_out_test_size = 0.2   # 20% of data for held-out test
        config.held_out_random_seed = 42  # For reproducible splits
        #
        # When ON:  Total 491 → Train+Val 393 (80%) | Held-Out Test 98 (20%)
        # When OFF: Total 491 → All used for CV (current behavior)
        # ===================================================

        pipeline = LSTMPipeline(config)
        results = pipeline.run_full_pipeline()

        print("\nTraining completed successfully!")
        print(f"New version created: {results['version_folder']}")
        print(f"Log file: {config.models_dir}/lstm_pipeline.log")
        print("Use this version folder name for analysis scripts!")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
