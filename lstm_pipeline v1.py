"""
Entry point to run the CGM LSTM binary classification pipeline.
"""

import os
import json
import logging

from cgm_lstm import Config, LSTMPipeline

# Logging configuration (matches your original settings)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("lstm_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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
        pipeline = LSTMPipeline(config)
        results = pipeline.run_full_pipeline()

        print("\nTraining completed successfully!")
        print(f"New version created: {results['version_folder']}")
        print("Use this version folder name for analysis scripts!")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
