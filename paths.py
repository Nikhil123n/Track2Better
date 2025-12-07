"""
Central path configuration for the CGM - AI-READI dataset.
All code should import paths from here instead of hard-coding absolute paths.
"""

import os

# Root folder of this git repo
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Root of the dataset inside the repo
PILOT_ROOT_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset")

# Subdirectories inside dataset
WEARABLE_BG_DIR = os.path.join(PILOT_ROOT_DATA_PATH, "wearable_blood_glucose")
CLEANED_DATA_PATH = os.path.join(PILOT_ROOT_DATA_PATH, "cleaned_data2")
UNCLEANED_DATA_DIR = os.path.join(PILOT_ROOT_DATA_PATH, "uncleaned_data")
LOG_DIR = os.path.join(PILOT_ROOT_DATA_PATH, "logs")

# Important files
MANIFEST_PATH = os.path.join(WEARABLE_BG_DIR, "manifest.tsv")
PARTICIPANTS_DATA_PATH = os.path.join(PILOT_ROOT_DATA_PATH, "participants.csv")
SUMMARY_PATH = os.path.join(CLEANED_DATA_PATH, "summarized_metrics_all_participants.csv") # used by xgboost_relabel_truehealthy

# Output / derived directories
AUGMENTED_OUTPUT_DIR = os.path.join(CLEANED_DATA_PATH, "augmented_batches")
PLOTS_OUTPUT_DIR = os.path.join(CLEANED_DATA_PATH, "cluster_true_healthy")

# Aliases for scripts that used different constant names
OUTPUT_PATH_VALID = CLEANED_DATA_PATH
OUTPUT_PATH_INVALID = UNCLEANED_DATA_DIR