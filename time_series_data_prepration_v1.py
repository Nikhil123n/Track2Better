import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# === Imports ===
import os
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# === FEATURE CONFIG ===
PILOT_ROOT_DATA_PATH = "C:/Users/nikhi/Box/AI-READI/nikhil working dataset/dataset/"  # change this to your own path
MANIFEST_PATH = os.path.join(PILOT_ROOT_DATA_PATH, "wearable_blood_glucose/manifest.tsv")
CLEANED_DATA_PATH = os.path.join(PILOT_ROOT_DATA_PATH, "cleaned_data2/")
# Load the summarized metrics CSV
SUMMARY_CSV = os.path.join(CLEANED_DATA_PATH, "summarized_metrics_all_participants.csv")
# Output paths
# LABELED_OUTPUT_DIR = os.path.join(CLEANED_DATA_PATH, "cleaned_batches_labeled")
AUGMENTED_OUTPUT_DIR = os.path.join(CLEANED_DATA_PATH, "augmented_batches")


# === Phase 1: Base Dataset Preparation ===
# Step 1: Relabel All Batches Using Final Labels
# We'll:
# Load summarized_metrics_all_participants.csv (which has the final study_group_cleaned labels).
# Loop through each batch_*.csv, match participant_id, and replace study_group with study_group_cleaned.
# Save each updated batch to a new cleaned_batches_labeled/ folder.


# === Phase 2: Data augmentation / Feature Engineering ===
# We’ll research and engineer:
# A. Time-Based Features (Derived from timestamp)
# - hour              → Hour of day (0–23)            → Captures circadian effects on glucose
# - day_fraction      → Time as fraction of 24h       → Normalized rhythm position
# - is_night          → 1 if hour in [0–6 or 22–23]    → Helps isolate night-time spikes
# - is_meal_time      → 1 if hour in [7–9, 12–14, 18–20] → Aligns with common eating times

# B. Rolling/Windowed Features (within 1hr/3hr windows)
# - glucose_rollmean_1h   → 12 points      → Smoothing of short-term trends
# - glucose_rollstd_1h    → 12 points      → Local variability (volatility)
# - glucose_diff          → First derivative (slope of change)
# - glucose_accel         → Second derivative (acceleration)

# C. Zoning/Excursion Flags (Discrete behavior)
# - above_140_flag    → High excursion
# - below_70_flag     → Hypoglycemia
# - spike_flag        → Sudden jump >30 mg/dL in 30 mins
# - dip_flag          → Sudden drop >30 mg/dL in 30 mins

# 💡 Advanced Ideas (Phase 2)
# - spectral_energy, ACF_lag1, post_meal_spike, cooling_duration (future ideas)

def add_rolling_features(df):
    """Add rolling features using vectorized operations for better performance."""
    try:
        # Sort by participant and time_index to ensure proper rolling calculations
        df = df.sort_values(['participant_id', 'time_index']).reset_index(drop=True)

        # Use vectorized operations with groupby transform
        df['glucose_rollmean_1h'] = df.groupby('participant_id')['blood_glucose_value'].transform(
            lambda x: x.rolling(window=12, min_periods=1).mean()
        )

        df['glucose_rollstd_1h'] = df.groupby('participant_id')['blood_glucose_value'].transform(
            lambda x: x.rolling(window=12, min_periods=1).std()
        )
        
        df['glucose_diff'] = df.groupby('participant_id')['blood_glucose_value'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        df['glucose_accel'] = df.groupby('participant_id')['glucose_diff'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        # Calculate glucose change rate (mg/dL per 5-minute interval)
        df['glucose_change_rate'] = df['glucose_diff'] / 5.0  # assuming 5-minute intervals
        
        # Fill any remaining NaN values in glucose_rollstd_1h with 0 (for first record of each participant)
        df['glucose_rollstd_1h'] = df['glucose_rollstd_1h'].fillna(0)
        
        return df
        
    except Exception as e:
        logging.error(f"Error in add_rolling_features: {e}")
        raise

def enrich_with_circadian(df):
    """Add circadian rhythm features."""
    try:
        # Normalize hour to 0–23 if needed
        df['hour'] = df['hour'] % 24

        # Add cyclic hour features
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Add is_sleep (tunable)
        df['is_sleep'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)

        return df
    except Exception as e:
        logging.error(f"Error in enrich_with_circadian: {e}")
        raise