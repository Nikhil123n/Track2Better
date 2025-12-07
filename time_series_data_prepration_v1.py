__CURR_FILE__ = "time_series_data_prepration_v1"

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

# === PATH CONFIG (centralized via paths.py) ===
# All dataset paths are defined in paths.py so this script is portable
from paths import PILOT_ROOT_DATA_PATH, MANIFEST_PATH, CLEANED_DATA_PATH, SUMMARY_PATH, AUGMENTED_OUTPUT_DIR, LOG_DIR
# Output paths
# LABELED_OUTPUT_DIR = os.path.join(CLEANED_DATA_PATH, "cleaned_batches_labeled")

# Create logs directory and Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG to see more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, __CURR_FILE__ + ".log")),  # Save logs to file
        logging.StreamHandler()  # Also print to console
    ]
)
logging.info(f"Logs will be saved to: {LOG_DIR}")

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

def extract_time_features(df):
    """Extract time-based features from time column without creating timestamp."""
    try:
        # Parse time string to extract hour
        # Assuming time format is HH:MM:SS
        df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.hour
        
        # Handle any failed time parsing
        if df['hour'].isnull().any():
            logging.warning(f"Failed to parse {df['hour'].isnull().sum()} time values")
            # Try alternative parsing
            df['hour'] = df['time'].str.split(':').str[0].astype(int, errors='ignore')
        
        # Create time-based features
        df['day_fraction'] = df['hour'] / 24 + pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.minute / 1440
        df['is_night'] = df['hour'].isin([0,1,2,3,4,5,22,23]).astype(int)
        df['is_meal_time'] = df['hour'].isin([7,8,9,12,13,14,18,19,20]).astype(int)
        
        # Fill any NaN values in day_fraction
        df['day_fraction'] = df['day_fraction'].fillna(df['hour'] / 24)
        
        return df
        
    except Exception as e:
        logging.error(f"Error in extract_time_features: {e}")
        raise
    
def process_and_augment_batches():
    """Combined function that labels and augments batches in one pass for efficiency."""
    try:
        os.makedirs(AUGMENTED_OUTPUT_DIR, exist_ok=True)

        # Load label map with error handling
        if not os.path.exists(SUMMARY_PATH):
            raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_PATH}")
        
        label_df = pd.read_csv(SUMMARY_PATH)
        # Validate required columns exist
        if 'participant_id' not in label_df.columns or 'study_group_cleaned' not in label_df.columns:
            raise ValueError("Required columns missing from summary CSV")
        
        label_map = dict(zip(label_df['participant_id'], label_df['study_group_cleaned']))
        logging.info(f"Loaded {len(label_map)} participant labels")

        # Process each batch file
        batch_files = sorted(glob(os.path.join(CLEANED_DATA_PATH, "batch_*.csv")))

        if not batch_files:
            raise FileNotFoundError("No batch files found")
        
        successful_batches = 0
        for file_path in batch_files:
            try:
                logging.info("\n" + " " * 60)
                logging.info(f"Processing {os.path.basename(file_path)}")

                df = pd.read_csv(file_path)
                
                # Step 1: Label mapping
                df['study_group'] = df['participant_id'].map(label_map)
                df.dropna(subset=['study_group'], inplace=True)

                # Step 2: Filter participants with at least 2138 records and keep only first 2138 records
                participant_counts = df['participant_id'].value_counts()
                valid_participants = participant_counts[participant_counts >= 2138].index
                df = df[df['participant_id'].isin(valid_participants)]
                df = df.groupby('participant_id').head(2138).reset_index(drop=True)
                
                logging.info(f"Filtered to {len(valid_participants)} participants with ≥2138 records")
                logging.info(f"Total records after filtering: {len(df)}")

                # Step 3: Feature engineering
                # Extract time features from time column (no timestamp creation)
                df = extract_time_features(df)

                # Add rolling/statistical features using vectorized operations
                df = add_rolling_features(df)

                # Add circadian features
                df = enrich_with_circadian(df)

                # Step 4: Save final augmented file
                fname = os.path.basename(file_path).replace(".csv", "_labeled_augmented.csv")
                out_path = os.path.join(AUGMENTED_OUTPUT_DIR, fname)
                df.to_csv(out_path, index=False)

                successful_batches += 1
                logging.info(f"Successfully saved: {out_path}")
                
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")
                continue
        
        logging.info(f"\nPipeline completed: {successful_batches}/{len(batch_files)} batches processed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


# === Visualization: Individual Participant Glucose Dynamics ===
def plot_glucose_dynamics(file_path, participant_id, save=True):
    """Plot glucose dynamics for a specific participant."""
    try:
        # Load one augmented batch
        df = pd.read_csv(file_path)

        # Choose one participant
        df_p = df[df['participant_id'] == participant_id].sort_values("time_index")
        
        if df_p.empty:
            logging.warning(f"No data found for participant {participant_id}")
            return

        # Plot key features
        plt.figure(figsize=(15, 10))

        # Glucose values
        plt.subplot(3, 1, 1)
        plt.plot(df_p['time_index'], df_p['blood_glucose_value'], label="Glucose", color="black")
        plt.plot(df_p['time_index'], df_p['glucose_rollmean_1h'], label="Roll Mean (1h)", color="blue")
        plt.fill_between(df_p['time_index'],
                        df_p['blood_glucose_value'] - df_p['glucose_rollstd_1h'],
                        df_p['blood_glucose_value'] + df_p['glucose_rollstd_1h'],
                        color='blue', alpha=0.2, label="±1 STD")
        plt.legend()
        plt.title(f"Participant {participant_id} - Glucose Dynamics")

        # Derivatives
        plt.subplot(3, 1, 2)
        plt.plot(df_p['time_index'], df_p['glucose_diff'], label="Glucose Diff", color="orange")
        plt.plot(df_p['time_index'], df_p['glucose_accel'], label="Glucose Accel", color="red")
        plt.legend()
        plt.title("First and Second Derivatives")

        # Circadian pattern
        plt.subplot(3, 1, 3)
        plt.plot(df_p['time_index'], df_p['hour'], label="Hour of Day", color="green")
        plt.fill_between(df_p['time_index'], 0, df_p['is_meal_time'] * 24,
                        color='purple', alpha=0.3, label="Meal Time")
        plt.fill_between(df_p['time_index'], 0, df_p['is_night'] * 24,
                        color='gray', alpha=0.2, label="Night Time")
        plt.legend()
        plt.title("Circadian Context")

        plt.tight_layout()

        # SAVE PLOT
        if save:
            save_dir = os.path.join(LOG_DIR, "participant_plots")
            os.makedirs(save_dir, exist_ok=True)

            outfile = os.path.join(
                save_dir,
                f"participant_{participant_id}_glucose_dynamics.png"
            )

            plt.savefig(outfile, dpi=300, bbox_inches="tight")
            logging.info(f"[SAVED] Glucose dynamics plot saved to: {outfile}")

        # SHOW PLOT
        plt.show()
        
    except Exception as e:
        logging.error(f"Error in plot_glucose_dynamics: {e}")


# === Visualization: Boxplots for Feature Distributions by Study Group ===
def plot_feature_distributions(file_path):
    """Plot feature distributions by study group."""
    try:
        # Load the augmented batch
        df = pd.read_csv(file_path)

        # Select all numeric features except participant_id, time_index, date, time, and study_group
        exclude_cols = ['participant_id', 'time_index', 'date', 'time', 'study_group']
        all_features = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        save_dir = os.path.join(LOG_DIR, "participant_plots")
        os.makedirs(save_dir, exist_ok=True)

        # Plot each feature individually by study group
        for feature in all_features:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df, x='study_group', y=feature, hue='study_group', dodge=False)
            plt.title(f"{feature} by Study Group")
            plt.xticks(rotation=20)
            plt.tight_layout()

            # === SAVE PLOT ===
            out_path = os.path.join(save_dir, f"{feature}_by_study_group.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            logging.info(f"[SAVED] Feature distribution plot saved: {out_path}")
            
            plt.show()
            
    except Exception as e:
        logging.error(f"Error in plot_feature_distributions: {e}")


# === Main Pipeline ===
if __name__ == "__main__":
    try:
        # Step 1: Relabel batches and Step 2: Augment batches with time-based, statistical, and circadian features
        process_and_augment_batches()

        # Optional: Visualizations
        plot_glucose_dynamics(
            file_path=os.path.join(AUGMENTED_OUTPUT_DIR, "batch_0_100_labeled_augmented.csv"),
            participant_id=1001
        )

        plot_feature_distributions(
            file_path=os.path.join(AUGMENTED_OUTPUT_DIR, "batch_0_100_labeled_augmented.csv")
        )
        
    except Exception as e:
        logging.error(f"Main pipeline execution failed: {e}")
        raise