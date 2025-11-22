import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG to see more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_processing.log"),  # Save logs to file
        logging.StreamHandler()  # Also print to console
    ]
)

# Mapping of old study group labels to new simplified labels
STUDY_GROUP_MAPPING = {
    "healthy": "healthy",
    "pre_diabetes_lifestyle_controlled": "pre_diabetes_lifestyle",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "oral_medication_and_or_non_insulin",
    "insulin_dependent": "insulin_dependent"
}

# Helper Function: Flatten JSON
def flatten_json(y):
    """Flatten a nested JSON structure."""
    out = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '_')
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '_')
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

# Helper Function: Read and Flatten JSON for a given Participant
def get_flatten_dict_from_path(pid, dfm, pilot_data_root):
    """Read and flatten CGM data for a participant."""
    pid_cgm = dfm[dfm['participant_id'] == pid]['glucose_filepath'].values[0]
    cgm_path = os.path.join(pilot_data_root, pid_cgm)
    
    logging.info(f"Opening file: {cgm_path}")
    try:
        with open(cgm_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found for participant {pid}")
        return []
    except json.JSONDecodeError:
        logging.error(f"JSON decode error for participant {pid}")
        return []

    # Flatten each CGM observation
    return [flatten_json(obs) for obs in data.get('body', {}).get('cgm', [])]


# Main Function: Extract and Curate Batches
def extract_and_curate_batches(MANIFEST_PATH, PILOT_DATA_ROOT, OUTPUT_PATH, batch_size=100, min_records=2138, participant_filter="valid"):
    """
    Extract and curate blood glucose data in batches.

    Args:
        MANIFEST_PATH (str): Path to the manifest.tsv file.
        PILOT_DATA_ROOT (str): Root directory of the dataset.
        OUTPUT_PATH (str): Directory to save output batch and combined CSV files.
        batch_size (int): Number of participants per batch.
        min_records (int): Minimum required glucose readings per participant.
        participant_filter (str): "valid" for Low/High==0, "invalid" for Low/High!=0, or pass a custom mask.
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Load Required Files
    dfm = pd.read_csv(MANIFEST_PATH, sep='\t')
    csv_path_low_high_counts = os.path.join(PILOT_DATA_ROOT, "cleaned_data", "low_high_counts.csv")    
    df_low_high_counts = pd.read_csv(csv_path_low_high_counts)
    df_participants = pd.read_csv(os.path.join(PILOT_DATA_ROOT, "participants.tsv"), sep='\t')

    # Filter Participants with Valid Records
    if participant_filter == "valid":
        df_filtered = df_low_high_counts[
            (df_low_high_counts['Low'] == 0) & 
            (df_low_high_counts['High'] == 0)
        ]
    elif participant_filter == "invalid":
        df_filtered = df_low_high_counts[
            (df_low_high_counts['Low'] != 0) | 
            (df_low_high_counts['High'] != 0)
        ]

    selected_pids = df_filtered['participant_id'].tolist()
    
    # Process Participants in Batches
    all_data_records = []
    # Outer loop to process participants in batches of size `batch_size`
    for batch_start in range(0, len(selected_pids), batch_size):
        batch_end = min(batch_start + batch_size, len(selected_pids))
        batch_pids = selected_pids[batch_start:batch_end]
        batch_data_records = []
        
        # Inner loop to process each participant in the batch
        for pid in batch_pids:
            records = get_flatten_dict_from_path(pid, dfm, PILOT_DATA_ROOT)
            if len(records) == 0:
                continue

            glucose_values = [r['blood_glucose_value'] for r in records]
            age = df_participants[df_participants['participant_id'] == pid]['age'].values[0]

            # Map the study group label
            study_group_raw = df_participants[df_participants['participant_id'] == pid]['study_group'].values[0]
            study_group = STUDY_GROUP_MAPPING.get(study_group_raw, study_group_raw)  # fallback to original if unmapped

            participant_records = [
                {
                    'participant_id': pid,
                    'time_index': i + 1,                    
                    'date': datetime.strptime(rec['effective_time_frame_time_interval_start_date_time'], "%Y-%m-%dT%H:%M:%S%z").strftime("%Y-%m-%d"),
                    'time': datetime.strptime(rec['effective_time_frame_time_interval_start_date_time'], "%Y-%m-%dT%H:%M:%S%z").strftime("%H:%M:%S"),
                    'age': age,
                    'blood_glucose_value': glucose_values[i],
                    # 'glucose_change_rate': glucose_values[i] - glucose_values[i - 1] if i > 0 else 0,
                    'study_group': study_group
                }
                for i, rec in enumerate(records)
            ]

            batch_data_records.extend(participant_records)

        # Save the batch data to a CSV file
        batch_df = pd.DataFrame(batch_data_records)
        batch_csv = os.path.join(OUTPUT_PATH, f"batch_{batch_start}_{batch_end}.csv")
        batch_df.to_csv(batch_csv, index=False)
        logging.info(f"Batch saved to: {batch_csv}")

        all_data_records.extend(batch_data_records)

    # Combine all batches into a single CSV file
    df_all = pd.DataFrame(all_data_records)
    all_csv_path = os.path.join(OUTPUT_PATH, "all_participants_blood_glucose_values.csv")
    df_all.to_csv(all_csv_path, index=False)
    logging.info(f"Combined CSV saved to: {all_csv_path}")


# Example usage
if __name__ == "__main__":
    MANIFEST_PATH = "C:/Users/nikhi/Box/AI-READI/nikhil working dataset/dataset/wearable_blood_glucose/manifest.tsv"
    PILOT_DATA_ROOT = "C:/Users/nikhi/Box/AI-READI/nikhil working dataset/dataset/"
    OUTPUT_PATH_VALID = "C:/Users/nikhi/Box/AI-READI/nikhil working dataset/dataset/cleaned_data2"
    OUTPUT_PATH_INVALID = "C:/Users/nikhi/Box/AI-READI/nikhil working dataset/dataset/uncleaned_data"

    # For valid participants (Low/High == 0)
    extract_and_curate_batches(MANIFEST_PATH, PILOT_DATA_ROOT, OUTPUT_PATH_VALID, batch_size=100, min_records=2138)

    # For invalid participants (Low/High != 0)
    extract_and_curate_batches(MANIFEST_PATH, PILOT_DATA_ROOT, OUTPUT_PATH_INVALID, batch_size=100, participant_filter="invalid")
