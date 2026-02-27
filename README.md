
# Time-Series Classification  
**Deep Learning Framework for Modeling Glucose Dynamics and Metabolic Health**

## Overview
This repository implements a deep-learning pipeline for analyzing **AI-READI** time-series data, primarily using **Continuous Glucose Monitoring (CGM)** signals and engineered temporal features.

The primary objective of this project is to build a **hybrid supervised-unsupervised binary classification model**, for distinguishing:
- True Healthy  
- Pre-diabetes / Non-Healthy  

using sequence models such as LSTM and complementary relabeling using XGBoost.

This replaces the earlier planned multiclass physiological modeling pipeline.

---

## Project Structure

```
├── cgm_batch_extraction.py
├── cluster_true_healthy v2.py
├── xgboost_relabel_truehealthy v2.py
├── time_series_data_prepration_v1.py
├── time_series_lstm_analyze_model v2.py
├── time_series_lstm_analyze_model v2.py
├── paths.py
├── lstm_pipeline v1.py
├── cgm_lstm/
│   ├── data.py
│   ├── model.py
│   ├── pipeline.py
│   └── viz.py
│
├── dataset/
│   ├── cleaned_data2/
│   │   ├── batch_*.csv
│   │   ├── augmented_batches/
│   │   └── summarized_metrics_all_participants.csv
│   ├── uncleaned_data/
│   └── wearable_blood_glucose/
│       └── manifest.tsv
│
├── models/
│
├── logs/
│   ├── time_series_lstm_analyze_model v2.log
│   ├── xgboost_relabel_truehealthy v2.log
│   └── participant_plots/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Description

- **Official Dataset Website:**  
  https://aireadi.org/

- **Dataset Description:**  
  The AI-READI dataset provides multi-modal health sensor data including Continuous Glucose Monitoring (CGM), activity, sleep, and physiological signals collected from 1067 participants across several metabolic groups.

This project specifically uses the CGM time-series component.

### Important Data Organization Constraints
To properly use this pipeline with the original dataset, your local `dataset/` directory MUST reflect the exact expected structure from the raw download. Please ensure you have:
1. `dataset/wearable_blood_glucose/manifest.tsv`
2. `dataset/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/` (containing all the raw participant JSON subdirectories)
3. `dataset/clinical_and_metadata/participants.tsv`

### Auto-Generated Counts 
The codebase utilizes a summary file called `low_high_counts.csv` to rapidly identify the total records and valid flags per participant. If this file does not exist when you run `cgm_batch_extraction.py`, the script will **automatically scan all 1,067 raw JSON files** to count records, detect Low/High bounds, extract matching start/end timestamps, and generate it dynamically. You do not manually need to insert a `low_high_counts.csv` file into the `dataset/` directory.

---

## Feature Engineering

This pipeline applies structured feature engineering on CGM sequences, including:

- Rolling mean and standard deviation (1-hour window)  
- Glucose derivatives (difference, acceleration)  
- Circadian rhythm features (sin/hour, cos/hour)  
- Night and meal-time flags  
- Day fraction encoding  
- Change rate features  

These engineered features are used to build the final LSTM sequences.

---

## Modeling Approach

### 1. Binary LSTM Classifier
Predicts:
- true_healthy  
- pre_diabetes_lifestyle  

using a two-layer LSTM model with:
- train/val/test splitting  
- early stopping  
- validation-calibrated classification threshold  
- class-weight balancing  

### 2. XGBoost Relabeling Pipeline
A separate iterative relabeling pipeline:
- Trains an XGBoost classifier on summary metrics  
- Identifies and relabels inconsistent participants  
- Saves confusion matrix and logs  
- Supports multi-stage refinement of study group labels  

### 3. Model Analysis
The analysis script provides:
- ROC, PR, MCC, Balanced Accuracy  
- SHAP and LIME explanations (if available)  
- Temporal pattern interpretation  
- Method comparison across importance metrics  

---

## Installation

```bash
git clone https://github.com/Nikhil123n/Track2Better.git
cd Track2Better
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Extract CGM JSON into CSV Batches
```bash
python cgm_batch_extraction.py
```

### 2. Run Clustering Algoithm
```bash
python "cluster_true_healthy v2.py"
```

### 3. Run XGBoost Relabeling
```bash
python "xgboost_relabel_truehealthy v2.py"
```

### 4. Augment and Engineer Features
```bash
python time_series_data_prepration_v1.py
```

### 5. Train Binary LSTM Model
```bash
python lstm_pipeline_v1.py
```

### 6. Analyze Trained Model
```bash
python "time_series_lstm_analyze_model v3.py"
```


```

---

## Version Tags

This repository uses Git tags to mark important milestones:

- **`v1.0-manuscript`** - Complete research codebase at manuscript submission (main branch)
  - Full training pipeline and analysis tools
  - Model: model_20260209_020732
  - Test Accuracy: 84%, ROC-AUC: 0.932
  
- **`v1.0-inference`** - Production-ready inference package (inference-standalone branch)
  - Self-contained deployment with pre-trained model
  - BioCompute Object (BCO) metadata included
  - Clone: `git clone --branch v1.0-inference https://github.com/Nikhil123n/Track2Better.git`

---

## Contributing  
Pull requests, suggestions, and improvements are welcome.
