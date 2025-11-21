
# AI-READI CGM Time-Series Classification  
**Deep Learning Framework for Modeling Glucose Dynamics and Metabolic Health**

## Overview
This repository contains the full pipeline for processing, feature engineering, and modeling Continuous Glucose Monitoring (CGM) time-series data from the **AI-READI** dataset. The dataset includes **1067 participants**, each with 5-minute interval glucose readings collected over multiple days.

The primary objective of this project is to build a **supervised LSTM-based classification model** that learns glucose dynamics and predicts:

- **Healthy**
- **Type-2 Diabetes (Lifestyle Controlled)**
- **Type-2 Diabetes (Oral Medication)**
- **Type-2 Diabetes (Insulin)**

As part of the extended research direction, we also identify:
- **True-healthy vs. Non-true healthy participants** using KMeans clustering on glucose variability metrics.
- **Responder vs. Non-responder patterns** using LSTM embeddings and representation learning.

---

## Project Structure

```
├── data/
│   ├── batch_0_100.csv
│   ├── batch_100_200.csv
│   ├── batch_200_300.csv
│   └── all_participants_blood_glucose_values.csv
│
├── preprocessing/
│   ├── load_cgm_data.py
│   ├── feature_engineering.py
│   └── circadian_features.py
│
├── models/
│   ├── lstm_classifier.py
│   ├── training_pipeline.py
│   └── embeddings_umap.py
│
├── results/
│   ├── model_weights/
│   ├── classification_reports/
│   └── plots/
│
└── README.md
```

---

## Data Description

The AI-READI CGM dataset includes:
- **5-minute interval blood glucose records**
- **5–15 days of recordings per participant**
- **Demographics (e.g., age)**
- **Condition labels**

Extracted features per timestamp include:
- `blood_glucose_value`
- `glucose_change_rate`
- `timestamp`, `time_index`
- `sin(hour)`, `cos(hour)` for circadian modeling
- Rolling mean & rolling standard deviation features

Each participant has **2069 aligned time-series points** after filtering and interpolation.

The classification target:  
`study_group` → {healthy, T2D-lifestyle, T2D-oral, T2D-insulin}

---

## Feature Engineering

This project includes:
- Time alignment across participants  
- Circadian features (sin/cos hour)  
- Variability measures (rolling mean/std)  
- Dynamic features (change rate, volatility)

---

## Modeling Approach

### **1. Baseline LSTM Classifier**
- Input: Raw glucose sequence + engineered features  
- Architecture:
  - 2-layer LSTM  
  - Dropout  
  - Fully connected layers  
- Output: 4-class prediction

### **2. True-Healthy Classifier**
A binary classifier predicting:
- `true_healthy`
- `not_true_healthy`

### **3. Responder Detection**
Using LSTM embeddings + UMAP/PCA clustering.

---

## Installation

```bash
git clone https://github.com/Nikhil123n/Track2Better.git
cd Track2Better
pip install -r requirements.txt
```

---

## Running the Pipeline

### Preprocess Data
```bash
python preprocessing/load_cgm_data.py
```

### Train LSTM
```bash
python models/training_pipeline.py
```

### Generate Embeddings
```bash
python models/embeddings_umap.py
```

---

## Citation

```
```

---

## Contributing
Pull requests and suggestions are welcome.

---
