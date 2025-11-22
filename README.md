
# Time-Series Classification  
**Deep Learning Framework for Modeling Glucose Dynamics and Metabolic Health**

## Overview
This repository implements a **multimodal deep-learning pipeline** for analyzing **AI-READI** time-series data, including **Continuous Glucose Monitoring (CGM)** signals and physiological features such as:
- Heart Rate  
- Respiration Rate  
- Stress Levels  
- Physical Activity  
- Sleep Patterns
  
The primary objective of this project is to build a **supervised LSTM-based classification model** that learns glucose dynamics and predicts:

- **Healthy**
- **Type-2 Diabetes (Lifestyle Controlled)**
- **Type-2 Diabetes (Oral Medication)**
- **Type-2 Diabetes (Insulin)**

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
│   ├── cgm_batch_creation.py
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
- **1067 participants**
- **5-minute interval blood glucose records**
- **5–15 days of recordings per participant**
- **Demographics (e.g., age)**
- **Multimodal sensor features**  
- **Demographic attributes**  
- **Study group labels**

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
python preprocessing/cgm_batch_creation.py
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
