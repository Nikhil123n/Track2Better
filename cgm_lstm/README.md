# CGM LSTM Pre-Diabetes Detection Pipeline

**Version:** 2.1.0
**Last Updated:** 2026-02-08
**Author:** Track2Better Team

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Data Requirements](#data-requirements)
5. [Configuration](#configuration)
6. [Running the Pipeline](#running-the-pipeline)
7. [Output Files](#output-files)
8. [Advanced Usage](#advanced-usage)
9. [Reproducibility](#reproducibility)
10. [Troubleshooting](#troubleshooting)
11. [Citation](#citation)

---

## Overview

This pipeline implements a **Conv+BiLSTM deep learning model** for detecting pre-diabetes from **continuous glucose monitoring (CGM) time-series data**. The model achieves:

- **94.4% ROC-AUC** on held-out test set
- **98.0% precision** for pre-diabetes detection
- **2.6% false positive rate** (minimal unnecessary alarms)
- **10.1% OGTT burden** (only 1 in 10 need expensive OGTT test)

### Key Features

✅ **Held-out test validation** (gold standard for medical AI)
✅ **3-tier confidence-based predictions** (high confidence / uncertain / healthy)
✅ **Temperature-scaled calibration** (reliable probability estimates)
✅ **Global threshold optimization** (single production-ready threshold)
✅ **Modular JSON outputs** (MLOps-ready deployment artifacts)

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/Track2Better.git
cd Track2Better

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify data files exist
ls data/  # Should see participant_augmented_data_*.csv

# 4. Run pipeline
python "lstm_pipeline v1.py"

# 5. Check results
ls models/model_$(date +%Y%m%d)_*/
```

**Expected runtime:** ~5-8 minutes on CPU

---

## Installation

### Prerequisites

- **Python:** 3.9 - 3.11 (tested on 3.10.11)
- **OS:** Windows, macOS, Linux
- **RAM:** Minimum 8GB (16GB recommended)
- **Disk:** ~500MB for models + outputs

### Dependencies

```bash
# Core ML libraries
pip install tensorflow==2.15.0
pip install scikit-learn==1.3.2
pip install numpy==1.24.3
pip install pandas==2.1.3

# Visualization
pip install matplotlib==3.8.2
pip install seaborn==0.13.0

# Optional (for notebook exploration)
pip install jupyter ipykernel
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Verify Installation

```python
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

---

## Data Requirements

### Input Data Format

The pipeline expects **9 CSV files** in the `./data/` directory:

```
data/
├── participant_augmented_data_1.csv
├── participant_augmented_data_2.csv
├── ...
└── participant_augmented_data_9.csv
```

### Required Columns

Each CSV must contain these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `participant_id` | str | Unique participant ID | `P001` |
| `date_time` | datetime | Timestamp | `2024-01-15 08:30:00` |
| `blood_glucose_value` | float | CGM reading (mg/dL) | `105.3` |
| `glucose_diff` | float | First derivative | `2.1` |
| `glucose_accel` | float | Second derivative | `-0.5` |
| `glucose_rollmean_1h` | float | 1-hour rolling mean | `102.7` |
| `glucose_rollstd_1h` | float | 1-hour rolling std | `8.3` |
| `sin_hour` | float | Time encoding (sin) | `0.707` |
| `cos_hour` | float | Time encoding (cos) | `0.707` |
| `is_fasting` | int | Fasting indicator | `0` or `1` |
| `is_night` | int | Night indicator | `0` or `1` |
| `label` | str | Ground truth | `pre_diabetes_lifestyle` or `true_healthy` |

### Data Validation

The pipeline automatically validates:
- ✅ All required columns present
- ✅ No missing participant IDs
- ✅ Sufficient data per participant (~7 days minimum)
- ✅ Valid label values

---

## Configuration

### Main Configuration File: `cgm_lstm/data.py`

```python
@dataclass
class Config:
    # Model architecture
    sequence_length: int = 2138    # ~7 days of 5-min intervals
    n_features: int = 9             # Number of input features
    lstm_units: int = 64            # LSTM hidden units
    dense_units: int = 32           # Dense layer units
    dropout_rate: float = 0.3       # Dropout for regularization
    l2_reg: float = 1e-3            # L2 regularization strength

    # Training
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 10

    # Cross-validation
    n_splits: int = 5               # 5-fold CV
    random_state: int = 42          # For reproducibility
    val_size_from_train: float = 0.2  # 20% validation split

    # Held-out test (NEW!)
    use_held_out_test: bool = True   # Enable gold-standard validation
    held_out_test_size: float = 0.2  # 20% held-out
    held_out_random_seed: int = 42   # For reproducible splits

    # Class balancing
    use_class_weights: bool = True   # Recommended
    use_smote: bool = False          # Not recommended (use class weights)

    # Calibration
    use_temperature_scaling: bool = True  # Post-hoc calibration
    calibration_bins: int = 10           # For ECE computation

    # Confidence zones (dynamic)
    confidence_low_offset: float = 0.08   # Zone 1 boundary
    confidence_high_offset: float = 0.22  # Zone 3 boundary
```

### Modifying Configuration

**Option 1: Edit `data.py` directly**

```python
# In cgm_lstm/data.py, change values in Config class
l2_reg: float = 1e-2  # Increase regularization
```

**Option 2: Override in `lstm_pipeline v1.py`**

```python
# In lstm_pipeline v1.py, after config creation
config = Config()
config.l2_reg = 1e-2  # Override specific values
```

---

## Running the Pipeline

### Basic Execution

```bash
python "lstm_pipeline v1.py"
```

### What Happens During Execution

```
Pipeline Stages:
├─ [1] Load and validate data                    (~10s)
├─ [2] Create LSTM sequences                     (~5s)
├─ [3] Split held-out test set (if enabled)      (~2s)
├─ [4] Run 5-fold cross-validation               (~4min)
│   ├─ Fold 1/5: Train → Validate → Test
│   ├─ Fold 2/5: Train → Validate → Test
│   ├─ Fold 3/5: Train → Validate → Test
│   ├─ Fold 4/5: Train → Validate → Test
│   └─ Fold 5/5: Train → Validate → Test
├─ [5] Train final production model              (~1min)
├─ [6] Evaluate on held-out test set             (~5s)
└─ [7] Generate deployment artifacts             (~2s)

Total time: ~5-8 minutes
```

### Monitoring Progress

Watch the console output:

```
[START] Starting LSTM Binary Classification Pipeline
[STEP 1] Loading and preparing data...
[OK] Loaded combined dataset: (1676192, 19)
[STEP 2] Creating LSTM sequences...
[OK] Created LSTM sequences | participants=491, valid=491
[HELD-OUT] Splitting data into train+val and held-out test set...
[HELD-OUT] Train+Val: 392 participants | Held-Out Test: 99 participants
[STEP 4] Running Cross-Validation...
[CV] Fold 1/5 starting...
  [TRAIN] Starting model training...
  [THRESH] Selected threshold on VAL: 0.3067 | Val AUC: 0.9402
  [CALIB] Temperature scaling fitted: T=1.0000
  [CV] Fold 1 done | ROC-AUC=0.9224 | Acc=0.8734
...
[STEP 6] Evaluating on held-out test set...
[HELD-OUT] ROC-AUC: 0.9435
[HELD-OUT] Pre-Diabetes Precision: 0.9804
[HELD-OUT] Detection Rate: 86.9%
[COMPLETE] Held-out test evaluation completed!
```

---

## Output Files

### Directory Structure

After running the pipeline, you'll find:

```
models/
└── model_20260208_185255/              ← Timestamped folder
    ├── lstm_pipeline.log               ← Complete execution log (NEW!)
    ├── model_metadata.json             ← High-level info (NEW!)
    ├── model_architecture.json         ← Model structure (NEW!)
    ├── training_results.json           ← Training history (NEW!)
    ├── evaluation_results.json         ← All metrics (NEW!)
    ├── deployment_config.json          ← Production config (NEW!)
    │
    ├── production_model/               ← Ready for deployment
    │   ├── best_model.keras            ← Trained weights
    │   ├── feature_scaler.npz          ← Normalization params
    │   ├── global_threshold.json       ← Decision threshold
    │   ├── confidence_thresholds.json  ← Zone boundaries
    │   ├── temperature_scaler.json     ← Calibration params
    │   ├── training_config.json        ← Hyperparameters
    │   └── DEPLOYMENT_README.md        ← How to deploy
    │
    ├── cross_validation/               ← CV results
    │   ├── cv_results.json             ← Aggregated CV metrics
    │   └── fold_1/                     ← Per-fold outputs
    │       ├── multicollinearity_fold_1.json
    │       ├── multicollinearity_corr_heatmap_fold_1.png
    │       ├── threshold_sensitivity.png
    │       └── reliability_curve.png
    │   └── fold_2/ ... fold_5/
    │
    ├── held_out_test/                  ← Gold-standard validation
    │   └── held_out_test_results.json  ← TRUE generalization metrics
    │
    └── processed_data/                 ← Preprocessed sequences
        ├── X_sequences.npy
        ├── y_labels.npy
        └── participant_ids.npy
```

### Key Files Explained

#### **1. `lstm_pipeline.log`** (NEW!)
Complete execution log with timestamps, now saved in model folder (not root).

```
2026-02-08 18:52:55,039 - cgm_lstm.pipeline - INFO - [START] Starting LSTM Pipeline
...
2026-02-08 18:56:01,754 - cgm_lstm.pipeline - INFO - [COMPLETE] Held-out test complete!
```

#### **2. `model_metadata.json`** (NEW!)
High-level model information for documentation.

```json
{
  "model_id": "model_20260208_185255",
  "version": "2.1.0",
  "purpose": "Pre-diabetes detection from CGM time-series",
  "known_limitations": [
    "Trained on 491 participants (small dataset)",
    "Not validated on external datasets"
  ]
}
```

#### **3. `model_architecture.json`** (NEW!)
Complete model structure for reproducibility.

```json
{
  "model_type": "Conv+BiLSTM",
  "total_parameters": 121729,
  "layers": [
    {"type": "Conv1D", "filters": 32, "kernel_size": 3},
    {"type": "Bidirectional", "units": 64, "cell": "LSTM"},
    ...
  ]
}
```

#### **4. `evaluation_results.json`** (NEW!)
Combined CV + held-out metrics.

```json
{
  "cross_validation": {
    "metrics": {
      "roc_auc": {"mean": 0.9190, "std": 0.0403}
    }
  },
  "held_out_test": {
    "metrics": {
      "roc_auc": 0.9435,
      "accuracy": 0.8788
    }
  }
}
```

#### **5. `deployment_config.json`** (NEW!)
Production-ready deployment configuration.

```json
{
  "deployment_status": "PRODUCTION_READY",
  "primary_metrics": {
    "roc_auc": 0.9435,
    "source": "held_out_test"
  },
  "monitoring_thresholds": {
    "roc_auc_min": 0.92
  }
}
```

---

## Advanced Usage

### Experiment Tracking

Each run creates a timestamped folder. Compare experiments:

```bash
# View all experiments
ls models/

# Compare ROC-AUC across runs
for dir in models/model_202602*/; do
  echo "$dir:"
  jq '.held_out_test.metrics.roc_auc' "$dir/evaluation_results.json"
done
```

### Hyperparameter Tuning

Create a grid search script:

```python
# experiment_grid.py
from cgm_lstm.data import Config

configs = [
    {"l2_reg": 1e-4, "lstm_units": 64},
    {"l2_reg": 1e-3, "lstm_units": 64},
    {"l2_reg": 1e-2, "lstm_units": 64},
    {"l2_reg": 1e-3, "lstm_units": 128},
]

for i, params in enumerate(configs):
    print(f"\n[EXPERIMENT {i+1}/{len(configs)}]")
    config = Config()
    for key, value in params.items():
        setattr(config, key, value)

    # Run pipeline with modified config
    # ... (run pipeline code here)
```

### Custom Data

If using your own CGM data:

```python
# 1. Ensure data matches required format
import pandas as pd
df = pd.read_csv("my_cgm_data.csv")

# 2. Validate columns
required = [
    'participant_id', 'date_time', 'blood_glucose_value',
    'glucose_diff', 'glucose_accel', 'glucose_rollmean_1h',
    'glucose_rollstd_1h', 'sin_hour', 'cos_hour',
    'is_fasting', 'is_night', 'label'
]
assert all(col in df.columns for col in required)

# 3. Save in expected location
df.to_csv("data/participant_augmented_data_1.csv", index=False)

# 4. Run pipeline
python "lstm_pipeline v1.py"
```

---

## Reproducibility

### Ensuring Reproducibility

The pipeline is designed for full reproducibility:

1. **Fixed Random Seeds**
   ```python
   random_state: int = 42
   held_out_random_seed: int = 42
   ```

2. **Environment Tracking**
   ```bash
   pip freeze > requirements.txt
   ```

3. **Data Versioning**
   ```bash
   # Hash data files
   md5sum data/*.csv > data_checksums.txt
   ```

4. **Configuration Logging**
   All configs saved in `training_config.json`

### Reproducing Results

To reproduce model `model_20260208_185255`:

```bash
# 1. Clone repository
git clone https://github.com/yourusername/Track2Better.git
cd Track2Better

# 2. Install exact dependencies
pip install -r requirements.txt

# 3. Verify data integrity
md5sum -c data_checksums.txt

# 4. Run with same config
# Config is already set in cgm_lstm/data.py (no changes needed)
python "lstm_pipeline v1.py"

# 5. Compare results
# Should see identical ROC-AUC (0.9190 ± 0.0403 for CV)
```

### Version Control

```bash
# Track changes
git add .
git commit -m "Experiment: L2=1e-3, held-out validation"
git tag -a v2.1.0 -m "Production-ready model"
git push origin v2.1.0
```

---

## Troubleshooting

### Common Issues

#### **1. Memory Error During Training**

```
MemoryError: Unable to allocate array
```

**Solution:** Reduce batch size

```python
# In cgm_lstm/data.py
batch_size: int = 8  # Reduce from 16
```

#### **2. Data Files Not Found**

```
FileNotFoundError: [Errno 2] No such file or directory: './data/participant_augmented_data_1.csv'
```

**Solution:** Verify data directory structure

```bash
ls data/  # Should list 9 CSV files
```

#### **3. TensorFlow GPU Warning**

```
Could not load dynamic library 'cudart64_**.dll'
```

**Solution:** This is just a warning (pipeline runs on CPU). To enable GPU:

```bash
# Install CUDA toolkit (optional)
# See: https://www.tensorflow.org/install/gpu
```

#### **4. Held-Out Test Disabled**

Pipeline skips held-out test evaluation.

**Solution:** Enable in config

```python
# In cgm_lstm/data.py
use_held_out_test: bool = True  # Must be True
```

#### **5. JSON Serialization Error**

```
TypeError: Object of type 'float32' is not JSON serializable
```

**Solution:** Already handled in `model_card.py` via `default=str` in `json.dump()`

### Getting Help

1. **Check logs:** `models/model_*/lstm_pipeline.log`
2. **GitHub Issues:** [Report a bug](https://github.com/yourusername/Track2Better/issues)
3. **Documentation:** [Full docs](./PHASE1_RESULTS_ANALYSIS.md)

---

## Performance Benchmarks

### Expected Metrics (Held-Out Test)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.944 | Excellent discrimination |
| **PR-AUC** | 0.893 | Strong precision-recall |
| **Pre-D Precision** | 98.0% | Very reliable positive predictions |
| **Pre-D Recall** | 82.0% | Catches 4 out of 5 directly |
| **False Positive Rate** | 2.6% | Minimal unnecessary alarms |
| **OGTT Burden** | 10.1% | Only 1 in 10 need expensive test |

### Runtime Benchmarks

| Hardware | Training Time | Inference Time (per sample) |
|----------|--------------|----------------------------|
| CPU (4-core) | ~6 minutes | ~20ms |
| GPU (NVIDIA RTX 3060) | ~2 minutes | ~5ms |

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{cgm_lstm_prediabetes_2026,
  title = {CGM LSTM Pre-Diabetes Detection Pipeline},
  author = {Track2Better Team},
  year = {2026},
  version = {2.1.0},
  url = {https://github.com/yourusername/Track2Better}
}
```

---

## License

MIT License - See [LICENSE](../LICENSE) file for details.

---

## Changelog

### Version 2.1.0 (2026-02-08)
- ✅ Added held-out test validation (gold standard)
- ✅ Implemented modular JSON generation (MLOps-ready)
- ✅ Moved log file to model folder
- ✅ Dynamic confidence zone boundaries
- ✅ Comprehensive README for reproducibility

### Version 2.0.0 (2026-02-08)
- ✅ Global threshold from OOF predictions
- ✅ Temperature scaling calibration
- ✅ 3-tier confidence-based system

### Version 1.0.0 (2026-02-08)
- ✅ Initial Conv+BiLSTM implementation
- ✅ 5-fold cross-validation

---

## Contact

For questions or collaboration:
- **Email:** your.email@example.com
- **GitHub:** [@yourusername](https://github.com/yourusername)
- **Issues:** [Report here](https://github.com/yourusername/Track2Better/issues)
