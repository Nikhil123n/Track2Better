# CGM Pre-Diabetes Detection - Inference

Production inference script for deploying the trained CGM pre-diabetes detection model.

## Quick Start

### Basic Usage

```bash
# Using model files in current directory (default)
python inference.py --csv test_set.csv

# Using model from different directory
python inference.py --csv test_set.csv --model ../models/model_20260209_020732/production_model
```

### With Output File

```bash
python inference.py --csv test_set.csv --output results.json
```

### Verify Setup

```bash
python inference.py --verify
```

## Production Model Files

The inference folder contains all necessary model files for standalone deployment:

| File | Description |
|------|-------------|
| best_model.keras | Trained Conv+BiLSTM model |
| feature_scaler.npz | Feature normalization parameters |
| global_threshold.json | Optimal decision boundary (0.3746) |
| confidence_thresholds.json | Dynamic thresholds for 3-tier prediction |
| temperature_scaler.json | Temperature scaling parameter (T=1.0) |
| training_config.json | Model training configuration |

All files are loaded automatically from the current directory when no `--model` parameter is specified.

## Input Data Format

### Pre-computed Features (Default)

CSV file with 2138 rows containing:

| Column | Description |
|--------|-------------|
| participant_id | Patient identifier |
| blood_glucose_value | CGM reading (mg/dL) |
| glucose_rollmean_1h | 1-hour rolling mean |
| glucose_rollstd_1h | 1-hour rolling std |
| glucose_diff | First derivative |
| glucose_accel | Second derivative |
| sin_hour | Circadian sine |
| cos_hour | Circadian cosine |
| is_meal_time | Meal indicator (0/1) |
| is_night | Night indicator (0/1) |

### Raw CGM Data (with --raw flag)

CSV file with 2138 rows containing:

| Column | Description |
|--------|-------------|
| participant_id | Patient identifier |
| time_index | Sequential index (0-2137) |
| time | Time in HH:MM:SS format |
| blood_glucose_value | CGM reading (mg/dL) |

Features will be automatically computed.

## Output Format

### Console Output

```
======================================================================
CGM PRE-DIABETES DETECTION - PREDICTION RESULTS
======================================================================

Patient Information:
   Participant ID: 1244
   Study Group: pre_diabetes_lifestyle_controlled
   Sequence Length: 2138 points
   Glucose Stats: 139.2 +/- 22.9 mg/dL
   Glucose Range: [80.0, 236.0] mg/dL

Ground Truth:
   True Label: Pre-Diabetes

Prediction:
   Status: Uncertain
   Confidence: Medium
   Risk Level: Medium

Probabilities:
   Healthy: 52.9%
   Pre-Diabetes: 47.1%

Uncertain Zone - Ground truth: Pre-Diabetes

Recommended Action:
   Secondary screening (OGTT) recommended for confirmation

======================================================================

UNCERTAIN: Pattern is near decision boundary.
Recommend: OGTT confirmation test within 2-4 weeks
======================================================================
```

### JSON Output (with --output flag)

```json
{
  "metadata": {
    "participant_id": "1244",
    "sequence_length": 2138,
    "glucose_mean": 139.2,
    "glucose_std": 22.9,
    "glucose_min": 80.0,
    "glucose_max": 236.0,
    "true_label": 0,
    "true_label_text": "Pre-Diabetes",
    "study_group": "pre_diabetes_lifestyle_controlled"
  },
  "prediction": {
    "prediction": "Uncertain",
    "confidence": "Medium",
    "action": "Secondary screening (OGTT) recommended for confirmation",
    "probability": 0.529,
    "probability_healthy": 0.529,
    "probability_prediabetes": 0.471,
    "class_label": null,
    "risk_level": "Medium"
  }
}
```

## Clinical Decision Rules

The model uses a 3-tier confidence-based system with **dynamic thresholds** computed during training:

### Threshold Computation

During training, the system:
1. Computes an optimal global threshold using **Youden's J statistic** on out-of-fold validation data
2. Creates three decision zones centered on this optimal threshold:
   - **Low threshold** = global_threshold - 0.08 (margin below)
   - **High threshold** = global_threshold + 0.22 (margin above)

For the current model (`model_20260209_020732`):
- Global threshold: **0.3746**
- Low threshold: **0.2946**
- High threshold: **0.5946**

### Decision Zones

**High Confidence Healthy (prob >= 0.5946)**
- **Prediction**: Healthy
- **Risk Level**: Low
- **Action**: Continue regular screening cycle (1-2 years)

**Uncertain (0.2946 <= prob < 0.5946)**
- **Prediction**: Uncertain
- **Risk Level**: Medium
- **Action**: Secondary screening (OGTT) recommended for confirmation

**High Confidence Pre-Diabetes (prob < 0.2946)**
- **Prediction**: Pre-Diabetes
- **Risk Level**: High
- **Action**: Immediate lifestyle intervention recommended

**Note**: Model predicts probability of "Healthy" class. Lower probability indicates higher pre-diabetes risk.

### Why Dynamic Thresholds?

- **Optimized for data**: Thresholds adapt to the model's actual probability distribution
- **Not arbitrary**: Based on Youden's J statistic (maximizes Sensitivity + Specificity - 1)
- **Validated**: Computed from out-of-fold validation predictions during cross-validation
- **Robust**: Zone widths and centers reflect real performance characteristics

## Model Components

The production model directory must contain:

```
production_model/
├── best_model.keras              # Trained model weights
├── feature_scaler.npz             # Feature normalization parameters
├── global_threshold.json          # Optimized decision threshold (Youden's J)
├── temperature_scaler.json        # Calibration parameter
├── confidence_thresholds.json     # Dynamic 3-tier decision thresholds
└── training_config.json           # Model architecture info
```

**Note**: If `confidence_thresholds.json` is missing, the system falls back to default thresholds (0.35, 0.65), but dynamic thresholds are strongly recommended for optimal performance.

## Feature Engineering Details

For raw CGM data (--raw flag), the script automatically computes:

### Rolling Statistics
- **glucose_rollmean_1h**: 1-hour moving average (12 points)
- **glucose_rollstd_1h**: 1-hour moving std deviation

### Derivatives
- **glucose_diff**: First derivative (change rate)
- **glucose_accel**: Second derivative (acceleration)

### Circadian Features
- **sin_hour**: Sine transformation of hour (0-24)
- **cos_hour**: Cosine transformation of hour (0-24)
- **is_meal_time**: Binary flag for meal hours (7-9, 12-14, 18-20)
- **is_night**: Binary flag for night hours (0-6, 22-23)

## Requirements

- Python 3.7+
- numpy>=1.24.0
- pandas>=2.0.0
- tensorflow>=2.12.0

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas tensorflow
```

## Advanced Usage

### Programmatic Use

```python
from inference import ProductionPredictor, load_precomputed_features

# Load data
X, metadata = load_precomputed_features('test_set.csv')

# Load model and predict
predictor = ProductionPredictor('../models/model_20260209_020732/production_model')
result = predictor.predict_with_confidence(X)

# Access results
print(f"Prediction: {result['prediction']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Action: {result['action']}")
```

### Batch Processing

```python
import glob
from inference import ProductionPredictor, load_precomputed_features

predictor = ProductionPredictor('path/to/production_model')

for csv_file in glob.glob('patients/*.csv'):
    try:
        X, metadata = load_precomputed_features(csv_file)
        result = predictor.predict_with_confidence(X)
        
        output_file = csv_file.replace('.csv', '_result.json')
        with open(output_file, 'w') as f:
            json.dump({'metadata': metadata, 'prediction': result}, f, indent=2)
            
    except Exception as e:
        print(f"Failed to process {csv_file}: {e}")
```

## Troubleshooting

### Invalid sequence length
**Error**: `Invalid sequence length: expected 2138, got XXX`

**Solution**: Model requires exactly 2138 time points (7.4 days at 5-minute intervals). Ensure your CGM data is complete.

### Missing features
**Error**: `Missing required features: ['feature_name']`

**Solution**: If using pre-computed mode (default), CSV must contain all 9 features. If using raw mode (--raw), CSV must contain participant_id, time_index, time, and blood_glucose_value.

### Model directory not found
**Error**: `Model directory not found: path/to/model`

**Solution**: Verify the path points to the production_model folder. List available models:
```bash
ls ../models/model_*/production_model
```

## Clinical Interpretation

### High Risk
Patient shows patterns consistent with pre-diabetes. Recommend lifestyle intervention, dietary counseling, and follow-up OGTT.

### Medium Risk (Uncertain)
Pattern is near decision boundary. Recommend OGTT confirmation test within 2-4 weeks to establish definitive diagnosis.

### Low Risk  
Patient shows healthy glucose control patterns. Recommend continuing annual screening cycle.

## Support

For issues or questions:
- Review training logs in `../models/model_YYYYMMDD_HHMMSS/lstm_pipeline.log`
- Check model performance in `../models/model_YYYYMMDD_HHMMSS/analysis/`
- See `DEPLOYMENT_README.md` in production_model folder

## License

Copyright 2026 Track2Better Team
