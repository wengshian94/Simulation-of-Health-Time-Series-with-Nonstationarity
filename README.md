# Simulation-of-Health-Time-Series-with-Nonstationarity

This project simulates health time series data with realistic nonstationarity patterns learned from real physiological data. It focuses on electrodermal activity (EDA) for stress classification tasks using the WESAD dataset.

---

## Project Structure

```text
data_process/
├── augment_sim_data.py         # Apply learned changes to simulated EDA
├── create_sim_data.py          # Generate baseline EDA using NeuroKit2
├── detect_changepoints.py      # Identify changepoints in real EDA using Trendet
├── learn_change_distributions.py  # Learn change types and magnitudes from real data
├── process_data.py             # Extract features from WESAD raw .pkl files
├── train_and_evaluate_models.py   # Evaluate ML models on real, simulated, and augmented data

data/
├── raw_data/                   # Place raw WESAD data (.pkl files here)
├── processed_data/             # Extracted statistical features from WESAD
└── sim_data/                   # Raw and augmented simulated data

models/
└── *.pkl                       # Saved models and change distributions
```

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries:
- `neurokit2`
- `trendet`
- `joblib`
- `scikit-learn`
- `pandas`, `numpy`

---

## Dataset Setup

### Step 1: Download WESAD Dataset

1. Visit https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html
2. Download and unzip the data
3. Place the WESAD folder inside:

```
data/raw_data/WESAD/
```

Expected format:

```
data/raw_data/WESAD/S2/S2.pkl
data/raw_data/WESAD/S3/S3.pkl
...
```

---

## How to Run the Pipeline

### 1. Preprocess real WESAD data

```bash
python data_process/process_data.py
```

This extracts EDA features from raw `.pkl` files and saves them as CSV.

---

### 2. Detect changepoints using Trendet

```bash
python data_process/detect_changepoints.py
```

---

### 3. Learn change distributions (type, mean/std deltas)

```bash
python data_process/learn_change_distributions.py
```

Saves:
- `models/change_type_classifier.pkl`
- `models/change_distributions.pkl`

---

### 4. Generate baseline simulated EDA

```bash
python data_process/create_sim_data.py
```

Creates:
- `data/sim_data/raw_sim_data.csv`

---

### 5. Augment simulated EDA using learned changes

```bash
python data_process/augment_sim_data.py
```

Creates:
- `data/sim_data/augmented_sim_data.csv`

---

### 6. Train and evaluate classifiers

```bash
python data_process/train_and_evaluate_models.py
```

Outputs classification metrics comparing:
- Real WESAD data
- Raw simulated data
- Augmented simulated data

---

## Summary

This project:
- Learns temporal changes in real EDA data (mean, variance)
- Injects those learned changes into synthetic data
- Evaluates how augmentation improves simulation realism and ML performance

---
