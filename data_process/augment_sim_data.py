import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# --- CONFIG ---
RAW_SIM_PATH = "../data/sim_data/raw_sim_data.csv"
CLASSIFIER_PATH = "../models/change_type_classifier.pkl"
DISTRIBUTIONS_PATH = "../models/change_distributions.pkl"
SAVE_PATH = "../data/sim_data/augmented_sim_data.csv"

SAMPLING_RATE = 4
WINDOW_SEC = 10
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SEC

# --- Load Data and Models ---
df = pd.read_csv(RAW_SIM_PATH)
clf = joblib.load(CLASSIFIER_PATH)
distributions = joblib.load(DISTRIBUTIONS_PATH)

scaler = StandardScaler()

# --- Prepare Features for Classification ---
features = df[['mean_eda', 'std_eda', 'min_eda', 'max_eda']].copy()
features.columns = ['mean_eda', 'std_eda', 'min_eda', 'max_eda']  # rename if needed
# Add diffs
features['mean_diff'] = features['mean_eda'].diff().fillna(0)
features['std_diff'] = features['std_eda'].diff().fillna(0)

X = features[['mean_eda', 'std_eda', 'min_eda', 'max_eda', 'mean_diff', 'std_diff']]
X_scaled = scaler.fit_transform(X)

# --- Predict Change Types ---
change_preds = clf.predict(X_scaled)
unique, counts = np.unique(change_preds, return_counts=True)
print("Change type predictions:", dict(zip(unique, counts)))


# --- Apply Changes to EDA ---
eda_aug = df['mean_eda'].copy().to_numpy()
window_means = df['mean_eda'].to_numpy()
window_stds = df['std_eda'].to_numpy()

for i, change_type in enumerate(change_preds):
    if change_type == 'none':
        continue

    duration = np.random.randint(3, 8)  # how many windows the change lasts
    end_idx = min(i + duration, len(eda_aug))

    current_mean = window_means[i]
    current_std = window_stds[i]

    if change_type in ['mean', 'both']:
        delta_mean = np.random.choice(distributions['mean'])
        for j in range(i, end_idx):
            scale = (j - i) / duration
            eda_aug[j] += (current_mean + delta_mean - eda_aug[j]) * scale

    if change_type in ['std', 'both']:
        delta_std = np.random.choice(distributions['std'])
        for j in range(i, end_idx):
            val = eda_aug[j]
            adjusted = (val - current_mean) * ((current_std + delta_std) / current_std) + current_mean
            eda_aug[j] = adjusted

# --- Save ---
df['augmented_eda'] = eda_aug
df.to_csv(SAVE_PATH, index=False)
print(f"Saved augmented data to {SAVE_PATH}")
