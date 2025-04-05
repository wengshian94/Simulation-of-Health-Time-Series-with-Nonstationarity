import pickle
import pandas as pd
import numpy as np
import os
from scipy.stats import mode

"""
Script to process data like in paper and save it into processed_data
"""
# Load subject data
def load_wesad_subject(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

# Preprocess EDA: window, extract features, assign label
def preprocess_eda_features(data, window_size_sec=10, sampling_rate=4, signal_type='wrist'):
    eda = data['signal'][signal_type]['EDA']  # 1D array
    labels = data['label']                # 1D array per second

    # Compute number of samples per window
    samples_per_window = window_size_sec * sampling_rate
    total_windows = len(eda) // samples_per_window

    features = []
    for i in range(total_windows):
        start = i * samples_per_window
        end = start + samples_per_window

        eda_window = eda[start:end]

        # Label for this window (using mode of per-second labels)
        label_idx_start = start // sampling_rate
        label_idx_end = end // sampling_rate
        window_labels = labels[label_idx_start:label_idx_end]
        label = mode(window_labels, keepdims=False).mode if len(window_labels) > 0 else 0

        features.append({
            'mean_eda': np.mean(eda_window),
            'std_eda': np.std(eda_window),
            'min_eda': np.min(eda_window),
            'max_eda': np.max(eda_window),
            'label': label
        })

    return pd.DataFrame(features)

if __name__ == "__main__":
    list_s = ['S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S13','S14','S15','S16','S17']
    processed_dir = "../data/processed_data/"
    chest_dir = os.path.join(processed_dir, 'chest')
    wrist_dir = os.path.join(processed_dir, 'wrist')

    #Ensure that the dirs are created
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
    if not os.path.exists(chest_dir):
        os.mkdir(chest_dir)
    if not os.path.exists(wrist_dir):
        os.mkdir(wrist_dir)

    for subject in list_s:
        print(f"Processing {subject}")
        file_path = f'../data/raw_data/WESAD/{subject}/{subject}.pkl'
        data = load_wesad_subject(file_path)
        wrist_eda_features_df = preprocess_eda_features(data, signal_type = 'wrist')
        chest_eda_features_df = preprocess_eda_features(data, signal_type = 'chest')

        wrist_filename = os.path.join(wrist_dir, f"{subject}.csv")
        chest_filename = os.path.join(chest_dir, f"{subject}.csv")
        
        wrist_eda_features_df.to_csv(wrist_filename)
        chest_eda_features_df.to_csv(chest_filename)
        print(f"File saved in : {chest_filename}")
        print(f"File saved in : {wrist_filename}\n")
    