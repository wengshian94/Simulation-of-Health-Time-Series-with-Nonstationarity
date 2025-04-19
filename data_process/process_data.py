import pickle
import pandas as pd
import numpy as np
import os
from scipy.stats import mode

"""
Script to process data like in paper and save it into processed_data
"""

list_s = ['S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S13','S14','S15','S16','S17']

# Load subject data
def load_wesad_subject(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

# Preprocess EDA: window, extract features, assign label
def preprocess_eda_features(data, window_size_sec=5, sampling_rate=4, signal_type='chest', include_3_cats=True):
    """
    Segments EDA signal into non-overlapping windows and extracts statistical features.
    Uses majority label within each window to assign a binary label (0=baseline, 1=stress).
    Args:
        data (dict): One subject's data loaded from WESAD .pkl
        window_size_sec (int): Length of each window in seconds (default: 10)
        sampling_rate (int): Sampling rate of EDA signal (default: 4 for wrist)
        signal_type (str): 'wrist' or 'chest'
        include_3_cats (bool): True to have 3 stress categories 1) Baseline 2) Stress 3) Amusement. False to only have 2 categories
    Returns:
        pd.DataFrame: DataFrame with features and binary labels
    """
    eda = data['signal'][signal_type]['EDA']
    labels = data['label']  # sampled at 700 Hz
    label_sampling_rate = 700  # As per WESAD dataset specifications
    
    samples_per_window = window_size_sec * sampling_rate
    total_windows = len(eda) // samples_per_window
    features = []
    
    for i in range(total_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        eda_window = eda[start:end]
        
        # Convert from EDA sample index to label index (accounting for different sampling rates)
        label_idx_start = int((start / sampling_rate) * label_sampling_rate)
        label_idx_end = int((end / sampling_rate) * label_sampling_rate)
        
        # Make sure indices are within bounds of labels array
        label_idx_start = min(label_idx_start, len(labels)-1)
        label_idx_end = min(label_idx_end, len(labels))
        
        label_window = labels[label_idx_start:label_idx_end]
        

        # Keep only baseline and stress (1 and 2)
        if include_3_cats:
            accepted_categories = [1, 2, 3]
        else:
            accepted_categories = [1,2]

        label_window = [l for l in label_window if l in accepted_categories]

        if len(label_window) == 0:
            continue  # skip window with undefined or unwanted labels
        
        majority_label = int(mode(label_window, keepdims=False).mode)
        #binary_label = 0 if majority_label == 1 else 1  # 0 = baseline, 1 = stress
        binary_label = majority_label
        
        features.append({
            'mean_eda': np.mean(eda_window),
            'std_eda': np.std(eda_window),
            'min_eda': np.min(eda_window),
            'max_eda': np.max(eda_window),
            'label': binary_label
        })
    
    return pd.DataFrame(features)

if __name__ == "__main__":
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
    