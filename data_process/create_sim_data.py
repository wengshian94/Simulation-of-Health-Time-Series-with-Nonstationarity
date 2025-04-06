import neurokit2 as nk
import numpy as np
import os
import pandas as pd
from process_data import list_s

def preprocess_eda_features(signal, labels, sampling_rate=4, window_sec=10):
    window_size = sampling_rate * window_sec
    features, window_labels = [], []

    for start in range(0, len(signal) - window_size + 1, window_size):
        end = start + window_size
        window = signal[start:end]
        label_window = labels[start:end]

        feature = {
            "mean_eda": np.mean(window),
            "std_eda": np.std(window),
            "min_eda": np.min(window),
            "maxmax_eda": np.max(window),
            "label": int(np.round(np.mean(label_window)))  # majority label in window
        }

        features.append(feature)

    return pd.DataFrame(features)



if __name__ == "__main__":
    # Simulate for len(list_s) individuals
    simulated_dataset = []
    for subject_id in range(len(list_s)):
        print(f'Sim: {subject_id}')
        df = simulate_eda_individual()
        df['subject'] = subject_id
        simulated_dataset.append(df)

    # Combine into one DataFrame
    sim_data = pd.concat(simulated_dataset, ignore_index=True)

    #Save file
    sim_dir = "../data/sim_data/"

    #Ensure that the dirs are created
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)
    sim_filename = os.path.join(sim_dir, 'raw_sim_data.csv')
    sim_data.to_csv(sim_filename)
    print(f"Data saved in: {sim_filename}")
    

