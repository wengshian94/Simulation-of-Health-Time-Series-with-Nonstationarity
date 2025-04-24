import neurokit2 as nk
import numpy as np
import os
import pandas as pd
from process_data import list_s

def preprocess_eda_features(signal, labels, sampling_rate=4, window_sec=10):
    window_size = sampling_rate * window_sec
    features = []

    for start in range(0, len(signal) - window_size + 1, window_size):
        end = start + window_size
        window = signal[start:end]
        label_window = labels[start:end]

        feature = {
            "mean_eda": np.mean(window),
            "std_eda": np.std(window),
            "min_eda": np.min(window),
            "max_eda": np.max(window),
            "label": int(np.round(np.mean(label_window)))  # dominant label in the window
        }

        features.append(feature)

    return pd.DataFrame(features)


def simulate_eda_individual(sampling_rate=4):
    # Durations from the paper
    baseline_duration = 1174  # in seconds
    stress_duration = 664     # in seconds

    # SCR peak counts drawn from uniform distributions
    scr_count_baseline = np.random.randint(1, 6)  # U(1,5)
    scr_count_stress = np.random.randint(6, 21)   # U(6,20)

    # Simulate EDA signals
    eda_baseline = nk.eda_simulate(duration=baseline_duration, sampling_rate=sampling_rate, scr_number=scr_count_baseline)

    eda_stress = nk.eda_simulate(duration=stress_duration, sampling_rate=sampling_rate, scr_number=scr_count_stress)

    # Concatenate signal and labels
    signal = np.concatenate([eda_baseline, eda_stress])
    labels = np.array([0] * len(eda_baseline) + [1] * len(eda_stress))

    return signal, labels


if __name__ == "__main__":
    simulated_dataset = []

    for subject_id in range(len(list_s)):
        print(f"Simulating subject {subject_id}")
        signal, labels = simulate_eda_individual()
        df = preprocess_eda_features(signal, labels, sampling_rate=4)
        df["subject"] = subject_id
        simulated_dataset.append(df)

    sim_data = pd.concat(simulated_dataset, ignore_index=True)

    sim_dir = "../data/sim_data/"
    os.makedirs(sim_dir, exist_ok=True)
    sim_filename = os.path.join(sim_dir, 'raw_sim_data.csv')
    sim_data.to_csv(sim_filename, index=False)

    print(f"Data saved in: {sim_filename}")
    print("--- Raw Simulated Data ---")
    print(sim_data["label"].value_counts())
    print(f"Total rows: {len(sim_data)}")
    print(f"Duplicate rows: {sim_data.duplicated().sum()}")
    print(sim_data.head())
