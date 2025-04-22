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


def simulate_eda_individual(length_sec=2500, sampling_rate=4):
    total_samples = length_sec * sampling_rate
    signal = []
    labels = []

    current_state = 0  # Start with baseline
    remaining = total_samples

    while remaining > 0:
        segment_duration = np.random.randint(20, 60) * sampling_rate
        segment_duration = min(segment_duration, remaining)

        eda_segment = nk.eda_simulate(duration=int(segment_duration / sampling_rate),
                                      sampling_rate=sampling_rate,
                                      noise=0.02)

        # Apply a random level to introduce nonstationarity
        eda_segment *= np.random.uniform(0.8, 1.2)

        # Flip stress state
        current_state = 1 - current_state if np.random.rand() > 0.3 else current_state

        # Occasionally introduce noisy/mixed labels (simulate real physiology)
        if np.random.rand() < 0.2:
            label_segment = np.random.binomial(1, 0.5, size=segment_duration)
        else:
            label_segment = np.full(segment_duration, current_state)

        signal.extend(eda_segment)
        labels.extend(label_segment)

        remaining -= segment_duration

    signal = np.array(signal[:total_samples])
    labels = np.array(labels[:total_samples])

    return preprocess_eda_features(signal, labels, sampling_rate=sampling_rate)


if __name__ == "__main__":
    simulated_dataset = []

    for subject_id in range(len(list_s)):
        print(f"Simulating subject {subject_id}")
        df = simulate_eda_individual()
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
