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
            "max_eda": np.max(window),
            "label": int(np.round(np.mean(label_window)))  # majority label in window
        }

        features.append(feature)

    return pd.DataFrame(features)


def simulate_eda_individual(length_sec=2400, sampling_rate=4):
    """
    Simulate EDA signal using NeuroKit2 for a single individual.
    - length_sec: Total duration in seconds
    - sampling_rate: Samples per second
    """
    signal = nk.eda_simulate(duration=length_sec, sampling_rate=sampling_rate, noise=0.05)

    # Create a random alternating binary label pattern (baseline=0, stress=1)
    total_samples = length_sec * sampling_rate
    segment_length = sampling_rate * 30  # 30-second blocks
    labels = np.zeros(total_samples)

    for i in range(0, total_samples, 2 * segment_length):
        stress_start = i + segment_length
        stress_end = min(i + 2 * segment_length, total_samples)
        labels[stress_start:stress_end] = 1

    return preprocess_eda_features(signal, labels, sampling_rate=sampling_rate)


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


