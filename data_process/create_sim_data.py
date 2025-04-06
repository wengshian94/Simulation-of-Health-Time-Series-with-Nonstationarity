import neurokit2 as nk
import numpy as np
import os
import pandas as pd
from process_data import list_s

def simulate_eda_individual(baseline_duration=1174, stress_duration=664, sampling_rate=4):
    # SCR peak counts: draw from uniform range
    scr_count_baseline = np.random.randint(1, 6)   # U(1,5)
    scr_count_stress = np.random.randint(6, 21)    # U(6,20)

    # Simulate baseline EDA
    eda_baseline = nk.eda_simulate(duration=baseline_duration,
                                   scr_number=scr_count_baseline,
                                   sampling_rate=sampling_rate)

    # Simulate stress EDA
    eda_stress = nk.eda_simulate(duration=stress_duration,
                                 scr_number=scr_count_stress,
                                 sampling_rate=sampling_rate)

    # Combine both segments
    eda_signal = np.concatenate([eda_baseline, eda_stress])
    labels = np.array([0] * len(eda_baseline) + [1] * len(eda_stress))  # 0=baseline, 1=stress

    df = pd.DataFrame({
        'EDA': eda_signal,
        'label': labels
    })

    return df

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
    sim_filename = os.path.join(sim_dir, 'sim_data.csv')
    sim_data.to_csv(sim_filename)
    print(f"Data saved in: {sim_filename}")
    

