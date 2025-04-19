import os
import pandas as pd
from trendet import identify_df_trends

INPUT_DIR = "../data/processed_data/chest"
OUTPUT_DIR = "../data/processed_data/chest_changepoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

subjects = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

for filename in subjects:
    print(f"Processing {filename}")
    df = pd.read_csv(os.path.join(INPUT_DIR, filename))
    df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='10s')  # assuming 10-second windows

    try:
        trend_df = identify_df_trends(
            df, column="mean_eda", window_size=5, identify="both"
        )
        output_file = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_chest_trends.csv"))
        trend_df.to_csv(output_file, index=False)
        print(f"Saved trend-labeled file: {output_file}")
    except Exception as e:
        print(f"Failed on {filename}: {e}")
