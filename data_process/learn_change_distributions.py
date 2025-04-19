import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

INPUT_DIR = "../data/processed_data/chest_changepoints"
MODEL_SAVE_PATH = "../models/change_type_classifier.pkl"
DIST_SAVE_PATH = "../models/change_distributions.pkl"

all_data = []

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(INPUT_DIR, fname))

    # Ensure both columns exist
    if "Up Trend" not in df.columns:
        df["Up Trend"] = None
    if "Down Trend" not in df.columns:
        df["Down Trend"] = None

    df = df[df["Up Trend"].notna() | df["Down Trend"].notna()].copy()

    def label_change(row):
        if pd.notna(row["Up Trend"]) and pd.notna(row["Down Trend"]):
            return "both"
        elif pd.notna(row["Up Trend"]):
            return "mean"
        elif pd.notna(row["Down Trend"]):
            return "std"
        else:
            return "none"

    df["change_type"] = df.apply(label_change, axis=1)
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

if "mean_diff" not in df_all.columns:
    df_all["mean_diff"] = df_all["mean_eda"].diff().fillna(0)
if "std_diff" not in df_all.columns:
    df_all["std_diff"] = df_all["std_eda"].diff().fillna(0)

X = df_all[["mean_eda", "std_eda", "min_eda", "max_eda", "mean_diff", "std_diff"]]

y = df_all["change_type"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(clf, MODEL_SAVE_PATH)

# Optionally save distributions of deltas
distributions = {
    "mean": df_all[df_all["change_type"] == "mean"]["mean_diff"].dropna().values,
    "std": df_all[df_all["change_type"] == "std"]["std_diff"].dropna().values,
    "both": df_all[df_all["change_type"] == "both"]["mean_diff"].dropna().values
}
joblib.dump(distributions, DIST_SAVE_PATH)
