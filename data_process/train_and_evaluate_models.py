import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_dataset(path, label_col='label'):
    df = pd.read_csv(path)
    features = df[['mean_eda', 'std_eda', 'min_eda', 'max_eda']]
    labels = df[label_col]
    return features, labels

def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return classification_report(y_test, preds, digits=2)

# --- Paths ---
real_dir = "../data/processed_data/chest/"
sim_path = "../data/sim_data/raw_sim_data.csv"
aug_path = "../data/sim_data/augmented_sim_data.csv"

# --- Aggregate all real WESAD subjects ---
X_real_all, y_real_all = [], []

for file in os.listdir(real_dir):
    if file.endswith(".csv"):
        X, y = load_dataset(os.path.join(real_dir, file))
        X_real_all.append(X)
        y_real_all.append(y)

X_real_combined = pd.concat(X_real_all, ignore_index=True)
y_real_combined = pd.concat(y_real_all, ignore_index=True)

print("---- Real WESAD (All Subjects) ----")
print(train_and_eval(X_real_combined, y_real_combined))

print("---- Simulated (Raw) ----")
X_sim, y_sim = load_dataset(sim_path)
print(train_and_eval(X_sim, y_sim))

print("---- Simulated (Augmented) ----")
X_aug, y_aug = load_dataset(aug_path)
print(train_and_eval(X_aug, y_aug))
