import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

INPUT_DIR = "../data/processed_data/chest_changepoints"
MODEL_PATH = "../models/changepoint_classifier.pkl"

data = []

# --- Step 1: Load and label changepoints ---
for file in os.listdir(INPUT_DIR):
    if file.endswith("_trends.csv"):
        df = pd.read_csv(os.path.join(INPUT_DIR, file))

        # Ensure both columns exist
        if "Up Trend" not in df.columns:
            df["Up Trend"] = None
        if "Down Trend" not in df.columns:
            df["Down Trend"] = None

        df['trend_regime'] = df[['Up Trend', 'Down Trend']].apply(lambda x: str(x['Up Trend']) + "_" + str(x['Down Trend']) , axis = 1)
        df['prev_trend_regime'] = df['trend_regime'].shift(1)
        df = df.dropna()
        
        # Create binary changepoint label
        df['changepoint'] = df.apply(lambda x: 1 if x.trend_regime != x.prev_trend_regime else 0, axis = 1)
        df['changepoint'] = df['changepoint'].astype(int)

        # Optional: Add diff features
        df["mean_diff"] = df["mean_eda"].diff().fillna(0)
        df["std_diff"] = df["std_eda"].diff().fillna(0)

        data.append(df)

df_all = pd.concat(data, ignore_index=True)

# --- Step 2: Train classifier ---
features = ["mean_eda", "std_eda", "min_eda", "max_eda", "mean_diff", "std_diff"]
X = df_all[features]
y = df_all["changepoint"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# --- Step 3: Evaluate ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Step 4: Save model ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
