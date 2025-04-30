import os
import pandas as pd
import numpy as np
from functools import reduce

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

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

def train_and_eval_metrics(X, y, model, modelname):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')

    # Create a MultiIndex for the columns
    columns = pd.MultiIndex.from_tuples([(modelname, 'Accuracy'), (modelname, 'F1')])

    # Create the DataFrame
    df = pd.DataFrame([[acc, f1]], columns=columns)
    return df

if __name__ == "__main__":
    # Define models
    models = {
        "LR": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    }

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
    list_df_real = []
    for modelname in models.keys():
        curr_model = models[modelname]
        curr_df_metrics = train_and_eval_metrics(X_real_combined, y_real_combined, curr_model, modelname)
        curr_df_metrics.index = ['Real Data']
        list_df_real.append(curr_df_metrics)
    df_real = reduce(lambda  left,right: pd.merge(left,right,left_index = True, right_index = True), list_df_real)

    print("---- Simulated (Raw) ----")
    X_sim, y_sim = load_dataset(sim_path)
    list_df_raw_stimulated = []
    for modelname in models.keys():
        curr_model = models[modelname]
        curr_df_metrics = train_and_eval_metrics(X_sim, y_sim, curr_model, modelname)
        curr_df_metrics.index = ['Raw Simulated']
        list_df_raw_stimulated.append(curr_df_metrics)
    df_raw_stimulated = reduce(lambda  left,right: pd.merge(left,right,left_index = True, right_index = True), list_df_raw_stimulated)

    print("---- Simulated (Augmented) ----")
    X_aug, y_aug = load_dataset(aug_path)
    list_df_augmented_stimulated = []
    for modelname in models.keys():
        curr_model = models[modelname]
        curr_df_metrics = train_and_eval_metrics(X_aug, y_aug, curr_model, modelname)
        curr_df_metrics.index = ['Our Method']
        list_df_augmented_stimulated.append(curr_df_metrics)
    df_augmented_stimulated  = reduce(lambda  left,right: pd.merge(left,right,left_index = True, right_index = True), list_df_augmented_stimulated)

    df_combined = pd.concat([df_raw_stimulated, df_augmented_stimulated, df_real])
    print(df_combined)
