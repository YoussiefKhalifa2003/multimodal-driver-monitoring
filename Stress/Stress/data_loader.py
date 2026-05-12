import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_emg_data():
    root_folder = "EMG_data_for_gestures-master"
    all_data = []
    all_labels = []
    file_count = 0
    sample_count = 0

    print(f"🔍 Scanning folder: {os.path.abspath(root_folder)}")

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(subdir, file)
                try:
                    df = pd.read_csv(path, sep='\\s+', skiprows=1)
                    if df.shape[1] != 10:
                        print(f"Skipping malformed file: {path}")
                        continue
                    X = df.iloc[:, 1:9].values  # Columns 2-9: EMG
                    y_raw = df.iloc[:, 9].values  # Column 10: Class
                    y = np.array([0 if label in [0, 1] else 1 for label in y_raw])

                    all_data.append(X)
                    all_labels.append(y)
                    file_count += 1
                    sample_count += len(y)
                except Exception as e:
                    print(f"Error reading {path}: {e}")

    if not all_data:
        raise ValueError("No valid data files found.")

    X_all = np.vstack(all_data)
    y_all = np.concatenate(all_labels)

    print(f"\n Loaded {file_count} files with {sample_count} total samples.")
    print(f"Stressed: {np.sum(y_all == 1)}, Not Stressed: {np.sum(y_all == 0)}\n")

    return X_all, y_all

def prepare_dataset(X, y, test_size=0.2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler

# ======== MAIN RUN =========
if __name__ == "__main__":
    X, y = load_emg_data()

    print("Feature matrix shape:", X.shape)
    print("Label array shape:", y.shape)
    print("Unique labels:", set(y))
