import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score


# === Set the base path to your dataset folder ===
base_path = r"C:\Users\MAJDA\OneDrive\Desktop\Spring 2245\Project detect\EMG_data_for_gestures-master"

# Lists to store all input features (X) and labels (y)
X_all, y_all = [], []

# === Loop through each numbered subfolder ===
for folder in sorted(os.listdir(base_path)):
    folder_path = os.path.join(base_path, folder)

    # Skip non-folders
    if not os.path.isdir(folder_path):
        continue

    # === Process all .txt files in the folder ===
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            print(f"Reading file: {file_path}")

            # Read the file, skip the first line (header), no column names
            df = pd.read_csv(file_path, delimiter=r"\s+", header=None, skiprows=1)

            # Extract EMG channels (columns 1 to 8) and the gesture class (column 9)
            X = df.iloc[:, 1:9]    # EMG features
            y = df.iloc[:, 9]      # Class label

            # Relabel: 0 and 1 → 0 (calm), 2–7 → 1 (stressed)
            y = y.apply(lambda val: 0 if val in [0, 1] else 1)

            # Add to dataset
            X_all.append(X)
            y_all.append(y)

# === Combine all data into single DataFrames ===
print("Combining data from all files...")
X_all = pd.concat(X_all, ignore_index=True)
y_all = pd.concat(y_all, ignore_index=True)

# === Normalize (standardize) the EMG signal features ===
print("Normalizing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# === Split data into training and testing sets (80/20) ===
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=42
)

# === Train a Random Forest classifier ===
print("Training classifier...")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# === Evaluate model performance ===
print("\n=== Evaluation on Test Set ===")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# === Save the model and scaler for future use ===
print("Saving model and scaler...")
joblib.dump(clf, "emg_stress_model.pkl")
joblib.dump(scaler, "emg_stress_scaler.pkl")
print("✅ Model and scaler saved successfully.")

# === Calculate individual metrics ===
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPrecision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"Accuracy:  {accuracy:.3f}")

# === Plot bar chart of metrics ===
plt.figure(figsize=(6, 4))
metrics = [precision, recall, accuracy]
labels = ["Precision", "Recall", "Accuracy"]
colors = ['#4CAF50', '#2196F3', '#FF9800']

plt.bar(labels, metrics, color=colors)
plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(metrics):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
