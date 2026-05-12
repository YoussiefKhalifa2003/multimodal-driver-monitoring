# retrain_model_enhanced.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from collections import Counter
import joblib

# Load EEG dataset
df = pd.read_csv("synthetic_eeg_dataset.csv")
X_raw = df.drop(columns=["label"]).values
y_raw = df["label"].values

# Convert labels to 'drowsy' and 'not_drowsy'
y_bin = np.array(['drowsy' if label == 'drowsy' else 'not_drowsy' for label in y_raw])

window_size = 50
features_list, labels_list = [], []

for i in range(window_size, len(X_raw)):
    window = X_raw[i - window_size:i]
    label_window = y_bin[i - window_size:i]
    means = np.mean(window, axis=0)
    stds = np.std(window, axis=0)
    diffs = np.mean(np.diff(window, axis=0), axis=0)
    features = np.concatenate([means, stds, diffs])
    label = Counter(label_window).most_common(1)[0][0]
    features_list.append(features)
    labels_list.append(label)

X_feat = np.array(features_list)
y_feat = np.array(labels_list)

# Balance classes
df_feat = pd.DataFrame(X_feat)
df_feat['label'] = y_feat
df_bal = df_feat.groupby("label").apply(lambda x: x.sample(n=min(len(x), 900), random_state=42)).reset_index(drop=True)

X_bal = df_bal.drop(columns=["label"]).values
y_bal = df_bal["label"].values

# Scale features and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "eeg_model_drowsy_binary.pkl")
joblib.dump(scaler, "eeg_scaler_drowsy_binary.pkl")
print("Model & scaler saved as eeg_model_drowsy_binary.pkl + eeg_scaler_drowsy_binary.pkl")
