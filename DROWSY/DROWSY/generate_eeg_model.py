import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic EEG dataset with distinct patterns for each class
n_samples = 1000
n_features = 10

# Create base data with normal distributions
not_drowsy_data = np.random.randn(n_samples//2, n_features) * 0.5  # Normal steady signal

# For drowsy data, create a base signal that occasionally has peaks/spikes
drowsy_data = np.random.randn(n_samples//2, n_features) * 0.5  # Base signal similar to not_drowsy

# Add occasional peaks to drowsy data
for i in range(0, n_samples//2, 10):  # Add peaks every ~10 samples
    # Create a peak in a random feature
    feature_idx = np.random.randint(0, n_features)
    # Add a peak over several consecutive samples to create a realistic pattern
    peak_length = np.random.randint(3, 7)
    peak_amplitude = np.random.uniform(1.5, 2.5)
    
    for j in range(min(peak_length, n_samples//2 - i)):
        if i + j < n_samples//2:
            drowsy_data[i + j, feature_idx] = peak_amplitude
            # Add some correlation to adjacent features
            if feature_idx > 0:
                drowsy_data[i + j, feature_idx-1] = peak_amplitude * 0.7
            if feature_idx < n_features - 1:
                drowsy_data[i + j, feature_idx+1] = peak_amplitude * 0.7

# Combine the data
X = np.vstack([drowsy_data, not_drowsy_data])
y = np.array(['drowsy'] * (n_samples//2) + ['not_drowsy'] * (n_samples//2))

# Shuffle the data
shuffle_idx = np.random.permutation(n_samples)
X = X[shuffle_idx]
y = y[shuffle_idx]

# Create DataFrame
feature_cols = [f'eeg_feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_cols)
df['label'] = y

# Save dataset to CSV
df.to_csv("synthetic_eeg_dataset.csv", index=False)
print("Synthetic EEG dataset created: synthetic_eeg_dataset.csv")

# Function to extract features the same way the visualizer does
def extract_features(window):
    means = np.mean(window, axis=0)
    stds = np.std(window, axis=0)
    maxs = np.max(window, axis=0)
    mins = np.min(window, axis=0)
    diffs = np.mean(np.diff(window, axis=0), axis=0)
    features = np.concatenate([means, stds, maxs, mins, diffs])
    return features

# Train a model with the same feature extraction logic
window_size = 50
X_processed = []

# Generate sliding windows and extract features
for i in range(window_size, n_samples):
    window_data = X[i-window_size:i, :]
    features = extract_features(window_data)
    X_processed.append(features)

X_processed = np.array(X_processed)
y_processed = y[window_size:]

# Scale the processed features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# Train a RandomForest model with less certainty
# Low n_estimators and max_features='sqrt' will make the model less confident
model = RandomForestClassifier(
    n_estimators=30,  # Fewer trees = more variability
    max_depth=3,      # Shallow trees = less certainty 
    max_features='sqrt',  # Subset of features = more randomness
    random_state=42,
    min_samples_leaf=5  # Require more samples per leaf = less overfit
)
model.fit(X_scaled, y_processed)

# Save model and scaler
joblib.dump(model, "eeg_model_drowsy_binary.pkl")
joblib.dump(scaler, "eeg_scaler_drowsy_binary.pkl")

print("Model saved: eeg_model_drowsy_binary.pkl")
print("Scaler saved: eeg_scaler_drowsy_binary.pkl")
print("\nNow you can run the visualizer script!") 