import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_emg_data, prepare_dataset

# === Load and prepare data ===
print("Loading EMG data...")
X, y = load_emg_data()
print("Preparing train/test split and scaling...")
X_train, X_test, y_train, y_test, scaler = prepare_dataset(X, y)

# === Define MLP model ===
print("Building MLP model...")
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.4),  # Higher dropout to fight overfitting
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Train with early stopping ===
print("🚀 Starting training...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=9,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# === Evaluate the model ===
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")

# === Save model and scaler ===
print("Saving model and scaler...")
os.makedirs("models", exist_ok=True)
model.save("models/stress_model.h5")
np.save("models/scaler_mean.npy", scaler.mean_)
np.save("models/scaler_scale.npy", scaler.scale_)

print("Training complete. Model saved to models/stress_model.h5")
