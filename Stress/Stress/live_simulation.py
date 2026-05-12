import numpy as np
import time
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from collections import deque
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

# === Load model and scaler ===
model = load_model("models/stress_model.h5")
scaler_mean = np.load("models/scaler_mean.npy")
scaler_scale = np.load("models/scaler_scale.npy")

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale
scaler.var_ = scaler.scale_ ** 2

# === Spotify integration ===
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
    redirect_uri=os.getenv('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:8888/callback'),
    scope='user-modify-playback-state user-read-playback-state user-read-currently-playing streaming'
))

CALM_PLAYLIST_URI = "spotify:playlist:CALM_PLAYLIST_ID"
REGULAR_PLAYLIST_URI = "spotify:playlist:REGULAR_PLAYLIST_ID"
current_music_state = None
last_check_time = time.time()

def play_playlist(uri):
    try:
        devices = sp.devices()
        if not devices["devices"]:
            print("No active Spotify device found.")
            return
        device_id = devices["devices"][0]["id"]
        sp.transfer_playback(device_id, force_play=True)
        sp.start_playback(device_id=device_id, context_uri=uri)
        print(f"🎵 Playing: {uri}")
    except Exception as e:
        print(f"Spotify Error: {e}")

# === Phase-controlled EMG generator ===
phase_durations = {"not_stressed": 20, "stressed": 20, "excess_stress": 20}
phase_order = ["not_stressed", "stressed", "excess_stress"]
current_phase_index = 0
phase_counter = 0

def generate_emg_sample(phase):
    if phase == "not_stressed":
        signal = np.random.normal(loc=0.00002, scale=0.000015, size=(8,))
        label = "NOT STRESSED"
    elif phase == "stressed":
        signal = np.random.normal(loc=0.00025, scale=0.00005, size=(8,))
        label = "STRESSED"
    else:
        signal = np.random.normal(loc=0.0004, scale=0.0001, size=(8,))
        signal += np.sin(np.linspace(0, np.pi, 8)) * 0.0002
        label = "EXCESS STRESS"
    return signal, label

# === Graph setup ===
window_size = 60
emg_history = [deque(maxlen=window_size) for _ in range(8)]
stress_markers = deque(maxlen=window_size)

plt.ion()
fig, axs = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
fig.subplots_adjust(hspace=0.5, wspace=0.3)
fig.suptitle("Live EMG - 8 Channels with Stress Detection", fontsize=18)

channel_lines = []
marker_lines = []

for i in range(8):
    ax = axs[i // 2][i % 2]
    ax.set_ylim(-0.0002, 0.001)
    ax.set_ylabel(f"CH{i+1}", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_title(f"Channel {i+1}", fontsize=12)
    ax.grid(True)

    line, = ax.plot([], [], label=f"EMG{i+1}", color='blue', linewidth=1.5)
    dot, = ax.plot([], [], 'ro', markersize=5, label="Stress")
    channel_lines.append(line)
    marker_lines.append(dot)

axs[3][0].set_xlabel("Time", fontsize=12)
axs[3][1].set_xlabel("Time", fontsize=12)

print("\n🧪 Starting live EMG simulation with Spotify...\n")

try:
    while True:
        # === Stress simulation ===
        current_phase = phase_order[current_phase_index]
        emg_sample, true_label = generate_emg_sample(current_phase)
        phase_counter += 1
        if phase_counter >= phase_durations[current_phase]:
            phase_counter = 0
            current_phase_index = (current_phase_index + 1) % len(phase_order)

        # === Predict ===
        scaled = scaler.transform([emg_sample])
        pred = model.predict(scaled, verbose=0)[0][0]
        predicted_label = "STRESSED" if pred >= 0.5 else "NOT STRESSED"
        is_stress = 1 if predicted_label == "STRESSED" else 0

        print(f"🟡 Phase: {current_phase:15s} | Simulated: {true_label:15s} | Predicted: {predicted_label:15s} (Conf: {pred:.2f})")

        # === Update graph data ===
        for i in range(8):
            emg_history[i].append(emg_sample[i])
        stress_markers.append(is_stress)

        # === Update plot ===
        for i in range(8):
            y = list(emg_history[i])
            x = list(range(len(y)))
            channel_lines[i].set_data(x, y)
            stress_indices = [idx for idx, s in enumerate(stress_markers) if s == 1]
            stress_vals = [y[j] for j in stress_indices if j < len(y)]
            marker_lines[i].set_data(stress_indices, stress_vals)
            axs[i // 2][i % 2].set_xlim(0, window_size)

        plt.pause(0.01)

        # === Check and update music every 10 seconds ===
        if time.time() - last_check_time >= 10:
            last_check_time = time.time()
            stress_count = sum(stress_markers)
            stress_ratio = stress_count / len(stress_markers)

            if stress_ratio >= 0.5:
                new_state = "STRESSED"
            else:
                new_state = "NOT_STRESSED"

            if new_state != current_music_state:
                current_music_state = new_state
                if current_music_state == "STRESSED":
                    play_playlist(CALM_PLAYLIST_URI)
                else:
                    play_playlist(REGULAR_PLAYLIST_URI)

        time.sleep(0.25)

except KeyboardInterrupt:
    print("\nSimulation stopped.")
    plt.ioff()
    plt.show()
