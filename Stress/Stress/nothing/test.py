import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load model and scaler
clf = joblib.load("emg_stress_model.pkl")
scaler = joblib.load("emg_stress_scaler.pkl")
num_channels = scaler.mean_.shape[0]

# Track Channel 1 over time
emg_history = []

# GUI Setup
root = tk.Tk()
root.title("Live EMG Stress Detector")
root.geometry("900x550")
root.configure(bg="#f4f4f9")

# Style
style = ttk.Style()
style.configure("TLabel", font=("Segoe UI", 12), background="#f4f4f9")
style.configure("TButton", font=("Segoe UI", 11), padding=5)

# Status Display
status_label = ttk.Label(root, text="Status: N/A", font=("Segoe UI", 16))
status_label.pack(pady=10)

# EMG Value Display
emg_label = ttk.Label(root, text="EMG: [ ]", font=("Courier New", 12))
emg_label.pack()

# Matplotlib Graph for Channel 1
fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
line, = ax.plot([], [], lw=2)
ax.set_ylim(-0.002, 0.002)
ax.set_xlim(0, 50)
ax.set_title("Live EMG (Channel 1)")
ax.set_ylabel("Amplitude")
ax.set_xlabel("Steps")
ax.grid(True)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

# Buttons
button_frame = ttk.Frame(root)
button_frame.pack(pady=20)

running = False

def start_loop():
    global running
    if not running:
        running = True
        update_loop()

def stop_loop():
    global running
    running = False
    root.destroy()

start_btn = ttk.Button(button_frame, text="Start Monitoring", command=start_loop)
start_btn.grid(row=0, column=0, padx=10)

quit_btn = ttk.Button(button_frame, text="Quit", command=stop_loop)
quit_btn.grid(row=0, column=1, padx=10)

# Loop logic
def update_loop():
    if not running:
        return

    # === Generate Synthetic EMG Sample ===
    emg_sample = np.random.normal(0, 0.0001, (1, num_channels))

    # Inject spike to simulate stress
    if np.random.rand() < 0.2:
        emg_sample += 0.001  # simulate movement/stress

    # Predict
    scaled_sample = scaler.transform(emg_sample)
    prediction = clf.predict(scaled_sample)[0]

    # Display results
    emg_label.config(text=f"EMG: {np.round(emg_sample[0], 6)}")
    if prediction == 1:
        status_label.config(text="🔴 Stress Detected", foreground="red")
    else:
        status_label.config(text="🟢 Calm", foreground="green")

    # Update graph
    emg_history.append(emg_sample[0][0])
    if len(emg_history) > 50:
        emg_history.pop(0)

    line.set_data(range(len(emg_history)), emg_history)
    ax.set_xlim(0, max(50, len(emg_history)))
    canvas.draw()

    root.after(500, update_loop)

# Run GUI
root.mainloop()
