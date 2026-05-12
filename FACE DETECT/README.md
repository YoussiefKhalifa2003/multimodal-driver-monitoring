# Module 1 — Facial Emotion Recognition

Part of the [Multimodal Driver Monitoring System](../../README.md) · IEEE RAAI 2025

---

## What this module does

Uses a live webcam feed to classify the driver's facial expression in real time. When a stable emotion is confirmed (≥ 5 consecutive consistent detections over a 10-second window), the system triggers the corresponding Spotify playlist via the Web API.

**Emotion → Music routing:**

| Detected Emotion | Playlist Genre |
|-----------------|----------------|
| `happy` | EDM |
| `sad` | Soul |
| `angry` | Rock |

---

## Model Architecture

A custom CNN trained from scratch on FER2013 grayscale face crops:

```
Input: (64, 64, 1)  ← grayscale face crop

Conv Block 1:  Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
Conv Block 2:  Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
Conv Block 3:  Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
Dense:         Flatten → Dense(256) → BN → Dropout(0.5) → Dense(3, softmax)
```

- **Optimizer:** Adam (lr=0.001, ReduceLROnPlateau)
- **Loss:** Categorical cross-entropy with class-weight balancing
- **Callbacks:** EarlyStopping (patience=10), ModelCheckpoint, TensorBoard
- **Augmentation:** rotation ±20°, width/height shift ±20%, zoom ±20%, horizontal flip, brightness [0.8–1.2]

---

## Scripts

| Script | Purpose |
|--------|---------|
| `download_fer2013.py` | Download FER2013 dataset from Kaggle |
| `train_facial_emotion_model.py` | Train CNN and save `facial_emotion_model.h5` |
| `evaluate_model.py` | Run evaluation and print metrics |
| `fusion_driver_demo.py` | **Main demo** — live webcam + Spotify integration |
| `simulated_state_generator.py` | Markov-chain simulator for testing without hardware |
| `script.py` | Dataset directory path utility |

---

## Setup

```bash
cd "FACE DETECT/FACE DETECT"

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# Configure Spotify (copy .env.example from repo root to here, or set env vars)
cp ../../.env.example .env
# Edit .env with your Spotify credentials

# 1. Download dataset
python download_fer2013.py

# 2. Train the model (saves facial_emotion_model.h5)
python train_facial_emotion_model.py

# 3. Run the live demo
python fusion_driver_demo.py
```

---

## Live Demo Controls

| Key | Action |
|-----|--------|
| `m` | Toggle Auto / Manual mode |
| `SPACE` | Capture emotion (Manual mode only) |
| `ESC` | Exit |

In **Auto mode** the system checks emotion every 10 seconds and changes music once a stable state (≥5 matching frames) is confirmed. An emotion intensity graph is rendered live in the top-right of the webcam window.

---

## Dependencies

See `requirements.txt`. Key packages: `tensorflow`, `opencv-python`, `mediapipe`, `spotipy`, `python-dotenv`.
