# Module 3 — EMG Stress Analysis

Part of the [Multimodal Driver Monitoring System](../../README.md) · IEEE RAAI 2025

---

## What this module does

Ingests 8-channel EMG signals from a MYO Thalmic bracelet and classifies the driver's physiological stress level using a binary MLP. A live simulation cycles through `not_stressed → stressed → excess_stress` phases, visualizes all 8 EMG channels in real time, and adjusts Spotify playback based on a rolling 60-sample stress ratio.

---

## Model Architecture

```
Input: 8 EMG channels (StandardScaler normalized)

Dense(64, relu) → Dropout(0.4)
Dense(32, relu) → Dropout(0.4)
Dense(1, sigmoid)

Optimizer: Adam
Loss:      Binary cross-entropy
Callback:  EarlyStopping (patience=5, restore_best_weights=True)
```

- Binary output: `≥ 0.5` → **STRESSED**, `< 0.5` → **NOT STRESSED**
- Saved to `models/stress_model.h5` + scaler arrays in `models/`

---

## Dataset

**EMG Gestures Dataset** — MYO Thalmic bracelet recordings

| Property | Detail |
|----------|--------|
| Subjects | 36 |
| Files | 72 (2 series per subject) |
| Channels | 8 EMG channels |
| Gesture classes | 0–7 (see `EMG_data_for_gestures-master/README.txt`) |
| Label mapping | Classes 0,1 (rest/unmarked) → `not_stressed (0)` · Classes 2–7 (active gestures) → `stressed (1)` |

> Original dataset reference: Lobov et al., *Sensors* 2018, doi: 10.3390/s18041122

---

## Scripts

| Script | Purpose |
|--------|---------|
| `data_loader.py` | Walk `EMG_data_for_gestures-master/`, parse `.txt` files, apply label mapping |
| `train_model.py` | Train MLP, evaluate on test split, save model + scaler |
| `live_simulation.py` | **Main demo** — phase-controlled EMG generator + 8-channel plot + Spotify |
| `graph.py` | Signal visualization utilities |
| `Accuracy graph.py` | Plot training accuracy / loss curves |
| `nothing/` | Experimental pipeline (`train.py`, `test.py`, `visual.py`) with `labeled_emg.csv` |

---

## Setup

```bash
cd "Stress/Stress"

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# Configure Spotify
cp ../../.env.example .env
# Edit .env with your Spotify credentials

# 1. Train the model (dataset is included in EMG_data_for_gestures-master/)
python train_model.py           # saves models/stress_model.h5

# 2. Run live stress simulation
python live_simulation.py
```

---

## Live Simulation

The simulation cycles through three physiological phases (20 samples each):

| Phase | EMG signal characteristics | True label |
|-------|---------------------------|------------|
| `not_stressed` | Low amplitude, μ=0.00002, σ=0.000015 | NOT STRESSED |
| `stressed` | Medium amplitude, μ=0.00025, σ=0.00005 | STRESSED |
| `excess_stress` | High amplitude + sinusoidal burst, μ=0.0004 | EXCESS STRESS |

Every 10 seconds the system computes the stress ratio over the last 60 samples:
- ≥50% stressed → play **calm playlist** (Spotify)
- <50% stressed → play **regular playlist** (Spotify)

The live plot shows all 8 EMG channels with red dot markers on stress-classified samples.

Press `Ctrl+C` to stop; the plot is shown in static mode.

---

## Dependencies

See `requirements.txt`. Key packages: `tensorflow`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `spotipy`, `python-dotenv`.
