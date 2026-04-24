# RT-MBAS — Real-Time Multimodal Behavioral Analytics System

A fully local, CPU-capable Python system that uses your webcam and MediaPipe
to detect face/hand landmarks in real time, extract behavioural features
(eye blink rate, head tilt, hand motion), classify your cognitive state
(Focused / Distracted / Stressed) with a scikit-learn model, and visualise
everything through a Streamlit dashboard. No API keys, no cloud, no GPU
required.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note — Python & MediaPipe version:**
> MediaPipe 0.10.30+ is required (older versions are not published to PyPI for
> Windows). The system uses the MediaPipe **Tasks API** internally — `mp.solutions`
> was removed in 0.10.30.

On first run the system downloads two small model files (~14 MB total) to
`ml/` automatically (no manual step needed).

---

## Workflow

### Step 1 — Collect data

```bash
python app/main.py
```

Sit in front of your webcam and move through different activity states:

| What to do | Expected label |
|---|---|
| Read quietly, minimal movement | Focused |
| Look around, fidget, pick up phone | Distracted |
| Lean forward, squint, tense up | Stressed |

Run for **5+ minutes per state** (150+ seconds each). Press **Q** to quit.
Frames are logged to `data/dataset.csv` and `data/sessions/<id>.csv`.

### Step 2 — Train the model

```bash
python ml/train.py
```

Requires at least 300 rows in `data/dataset.csv` covering all three labels.
Prints accuracy, classification report, and confusion matrix, then writes
`ml/model.pkl` and `ml/scaler.pkl`.

### Step 3 — Run with live inference

```bash
python app/main.py
```

The model is loaded automatically. The HUD now shows the predicted state and
confidence percentage in real time.

### Step 4 — View the analytics dashboard

```bash
streamlit run dashboard/app.py
```

Opens in your browser at `http://localhost:8501`.

---

## Project Structure

```
rt-mbas/
├── app/
│   ├── main.py        — real-time loop (entry point)
│   ├── camera.py      — WebcamHandler (OpenCV)
│   ├── landmarks.py   — LandmarkDetector (MediaPipe Tasks API)
│   ├── features.py    — FeatureExtractor (stateful, 15 features)
│   ├── inference.py   — InferenceEngine (RandomForest + smoothing)
│   └── config.py      — global constants
├── data/
│   ├── dataset.csv    — cumulative training data (auto-created)
│   └── sessions/      — per-session CSVs (auto-created)
├── ml/
│   ├── preprocess.py  — DataPreprocessor (load, engineer, split, scale)
│   ├── train.py       — training script
│   ├── model.pkl      — saved model (after training)
│   ├── scaler.pkl     — saved scaler (after training)
│   ├── face_landmarker.task   — MediaPipe model (auto-downloaded)
│   └── hand_landmarker.task   — MediaPipe model (auto-downloaded)
├── analysis/
│   └── plots.py       — file-saving visualisation functions
├── dashboard/
│   └── app.py         — Streamlit dashboard (4 tabs)
└── utils/
    ├── helpers.py     — euclidean_distance, heuristic labels, ensure_dirs
    └── logger.py      — SessionLogger (CSV appender)
```

---

## Features Extracted Per Frame

| Feature | Description |
|---|---|
| `eye_aspect_ratio` | Mean EAR for both eyes (blink/fatigue proxy) |
| `blink_indicator` | 1 if EAR < 0.21 |
| `mouth_open_ratio` | Mouth gap / face height |
| `head_tilt_angle` | Degrees from horizontal (eye-corner axis) |
| `face_motion_delta` | Nose tip displacement vs. previous frame |
| `hand_velocity` | Wrist displacement vs. previous frame (px/frame) |
| `hand_acceleration` | Change in velocity frame-to-frame |
| `gesture_activity_score` | Mean distance of all 21 landmarks from wrist |
| `idle_time_ratio` | Fraction of recent frames with velocity < 5 px/f |
| `rolling_mean_ear` | 30-frame rolling mean of EAR |
| `rolling_std_ear` | 30-frame rolling std of EAR |
| `rolling_mean_velocity` | 30-frame rolling mean of hand velocity |
| `rolling_std_velocity` | 30-frame rolling std of hand velocity |
| `motion_entropy` | Shannon entropy of velocity distribution (rolling) |

---

## How to Interpret Predictions

- **Focused** — sustained gaze, minimal hand motion, stable head angle.
- **Distracted** — frequent hand movement (phone, typing), large head turns.
- **Stressed** — reduced EAR (squinting/staring), low motion but elevated
  muscle tension indicators.

Predictions are smoothed over the last 10 frames (majority vote) to avoid
rapid label flickering.

---

## Heuristic Labels (data-collection mode)

When no model exists, `generate_heuristic_label()` assigns labels using:

```
EAR < 0.20 AND hand_velocity < 5  →  Stressed
hand_velocity > 30 OR face_motion > 20  →  Distracted
otherwise  →  Focused
```

These rules bootstrap a usable dataset but are intentionally simple.
After training, the ML model replaces the heuristic for all predictions.

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| **Overview** | Total frames, session duration, label pie chart, EAR time series |
| **Feature Trends** | Multi-feature line chart with optional rolling-mean overlay |
| **Model Insights** | Feature importances bar chart, confusion matrix image |
| **Raw Data** | Searchable/filterable table, CSV download button |

---

## Troubleshooting

**"Error: Cannot open camera at index 0"**
→ Change `WEBCAM_INDEX` in `app/config.py`. Try `1` or `2` for external webcams.

**"AttributeError: module 'mediapipe' has no attribute 'solutions'"**
→ This project requires mediapipe ≥ 0.10.30 (Tasks API). Run:
```bash
pip install "mediapipe>=0.10.30"
```
Older versions are not available on Windows PyPI.

**"Only N rows found — need at least 300"**
→ Collect more data. Run `python app/main.py` across all three states until
`data/dataset.csv` has 100+ rows per label.

**"Missing labels: {'Stressed'}"**
→ The model requires examples of every class. Act out the missing state in
front of the webcam, then retrain.

**"Low accuracy after training"**
→ Ensure balanced data (similar row count per label). Overly imbalanced
datasets cause the model to ignore minority classes. Re-collect with deliberate
focus on under-represented states.

**Dashboard shows blank charts**
→ Make sure you are running `streamlit run dashboard/app.py` from the project
root (`rt-mbas/`), not from inside `dashboard/`.
