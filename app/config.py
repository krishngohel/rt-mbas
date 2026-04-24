"""Global constants used across all RT-MBAS modules."""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
ML_DIR = ROOT_DIR / "ml"

# ── Camera ────────────────────────────────────────────────────────────────────
WEBCAM_INDEX = 0
TARGET_FPS = 30
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ── Feature extraction ────────────────────────────────────────────────────────
FEATURE_WINDOW_SIZE = 30   # frames for rolling statistics
SMOOTHING_WINDOW = 10      # frames for prediction smoothing

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = ROOT_DIR / "data" / "dataset.csv"
SESSIONS_PATH = ROOT_DIR / "data" / "sessions"
MODEL_PATH = ML_DIR / "model.pkl"
SCALER_PATH = ML_DIR / "scaler.pkl"

# MediaPipe Tasks API model files (auto-downloaded on first run)
FACE_LANDMARKER_PATH = ML_DIR / "face_landmarker.task"
HAND_LANDMARKER_PATH = ML_DIR / "hand_landmarker.task"

# ── ML ────────────────────────────────────────────────────────────────────────
LABELS = ["Focused", "Distracted", "Stressed"]

# ── MediaPipe confidence thresholds ───────────────────────────────────────────
FACE_MIN_DETECTION_CONFIDENCE = 0.7
FACE_MIN_TRACKING_CONFIDENCE = 0.7
HAND_MIN_DETECTION_CONFIDENCE = 0.7
HAND_MIN_TRACKING_CONFIDENCE = 0.7
