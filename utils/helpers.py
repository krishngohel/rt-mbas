"""Standalone utility functions shared across RT-MBAS modules."""
import math
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def euclidean_distance(p1, p2) -> float:
    """Compute Euclidean distance between two points (2D or 3D tuples)."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def normalize_landmark(landmark, frame_w: int, frame_h: int) -> tuple:
    """Convert a normalized MediaPipe landmark (x, y, z) to pixel coordinates."""
    x, y, z = landmark
    return (x * frame_w, y * frame_h, z)


def smooth_predictions(prediction_buffer, window: int):
    """Return the most common prediction in the last *window* frames."""
    recent = list(prediction_buffer)[-window:]
    if not recent:
        return None
    return Counter(recent).most_common(1)[0][0]


def exponential_smoothing(value: float, prev_smoothed, alpha: float = 0.3) -> float:
    """Apply exponential smoothing: alpha * value + (1 - alpha) * prev_smoothed."""
    if prev_smoothed is None:
        return float(value)
    return alpha * float(value) + (1.0 - alpha) * float(prev_smoothed)


def generate_heuristic_label(features: dict) -> str:
    """
    Rule-based label assignment for bootstrapping the dataset.

    Rules (in priority order):
      Stressed    — low EAR AND very low hand movement
      Distracted  — high hand velocity OR large face motion
      Focused     — everything else
    """
    ear = features.get("eye_aspect_ratio", 0.25)
    vel = features.get("hand_velocity", 0.0)
    face_motion = features.get("face_motion_delta", 0.0)

    if ear < 0.20 and vel < 5.0:
        return "Stressed"
    if vel > 30.0 or face_motion > 20.0:
        return "Distracted"
    return "Focused"


def ensure_dirs() -> None:
    """Create required project directories if they do not already exist."""
    for subdir in ("data", "data/sessions", "ml", "analysis"):
        (_ROOT / subdir).mkdir(parents=True, exist_ok=True)
