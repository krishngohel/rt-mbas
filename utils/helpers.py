"""Standalone utility functions shared across RT-MBAS modules."""
import math
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def euclidean_distance(p1, p2) -> float:
    """Euclidean distance between two 2-D or 3-D coordinate tuples."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def normalize_landmark(landmark, frame_w: int, frame_h: int) -> tuple:
    """Convert a normalised MediaPipe (x, y, z) landmark to pixel coordinates."""
    x, y, z = landmark
    return (x * frame_w, y * frame_h, z)


def smooth_predictions(prediction_buffer, window: int):
    """Return the most common prediction in the last *window* frames."""
    recent = list(prediction_buffer)[-window:]
    if not recent:
        return None
    return Counter(recent).most_common(1)[0][0]


def exponential_smoothing(value: float, prev_smoothed, alpha: float = 0.3) -> float:
    """Exponential moving average: alpha * value + (1 - alpha) * prev_smoothed."""
    if prev_smoothed is None:
        return float(value)
    return alpha * float(value) + (1.0 - alpha) * float(prev_smoothed)


def generate_heuristic_label(features: dict) -> str:
    """
    Rule-based label used to bootstrap the training dataset.

    Calibrated against 6-point EAR ranges:
      - Eyes open, relaxed : EAR ~0.28-0.38
      - Mild squint/fatigue: EAR ~0.22-0.27
      - Heavy squint       : EAR ~0.17-0.22
      - Nearly closed      : EAR < 0.17

    Rules evaluated in priority order — first match wins.
    """
    ear         = features.get("eye_aspect_ratio",  0.28)
    vel         = features.get("hand_velocity",     0.0)
    face_motion = features.get("face_motion_delta", 0.0)
    tilt        = abs(features.get("head_tilt_angle", 0.0))

    # No face detected (EAR == 0 exactly) — default label, don't infer state
    if ear <= 0.01:
        return "Focused"

    # Exhausted — barely-open eyes with near-zero movement
    if ear < 0.17 and vel < 5.0 and face_motion < 2.0:
        return "Exhausted"

    # Angry — squinting combined with active body movement (tension/agitation)
    if ear < 0.24 and (vel > 15.0 or face_motion > 8.0):
        return "Angry"

    # Stressed — squinting, still, facing forward
    if ear < 0.23 and vel < 8.0 and tilt < 12.0:
        return "Stressed"

    # Sad — mildly low EAR, very little movement, face forward
    if ear < 0.27 and vel < 5.0 and face_motion < 3.0:
        return "Sad"

    # Distracted — high movement or pronounced head turn (check before Confused)
    if vel > 22.0 or face_motion > 10.0 or tilt > 18.0:
        return "Distracted"

    # Confused — noticeable head tilt without large movement
    if tilt > 10.0:
        return "Confused"

    return "Focused"


def ensure_dirs() -> None:
    """Create required project directories if they do not already exist."""
    for subdir in ("data", "data/sessions", "ml", "analysis"):
        (_ROOT / subdir).mkdir(parents=True, exist_ok=True)
