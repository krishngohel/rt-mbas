"""
Stateful feature extractor for RT-MBAS.

Computes face, hand, and temporal features from per-frame landmark dicts.
All landmark coordinates are normalized (0-1); pixel conversion uses frame_shape.
"""
import time
import numpy as np
from collections import deque

from app.config import FEATURE_WINDOW_SIZE
from utils.helpers import euclidean_distance, normalize_landmark, exponential_smoothing


class FeatureExtractor:
    """
    Stateful extractor that computes behavioural features from landmark data.

    Maintains rolling buffers and previous-frame state across calls to extract().
    """

    def __init__(self):
        """Initialize rolling buffers and all per-frame state variables."""
        self._ear_buffer: deque = deque(maxlen=FEATURE_WINDOW_SIZE)
        self._velocity_buffer: deque = deque(maxlen=FEATURE_WINDOW_SIZE)
        self._idle_buffer: deque = deque(maxlen=FEATURE_WINDOW_SIZE)

        self._prev_nose: tuple | None = None
        self._prev_wrist: tuple | None = None
        self._prev_velocity: float = 0.0
        self._smoothed_ear: float | None = None
        self._smoothed_velocity: float | None = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _px(self, landmarks: list, idx: int, fw: int, fh: int) -> tuple:
        """Convert a normalized landmark at *idx* to pixel coordinates."""
        return normalize_landmark(landmarks[idx], fw, fh)

    def _eye_aspect_ratio(self, lm: list, fw: int, fh: int) -> float:
        """
        Compute mean Eye Aspect Ratio (EAR) for both eyes.

        Left eye vertical pair  : landmarks 159 (upper), 145 (lower)
        Left eye horizontal pair: landmarks 33  (left),  133 (right)
        Right eye vertical pair : landmarks 386 (upper), 374 (lower)
        Right eye horizontal pair: landmarks 362 (left), 263 (right)
        EAR = vertical_distance / horizontal_distance
        """
        left_v = euclidean_distance(self._px(lm, 159, fw, fh), self._px(lm, 145, fw, fh))
        left_h = euclidean_distance(self._px(lm, 33,  fw, fh), self._px(lm, 133, fw, fh))
        left_ear = left_v / (left_h + 1e-6)

        right_v = euclidean_distance(self._px(lm, 386, fw, fh), self._px(lm, 374, fw, fh))
        right_h = euclidean_distance(self._px(lm, 362, fw, fh), self._px(lm, 263, fw, fh))
        right_ear = right_v / (right_h + 1e-6)

        return (left_ear + right_ear) / 2.0

    def _blink_indicator(self, ear: float) -> int:
        """Return 1 if EAR signals a blink (< 0.21), else 0."""
        return 1 if ear < 0.21 else 0

    def _mouth_open_ratio(self, lm: list, fw: int, fh: int) -> float:
        """Ratio of mouth gap (13→14) to face height (10→152)."""
        mouth = euclidean_distance(self._px(lm, 13, fw, fh), self._px(lm, 14, fw, fh))
        face_h = euclidean_distance(self._px(lm, 10, fw, fh), self._px(lm, 152, fw, fh))
        return mouth / (face_h + 1e-6)

    def _head_tilt_angle(self, lm: list, fw: int, fh: int) -> float:
        """Angle in degrees of the eye-corner axis (landmark 33→263) vs. horizontal."""
        left = self._px(lm, 33, fw, fh)
        right = self._px(lm, 263, fw, fh)
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    def _face_motion_delta(self, lm: list, fw: int, fh: int) -> float:
        """Euclidean displacement of nose tip (landmark 1) between consecutive frames."""
        nose = self._px(lm, 1, fw, fh)
        delta = euclidean_distance(nose, self._prev_nose) if self._prev_nose is not None else 0.0
        self._prev_nose = nose
        return delta

    def _hand_velocity(self, hand_lm: list | None, fw: int, fh: int) -> float:
        """Pixel displacement of wrist (landmark 0) between consecutive frames."""
        if hand_lm is None:
            self._prev_wrist = None
            return 0.0
        wrist = normalize_landmark(hand_lm[0], fw, fh)
        vel = euclidean_distance(wrist, self._prev_wrist) if self._prev_wrist is not None else 0.0
        self._prev_wrist = wrist
        return vel

    def _hand_acceleration(self, velocity: float) -> float:
        """Absolute change in hand velocity between consecutive frames."""
        accel = abs(velocity - self._prev_velocity)
        self._prev_velocity = velocity
        return accel

    def _gesture_activity_score(self, hand_lm: list | None, fw: int, fh: int) -> float:
        """Mean distance of all 21 landmarks from the wrist — measures hand spread."""
        if hand_lm is None:
            return 0.0
        wrist = normalize_landmark(hand_lm[0], fw, fh)
        pts = [normalize_landmark(lm, fw, fh) for lm in hand_lm]
        return float(np.mean([euclidean_distance(p, wrist) for p in pts]))

    def _idle_time_ratio(self, velocity: float) -> float:
        """Fraction of frames in rolling window where hand_velocity < 5 px/frame."""
        self._idle_buffer.append(1 if velocity < 5.0 else 0)
        return float(sum(self._idle_buffer)) / len(self._idle_buffer)

    def _motion_entropy(self) -> float:
        """Shannon entropy of hand_velocity values in the rolling buffer."""
        arr = np.array(list(self._velocity_buffer))
        if len(arr) < 2:
            return 0.0
        n_bins = min(10, len(arr))
        counts, _ = np.histogram(arr, bins=n_bins)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts[counts > 0] / total
        return float(-np.sum(probs * np.log(probs + 1e-9)))

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, landmarks_dict: dict, frame_shape: tuple) -> dict:
        """
        Compute all behavioural features for one frame.

        Args:
            landmarks_dict: {"face": [...], "left_hand": [...], "right_hand": [...]}
                            Values are lists of normalized (x, y, z) tuples or None.
            frame_shape: frame.shape tuple (height, width, channels).

        Returns:
            Flat dict with keys:
              timestamp, eye_aspect_ratio, blink_indicator, mouth_open_ratio,
              head_tilt_angle, face_motion_delta, hand_velocity, hand_acceleration,
              gesture_activity_score, idle_time_ratio, rolling_mean_ear,
              rolling_std_ear, rolling_mean_velocity, rolling_std_velocity,
              motion_entropy
        """
        fh, fw = frame_shape[:2]
        face_lm = landmarks_dict.get("face")
        left_hand = landmarks_dict.get("left_hand")
        right_hand = landmarks_dict.get("right_hand")
        hand_lm = right_hand if right_hand is not None else left_hand

        # ── Face features ─────────────────────────────────────────────────────
        if face_lm is not None:
            ear = self._eye_aspect_ratio(face_lm, fw, fh)
            blink = self._blink_indicator(ear)
            mouth_ratio = self._mouth_open_ratio(face_lm, fw, fh)
            tilt = self._head_tilt_angle(face_lm, fw, fh)
            face_motion = self._face_motion_delta(face_lm, fw, fh)
        else:
            ear = 0.0
            blink = 0
            mouth_ratio = 0.0
            tilt = 0.0
            face_motion = 0.0
            self._prev_nose = None

        # ── Hand features ─────────────────────────────────────────────────────
        velocity = self._hand_velocity(hand_lm, fw, fh)
        acceleration = self._hand_acceleration(velocity)
        gesture_score = self._gesture_activity_score(hand_lm, fw, fh)
        idle_ratio = self._idle_time_ratio(velocity)

        # ── Rolling / temporal ────────────────────────────────────────────────
        self._ear_buffer.append(ear)
        self._velocity_buffer.append(velocity)

        mean_ear = float(np.mean(self._ear_buffer))
        std_ear = float(np.std(self._ear_buffer))
        mean_vel = float(np.mean(self._velocity_buffer))
        std_vel = float(np.std(self._velocity_buffer))
        entropy = self._motion_entropy()

        self._smoothed_ear = exponential_smoothing(ear, self._smoothed_ear)
        self._smoothed_velocity = exponential_smoothing(velocity, self._smoothed_velocity)

        return {
            "timestamp": time.time(),
            "eye_aspect_ratio": round(float(ear), 5),
            "blink_indicator": int(blink),
            "mouth_open_ratio": round(float(mouth_ratio), 5),
            "head_tilt_angle": round(float(tilt), 3),
            "face_motion_delta": round(float(face_motion), 3),
            "hand_velocity": round(float(velocity), 3),
            "hand_acceleration": round(float(acceleration), 3),
            "gesture_activity_score": round(float(gesture_score), 4),
            "idle_time_ratio": round(float(idle_ratio), 4),
            "rolling_mean_ear": round(mean_ear, 5),
            "rolling_std_ear": round(std_ear, 5),
            "rolling_mean_velocity": round(mean_vel, 3),
            "rolling_std_velocity": round(std_vel, 3),
            "motion_entropy": round(entropy, 4),
        }
