"""
Stateful behavioural feature extractor for RT-MBAS.

EAR uses the standard 6-landmark formula (Soukupova & Cech, 2016) adapted
for MediaPipe's 478-landmark face mesh.  Two vertical pairs per eye give a
more stable estimate than a single pair, especially under small head rotations.

Eye landmark sets (viewer's perspective):
  Right eye (person's right, appears left on screen):
    outer=33, upper={160,158}, inner=133, lower={153,144}
  Left eye (person's left, appears right on screen):
    outer=263, upper={385,387}, inner=362, lower={373,380}

EAR = (||v1-v6|| + ||v2-v5||) / (2 * ||h1-h4||)
"""
import time
import numpy as np
from collections import deque

from app.config import FEATURE_WINDOW_SIZE
from utils.helpers import euclidean_distance, normalize_landmark, exponential_smoothing

# 6-point eye landmark indices: [outer, upper1, upper2, inner, lower1, lower2]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]


class FeatureExtractor:
    """
    Stateful extractor — maintains rolling buffers and per-frame delta state.

    Call extract() once per display frame; all state is updated in place.
    """

    def __init__(self):
        self._ear_buf:  deque = deque(maxlen=FEATURE_WINDOW_SIZE)
        self._vel_buf:  deque = deque(maxlen=FEATURE_WINDOW_SIZE)
        self._idle_buf: deque = deque(maxlen=FEATURE_WINDOW_SIZE)

        self._prev_nose:  tuple | None = None
        self._prev_wrist: tuple | None = None
        self._prev_vel:   float = 0.0
        self._ema_ear:    float | None = None
        self._ema_vel:    float | None = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _px(self, lm, idx, fw, fh) -> tuple:
        return normalize_landmark(lm[idx], fw, fh)

    def _eye_aspect_ratio(self, lm, fw, fh) -> float:
        """6-point EAR averaged over both eyes."""
        def _ear(pts):
            v1 = euclidean_distance(self._px(lm, pts[1], fw, fh),
                                    self._px(lm, pts[5], fw, fh))
            v2 = euclidean_distance(self._px(lm, pts[2], fw, fh),
                                    self._px(lm, pts[4], fw, fh))
            h  = euclidean_distance(self._px(lm, pts[0], fw, fh),
                                    self._px(lm, pts[3], fw, fh))
            return (v1 + v2) / (2.0 * h + 1e-6)

        return (_ear(_RIGHT_EYE) + _ear(_LEFT_EYE)) / 2.0

    def _mouth_open_ratio(self, lm, fw, fh) -> float:
        """Vertical mouth gap (13->14) normalised by face height (10->152)."""
        gap  = euclidean_distance(self._px(lm, 13, fw, fh), self._px(lm, 14, fw, fh))
        face = euclidean_distance(self._px(lm, 10, fw, fh), self._px(lm, 152, fw, fh))
        return gap / (face + 1e-6)

    def _head_tilt_angle(self, lm, fw, fh) -> float:
        """
        Angle (degrees) of the eye-corner axis (landmark 33 -> 263) vs horizontal.
        Near 0 = upright face; positive = face tilted clockwise (right ear down).
        """
        left  = self._px(lm, 33,  fw, fh)
        right = self._px(lm, 263, fw, fh)
        return float(np.degrees(np.arctan2(right[1] - left[1],
                                           right[0] - left[0] + 1e-6)))

    def _face_motion(self, lm, fw, fh) -> float:
        nose = self._px(lm, 1, fw, fh)
        d = euclidean_distance(nose, self._prev_nose) if self._prev_nose is not None else 0.0
        self._prev_nose = nose
        return d

    def _hand_velocity(self, hand_lm, fw, fh) -> float:
        if hand_lm is None:
            self._prev_wrist = None
            return 0.0
        wrist = normalize_landmark(hand_lm[0], fw, fh)
        v = euclidean_distance(wrist, self._prev_wrist) if self._prev_wrist is not None else 0.0
        self._prev_wrist = wrist
        return v

    def _gesture_score(self, hand_lm, fw, fh) -> float:
        if hand_lm is None:
            return 0.0
        wrist = normalize_landmark(hand_lm[0], fw, fh)
        pts   = [normalize_landmark(lm, fw, fh) for lm in hand_lm]
        return float(np.mean([euclidean_distance(p, wrist) for p in pts]))

    @staticmethod
    def _entropy(buf: deque) -> float:
        arr = np.array(list(buf))
        if len(arr) < 2:
            return 0.0
        counts, _ = np.histogram(arr, bins=min(10, len(arr)))
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts[counts > 0] / total
        return float(-np.sum(p * np.log(p + 1e-9)))

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, landmarks_dict: dict, frame_shape: tuple) -> dict:
        """
        Compute all 15 behavioural features from one frame's landmark dict.

        Args:
            landmarks_dict: {"face": [...], "left_hand": [...], "right_hand": [...]}
                            Values are lists of normalised (x, y, z) tuples or None.
            frame_shape:    frame.shape tuple — used to convert to pixel coords.

        Returns:
            Flat feature dict.  All values are Python floats / ints.
        """
        fh, fw = frame_shape[:2]
        face_lm  = landmarks_dict.get("face")
        right_lm = landmarks_dict.get("right_hand")
        left_lm  = landmarks_dict.get("left_hand")
        hand_lm  = right_lm if right_lm is not None else left_lm

        # ── Face ──────────────────────────────────────────────────────────────
        if face_lm is not None and len(face_lm) > 387:
            ear         = self._eye_aspect_ratio(face_lm, fw, fh)
            blink       = int(ear < 0.21)
            mouth_ratio = self._mouth_open_ratio(face_lm, fw, fh)
            tilt        = self._head_tilt_angle(face_lm, fw, fh)
            face_motion = self._face_motion(face_lm, fw, fh)
        else:
            ear = blink = mouth_ratio = tilt = face_motion = 0.0
            self._prev_nose = None

        # ── Hand ──────────────────────────────────────────────────────────────
        vel    = self._hand_velocity(hand_lm, fw, fh)
        accel  = abs(vel - self._prev_vel)
        self._prev_vel = vel
        gesture = self._gesture_score(hand_lm, fw, fh)

        self._idle_buf.append(1 if vel < 5.0 else 0)
        idle_ratio = sum(self._idle_buf) / len(self._idle_buf)

        # ── Rolling / temporal ────────────────────────────────────────────────
        self._ear_buf.append(ear)
        self._vel_buf.append(vel)

        mean_ear = float(np.mean(self._ear_buf))
        std_ear  = float(np.std(self._ear_buf))
        mean_vel = float(np.mean(self._vel_buf))
        std_vel  = float(np.std(self._vel_buf))
        entropy  = self._entropy(self._vel_buf)

        self._ema_ear = exponential_smoothing(ear, self._ema_ear)
        self._ema_vel = exponential_smoothing(vel, self._ema_vel)

        return {
            "timestamp":             time.time(),
            "eye_aspect_ratio":      round(float(ear),        5),
            "blink_indicator":       int(blink),
            "mouth_open_ratio":      round(float(mouth_ratio), 5),
            "head_tilt_angle":       round(float(tilt),        3),
            "face_motion_delta":     round(float(face_motion), 3),
            "hand_velocity":         round(float(vel),         3),
            "hand_acceleration":     round(float(accel),       3),
            "gesture_activity_score":round(float(gesture),     4),
            "idle_time_ratio":       round(float(idle_ratio),  4),
            "rolling_mean_ear":      round(mean_ear,           5),
            "rolling_std_ear":       round(std_ear,            5),
            "rolling_mean_velocity": round(mean_vel,           3),
            "rolling_std_velocity":  round(std_vel,            3),
            "motion_entropy":        round(entropy,            4),
        }
