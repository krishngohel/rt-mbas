"""
MediaPipe Tasks-based landmark detector for RT-MBAS.

Uses the Tasks API (mediapipe >= 0.10.30) which replaced mp.solutions.
Model files are auto-downloaded to ml/ on first run (~14 MB total).
"""
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from app.config import (
    FACE_MIN_DETECTION_CONFIDENCE,
    FACE_MIN_TRACKING_CONFIDENCE,
    HAND_MIN_DETECTION_CONFIDENCE,
    HAND_MIN_TRACKING_CONFIDENCE,
    FACE_LANDMARKER_PATH,
    HAND_LANDMARKER_PATH,
)

# ── Model download URLs ───────────────────────────────────────────────────────
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Face mesh connections (oval + eyes + lips) ────────────────────────────────
_FACE_CONNECTIONS = frozenset([
    # Face oval
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
    # Right eye
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
    # Left eye
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263),
    # Lips outer
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270),
    (270, 409), (409, 291),
    # Lips inner
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317),
    (317, 402), (402, 318), (318, 324), (324, 308), (78, 191), (191, 80),
    (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310),
    (310, 415), (415, 308),
])

# ── Hand skeleton connections (21 landmarks) ──────────────────────────────────
_HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
])


def _ensure_model(path, url: str) -> None:
    """Download a MediaPipe task model file if it is not already present."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} (first run only) ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, path)
    print("done.")


class LandmarkDetector:
    """Detects face mesh and hand landmarks using the MediaPipe Tasks API."""

    def __init__(self):
        """Download models if needed and initialize FaceLandmarker + HandLandmarker."""
        _ensure_model(FACE_LANDMARKER_PATH, _FACE_MODEL_URL)
        _ensure_model(HAND_LANDMARKER_PATH, _HAND_MODEL_URL)

        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
            num_faces=1,
            min_face_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=FACE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_MIN_TRACKING_CONFIDENCE,
        )
        self._face = mp_vision.FaceLandmarker.create_from_options(face_opts)

        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
            num_hands=2,
            min_hand_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE,
        )
        self._hands = mp_vision.HandLandmarker.create_from_options(hand_opts)

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run face mesh and hand detection on a BGR frame.

        Returns:
            {
                "face":       list of (x, y, z) normalized tuples, or None
                "left_hand":  list of (x, y, z) normalized tuples, or None
                "right_hand": list of (x, y, z) normalized tuples, or None
            }
        Never raises — returns all-None dict on any detection error.
        """
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            face_result = self._face.detect(mp_img)
            hand_result = self._hands.detect(mp_img)

            face_lm = None
            if face_result.face_landmarks:
                face_lm = [
                    (lm.x, lm.y, lm.z)
                    for lm in face_result.face_landmarks[0]
                ]

            left_hand = right_hand = None
            if hand_result.hand_landmarks:
                for i, hand_lm in enumerate(hand_result.hand_landmarks):
                    pts = [(lm.x, lm.y, lm.z) for lm in hand_lm]
                    label = hand_result.handedness[i][0].category_name
                    if label == "Left":
                        left_hand = pts
                    else:
                        right_hand = pts

            return {"face": face_lm, "left_hand": left_hand, "right_hand": right_hand}

        except Exception:
            return {"face": None, "left_hand": None, "right_hand": None}

    def draw_landmarks(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """
        Draw face mesh and hand skeletons onto a copy of *frame* using OpenCV.

        Returns the annotated frame (original is not modified).
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Face ─────────────────────────────────────────────────────────────
        face_lm = landmarks_dict.get("face")
        if face_lm:
            n = len(face_lm)
            for a, b in _FACE_CONNECTIONS:
                if a < n and b < n:
                    pt1 = (int(face_lm[a][0] * w), int(face_lm[a][1] * h))
                    pt2 = (int(face_lm[b][0] * w), int(face_lm[b][1] * h))
                    cv2.line(out, pt1, pt2, (0, 200, 0), 1, cv2.LINE_AA)

        # ── Hands ─────────────────────────────────────────────────────────────
        hand_colors = {
            "left_hand": (255, 120, 0),
            "right_hand": (0, 120, 255),
        }
        for key, color in hand_colors.items():
            hand_lm = landmarks_dict.get(key)
            if hand_lm:
                for a, b in _HAND_CONNECTIONS:
                    pt1 = (int(hand_lm[a][0] * w), int(hand_lm[a][1] * h))
                    pt2 = (int(hand_lm[b][0] * w), int(hand_lm[b][1] * h))
                    cv2.line(out, pt1, pt2, color, 2, cv2.LINE_AA)
                for lm in hand_lm:
                    cv2.circle(
                        out,
                        (int(lm[0] * w), int(lm[1] * h)),
                        4, (255, 255, 255), -1,
                    )

        return out

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._face.close()
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
