"""
Threaded MediaPipe landmark detector — VIDEO mode, full face mesh overlay.

Topology reference: mediapipe.dev/solutions/face_mesh
  Landmarks 0-467  : Face mesh (468 points)
  Landmarks 468-477: Iris refinement (10 points, 5 per eye)

Detection runs in a background daemon thread.  The main thread calls
update() each frame (non-blocking) and get_latest() to read results.
"""
import threading
import time
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
    DETECT_WIDTH,
    DETECT_HEIGHT,
)

_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ═════════════════════════════════════════════════════════════════════════════
#  Face mesh connection topology
#  All sets are frozenset of (landmark_a, landmark_b) pairs.
#  Source: mediapipe/python/solutions/face_mesh_connections.py
# ═════════════════════════════════════════════════════════════════════════════

FACE_OVAL = frozenset([
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
])

FACE_LEFT_EYE = frozenset([
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263),
])

FACE_RIGHT_EYE = frozenset([
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
])

FACE_LEFT_EYEBROW = frozenset([
    (276, 283), (283, 282), (282, 295), (295, 285),
    (300, 293), (293, 334), (334, 296), (296, 336),
])

FACE_RIGHT_EYEBROW = frozenset([
    (46, 53), (53, 52), (52, 65), (65, 55),
    (70, 63), (63, 105), (105, 66), (66, 107),
])

FACE_NOSE = frozenset([
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
    (4, 1), (1, 19), (19, 94), (94, 2), (2, 97),
    (97, 98), (2, 326), (326, 327), (327, 294), (294, 278),
])

FACE_LIPS_OUTER = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
])

FACE_LIPS_INNER = frozenset([
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
])

# Iris landmark rings (requires refine_landmarks / 478-point model)
FACE_RIGHT_IRIS = [(469, 470), (470, 471), (471, 472), (472, 469)]
FACE_LEFT_IRIS  = [(474, 475), (475, 476), (476, 477), (477, 474)]

# ── Drawing style ─────────────────────────────────────────────────────────────
# (group, colour BGR, line thickness)
_FACE_GROUPS = [
    (FACE_OVAL,         (0,   220,   0),   1),   # green
    (FACE_RIGHT_EYE,    (0,   230, 230),   1),   # cyan
    (FACE_LEFT_EYE,     (0,   230, 230),   1),   # cyan
    (FACE_RIGHT_EYEBROW,(0,   200, 255),   1),   # amber
    (FACE_LEFT_EYEBROW, (0,   200, 255),   1),   # amber
    (FACE_NOSE,         (200, 200, 200),   1),   # light grey
    (FACE_LIPS_OUTER,   (80,  80,  255),   1),   # red-pink
    (FACE_LIPS_INNER,   (120, 120, 255),   1),   # lighter red
]

# ── Hand skeleton ─────────────────────────────────────────────────────────────
_HAND_SKELETON = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),        # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),        # index
    (0, 9),  (9, 10), (10, 11),(11, 12),       # middle
    (0, 13),(13, 14),(14, 15),(15, 16),        # ring
    (0, 17),(17, 18),(18, 19),(19, 20),        # pinky
    (5, 9),  (9, 13),(13, 17),                 # palm arch
]
_FINGERTIP_IDX = {4, 8, 12, 16, 20}   # distinct colour for fingertips

_EMPTY = {"face": None, "left_hand": None, "right_hand": None}


# ─────────────────────────────────────────────────────────────────────────────

def _ensure_model(path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, path)
    print("done.")


def _lm_px(lm_list, idx: int, w: int, h: int) -> tuple[int, int]:
    """Convert a normalised landmark to integer pixel coordinates."""
    lm = lm_list[idx]
    return (int(lm[0] * w), int(lm[1] * h))


def _draw_connections(
    img: np.ndarray,
    lm: list,
    pairs,
    color: tuple,
    thickness: int,
    n: int,
) -> None:
    """Draw a set of (a, b) connection pairs — skips any index out of range."""
    for a, b in pairs:
        if a < n and b < n:
            cv2.line(img, _lm_px(lm, a, img.shape[1], img.shape[0]),
                         _lm_px(lm, b, img.shape[1], img.shape[0]),
                     color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────

class LandmarkDetector:
    """
    Threaded face-mesh + hand-landmark detector using MediaPipe Tasks (VIDEO mode).

    Public interface
    ----------------
    update(frame_bgr)   -- push latest camera frame (non-blocking)
    get_latest()        -- last computed result dict
    draw_landmarks(...) -- annotate a frame in-place copy; returns new ndarray
    close()             -- stop thread + release resources
    """

    def __init__(self):
        _ensure_model(FACE_LANDMARKER_PATH, _FACE_MODEL_URL)
        _ensure_model(HAND_LANDMARKER_PATH, _HAND_MODEL_URL)

        RunningMode = mp_vision.RunningMode

        # FaceLandmarker — VIDEO mode enables inter-frame tracking
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=FACE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_MIN_TRACKING_CONFIDENCE,
        )
        self._face = mp_vision.FaceLandmarker.create_from_options(face_opts)

        # HandLandmarker — VIDEO mode
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE,
        )
        self._hands = mp_vision.HandLandmarker.create_from_options(hand_opts)

        # Shared state
        self._cond        = threading.Condition()
        self._pending: tuple | None = None

        self._result_lock = threading.Lock()
        self._latest: dict = dict(_EMPTY)

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True,
                                         name="mp-detect")
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame_bgr: np.ndarray) -> None:
        """Push a new frame to the detection thread.  Drops frame if thread busy."""
        ts_ms = int(time.monotonic() * 1000)
        with self._cond:
            self._pending = (frame_bgr, ts_ms)
            self._cond.notify()

    def get_latest(self) -> dict:
        """Return the most recently computed landmark result (thread-safe)."""
        with self._result_lock:
            return dict(self._latest)

    # ── Background detection loop ─────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            with self._cond:
                while self._pending is None and self._running:
                    self._cond.wait(timeout=0.05)
                if not self._running:
                    break
                frame_bgr, ts_ms = self._pending
                self._pending = None

            try:
                small  = cv2.resize(frame_bgr, (DETECT_WIDTH, DETECT_HEIGHT))
                rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                face_res = self._face.detect_for_video(mp_img, ts_ms)
                hand_res = self._hands.detect_for_video(mp_img, ts_ms)

                face_lm = None
                if face_res.face_landmarks:
                    face_lm = [(lm.x, lm.y, lm.z)
                                for lm in face_res.face_landmarks[0]]

                left_hand = right_hand = None
                if hand_res.hand_landmarks:
                    for i, hlm in enumerate(hand_res.hand_landmarks):
                        pts = [(lm.x, lm.y, lm.z) for lm in hlm]
                        if hand_res.handedness[i][0].category_name == "Left":
                            left_hand = pts
                        else:
                            right_hand = pts

                with self._result_lock:
                    self._latest = {
                        "face":       face_lm,
                        "left_hand":  left_hand,
                        "right_hand": right_hand,
                    }
            except Exception:
                pass  # never crash the detection thread

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_landmarks(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Render the full face mesh (468 landmarks) and hand skeletons (21 per hand)
        onto a copy of *frame*.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Face mesh ─────────────────────────────────────────────────────────
        face_lm = result.get("face")
        if face_lm:
            n = len(face_lm)

            # Pre-compute all pixel coords in one vectorised pass — avoids
            # repeated int(lm[i]*w) calls across every drawing step.
            lm_arr = np.array([[lm[0], lm[1]] for lm in face_lm[:n]], dtype=np.float32)
            px = (lm_arr * np.array([w, h], dtype=np.float32)).astype(np.int32)
            np.clip(px[:, 0], 0, w - 1, out=px[:, 0])
            np.clip(px[:, 1], 0, h - 1, out=px[:, 1])

            # 1 · Landmark dots — single numpy write instead of 468 cv2.circle calls
            dots = px[:min(468, n)]
            out[dots[:, 1], dots[:, 0]] = (0, 180, 0)

            # 2 · Coloured connection groups
            for group, color, thickness in _FACE_GROUPS:
                for a, b in group:
                    if a < n and b < n:
                        cv2.line(out, (px[a, 0], px[a, 1]),
                                      (px[b, 0], px[b, 1]),
                                 color, thickness, cv2.LINE_8)

            # 3 · Iris rings
            if n >= 478:
                for ring, color in [
                    (FACE_RIGHT_IRIS, (0, 200, 255)),
                    (FACE_LEFT_IRIS,  (0, 200, 255)),
                ]:
                    for a, b in ring:
                        if a < n and b < n:
                            cv2.line(out, (px[a, 0], px[a, 1]),
                                          (px[b, 0], px[b, 1]),
                                     color, 1, cv2.LINE_8)

        # ── Hands ─────────────────────────────────────────────────────────────
        for key, skel_color, tip_color in (
            ("left_hand",  (255, 180,  30), (255, 80,   0)),
            ("right_hand", ( 30, 180, 255), ( 0, 100, 255)),
        ):
            hand_lm = result.get(key)
            if not hand_lm:
                continue

            hlm_arr = np.array([[lm[0], lm[1]] for lm in hand_lm], dtype=np.float32)
            hpx = (hlm_arr * np.array([w, h], dtype=np.float32)).astype(np.int32)
            np.clip(hpx[:, 0], 0, w - 1, out=hpx[:, 0])
            np.clip(hpx[:, 1], 0, h - 1, out=hpx[:, 1])

            for a, b in _HAND_SKELETON:
                cv2.line(out, (hpx[a, 0], hpx[a, 1]),
                              (hpx[b, 0], hpx[b, 1]),
                         skel_color, 2, cv2.LINE_8)

            for i in range(len(hand_lm)):
                color = tip_color if i in _FINGERTIP_IDX else (255, 255, 255)
                r     = 6 if i in _FINGERTIP_IDX or i == 0 else 3
                pt    = (hpx[i, 0], hpx[i, 1])
                cv2.circle(out, pt, r, color, -1)
                cv2.circle(out, pt, r, (0, 0, 0), 1)

        return out

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop the detection thread and release MediaPipe resources."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self._face.close()
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
