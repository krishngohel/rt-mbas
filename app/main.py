"""
RT-MBAS — Real-Time Multimodal Behavioral Analytics System
Run with:  python app/main.py
Controls:  Q = quit

Performance design
------------------
MediaPipe detection runs in a background daemon thread (see landmarks.py).
The main loop only does: grab frame -> draw landmarks -> warning overlay -> HUD -> imshow.
Feature extraction and logging happen every LOG_EVERY_N frames to cut I/O cost.
Camera buffer is set to 1 so we always get the freshest frame.
"""
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.camera import WebcamHandler
from app.landmarks import LandmarkDetector
from app.features import FeatureExtractor
from app.inference import InferenceEngine
from app.config import (
    WEBCAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS, LOG_EVERY_N,
    DATA_PATH, SESSIONS_PATH,
)
from utils.helpers import generate_heuristic_label, ensure_dirs
from utils.logger import SessionLogger

_STATE_COLORS = {
    "Focused":    (0, 220, 0),      # green
    "Distracted": (0, 165, 255),    # orange
    "Stressed":   (0, 60, 230),     # red-orange
    "Confused":   (0, 210, 255),    # amber-yellow
    "Sad":        (210, 90, 0),     # deep blue
    "Angry":      (0, 0, 255),      # pure red
    "Exhausted":  (130, 80, 160),   # muted purple
}

# ── Face quality thresholds ───────────────────────────────────────────────────
# IOD = inter-ocular pixel distance (landmarks 33–263) / frame_width.
# Thresholds reduced by 7.5 % vs. v1 so the optimal viewing distance is
# pushed 7.5 % further back — more of the upper body is in frame.
_IOD_GOOD = 0.093   # > 9.3 % → comfortable range  (was 0.10)
_IOD_WARN = 0.060   # 6.0–9.3 % → "move closer"    (was 0.065)
#                    < 6.0 % or no face → critical

# ── Controls shown in the legend ─────────────────────────────────────────────
_CONTROLS = [
    ("Q", "Quit"),
    ("T", "Start / stop training"),
    ("H", "Toggle this legend"),
]

# ── Guided training mode ──────────────────────────────────────────────────────
# 21 prompts (3 per emotion × 7 emotions).  Each prompt: 3 s countdown then
# 5 s of data collection — ~2.8 min total for a full training run.
_TRAIN_COUNTDOWN_S = 3.0
_TRAIN_COLLECT_S   = 5.0

# (emotion, headline shown large, detail shown small)
_TRAINING_PROMPTS = [
    # Focused × 3
    ("Focused",    "Look at screen normally",
                   "Relaxed, eyes open, natural sitting posture"),
    ("Focused",    "Lean slightly forward — reading",
                   "Forward lean as if reading carefully, still relaxed"),
    ("Focused",    "Gentle head tilt, still focused",
                   "5-10 degree tilt, eyes remain on screen"),
    # Distracted × 3
    ("Distracted", "Look right — something caught your eye",
                   "Turn head 30-45 deg to the right, gaze off screen"),
    ("Distracted", "Look left — glance away",
                   "Turn head 30-45 deg to the left, away from screen"),
    ("Distracted", "Look up-right — mind wandering",
                   "Gaze upward-right as if daydreaming"),
    # Stressed × 3
    ("Stressed",   "Slight squint, stay still",
                   "Narrow eyes mildly, tense expression, face forward"),
    ("Stressed",   "Forward gaze, tense posture",
                   "Eyes narrowed, stiff neck, minimal movement"),
    ("Stressed",   "Intense focus — furrowed brow",
                   "Squint hard, very still, face straight ahead"),
    # Confused × 3
    ("Confused",   "Tilt head ~15 deg to the right",
                   "Classic quizzical tilt, slightly raised brow"),
    ("Confused",   "Tilt head ~15 deg to the left",
                   "Quizzical tilt to the other side"),
    ("Confused",   "Head tilt with uncertain look",
                   "Moderate tilt either side, one brow raised"),
    # Sad × 3
    ("Sad",        "Droopy eyes, low energy",
                   "Heavy eyelids, downcast gaze, minimal movement"),
    ("Sad",        "Downcast gaze, very slow",
                   "Look slightly down, low energy, nearly still"),
    ("Sad",        "Heavy, tired expression",
                   "Mildly drooping eyes, slouched, very still"),
    # Angry × 3
    ("Angry",      "Squint hard + sharp hand movement",
                   "Narrow eyes, then tap desk or gesture quickly"),
    ("Angry",      "Furrowed brow + sudden gesture",
                   "Intense look, then make a quick hand motion"),
    ("Angry",      "Tense expression + firm movement",
                   "Squint and clench, then move hand sharply"),
    # Exhausted × 3
    ("Exhausted",  "Heavy-lidded eyes, slow blink",
                   "Let eyes droop, blink slowly, very still"),
    ("Exhausted",  "Eyes nearly closed",
                   "Eyes barely open, minimal movement, slumped slightly"),
    ("Exhausted",  "Maximum tiredness — eyes at minimum",
                   "As closed as possible while awake — do not move"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Text helper
# ─────────────────────────────────────────────────────────────────────────────

def _text(
    frame: np.ndarray,
    txt: str,
    pos: tuple,
    color: tuple = (255, 255, 255),
    scale: float = 0.6,
    thickness: int = 1,
) -> None:
    """Outlined text — readable on any background colour."""
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  Face quality assessment
# ─────────────────────────────────────────────────────────────────────────────

def _face_quality(landmarks: dict, fw: int, fh: int) -> dict:
    """
    Assess face visibility and distance from the camera.

    Accepts the full landmark dict so it can distinguish between
    "face genuinely absent from frame" and "hands are covering the face".
    When one or both hands are visible but the face is not detected, no
    warning is raised — the occlusion is intentional.

    Uses inter-ocular distance (IOD) between outer eye corners
    (landmarks 33 and 263) normalised by frame width as the distance proxy.

    Returns
    -------
    dict with keys:
      status    : "ok" | "warn" | "critical"
      message   : human-readable string (empty when status == "ok")
      iod_ratio : IOD / frame_width  (0.0 when no face)
    """
    face_lm = landmarks.get("face")
    hands_present = (
        landmarks.get("left_hand") is not None
        or landmarks.get("right_hand") is not None
    )

    if face_lm is None or len(face_lm) < 264:
        # Hands visible → face is covered, not out of frame — suppress warning
        if hands_present:
            return {"status": "ok", "message": "", "iod_ratio": 0.0}
        return {"status": "critical", "message": "No face detected", "iod_ratio": 0.0}

    # Inter-ocular distance
    lx = face_lm[33][0] * fw;  ly = face_lm[33][1] * fh
    rx = face_lm[263][0] * fw; ry = face_lm[263][1] * fh
    iod = math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)
    iod_ratio = iod / fw

    # Face partially out of frame — nose tip (lm 1) as centre proxy
    nx, ny = face_lm[1][0], face_lm[1][1]
    edge = 0.08
    if nx < edge or nx > 1.0 - edge or ny < edge or ny > 1.0 - edge:
        return {"status": "warn",
                "message": "Center your face in frame",
                "iod_ratio": iod_ratio}

    if iod_ratio < _IOD_WARN:
        return {"status": "critical",
                "message": "Too far  -  move closer to the camera",
                "iod_ratio": iod_ratio}

    if iod_ratio < _IOD_GOOD:
        return {"status": "warn",
                "message": "Move a little closer for best accuracy",
                "iod_ratio": iod_ratio}

    return {"status": "ok", "message": "", "iod_ratio": iod_ratio}


# ─────────────────────────────────────────────────────────────────────────────
#  Warning drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_guide_oval(frame: np.ndarray, frame_n: int) -> None:
    """
    Draw a dashed face-position guide oval when no face is detected.
    The oval blinks at ~1.5 Hz to attract attention.
    Modifies *frame* in-place.
    """
    h, w = frame.shape[:2]
    blink = (frame_n // 20) % 2 == 0
    cx, cy = w // 2, int(h * 0.42)
    rx = w // 8
    ry = int(h * 0.28)
    color = (0, 210, 0) if blink else (0, 100, 0)

    # Dashed oval — every other 12-degree arc segment
    for deg in range(0, 360, 18):
        cv2.ellipse(frame, (cx, cy), (rx, ry),
                    0, deg, deg + 10, color, 2, cv2.LINE_AA)

    # Minimal eye markers
    eye_y  = cy - ry // 3
    eye_dx = rx // 2
    for ex in (cx - eye_dx, cx + eye_dx):
        cv2.circle(frame, (ex, eye_y), 7, color, 1, cv2.LINE_AA)

    # Nose bridge hint
    cv2.line(frame, (cx, cy - ry // 6), (cx, cy + ry // 8), color, 1, cv2.LINE_AA)

    # Instruction label below oval
    if blink:
        _text(frame, "Position face here",
              (cx - 118, cy + ry + 32), color, scale=0.68, thickness=2)


def _draw_distance_bar(frame: np.ndarray, iod_ratio: float) -> None:
    """
    Vertical distance meter on the right edge of the frame.
    Green = close / good, amber = warning, red = too far / no face.
    """
    h, w = frame.shape[:2]
    bx   = w - 30
    bh   = h // 3
    by0  = h // 2 - bh // 2
    by1  = by0 + bh

    # Track background
    cv2.rectangle(frame, (bx, by0), (bx + 14, by1), (30, 30, 30), -1)
    cv2.rectangle(frame, (bx, by0), (bx + 14, by1), (100, 100, 100),  1)

    # Fill level — map [0, _IOD_GOOD * 1.5] → [0, bh]
    max_ratio = _IOD_GOOD * 1.5
    fill_f    = min(iod_ratio / max_ratio, 1.0)
    fill_px   = int(bh * fill_f)
    fy0       = by1 - fill_px

    if fill_f >= 0.67:
        bar_color = (0, 200, 0)    # green — good range
    elif fill_f >= 0.44:
        bar_color = (0, 160, 255)  # amber — warning
    else:
        bar_color = (0, 0, 220)    # red — too far / no face

    if fill_px > 0:
        cv2.rectangle(frame, (bx, fy0), (bx + 14, by1), bar_color, -1)

    # Threshold tick mark (where "good" begins)
    tick_y = by1 - int(bh * (_IOD_GOOD / max_ratio))
    cv2.line(frame, (bx - 4, tick_y), (bx + 18, tick_y), (255, 255, 255), 1)

    # Labels
    _text(frame, "DIST", (bx - 2, by0 - 12), (180, 180, 180), scale=0.38)
    _text(frame, "OK",   (bx - 2, by0 - 26), (0, 200, 0),     scale=0.36)


def _draw_face_warning(
    frame: np.ndarray,
    quality: dict,
    frame_n: int,
) -> None:
    """
    Render all face-quality visual cues onto *frame* in-place.

    Layers applied (back to front):
      critical  → dark vignette + dashed guide oval + blinking red pill
      warn      → amber top banner with warning triangle
      both      → distance bar on right edge
    """
    status  = quality["status"]
    message = quality["message"]
    blink   = (frame_n // 20) % 2 == 0
    h, w    = frame.shape[:2]

    if status == "critical":
        # ── Vignette — darken the frame to shift focus to the guide ──────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # ── Dashed guide oval ─────────────────────────────────────────────────
        _draw_guide_oval(frame, frame_n)

        # ── Shoulder guide arcs ───────────────────────────────────────────────
        cx   = w // 2
        ry   = int(h * 0.28)
        cy   = int(h * 0.42)
        s_y  = cy + ry + 24
        s_rx = w // 24
        s_ry = s_rx // 2
        g_c  = (0, 210, 0) if blink else (0, 100, 0)
        for sx in (cx - w // 8, cx + w // 8):
            cv2.ellipse(frame, (sx, s_y), (s_rx, s_ry),
                        0, 180, 360, g_c, 2, cv2.LINE_AA)

        # ── Red warning pill at bottom ────────────────────────────────────────
        if blink:
            pill_w = 400
            px0 = w // 2 - pill_w // 2
            ov2 = frame.copy()
            cv2.rectangle(ov2, (px0, h - 60), (px0 + pill_w, h - 22),
                          (0, 0, 170), -1)
            cv2.addWeighted(ov2, 0.78, frame, 0.22, 0, frame)
        _text(frame, message,
              (w // 2 - 200, h - 33), (255, 255, 255), scale=0.72, thickness=2)

    elif status == "warn":
        # ── Amber top banner ──────────────────────────────────────────────────
        banner_color = (0, 145, 255) if blink else (0, 100, 195)
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 48), banner_color, -1)
        cv2.addWeighted(ov, 0.68, frame, 0.32, 0, frame)

        # Warning triangle (filled white, amber !)
        pts = np.array(
            [(w // 2 - 152, 40), (w // 2 - 132, 8), (w // 2 - 112, 40)],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], (255, 255, 255))
        _text(frame, "!", (w // 2 - 140, 37), (0, 100, 200), scale=0.60)

        _text(frame, message,
              (w // 2 - 100, 33), (255, 255, 255), scale=0.65)

    # ── Distance bar — shown whenever face is not clearly in range ────────────
    if status != "ok":
        _draw_distance_bar(frame, quality["iod_ratio"])


# ─────────────────────────────────────────────────────────────────────────────
#  Controls legend
# ─────────────────────────────────────────────────────────────────────────────

def _draw_controls_legend(frame: np.ndarray) -> None:
    """Compact key-binding legend in the bottom-right corner."""
    h, w  = frame.shape[:2]
    row_h = 22
    pad   = 7
    pw    = 190                          # panel width
    ph    = pad + len(_CONTROLS) * row_h + pad
    px0   = w - pw - 42                  # 42 px gap clears the distance bar
    py0   = h - ph - 8

    ov = frame.copy()
    cv2.rectangle(ov, (px0, py0), (px0 + pw, py0 + ph), (12, 12, 12), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (px0, py0), (px0 + pw, py0 + ph), (60, 60, 60), 1)

    for i, (key_str, desc) in enumerate(_CONTROLS):
        y  = py0 + pad + i * row_h + row_h // 2
        kx = px0 + 8
        cv2.rectangle(frame, (kx, y - 8),      (kx + 18, y + 8), (55, 55, 55), -1)
        cv2.rectangle(frame, (kx, y - 8),      (kx + 18, y + 8), (120, 120, 120), 1)
        _text(frame, key_str, (kx + 3,  y + 5), (210, 210, 210), scale=0.44)
        _text(frame, desc,    (kx + 26, y + 5), (155, 155, 155), scale=0.44)


# ─────────────────────────────────────────────────────────────────────────────
#  Training mode helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_training_overlay(
    frame: np.ndarray,
    idx: int,
    phase: str,
    elapsed: float,
) -> None:
    """Render the guided-training UI on top of the live feed."""
    h, w = frame.shape[:2]
    n_prompts = len(_TRAINING_PROMPTS)

    # Dark banner across the top
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 128), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)

    if phase == "complete" or idx >= n_prompts:
        _text(frame, "Training complete!  Saving data...",
              (w // 2 - 280, 68), (0, 230, 80), scale=1.0, thickness=2)
        return

    emotion, headline, detail = _TRAINING_PROMPTS[idx]
    color = _STATE_COLORS.get(emotion, (255, 255, 255))

    # Progress counter (top-left)
    _text(frame, f"Prompt {idx + 1} / {n_prompts}",
          (14, 36), (180, 180, 180), scale=0.58)

    # Emotion name (centre)
    label_txt = f"TRAINING  —  {emotion.upper()}"
    _text(frame, label_txt, (w // 2 - 190, 44), color, scale=0.88, thickness=2)

    # Instruction lines
    _text(frame, headline, (w // 2 - 190, 76), (240, 240, 240), scale=0.62)
    _text(frame, detail,   (w // 2 - 190, 102), (155, 155, 155), scale=0.48)

    if phase == "countdown":
        remaining = max(0.0, _TRAIN_COUNTDOWN_S - elapsed)
        _text(frame, f"Get ready…  {remaining:.1f}s",
              (w // 2 - 130, h // 2 + 20), (0, 210, 255), scale=1.15, thickness=2)

    elif phase == "collect":
        progress  = min(elapsed / _TRAIN_COLLECT_S, 1.0)
        bar_w     = 500
        bx        = w // 2 - bar_w // 2
        by        = h - 58
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 18), (35, 35, 35), -1)
        cv2.rectangle(frame, (bx, by),
                      (bx + int(bar_w * progress), by + 18), (0, 200, 100), -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 18), (90, 90, 90), 1)
        remaining = max(0.0, _TRAIN_COLLECT_S - elapsed)
        _text(frame, f"Recording…  {remaining:.1f}s",
              (w // 2 - 100, h - 72), (0, 255, 120), scale=0.75, thickness=2)

    # Stop hint (bottom-right)
    _text(frame, "Press T to stop early",
          (w - 262, h - 14), (120, 120, 120), scale=0.48)


def _save_training_data(rows: list) -> None:
    """Write collected training rows to a user file and append to the main dataset."""
    if not rows:
        print("No training data collected.")
        return

    import os
    import pandas as pd
    from datetime import datetime

    df = pd.DataFrame(rows)
    ts       = datetime.now().strftime("%Y%m%d%H%M%S")
    username = os.environ.get("RTMBAS_USER", "user")

    SESSIONS_PATH.mkdir(parents=True, exist_ok=True)
    user_path = SESSIONS_PATH / f"{ts}_training_{username}.csv"
    df.to_csv(user_path, index=False)

    has_header = DATA_PATH.exists() and DATA_PATH.stat().st_size > 0
    df.to_csv(DATA_PATH, mode="a", index=False, header=not has_header)

    print(f"\nTraining saved: {len(df)} rows")
    print(f"  User file : {user_path}")
    print(f"  Dataset   : {DATA_PATH}")
    print("Run  python ml/train.py  to retrain with your new data.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    """Main loop: grab -> push to detector -> draw -> warn -> HUD -> show."""
    ensure_dirs()

    cam = WebcamHandler(
        index=WEBCAM_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=TARGET_FPS,
    )
    if not cam.start():
        return

    # Minimise capture latency: single-frame buffer + MJPG codec
    cam.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    detector  = LandmarkDetector()
    extractor = FeatureExtractor()
    logger    = SessionLogger()
    engine    = InferenceEngine()

    if engine.is_loaded():
        print("Model loaded — live inference active.")
    else:
        print("No model — collecting data.  Run ml/train.py when ready.")

    fps_buf     = deque(maxlen=60)
    last_t      = time.perf_counter()
    frame_n     = 0
    last_result: dict | None = None
    last_label:  str  = "Focused"

    show_legend  = True   # toggled by H

    # Training mode state
    _tm_active   = False
    _tm_idx      = 0
    _tm_phase    = "idle"   # "countdown" | "collect" | "complete"
    _tm_phase_t0 = 0.0
    _tm_rows: list = []

    try:
        while True:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                continue

            frame_n += 1
            now = time.perf_counter()
            fps_buf.append(now - last_t)
            last_t = now
            fps = len(fps_buf) / (sum(fps_buf) or 1e-9)

            # Non-blocking push to detection thread
            detector.update(frame)
            landmarks = detector.get_latest()

            # Feature extraction + inference (both < 0.3 ms — run every frame)
            features = extractor.extract(landmarks, frame.shape)
            result   = engine.predict(features)
            if result:
                last_result = result

            # ── Training mode state machine ───────────────────────────────────
            if _tm_active:
                now_t   = time.perf_counter()
                elapsed = now_t - _tm_phase_t0

                if _tm_phase == "countdown" and elapsed >= _TRAIN_COUNTDOWN_S:
                    _tm_phase    = "collect"
                    _tm_phase_t0 = now_t
                    elapsed      = 0.0

                if _tm_phase == "collect":
                    if frame_n % LOG_EVERY_N == 0:
                        _tm_rows.append(
                            {**features, "label": _TRAINING_PROMPTS[_tm_idx][0]}
                        )
                    elapsed = now_t - _tm_phase_t0
                    if elapsed >= _TRAIN_COLLECT_S:
                        _tm_idx += 1
                        if _tm_idx >= len(_TRAINING_PROMPTS):
                            _tm_phase    = "complete"
                            _tm_phase_t0 = now_t
                            _save_training_data(_tm_rows)
                            _tm_rows = []
                        else:
                            _tm_phase    = "countdown"
                            _tm_phase_t0 = now_t

                elif _tm_phase == "complete":
                    if now_t - _tm_phase_t0 > 3.0:
                        _tm_active = False
                        _tm_phase  = "idle"

            else:
                # Regular heuristic logging — suppressed during training
                if frame_n % LOG_EVERY_N == 0:
                    last_label = generate_heuristic_label(features)
                    logger.log_frame({**features, "label": last_label})

            # ── Render layers ─────────────────────────────────────────────────
            annotated = detector.draw_landmarks(frame, landmarks)
            h, w = annotated.shape[:2]

            # 1 · Face quality warning (vignette / oval / banner / bar)
            quality = _face_quality(landmarks, w, h)
            _draw_face_warning(annotated, quality, frame_n)

            # 2 · HUD — FPS (top left)
            _text(annotated, f"FPS: {fps:.0f}", (10, 28), (0, 220, 0), scale=0.65)

            # 3 · HUD — State or training overlay
            if _tm_active:
                _draw_training_overlay(
                    annotated, _tm_idx, _tm_phase,
                    time.perf_counter() - _tm_phase_t0,
                )
            else:
                label_y = 72 if quality["status"] == "warn" else 28
                if last_result and quality["status"] == "ok":
                    state = last_result["prediction"]
                    conf  = last_result["confidence"]
                    color = _STATE_COLORS.get(state, (255, 255, 255))
                    _text(annotated, f"State: {state}",
                          (w // 2 - 110, label_y), color, scale=0.75, thickness=2)
                    _text(annotated, f"Confidence: {conf:.0%}",
                          (w // 2 - 110, label_y + 26), (210, 210, 210), scale=0.55)
                elif quality["status"] == "ok":
                    _text(annotated, "Collecting Data",
                          (w // 2 - 95, label_y), (200, 200, 200), scale=0.65, thickness=2)
                    _text(annotated, f"Heuristic: {last_label}",
                          (w // 2 - 95, label_y + 26),
                          _STATE_COLORS.get(last_label, (200, 200, 200)), scale=0.55)

            # 4 · HUD — Live metrics (bottom left) — hidden when no face or training
            if quality["status"] == "ok" and not _tm_active:
                ear_v  = features.get("eye_aspect_ratio", 0.0)
                vel_v  = features.get("hand_velocity",    0.0)
                tilt_v = features.get("head_tilt_angle",  0.0)
                _text(annotated, f"EAR:  {ear_v:.3f}",       (10, h - 60))
                _text(annotated, f"Vel:  {vel_v:.1f} px/f",  (10, h - 38))
                _text(annotated, f"Tilt: {tilt_v:.1f} deg",  (10, h - 16))

            # 5 · Controls legend (bottom-right, hidden during training)
            if show_legend and not _tm_active:
                _draw_controls_legend(annotated)

            cv2.imshow("RT-MBAS", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("h") or key == ord("H"):
                show_legend = not show_legend
            elif key == ord("t") or key == ord("T"):
                if not _tm_active:
                    _tm_active   = True
                    _tm_idx      = 0
                    _tm_phase    = "countdown"
                    _tm_phase_t0 = time.perf_counter()
                    _tm_rows     = []
                    print("\n=== TRAINING MODE — press T again to stop early ===")
                    print(f"Set RTMBAS_USER env var to tag files with your name.")
                else:
                    _tm_active = False
                    _tm_phase  = "idle"
                    _save_training_data(_tm_rows)
                    _tm_rows   = []

    finally:
        logger.close()
        detector.close()
        cam.release()
        print(f"Session {logger.session_id} complete.")


if __name__ == "__main__":
    run()
