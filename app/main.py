"""
RT-MBAS — Real-Time Multimodal Behavioral Analytics System
Run with:  python app/main.py
Controls:  Q = quit
"""
import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.camera import WebcamHandler
from app.landmarks import LandmarkDetector
from app.features import FeatureExtractor
from app.inference import InferenceEngine
from app.config import WEBCAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS
from utils.helpers import generate_heuristic_label, ensure_dirs
from utils.logger import SessionLogger

_STATE_COLORS = {
    "Focused":    (0, 220, 0),
    "Distracted": (0, 165, 255),
    "Stressed":   (0, 0, 220),
}


def _put_text(
    frame: np.ndarray,
    text: str,
    pos: tuple,
    color: tuple = (255, 255, 255),
    scale: float = 0.6,
    thickness: int = 1,
) -> None:
    """Draw text with a dark outline so it is readable on any background."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def run() -> None:
    """Main real-time loop: capture → detect → extract → log → infer → display."""
    ensure_dirs()

    cam = WebcamHandler(
        index=WEBCAM_INDEX,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=TARGET_FPS,
    )
    if not cam.start():
        return

    detector = LandmarkDetector()
    extractor = FeatureExtractor()
    logger = SessionLogger()
    engine = InferenceEngine()

    if engine.is_loaded():
        print("Model loaded — running with live inference.")
    else:
        print("No model found — collecting data in heuristic-label mode.")
        print("Run  python ml/train.py  after collecting enough data.")

    frame_count = 0
    fps_display = 0.0
    fps_timer = time.time()

    try:
        while True:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                continue

            # ── FPS counter ───────────────────────────────────────────────────
            frame_count += 1
            now = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_timer = now

            # ── Core pipeline ─────────────────────────────────────────────────
            landmarks = detector.detect(frame)
            annotated = detector.draw_landmarks(frame, landmarks)
            features = extractor.extract(landmarks, frame.shape)

            heuristic = generate_heuristic_label(features)
            logger.log_frame({**features, "label": heuristic})

            result = engine.predict(features)

            # ── HUD overlay ───────────────────────────────────────────────────
            h, w = annotated.shape[:2]

            # FPS — top left, green
            _put_text(annotated, f"FPS: {fps_display:.1f}", (10, 30), (0, 220, 0))

            # State — top centre
            if result:
                state = result["prediction"]
                conf_str = f"{result['confidence']:.0%}"
                color = _STATE_COLORS.get(state, (255, 255, 255))
                _put_text(
                    annotated,
                    f"State: {state}",
                    (w // 2 - 100, 30),
                    color, scale=0.75, thickness=2,
                )
                _put_text(
                    annotated,
                    f"Confidence: {conf_str}",
                    (w // 2 - 100, 58),
                    (220, 220, 220), scale=0.55,
                )
            else:
                _put_text(
                    annotated,
                    "No Model - Collecting Data",
                    (w // 2 - 160, 30),
                    (200, 200, 200), scale=0.65, thickness=2,
                )
                _put_text(
                    annotated,
                    f"Heuristic: {heuristic}",
                    (w // 2 - 160, 58),
                    _STATE_COLORS.get(heuristic, (220, 220, 220)), scale=0.55,
                )

            # EAR + hand velocity — bottom left
            ear_val = features.get("eye_aspect_ratio", 0.0)
            vel_val = features.get("hand_velocity", 0.0)
            _put_text(annotated, f"EAR: {ear_val:.3f}", (10, h - 40))
            _put_text(annotated, f"Hand vel: {vel_val:.1f} px/f", (10, h - 15))

            cv2.imshow("RT-MBAS", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        logger.close()
        detector.close()
        cam.release()
        print(f"Session {logger.session_id} complete.")


if __name__ == "__main__":
    run()
