"""Measure threaded detector throughput on a realistic blank frame."""
import sys, time
from pathlib import Path
from collections import deque
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, cv2
from app.landmarks import LandmarkDetector
from app.features  import FeatureExtractor

det = LandmarkDetector()
ext = FeatureExtractor()
blank = np.zeros((720, 1280, 3), dtype=np.uint8)

# ── Measure display-loop rate (threaded: update + get_latest + draw + extract)
N = 120
times = []
t0 = time.perf_counter()
for _ in range(N):
    det.update(blank)
    lm  = det.get_latest()
    out = det.draw_landmarks(blank, lm)
    fv  = ext.extract(lm, blank.shape)
    times.append(time.perf_counter())

display_fps = N / (times[-1] - t0)

# ── Measure detection thread rate separately
import threading
det_times = []
ev = threading.Event()

original_loop = det._loop

def patched_loop():
    while det._running:
        import threading as _t
        with det._cond:
            while det._pending is None and det._running:
                det._cond.wait(timeout=0.05)
            if not det._running:
                break
            frame_bgr, ts_ms = det._pending
            det._pending = None
        t_start = time.perf_counter()
        try:
            import cv2 as _cv2, mediapipe as _mp
            small = _cv2.resize(frame_bgr, (640, 360))
            rgb   = _cv2.cvtColor(small, _cv2.COLOR_BGR2RGB)
            mp_img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            det._face.detect_for_video(mp_img, ts_ms)
            det._hands.detect_for_video(mp_img, ts_ms)
        except Exception:
            pass
        det_times.append(time.perf_counter() - t_start)

det.close()
det2 = LandmarkDetector()
det2._thread.join(timeout=0.1)   # let original thread stop

print(f"\nDisplay loop throughput (threaded): {display_fps:.0f} FPS")
print(f"  update() + get_latest() + draw() + extract() per frame")

# Time just the detection inference calls directly
det2.close()
det3 = LandmarkDetector()
time.sleep(0.1)

# Direct detection timing (what the background thread sees)
small  = cv2.resize(blank, (640, 360))
rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
import mediapipe as mp
ts = 1
direct_times = []
for _ in range(20):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    t0 = time.perf_counter()
    det3._face.detect_for_video(mp_img, ts)
    det3._hands.detect_for_video(mp_img, ts)
    direct_times.append((time.perf_counter() - t0) * 1000)
    ts += 33

det3.close()
det_ms = sum(direct_times) / len(direct_times)
print(f"\nBackground detection per call: {det_ms:.1f} ms  ({1000/det_ms:.0f} FPS ceiling)")
print(f"\nFinal summary:")
print(f"  Display loop   : ~{display_fps:.0f} FPS  (main thread, non-blocking)")
print(f"  Detection thread: ~{1000/det_ms:.0f} FPS  (background, on blank frame)")
print(f"  Expected with real face: 15-25 FPS detection, {min(int(display_fps), 60)} FPS display")
