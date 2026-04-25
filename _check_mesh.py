"""Verify face mesh connection groups and draw on a synthetic 478-landmark frame."""
import sys, time
sys.path.insert(0, '.')
import numpy as np, cv2
from app.landmarks import (
    LandmarkDetector,
    FACE_OVAL, FACE_LEFT_EYE, FACE_RIGHT_EYE,
    FACE_LEFT_EYEBROW, FACE_RIGHT_EYEBROW,
    FACE_NOSE, FACE_LIPS_OUTER, FACE_LIPS_INNER,
    FACE_RIGHT_IRIS, FACE_LEFT_IRIS,
)

groups = {
    "FACE_OVAL":          FACE_OVAL,
    "FACE_LEFT_EYE":      FACE_LEFT_EYE,
    "FACE_RIGHT_EYE":     FACE_RIGHT_EYE,
    "FACE_LEFT_EYEBROW":  FACE_LEFT_EYEBROW,
    "FACE_RIGHT_EYEBROW": FACE_RIGHT_EYEBROW,
    "FACE_NOSE":          FACE_NOSE,
    "FACE_LIPS_OUTER":    FACE_LIPS_OUTER,
    "FACE_LIPS_INNER":    FACE_LIPS_INNER,
}

print("Connection groups:")
total = 0
for name, g in groups.items():
    top = max(max(a, b) for a, b in g)
    print(f"  {name:22s}  {len(g):3d} pairs  max_idx={top}")
    total += len(g)

iris_pairs = list(FACE_RIGHT_IRIS) + list(FACE_LEFT_IRIS)
print(f"  {'IRIS (both)':22s}  {len(iris_pairs):3d} pairs  max_idx={max(max(a,b) for a,b in iris_pairs)}")
print(f"  {'TOTAL':22s}  {total+len(iris_pairs):3d} pairs")

# All indices referenced are within 0-477
all_idx = set()
for g in groups.values():
    for a, b in g:
        all_idx.add(a); all_idx.add(b)
for a, b in iris_pairs:
    all_idx.add(a); all_idx.add(b)
print(f"\n  All landmark indices in range [0, 477]: {max(all_idx) <= 477}")
print(f"  Unique landmarks referenced: {len(all_idx)}")

# Synthetic draw test — 478 face landmarks + 21 hand landmarks
print("\nDraw test on synthetic 478-pt face + 2 hands ...")
face_lm = [(0.4 + 0.02 * (i % 10), 0.5 + 0.02 * (i // 10), 0.0) for i in range(478)]
hand_lm = [(0.8 + 0.01 * i, 0.6 + 0.01 * i, 0.0) for i in range(21)]
result = {"face": face_lm, "left_hand": hand_lm, "right_hand": None}

blank = np.zeros((720, 1280, 3), dtype=np.uint8)
det = LandmarkDetector()
time.sleep(0.1)   # let thread start

annotated = det.draw_landmarks(blank, result)
assert annotated.shape == blank.shape, "Shape mismatch"
assert not np.array_equal(annotated, blank), "Nothing was drawn"
print("  draw_landmarks: OK")

# Timing
N = 50
t0 = time.perf_counter()
for _ in range(N):
    det.draw_landmarks(blank, result)
ms = (time.perf_counter() - t0) / N * 1000
print(f"  draw_landmarks avg time: {ms:.2f} ms ({1000/ms:.0f} FPS budget)")

det.close()
print("\nAll checks passed.")
