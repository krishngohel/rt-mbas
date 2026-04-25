"""Verify the three changes: distance threshold, 7-state labels, hand-covers-face."""
import sys, math
sys.path.insert(0, '.')

import importlib.util
spec = importlib.util.spec_from_file_location("main", "app/main.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from utils.helpers import generate_heuristic_label
from app.config import LABELS

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
fails = []

def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f" -- {detail}" if detail else ""))
        fails.append(name)

# ── 1. Distance thresholds reduced by 7.5 % ──────────────────────────────────
print("\n[Distance thresholds (-7.5 %)]")
check("IOD_GOOD == 0.093", abs(mod._IOD_GOOD - 0.093) < 1e-6, mod._IOD_GOOD)
check("IOD_WARN == 0.060", abs(mod._IOD_WARN - 0.060) < 1e-6, mod._IOD_WARN)
check("IOD_GOOD < 0.10 (further than before)", mod._IOD_GOOD < 0.10)
check("IOD_WARN < 0.065 (further than before)", mod._IOD_WARN < 0.065)

# ── 2. 7-state label system ───────────────────────────────────────────────────
print("\n[7-state label set]")
expected = {"Focused","Distracted","Stressed","Confused","Sad","Angry","Exhausted"}
check("config.LABELS has 7 states", set(LABELS) == expected, LABELS)
check("STATE_COLORS covers all 7", set(mod._STATE_COLORS.keys()) == expected)

print("\n[Heuristic — 7 distinct states]")
def feat(**kw):
    base = {"eye_aspect_ratio":0.30,"hand_velocity":2.0,
            "face_motion_delta":1.0,"head_tilt_angle":2.0}
    base.update(kw)
    return base

check("Focused  (normal)", generate_heuristic_label(feat()) == "Focused")
check("Exhausted (EAR<0.17, still)",
      generate_heuristic_label(feat(eye_aspect_ratio=0.14, hand_velocity=1, face_motion_delta=0.5)) == "Exhausted")
check("Angry    (squint+high vel)",
      generate_heuristic_label(feat(eye_aspect_ratio=0.20, hand_velocity=20)) == "Angry")
check("Stressed (squint, still, forward)",
      generate_heuristic_label(feat(eye_aspect_ratio=0.20, hand_velocity=4, head_tilt_angle=3)) == "Stressed")
check("Sad      (low EAR, near-zero motion)",
      generate_heuristic_label(feat(eye_aspect_ratio=0.25, hand_velocity=2, face_motion_delta=1.0)) == "Sad")
check("Distracted (high vel)",
      generate_heuristic_label(feat(hand_velocity=30)) == "Distracted")
check("Distracted (large tilt >18)",
      generate_heuristic_label(feat(head_tilt_angle=20)) == "Distracted")
check("Confused (tilt 10-18)",
      generate_heuristic_label(feat(head_tilt_angle=14)) == "Confused")
check("No face (EAR=0) -> Focused",
      generate_heuristic_label(feat(eye_aspect_ratio=0.0)) == "Focused")

# ── 3. Hand-covers-face suppresses warning ────────────────────────────────────
print("\n[Hand-covers-face logic]")
W, H = 1280, 720

# No face, no hands -> critical
q = mod._face_quality({"face": None, "left_hand": None, "right_hand": None}, W, H)
check("No face, no hands -> critical", q["status"] == "critical")

# No face, left hand visible -> ok (suppressed)
fake_hand = [(0.5, 0.5, 0.0)] * 21
q = mod._face_quality({"face": None, "left_hand": fake_hand, "right_hand": None}, W, H)
check("No face, left hand  -> ok (suppressed)", q["status"] == "ok")

# No face, right hand visible -> ok (suppressed)
q = mod._face_quality({"face": None, "left_hand": None, "right_hand": fake_hand}, W, H)
check("No face, right hand -> ok (suppressed)", q["status"] == "ok")

# Both hands, no face -> ok (suppressed)
q = mod._face_quality({"face": None, "left_hand": fake_hand, "right_hand": fake_hand}, W, H)
check("No face, both hands -> ok (suppressed)", q["status"] == "ok")

# Good face, hand present -> still assesses face properly
def make_face(iod_frac, nx=0.5, ny=0.5, n=478):
    lm = [(0.5, 0.5, 0.0)] * n
    lm = list(lm)
    half = (iod_frac * W) / 2 / W
    lm[33]  = (0.5 - half, 0.5, 0.0)
    lm[263] = (0.5 + half, 0.5, 0.0)
    lm[1]   = (nx, ny, 0.0)
    return lm

q = mod._face_quality({"face": make_face(0.13), "left_hand": fake_hand, "right_hand": None}, W, H)
check("Good face + hand    -> ok", q["status"] == "ok")

q = mod._face_quality({"face": make_face(0.04), "left_hand": fake_hand, "right_hand": None}, W, H)
check("Far face + hand     -> critical (distance, not hand)", q["status"] == "critical")

print()
if fails:
    print(f"\033[91m{len(fails)} FAILED: {fails}\033[0m")
    sys.exit(1)
else:
    print("\033[92mAll checks passed.\033[0m")
