"""Smoke-test every face warning code path without a camera."""
import sys
sys.path.insert(0, '.')
import numpy as np, cv2

# Import the private helpers directly from main
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("main", "app/main.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_face_quality    = mod._face_quality
_draw_face_warning = mod._draw_face_warning
_draw_guide_oval = mod._draw_guide_oval
_draw_distance_bar = mod._draw_distance_bar

W, H = 1280, 720

def make_face(iod_fraction, nose_x=0.5, nose_y=0.5, n=478):
    """Build a minimal fake landmark list with controlled IOD."""
    lm = [(0.5, 0.5, 0.0)] * n
    half = (iod_fraction * W) / 2 / W
    lm = list(lm)
    lm[33]  = (0.5 - half, 0.5, 0.0)   # right outer eye corner
    lm[263] = (0.5 + half, 0.5, 0.0)   # left  outer eye corner
    lm[1]   = (nose_x, nose_y, 0.0)    # nose tip
    return lm

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []

def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f" -- {detail}" if detail else ""))
        failures.append(name)

print("\n[_face_quality]")

# No face
q = _face_quality(None, W, H)
check("None face -> critical", q["status"] == "critical")
check("None face -> iod_ratio 0", q["iod_ratio"] == 0.0)

# Good distance  (IOD = 13 % of width)
q = _face_quality(make_face(0.13), W, H)
check("Good IOD -> ok",          q["status"] == "ok")
check("Good IOD -> no message",  q["message"] == "")
check("Good IOD -> iod_ratio",   0.12 < q["iod_ratio"] < 0.15)

# Warning distance (IOD = 8 % of width, within 6.5-10%)
q = _face_quality(make_face(0.08), W, H)
check("Warn IOD -> warn",        q["status"] == "warn")

# Too far (IOD = 4 % of width)
q = _face_quality(make_face(0.04), W, H)
check("Far IOD -> critical",     q["status"] == "critical")
check("Far IOD message correct", "closer" in q["message"].lower())

# Face near edge (nose_x = 0.03 — 3% from left)
q = _face_quality(make_face(0.13, nose_x=0.03), W, H)
check("Near edge -> warn",       q["status"] == "warn")
check("Near edge message",       "center" in q["message"].lower() or "Center" in q["message"])

print("\n[_draw_face_warning — render without crash]")
blank = np.zeros((H, W, 3), dtype=np.uint8)

# Critical path (no face)
try:
    f = blank.copy()
    _draw_face_warning(f, {"status": "critical", "message": "No face detected", "iod_ratio": 0.0}, frame_n=10)
    _draw_face_warning(f, {"status": "critical", "message": "No face detected", "iod_ratio": 0.0}, frame_n=30)
    check("critical blink=off renders", not np.array_equal(f, blank))
except Exception as e:
    check("critical path", False, str(e))

# Warn path
try:
    f = blank.copy()
    _draw_face_warning(f, {"status": "warn", "message": "Move closer", "iod_ratio": 0.07}, frame_n=10)
    check("warn path renders", not np.array_equal(f, blank))
except Exception as e:
    check("warn path", False, str(e))

# OK path — frame must NOT be modified
try:
    f = blank.copy()
    _draw_face_warning(f, {"status": "ok", "message": "", "iod_ratio": 0.14}, frame_n=10)
    check("ok path: no modification", np.array_equal(f, blank))
except Exception as e:
    check("ok path", False, str(e))

# Distance bar rendering
try:
    f = blank.copy()
    _draw_distance_bar(f, 0.0)
    _draw_distance_bar(f, 0.065)
    _draw_distance_bar(f, 0.12)
    check("distance bar renders", not np.array_equal(f, blank))
except Exception as e:
    check("distance bar", False, str(e))

# Guide oval (both blink states)
try:
    for fn in [10, 30]:
        f = blank.copy()
        _draw_guide_oval(f, fn)
    check("guide oval renders", not np.array_equal(f, blank))
except Exception as e:
    check("guide oval", False, str(e))

print("\n[IOD threshold sanity]")
check("IOD_GOOD > IOD_WARN", mod._IOD_GOOD > mod._IOD_WARN)
check("IOD_WARN > 0",        mod._IOD_WARN > 0)

print()
if failures:
    print(f"\033[91m{len(failures)} FAILED: {failures}\033[0m")
    sys.exit(1)
else:
    print("\033[92mAll checks passed.\033[0m")
