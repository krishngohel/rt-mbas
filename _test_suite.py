"""
RT-MBAS internal test suite — runs without a camera.
Usage: python _test_suite.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


# ── helpers ───────────────────────────────────────────────────────────────────
print("\n[helpers]")
from utils.helpers import (
    euclidean_distance, normalize_landmark, smooth_predictions,
    exponential_smoothing, generate_heuristic_label, ensure_dirs,
)
from collections import deque

check("euclidean_distance 3-4-5", abs(euclidean_distance((0,0,0),(3,4,0)) - 5.0) < 1e-9)
check("euclidean_distance 2D", abs(euclidean_distance((0,0),(3,4)) - 5.0) < 1e-9)
check("normalize_landmark", normalize_landmark((0.5, 0.5, 0.0), 640, 480) == (320.0, 240.0, 0.0))

buf = deque(["Focused"]*8 + ["Distracted"]*2, maxlen=10)
check("smooth_predictions majority", smooth_predictions(buf, 10) == "Focused")
check("smooth_predictions empty", smooth_predictions(deque(), 10) is None)

check("exp_smoothing None seed", exponential_smoothing(1.0, None) == 1.0)
check("exp_smoothing value", abs(exponential_smoothing(0.0, 1.0, alpha=0.5) - 0.5) < 1e-9)

stressed = generate_heuristic_label({"eye_aspect_ratio": 0.15, "hand_velocity": 2.0, "face_motion_delta": 0.0})
distracted = generate_heuristic_label({"eye_aspect_ratio": 0.30, "hand_velocity": 50.0, "face_motion_delta": 0.0})
focused = generate_heuristic_label({"eye_aspect_ratio": 0.30, "hand_velocity": 5.0, "face_motion_delta": 0.0})
check("heuristic Stressed", stressed == "Stressed", stressed)
check("heuristic Distracted", distracted == "Distracted", distracted)
check("heuristic Focused", focused == "Focused", focused)

ensure_dirs()
check("ensure_dirs created data/", Path("data").is_dir())
check("ensure_dirs created data/sessions/", Path("data/sessions").is_dir())
check("ensure_dirs created ml/", Path("ml").is_dir())
check("ensure_dirs created analysis/", Path("analysis").is_dir())


# ── config ────────────────────────────────────────────────────────────────────
print("\n[config]")
from app.config import (
    WEBCAM_INDEX, TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT,
    FEATURE_WINDOW_SIZE, SMOOTHING_WINDOW,
    DATA_PATH, SESSIONS_PATH, MODEL_PATH, SCALER_PATH,
    FACE_LANDMARKER_PATH, HAND_LANDMARKER_PATH, LABELS,
)
check("WEBCAM_INDEX == 0", WEBCAM_INDEX == 0)
check("TARGET_FPS == 60", TARGET_FPS == 60)
check("FRAME_WIDTH == 1280", FRAME_WIDTH == 1280)
check("FEATURE_WINDOW_SIZE == 30", FEATURE_WINDOW_SIZE == 30)
check("SMOOTHING_WINDOW == 10", SMOOTHING_WINDOW == 10)
check("LABELS correct", LABELS == ["Focused", "Distracted", "Stressed"])
check("DATA_PATH is Path", isinstance(DATA_PATH, Path))
check("MODEL_PATH suffix", MODEL_PATH.suffix == ".pkl")
check("FACE_LANDMARKER_PATH suffix", FACE_LANDMARKER_PATH.suffix == ".task")


# ── FeatureExtractor (synthetic landmarks) ────────────────────────────────────
print("\n[FeatureExtractor — synthetic landmarks]")
from app.features import FeatureExtractor

# Build minimal fake landmark list: 478 entries, all at centre of frame
_N_FACE = 478
_fake_face = [(0.5, 0.5, 0.0)] * _N_FACE
_fake_hand = [(0.5, 0.5, 0.0)] * 21

frame_shape = (720, 1280, 3)
extractor = FeatureExtractor()

# Frame with face + hand
lm1 = {"face": _fake_face, "left_hand": None, "right_hand": _fake_hand}
fv1 = extractor.extract(lm1, frame_shape)
check("extract returns dict", isinstance(fv1, dict))
check("timestamp present", "timestamp" in fv1)
check("eye_aspect_ratio present", "eye_aspect_ratio" in fv1)
check("hand_velocity present", "hand_velocity" in fv1)
check("motion_entropy present", "motion_entropy" in fv1)
check("rolling_mean_velocity key", "rolling_mean_velocity" in fv1)

# Frame with no landmarks
lm0 = {"face": None, "left_hand": None, "right_hand": None}
fv0 = extractor.extract(lm0, frame_shape)
check("no-landmark EAR == 0", fv0["eye_aspect_ratio"] == 0.0)
check("no-landmark velocity == 0", fv0["hand_velocity"] == 0.0)

# EAR should be numeric and in sane range
check("EAR >= 0", fv1["eye_aspect_ratio"] >= 0.0)
check("mouth_open_ratio >= 0", fv1["mouth_open_ratio"] >= 0.0)
check("idle_time_ratio in [0,1]", 0.0 <= fv1["idle_time_ratio"] <= 1.0)

# Two frames with moving wrist → velocity > 0 on second call
extractor2 = FeatureExtractor()
lm_a = {"face": None, "left_hand": None, "right_hand": [(0.1, 0.1, 0.0)] * 21}
lm_b = {"face": None, "left_hand": None, "right_hand": [(0.9, 0.9, 0.0)] * 21}
extractor2.extract(lm_a, frame_shape)
fv_b = extractor2.extract(lm_b, frame_shape)
check("hand_velocity > 0 after movement", fv_b["hand_velocity"] > 0.0,
      f"got {fv_b['hand_velocity']}")


# ── LandmarkDetector (model files) ────────────────────────────────────────────
print("\n[LandmarkDetector — model download / init]")
try:
    from app.landmarks import LandmarkDetector
    det = LandmarkDetector()
    check("FaceLandmarker created", det._face is not None)
    check("HandLandmarker created", det._hands is not None)
    check("face_landmarker.task exists", FACE_LANDMARKER_PATH.exists())
    check("hand_landmarker.task exists", HAND_LANDMARKER_PATH.exists())

    # update() + get_latest() on a blank frame must not raise
    import cv2, time as _time
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    det.update(blank)
    _time.sleep(0.15)          # allow detection thread one cycle
    result = det.get_latest()
    check("get_latest returns dict", isinstance(result, dict))
    check("result has 'face' key", "face" in result)
    check("result has 'left_hand' key", "left_hand" in result)
    check("result has 'right_hand' key", "right_hand" in result)
    check("blank frame -> face is None", result["face"] is None)

    # draw_landmarks must not raise
    annotated = det.draw_landmarks(blank, result)
    check("draw_landmarks returns ndarray", isinstance(annotated, np.ndarray))
    check("draw_landmarks same shape", annotated.shape == blank.shape)

    det.close()
    check("close() ran without error", True)
except Exception as e:
    check("LandmarkDetector init/detect/draw", False, str(e))


# ── InferenceEngine ────────────────────────────────────────────────────────────
print("\n[InferenceEngine]")
from app.inference import InferenceEngine
from app.config import MODEL_PATH, SCALER_PATH

if MODEL_PATH.exists():
    # Model is trained — verify it loads and predicts cleanly
    engine = InferenceEngine()
    check("is_loaded() True when model present", engine.is_loaded())
    result_inf = engine.predict(fv1)
    check("predict() returns dict", isinstance(result_inf, dict))
    check("predict() has 'prediction' key", "prediction" in (result_inf or {}))
    check("predict() has 'confidence' key", "confidence" in (result_inf or {}))
    # Test no-model path by using a non-existent path
    import joblib
    _tmp_model = MODEL_PATH.with_suffix(".pkl.bak")
    _tmp_scaler = SCALER_PATH.with_suffix(".pkl.bak")
    MODEL_PATH.rename(_tmp_model)
    SCALER_PATH.rename(_tmp_scaler)
    engine_none = InferenceEngine()
    check("is_loaded() False when model missing", not engine_none.is_loaded())
    check("predict() returns None when not loaded", engine_none.predict(fv1) is None)
    _tmp_model.rename(MODEL_PATH)
    _tmp_scaler.rename(SCALER_PATH)
else:
    engine = InferenceEngine()
    check("is_loaded() False when no model", not engine.is_loaded())
    check("predict() returns None when not loaded", engine.predict(fv1) is None)
    check("predict() returns dict skipped (no model)", True)


# ── DataPreprocessor ──────────────────────────────────────────────────────────
print("\n[DataPreprocessor]")
from ml.preprocess import DataPreprocessor

# Build a small synthetic dataset
np.random.seed(0)
n = 60
synthetic = pd.DataFrame({
    "timestamp": np.arange(n, dtype=float),
    "eye_aspect_ratio": np.random.uniform(0.15, 0.35, n),
    "blink_indicator": np.random.randint(0, 2, n),
    "mouth_open_ratio": np.random.uniform(0.01, 0.15, n),
    "head_tilt_angle": np.random.uniform(-10, 10, n),
    "face_motion_delta": np.random.uniform(0, 5, n),
    "hand_velocity": np.random.uniform(0, 50, n),
    "hand_acceleration": np.random.uniform(0, 10, n),
    "gesture_activity_score": np.random.uniform(0, 100, n),
    "idle_time_ratio": np.random.uniform(0, 1, n),
    "rolling_mean_ear": np.random.uniform(0.2, 0.35, n),
    "rolling_std_ear": np.random.uniform(0, 0.05, n),
    "rolling_mean_velocity": np.random.uniform(0, 30, n),
    "rolling_std_velocity": np.random.uniform(0, 10, n),
    "motion_entropy": np.random.uniform(0, 2, n),
    "label": (["Focused"] * 20) + (["Distracted"] * 20) + (["Stressed"] * 20),
})

prep = DataPreprocessor()

# Save and reload to test load()
tmp_path = Path("data/_test_synthetic.csv")
synthetic.to_csv(tmp_path, index=False)
loaded = prep.load(tmp_path)
check("load() returns DataFrame", isinstance(loaded, pd.DataFrame))
check("load() has correct rows", len(loaded) == n)

engineered = prep.engineer_features(loaded)
check("engineer_features adds lag cols", "eye_aspect_ratio_lag1" in engineered.columns)
check("engineer_features adds lag5", "hand_velocity_lag5" in engineered.columns)

feat_cols = prep.get_feature_columns(engineered)
check("get_feature_columns excludes timestamp", "timestamp" not in feat_cols)
check("get_feature_columns excludes label", "label" not in feat_cols)
check("get_feature_columns has lag cols", "eye_aspect_ratio_lag1" in feat_cols)

X_train, X_test, y_train, y_test = prep.split(engineered)
check("split returns 4 objects", X_train is not None)
check("train size ~80%", abs(len(X_train) / n - 0.8) < 0.05)
check("test size ~20%", abs(len(X_test) / n - 0.2) < 0.05)
check("all labels in train", set(y_train.unique()) == {"Focused", "Distracted", "Stressed"})

X_tr_s, X_te_s = prep.scale(X_train, X_test)
check("scale returns ndarrays", isinstance(X_tr_s, np.ndarray))
check("scale shape matches", X_tr_s.shape == (len(X_train), len(feat_cols)))
check("scaler.pkl written", SCALER_PATH.exists())

tmp_path.unlink(missing_ok=True)


# ── SessionLogger ─────────────────────────────────────────────────────────────
print("\n[SessionLogger]")
# Temporarily rename dataset.csv so logger creates a fresh one
real_csv = Path("data/dataset.csv")
backup = Path("data/_dataset_backup.csv")
if real_csv.exists():
    real_csv.rename(backup)

from utils.logger import SessionLogger
logger = SessionLogger()
row = {**fv1, "label": "Focused"}
logger.log_frame(row)
logger.log_frame({**fv1, "label": "Distracted"})
logger.close()

check("dataset.csv created", real_csv.exists())
df_log = pd.read_csv(real_csv)
check("dataset.csv has 2 rows", len(df_log) == 2)
check("dataset.csv has label col", "label" in df_log.columns)
check("session csv created", logger._session_path.exists())
df_sess = pd.read_csv(logger._session_path)
check("session csv has 2 rows", len(df_sess) == 2)

# Restore
real_csv.unlink(missing_ok=True)
if backup.exists():
    backup.rename(real_csv)


# ── analysis/plots (no display) ───────────────────────────────────────────────
print("\n[analysis.plots — save-to-file]")
from analysis.plots import (
    plot_feature_timeseries, plot_correlation_heatmap,
    plot_distributions, plot_label_distribution,
    plot_feature_importance,
)
from sklearn.ensemble import RandomForestClassifier

plot_df = synthetic.copy()
_out = Path("analysis/_test_plot.png")

try:
    plot_feature_timeseries(plot_df, str(_out))
    check("plot_feature_timeseries saved", _out.exists())
    _out.unlink(missing_ok=True)
except Exception as e:
    check("plot_feature_timeseries", False, str(e))

try:
    plot_correlation_heatmap(plot_df, str(_out))
    check("plot_correlation_heatmap saved", _out.exists())
    _out.unlink(missing_ok=True)
except Exception as e:
    check("plot_correlation_heatmap", False, str(e))

try:
    plot_distributions(plot_df, str(_out))
    check("plot_distributions saved", _out.exists())
    _out.unlink(missing_ok=True)
except Exception as e:
    check("plot_distributions", False, str(e))

try:
    plot_label_distribution(plot_df, str(_out))
    check("plot_label_distribution saved", _out.exists())
    _out.unlink(missing_ok=True)
except Exception as e:
    check("plot_label_distribution", False, str(e))

try:
    # Quick RF for importance test
    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    _feat_cols_s = [c for c in feat_cols if c in synthetic.columns]
    rf.fit(synthetic[_feat_cols_s].values, synthetic["label"].values)
    plot_feature_importance(rf, _feat_cols_s, str(_out))
    check("plot_feature_importance saved", _out.exists())
    _out.unlink(missing_ok=True)
except Exception as e:
    check("plot_feature_importance", False, str(e))


# ── Summary ───────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"\033[91m{len(failures)} test(s) FAILED: {failures}\033[0m")
    sys.exit(1)
else:
    print("\033[92mAll tests passed.\033[0m")
