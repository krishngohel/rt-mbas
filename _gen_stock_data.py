"""
Generate synthetic stock training data for RT-MBAS.

Run once:  python _gen_stock_data.py
Output:    data/stock_dataset.csv  (3500 rows, 500 per emotion)

Values are calibrated against the live FeatureExtractor output ranges so
the stock data is in-distribution with real captured frames.
"""
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_PATH = Path("data/stock_dataset.csv")
ROWS_PER_EMOTION = 500
SEQ_LEN = 30  # one full rolling-window's worth of frames per sequence

# ── Per-emotion feature distributions ─────────────────────────────────────────
# Keys: ear, vel, fm (face_motion), tilt, mo (mouth_open)
# Each value: (mean, std, lo_clip, hi_clip)
# Calibrated against heuristic thresholds in utils/helpers.py
SPECS = {
    "Focused": dict(
        ear=(0.330, 0.035, 0.260, 0.400),
        vel=(2.5,   1.8,   0.0,   8.0),
        fm= (0.6,   0.45,  0.0,   2.5),
        tilt=(0.0,  5.0,  -10.0,  10.0),
        mo= (0.058, 0.018, 0.018, 0.130),
    ),
    "Distracted": dict(
        ear=(0.320, 0.040, 0.240, 0.400),
        vel=(23.0,  6.5,   13.0,  42.0),
        fm= (11.5,  3.0,    7.0,  20.0),
        tilt=(19.0, 7.0,   11.0,  36.0),   # magnitude; sign flipped below
        mo= (0.070, 0.022,  0.030, 0.145),
    ),
    "Stressed": dict(
        ear=(0.200, 0.018, 0.155, 0.235),
        vel=(4.0,   1.5,   1.0,   8.0),
        fm= (0.75,  0.45,  0.1,   2.2),
        tilt=(0.0,  4.0,  -8.0,   8.0),
        mo= (0.040, 0.012, 0.014, 0.080),
    ),
    "Confused": dict(
        ear=(0.300, 0.030, 0.240, 0.360),
        vel=(4.0,   2.0,   1.0,   9.0),
        fm= (2.0,   1.0,   0.4,   4.5),
        tilt=(13.5, 3.0,   9.5,  18.5),    # magnitude; sign flipped below
        mo= (0.062, 0.020, 0.024, 0.110),
    ),
    "Sad": dict(
        ear=(0.245, 0.015, 0.205, 0.280),
        vel=(1.5,   0.9,   0.0,   4.5),
        fm= (0.7,   0.40,  0.05,  2.5),
        tilt=(0.0,  3.0,  -5.0,   5.0),
        mo= (0.035, 0.010, 0.012, 0.075),
    ),
    "Angry": dict(
        ear=(0.205, 0.018, 0.155, 0.240),
        vel=(22.0,  5.0,   13.0,  36.0),
        fm= (10.0,  3.0,    6.0,  18.0),
        tilt=(0.0,  4.0,  -8.0,   8.0),
        mo= (0.080, 0.025,  0.030, 0.155),
    ),
    "Exhausted": dict(
        ear=(0.135, 0.020, 0.068, 0.180),
        vel=(1.0,   0.65,  0.0,   3.0),
        fm= (0.30,  0.20,  0.02,  1.2),
        tilt=(0.0,  3.0,  -5.0,   5.0),
        mo= (0.025, 0.008, 0.008, 0.060),
    ),
}


def _gen_sequence(label: str) -> list:
    s = SPECS[label]

    def _arr(key):
        m, sd, lo, hi = s[key]
        return np.clip(np.abs(RNG.normal(m, sd, SEQ_LEN)) if lo >= 0 else RNG.normal(m, sd, SEQ_LEN), lo, hi)

    ear_arr  = _arr("ear")
    vel_arr  = _arr("vel")
    fm_arr   = _arr("fm")
    mo_arr   = _arr("mo")

    # Tilt: alternate sign for Distracted/Confused to cover both sides
    raw_tilt = _arr("tilt")
    if label in ("Distracted", "Confused"):
        if RNG.random() < 0.5:
            raw_tilt = -raw_tilt
    tilt_arr = raw_tilt

    # Rolling stats across the 30-frame window
    r_ear_mean = float(ear_arr.mean())
    r_ear_std  = max(float(ear_arr.std()), 0.001)
    r_vel_mean = float(vel_arr.mean())
    r_vel_std  = max(float(vel_arr.std()), 0.001)

    # Motion entropy from velocity histogram
    max_v = max(float(vel_arr.max()), 1.0)
    hist, _ = np.histogram(vel_arr, bins=10, range=(0.0, max_v))
    p = hist / hist.sum()
    p_nz = p[p > 0]
    entropy = float(-np.sum(p_nz * np.log(p_nz + 1e-12)))

    rows = []
    for i in range(SEQ_LEN):
        ear  = float(ear_arr[i])
        vel  = float(vel_arr[i])
        prev = float(vel_arr[i - 1]) if i > 0 else vel
        accel = abs(vel - prev)
        gest  = max(0.0, vel * 0.285 + float(RNG.normal(0.0, 0.04)))
        idle  = float(np.mean(vel_arr[: i + 1] < 5.0))

        rows.append({
            "timestamp":              float(i) * 0.0333,
            "eye_aspect_ratio":       round(ear, 5),
            "blink_indicator":        int(ear < 0.21),
            "mouth_open_ratio":       round(float(mo_arr[i]), 5),
            "head_tilt_angle":        round(float(tilt_arr[i]), 3),
            "face_motion_delta":      round(float(fm_arr[i]), 3),
            "hand_velocity":          round(vel, 3),
            "hand_acceleration":      round(accel, 3),
            "gesture_activity_score": round(gest, 4),
            "idle_time_ratio":        round(idle, 4),
            "rolling_mean_ear":       round(r_ear_mean, 5),
            "rolling_std_ear":        round(r_ear_std, 5),
            "rolling_mean_velocity":  round(r_vel_mean, 3),
            "rolling_std_velocity":   round(r_vel_std, 3),
            "motion_entropy":         round(entropy, 4),
            "label":                  label,
        })
    return rows


def main() -> None:
    by_label: dict = defaultdict(list)

    for label in SPECS:
        while len(by_label[label]) < ROWS_PER_EMOTION:
            by_label[label].extend(_gen_sequence(label))

    rows = []
    for label, label_rows in by_label.items():
        rows.extend(label_rows[:ROWS_PER_EMOTION])

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Stock data written: {len(df)} rows -> {OUT_PATH}")
    print(df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
