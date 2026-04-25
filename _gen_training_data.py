"""Generate a realistic synthetic dataset large enough to train on."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

np.random.seed(42)

def make_class(label, n, ear_mu, vel_mu, tilt_mu):
    return pd.DataFrame({
        "timestamp": np.cumsum(np.random.uniform(0.03, 0.04, n)),
        "eye_aspect_ratio":       np.clip(np.random.normal(ear_mu, 0.04, n), 0.05, 0.60),
        "blink_indicator":        (np.random.normal(ear_mu, 0.04, n) < 0.21).astype(int),
        "mouth_open_ratio":       np.clip(np.random.normal(0.05, 0.02, n), 0.0, 0.3),
        "head_tilt_angle":        np.random.normal(tilt_mu, 3.0, n),
        "face_motion_delta":      np.abs(np.random.normal(2.0, 2.0, n)),
        "hand_velocity":          np.abs(np.random.normal(vel_mu, 10.0, n)),
        "hand_acceleration":      np.abs(np.random.normal(2.0, 1.5, n)),
        "gesture_activity_score": np.abs(np.random.normal(40.0, 15.0, n)),
        "idle_time_ratio":        np.clip(np.random.normal(0.5, 0.2, n), 0.0, 1.0),
        "rolling_mean_ear":       np.clip(np.random.normal(ear_mu, 0.03, n), 0.05, 0.60),
        "rolling_std_ear":        np.abs(np.random.normal(0.02, 0.01, n)),
        "rolling_mean_velocity":  np.abs(np.random.normal(vel_mu, 8.0, n)),
        "rolling_std_velocity":   np.abs(np.random.normal(5.0, 2.0, n)),
        "motion_entropy":         np.abs(np.random.normal(1.2, 0.4, n)),
        "label":                  label,
    })

focused    = make_class("Focused",    150, ear_mu=0.28, vel_mu=8.0,  tilt_mu=2.0)
distracted = make_class("Distracted", 150, ear_mu=0.27, vel_mu=40.0, tilt_mu=12.0)
stressed   = make_class("Stressed",   150, ear_mu=0.18, vel_mu=3.0,  tilt_mu=1.0)

df = pd.concat([focused, distracted, stressed], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

out = Path("data/dataset.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Written {len(df)} rows to {out}")
print(df["label"].value_counts().to_string())
