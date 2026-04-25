"""
Real-time inference engine for RT-MBAS.

Loads a trained RandomForest model and scaler, reconstructs lag features
on-the-fly from a rolling history buffer, and returns smoothed predictions.
"""
import numpy as np
import pandas as pd
import joblib
from collections import deque, Counter

from app.config import MODEL_PATH, SCALER_PATH, SMOOTHING_WINDOW


class InferenceEngine:
    """
    Loads model.pkl + scaler.pkl and runs smoothed per-frame inference.

    Lag features (lag-1, lag-5) are reconstructed from an internal history
    deque so no preprocessing step is required at runtime.
    """

    def __init__(self):
        """Load model and scaler; initialize smoothing buffer and feature history."""
        self._model = None
        self._scaler = None
        self._feature_names: list | None = None
        self._prediction_buffer: deque = deque(maxlen=SMOOTHING_WINDOW)
        self._feature_history: deque = deque(maxlen=10)
        self._loaded: bool = False
        self._load()

    def _load(self) -> None:
        """Attempt to load model.pkl and scaler.pkl from the configured paths."""
        try:
            self._model = joblib.load(MODEL_PATH)
            self._scaler = joblib.load(SCALER_PATH)
            if hasattr(self._model, "feature_names_in_"):
                self._feature_names = list(self._model.feature_names_in_)
            self._loaded = True
        except Exception:
            self._loaded = False

    def _build_row(self, feature_dict: dict) -> dict:
        """
        Augment the current feature dict with lag-1 and lag-5 values for the
        columns the model was trained on (eye_aspect_ratio, hand_velocity).
        """
        row = dict(feature_dict)
        history = list(self._feature_history)

        lag1 = history[-1] if len(history) >= 1 else feature_dict
        lag5 = history[-5] if len(history) >= 5 else feature_dict

        for col in ("eye_aspect_ratio", "hand_velocity"):
            row[f"{col}_lag1"] = lag1.get(col, 0.0)
            row[f"{col}_lag5"] = lag5.get(col, 0.0)

        return row

    def predict(self, feature_dict: dict) -> dict | None:
        """
        Run inference on one frame's feature dict.

        Args:
            feature_dict: output of FeatureExtractor.extract()

        Returns:
            dict with keys:
              "prediction"     — majority-voted label over smoothing window
              "confidence"     — probability of the raw (unsmoothed) top class
              "raw_prediction" — label from the current frame only
            or None if no model is loaded.
        """
        if not self._loaded:
            return None

        self._feature_history.append(feature_dict)
        row = self._build_row(feature_dict)

        cols = self._feature_names or [
            k for k in row if k not in ("timestamp", "label")
        ]
        X = pd.DataFrame([[row.get(c, 0.0) for c in cols]], columns=cols)
        X_scaled = pd.DataFrame(self._scaler.transform(X), columns=cols)

        raw_pred = self._model.predict(X_scaled)[0]
        probs = self._model.predict_proba(X_scaled)[0]
        confidence = float(probs.max())

        self._prediction_buffer.append(raw_pred)
        smoothed = Counter(self._prediction_buffer).most_common(1)[0][0]

        return {
            "prediction": smoothed,
            "confidence": confidence,
            "raw_prediction": raw_pred,
        }

    def is_loaded(self) -> bool:
        """Return True if model and scaler were loaded successfully."""
        return self._loaded
