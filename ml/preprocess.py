"""
Data loading, cleaning, feature engineering, splitting, and scaling for RT-MBAS.
Run indirectly via ml/train.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from app.config import SCALER_PATH


class DataPreprocessor:
    """Handles the full preprocessing pipeline from raw CSV to scaled train/test sets."""

    def __init__(self):
        """Initialize with a fresh StandardScaler."""
        self._scaler = StandardScaler()

    def load(self, path) -> pd.DataFrame:
        """
        Load CSV data, drop rows with >20% missing values, and fill remaining
        NaN entries with each column's median.

        Args:
            path: file path (str or Path) to the dataset CSV.

        Returns:
            Cleaned pandas DataFrame.
        """
        df = pd.read_csv(path)
        min_valid = int(len(df.columns) * 0.80)   # keep rows with >=80% non-null
        df = df.dropna(thresh=min_valid)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag-1 and lag-5 features for eye_aspect_ratio and hand_velocity.

        Lag features capture recent temporal trends without requiring a recurrent model.

        Args:
            df: input DataFrame (must contain the base columns).

        Returns:
            New DataFrame with four additional lag columns appended.
        """
        df = df.copy()
        for col in ("eye_aspect_ratio", "hand_velocity"):
            if col in df.columns:
                df[f"{col}_lag1"] = df[col].shift(1).fillna(0.0)
                df[f"{col}_lag5"] = df[col].shift(5).fillna(0.0)
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Return all numeric column names except 'timestamp' and 'label'.

        Args:
            df: DataFrame after feature engineering.

        Returns:
            List of feature column name strings.
        """
        exclude = {"timestamp", "label"}
        return [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    def split(
        self,
        df: pd.DataFrame,
        target_col: str = "label",
        test_size: float = 0.2,
    ):
        """
        Perform a stratified train/test split.

        Args:
            df: fully engineered DataFrame.
            target_col: name of the label column.
            test_size: fraction of data to hold out for testing.

        Returns:
            X_train, X_test, y_train, y_test (DataFrames / Series).
        """
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols]
        y = df[target_col]
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=42,
        )

    def scale(self, X_train, X_test):
        """
        Fit StandardScaler on training data, transform both sets, and save the
        fitted scaler to SCALER_PATH for use during inference.

        Args:
            X_train: training feature matrix (DataFrame or ndarray).
            X_test:  test feature matrix.

        Returns:
            (X_train_scaled, X_test_scaled) as numpy arrays.
        """
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, SCALER_PATH)
        return X_train_scaled, X_test_scaled
