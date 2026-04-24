"""
RT-MBAS training script.
Usage:  python ml/train.py

Loads data/dataset.csv, trains a RandomForestClassifier, prints evaluation
metrics, saves a confusion-matrix plot, and writes ml/model.pkl.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ml.preprocess import DataPreprocessor
from app.config import DATA_PATH, MODEL_PATH, LABELS

_ANALYSIS_DIR = Path(__file__).parent.parent / "analysis"


def _save_confusion_matrix(cm: np.ndarray, labels: list, output_path: Path) -> None:
    """Render and save a labelled confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Confusion matrix saved → {output_path}")


def main() -> None:
    """Load data, validate, train, evaluate, and persist the model."""
    if not DATA_PATH.exists():
        print(
            f"No dataset found at {DATA_PATH}.\n"
            "Run  python app/main.py  first to collect behavioural data."
        )
        sys.exit(1)

    prep = DataPreprocessor()
    df = prep.load(DATA_PATH)

    # ── Minimum data checks ───────────────────────────────────────────────────
    if len(df) < 300:
        print(
            f"Only {len(df)} rows in dataset — need at least 300 "
            "(~100 per class).\n"
            "Keep running app/main.py across different activity states."
        )
        sys.exit(1)

    present = set(df["label"].unique())
    missing = set(LABELS) - present
    if missing:
        print(
            f"Missing labels: {missing}.\n"
            "Collect data covering all three states: Focused, Distracted, Stressed."
        )
        sys.exit(1)

    # ── Feature engineering ───────────────────────────────────────────────────
    df = prep.engineer_features(df)
    X_train, X_test, y_train, y_test = prep.split(df)
    X_train_s, X_test_s = prep.scale(X_train, X_test)
    feature_names = list(X_train.columns)

    print(f"Training on {len(X_train)} samples | {len(feature_names)} features ...")

    # ── Model training ────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    # Store feature names so InferenceEngine can align columns at runtime
    model.feature_names_in_ = np.array(feature_names)

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    print("Confusion Matrix:")
    print(cm)

    # ── Persist artefacts ─────────────────────────────────────────────────────
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrix(cm, LABELS, _ANALYSIS_DIR / "confusion_matrix.png")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    print("Run  python app/main.py        for live inference.")
    print("Run  streamlit run dashboard/app.py  for the analytics dashboard.")


if __name__ == "__main__":
    main()
