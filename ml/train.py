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
import pandas as pd
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
_STOCK_PATH   = DATA_PATH.parent / "stock_dataset.csv"


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
    print(f"Confusion matrix saved -> {output_path}")


def main() -> None:
    """Load data, validate, train, evaluate, and persist the model."""
    prep = DataPreprocessor()
    dfs  = []

    if _STOCK_PATH.exists():
        df_stock = prep.load(_STOCK_PATH)
        dfs.append(df_stock)
        print(f"Stock data : {len(df_stock):>6} rows  ({_STOCK_PATH.name})")
    else:
        print(f"No stock data found at {_STOCK_PATH} — run _gen_stock_data.py to create it.")

    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        df_user = prep.load(DATA_PATH)
        dfs.append(df_user)
        print(f"User data  : {len(df_user):>6} rows  ({DATA_PATH.name})")

    if not dfs:
        print(
            "No data found. Run  python _gen_stock_data.py  for stock data,\n"
            "or  python app/main.py  to collect live behavioural data."
        )
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    print(f"Total      : {len(df):>6} rows\n")

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
    if len(present) < 2:
        print(
            f"Need at least 2 distinct labels to train — only found: {present}.\n"
            "Run app/main.py across multiple emotional states first."
        )
        sys.exit(1)
    if missing:
        print(
            f"Note: {len(missing)} label(s) not yet collected: {sorted(missing)}\n"
            "The model will train on the labels present. Collect the missing states\n"
            "and retrain for full coverage."
        )

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
    print(f"\nModel saved -> {MODEL_PATH}")
    print("Run  python app/main.py        for live inference.")
    print("Run  streamlit run dashboard/app.py  for the analytics dashboard.")


if __name__ == "__main__":
    main()
