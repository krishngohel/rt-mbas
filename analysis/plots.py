"""
Standalone visualization functions for RT-MBAS.

All functions accept a pandas DataFrame and an output path string, save
a PNG to that path, and return nothing. They never display windows.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _numeric_features(df: pd.DataFrame) -> list:
    """Return numeric columns, excluding timestamp and binary indicators."""
    skip = {"timestamp", "blink_indicator"}
    return [
        c for c in df.select_dtypes(include=[float, int]).columns
        if c not in skip
    ]


def plot_feature_timeseries(df: pd.DataFrame, output_path: str) -> None:
    """
    Save line plots of all numeric features over frame index.

    Each feature gets its own sub-plot with a shared x-axis. The resulting
    PNG gives a quick visual scan of how every signal evolves over time.

    Args:
        df: DataFrame containing at least one numeric feature column.
        output_path: destination file path for the saved PNG.
    """
    cols = _numeric_features(df)
    n = len(cols)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    x = np.arange(len(df))
    for ax, col in zip(axes, cols):
        ax.plot(x, df[col].values, linewidth=0.8, alpha=0.85, color="#3498db")
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Feature Time Series", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a seaborn heatmap of feature–feature Pearson correlations.

    Args:
        df: DataFrame containing numeric feature columns.
        output_path: destination file path for the saved PNG.
    """
    cols = _numeric_features(df)
    if len(cols) < 2:
        return

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(max(10, len(cols)), max(8, len(cols) - 2)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a histogram grid showing the distribution of every numeric feature.

    Args:
        df: DataFrame containing numeric feature columns.
        output_path: destination file path for the saved PNG.
    """
    cols = _numeric_features(df)
    if not cols:
        return

    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.75,
                     color="#3498db")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count", fontsize=8)
        axes[i].grid(True, alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_label_distribution(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a bar chart showing per-label frame counts.

    Args:
        df: DataFrame with a 'label' column.
        output_path: destination file path for the saved PNG.
    """
    if "label" not in df.columns:
        return

    counts = df["label"].value_counts()
    palette = {"Focused": "#2ecc71", "Distracted": "#e67e22", "Stressed": "#e74c3c"}
    bar_colors = [palette.get(lbl, "#95a5a6") for lbl in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="black",
                  alpha=0.9)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.values.max() * 0.01,
            str(val),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("Label Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Behavioural State")
    ax.set_ylabel("Frame Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_feature_importance(model, feature_names: list, output_path: str) -> None:
    """
    Save a horizontal bar chart of RandomForest feature importances, sorted
    ascending so the most important feature sits at the top of the chart.

    Args:
        model: fitted RandomForestClassifier with .feature_importances_.
        feature_names: list of feature name strings matching model columns.
        output_path: destination file path for the saved PNG.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.38)))
    ax.barh(sorted_names, sorted_vals, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
