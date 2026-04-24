"""
RT-MBAS Behavioral Analytics Dashboard.
Run with:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis.plots import (
    plot_feature_importance,
    plot_label_distribution,
)

# ── Path constants (no app/ imports) ─────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_DATA_PATH = _ROOT / "data" / "dataset.csv"
_SESSIONS_DIR = _ROOT / "data" / "sessions"
_MODEL_PATH = _ROOT / "ml" / "model.pkl"
_CM_PATH = _ROOT / "analysis" / "confusion_matrix.png"
_FI_PATH = _ROOT / "analysis" / "feature_importance.png"

_LABELS = ["Focused", "Distracted", "Stressed"]
_STATE_COLORS = {
    "Focused": "#2ecc71",
    "Distracted": "#e67e22",
    "Stressed": "#e74c3c",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def _load_csv(path) -> pd.DataFrame | None:
    """Load a CSV from a file path or uploaded file object."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return None


@st.cache_resource
def _load_model():
    """Load ml/model.pkl if it exists."""
    if not _MODEL_PATH.exists():
        return None
    try:
        return joblib.load(_MODEL_PATH)
    except Exception:
        return None


def _numeric_cols(df: pd.DataFrame) -> list:
    skip = {"timestamp", "blink_indicator"}
    return [c for c in df.select_dtypes(include=[float, int]).columns if c not in skip]


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RT-MBAS Dashboard",
    page_icon="🧠",
    layout="wide",
)
st.title("RT-MBAS Behavioral Analytics Dashboard")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload a session CSV", type="csv")

    session_options = ["All sessions (dataset.csv)"]
    if _SESSIONS_DIR.exists():
        session_files = sorted(_SESSIONS_DIR.glob("*.csv"), reverse=True)
        session_options += [f.stem for f in session_files]

    selected_session = st.selectbox("Or pick a session", session_options)
    st.divider()
    st.caption("All processing is local — no internet required.")

# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded is not None:
    df = _load_csv(uploaded)
elif selected_session != "All sessions (dataset.csv)":
    df = _load_csv(_SESSIONS_DIR / f"{selected_session}.csv")
elif _DATA_PATH.exists():
    df = _load_csv(_DATA_PATH)
else:
    df = None

if df is None or df.empty:
    st.warning(
        "No data found. Run `python app/main.py` to collect data, then refresh."
    )
    st.stop()

num_cols = _numeric_cols(df)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Feature Trends", "Model Insights", "Raw Data"]
)

# ═══════════════════════════════ Tab 1: Overview ══════════════════════════════
with tab1:
    total = len(df)

    # Compute duration
    duration_str = "—"
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        secs = df["timestamp"].max() - df["timestamp"].min()
        m, s = divmod(int(secs), 60)
        duration_str = f"{m}m {s}s"

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Frames", f"{total:,}")
    col2.metric("Session Duration", duration_str)
    if "label" in df.columns:
        dominant = df["label"].mode().iloc[0]
        col3.metric("Dominant State", dominant)

    st.divider()

    left, right = st.columns(2)

    # Label distribution pie chart
    with left:
        if "label" in df.columns:
            counts = df["label"].value_counts().reset_index()
            counts.columns = ["State", "Count"]
            fig_pie = px.pie(
                counts,
                names="State",
                values="Count",
                title="Label Distribution",
                color="State",
                color_discrete_map=_STATE_COLORS,
                hole=0.3,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No 'label' column found in this dataset.")

    # EAR over time
    with right:
        if "eye_aspect_ratio" in df.columns:
            fig_ear = px.line(
                df.reset_index(),
                x="index",
                y="eye_aspect_ratio",
                title="Eye Aspect Ratio Over Time",
                labels={"index": "Frame", "eye_aspect_ratio": "EAR"},
            )
            fig_ear.add_hline(
                y=0.21,
                line_dash="dash",
                line_color="red",
                annotation_text="Blink threshold (0.21)",
            )
            st.plotly_chart(fig_ear, use_container_width=True)

# ══════════════════════════ Tab 2: Feature Trends ════════════════════════════
with tab2:
    st.subheader("Multi-Feature Trend")

    default_features = [
        c for c in ["eye_aspect_ratio", "hand_velocity", "face_motion_delta"]
        if c in df.columns
    ]
    selected = st.multiselect(
        "Features to display",
        options=num_cols,
        default=default_features,
    )
    show_rolling = st.checkbox("Show rolling mean overlay (window = 30 frames)", value=False)

    if selected:
        fig_trend = go.Figure()
        x = list(range(len(df)))
        palette = px.colors.qualitative.Plotly

        for i, col in enumerate(selected):
            color = palette[i % len(palette)]
            y = df[col].values
            fig_trend.add_trace(go.Scatter(
                x=x, y=y,
                name=col,
                line=dict(color=color, width=1),
                opacity=0.8,
            ))
            if show_rolling:
                rolled = pd.Series(y).rolling(30, min_periods=1).mean()
                fig_trend.add_trace(go.Scatter(
                    x=x, y=rolled,
                    name=f"{col} (mean)",
                    line=dict(color=color, width=2, dash="dot"),
                    opacity=1.0,
                    showlegend=True,
                ))

        fig_trend.update_layout(
            height=480,
            xaxis_title="Frame",
            yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Correlation Heatmap")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        fig_corr.update_traces(textfont_size=9)
        st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════ Tab 3: Model Insights ════════════════════════════
with tab3:
    st.subheader("Model Insights")
    model = _load_model()

    if model is None:
        st.info(
            "No trained model found at `ml/model.pkl`.\n\n"
            "Run `python ml/train.py` after collecting at least 300 frames of data."
        )
    else:
        st.success(f"Model loaded from `{_MODEL_PATH.relative_to(_ROOT)}`")

        # Feature importances bar chart (plotly — interactive)
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
            importances = model.feature_importances_
            fi_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=True)
            )
            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importances (Random Forest)",
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig_fi.update_layout(height=max(400, len(feature_names) * 24))
            st.plotly_chart(fig_fi, use_container_width=True)

            # Also save a static PNG for train.py compatibility
            if not _FI_PATH.exists():
                try:
                    plot_feature_importance(model, feature_names, str(_FI_PATH))
                except Exception:
                    pass

        # Confusion matrix image (generated by train.py)
        st.subheader("Confusion Matrix")
        if _CM_PATH.exists():
            st.image(str(_CM_PATH), caption="Confusion matrix from last training run")
        else:
            st.caption(
                "No confusion matrix image yet — it is generated automatically "
                "when you run `python ml/train.py`."
            )

# ══════════════════════════ Tab 4: Raw Data ══════════════════════════════════
with tab4:
    st.subheader("Raw Data")

    # Label filter
    filter_options = ["All"]
    if "label" in df.columns:
        filter_options += sorted(df["label"].dropna().unique().tolist())
    chosen_label = st.selectbox("Filter by label", filter_options)

    filtered = df if chosen_label == "All" else df[df["label"] == chosen_label]

    st.dataframe(filtered, use_container_width=True, height=420)
    st.caption(f"Showing {len(filtered):,} of {len(df):,} rows")

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered CSV",
        data=csv_bytes,
        file_name="rt_mbas_export.csv",
        mime="text/csv",
    )
