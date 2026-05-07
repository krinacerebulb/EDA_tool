"""Small helper utilities shared across modules."""

import base64
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string (for HTML report)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def split_columns(df: pd.DataFrame):
    """Return (numeric_cols, categorical_cols, datetime_cols) based on dtypes."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    return numeric_cols, categorical_cols, datetime_cols


def human_bytes(num: int) -> str:
    """Format a byte count into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def plotly_template() -> str:
    """Return the active Plotly template based on the user's theme choice.

    Reads ``st.session_state['theme']`` if Streamlit is available; defaults to
    the light template otherwise. Safe to call from non-UI modules.
    """
    try:
        import streamlit as st

        if st.session_state.get("theme") == "Dark":
            return "plotly_dark"
    except Exception:
        pass
    return "plotly_white"
