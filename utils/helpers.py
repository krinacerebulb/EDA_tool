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
    """Format a byte count into a human-readable string (always 2 decimals)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TB"


def plotly_template() -> str:
    """Return the active Plotly template. Light theme only."""
    return "plotly_white"


def safe_dataframe(df, **kwargs):
    """PyArrow-safe ``st.dataframe`` wrapper.

    Why this exists: ``st.dataframe`` serialises through PyArrow. If the
    frame still has a mixed-type object column (e.g. an on-the-fly summary
    frame built outside the loader's sanitization path), PyArrow raises
    ``ArrowInvalid`` and the whole Streamlit page errors. This wrapper runs
    the input through ``make_arrow_safe`` first, so rendering is guaranteed
    to succeed.

    Use ``safe_dataframe(df, width="stretch")`` anywhere you would have
    called ``st.dataframe(df, width="stretch")`` — especially for frames
    built from heterogeneous sources (preview tables, conversion summaries,
    user-built previews).
    """
    import streamlit as st
    try:
        from modules.data_sanitization import make_arrow_safe
        safe = make_arrow_safe(df)
        return st.dataframe(safe, **kwargs)
    except Exception as exc:
        # Absolute last resort — full stringification.
        try:
            from modules.data_sanitization import force_stringify
            return st.dataframe(force_stringify(df), **kwargs)
        except Exception:
            st.warning(f"Could not render DataFrame: {exc}")
            return None


def compress_strings_to_category(
    df: pd.DataFrame,
    max_unique_ratio: float = 0.5,
    min_rows: int = 1000,
) -> pd.DataFrame:
    """Convert low-cardinality object columns to ``category`` in-place-by-return.

    Big win for memory on industrial datasets: a 500k-row column with 100
    unique strings drops from ~25 MB to a few hundred KB. Downstream code
    that uses ``select_dtypes(include=["object", "category", "bool"])`` still
    sees the column, so behaviour is preserved.

    Skipped automatically for small DataFrames (below ``min_rows``) where the
    overhead wouldn't pay off.
    """
    if len(df) < min_rows:
        return df
    object_cols = df.select_dtypes(include=["object"]).columns
    if object_cols.empty:
        return df
    for col in object_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique == 0:
            continue
        if nunique / len(df) <= max_unique_ratio:
            df[col] = df[col].astype("category")
    return df
