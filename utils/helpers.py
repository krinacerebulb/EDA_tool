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
    """Return the active Plotly template. Light theme only."""
    return "plotly_white"


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
