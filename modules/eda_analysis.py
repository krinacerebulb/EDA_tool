"""Descriptive statistics for numeric and categorical columns.

Public functions are wrapped in ``@st.cache_data`` so heavy stats are computed
once per DataFrame content-hash and reused across reruns.
"""

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def dataset_overview(df: pd.DataFrame) -> dict:
    """High-level facts about the dataset."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "total_missing": int(df.isna().sum().sum()),
    }


@st.cache_data(show_spinner=False)
def numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, std, min, max (and a few extras) for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()

    stats = pd.DataFrame({
        "Mean": numeric.mean(),
        "Median": numeric.median(),
        "Std": numeric.std(),
        "Min": numeric.min(),
        "Max": numeric.max(),
        "Skew": numeric.skew(),
    }).round(3)
    stats.index.name = "Column"
    return stats.reset_index()


@st.cache_data(show_spinner=False)
def categorical_statistics(df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    For each categorical column, return a small DataFrame of the top-N
    value counts plus a top-level summary row (unique count, mode).
    """
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    result = {}
    for col in cat_cols:
        series = df[col].dropna()
        if series.empty:
            result[col] = {
                "unique": 0,
                "mode": None,
                "top_values": pd.DataFrame(columns=["Value", "Count", "Percent"]),
            }
            continue
        counts = series.value_counts().head(top_n)
        top_values = pd.DataFrame({
            "Value": counts.index.astype(str),
            "Count": counts.values,
            "Percent": (counts.values / len(series) * 100).round(2),
        })
        result[col] = {
            "unique": int(series.nunique()),
            "mode": str(series.mode().iloc[0]),
            "top_values": top_values,
        }
    return result
