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


def _round2(v) -> float:
    """Round to 2 decimals; preserves NaN for missing-value semantics."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float("nan")
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return float("nan")


@st.cache_data(show_spinner=False)
def numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Full ``df.describe()``-style summary for numeric columns.

    Columns: ``Column, Count, Missing, Missing %, Unique, Mean, Std, Min,
    25%, 50%, 75%, Max, Skew, Kurtosis``. All numeric values rounded to
    two decimals for clean display. Skew / kurtosis only meaningful when
    there are enough observations (≥3 / ≥4); otherwise NaN.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()

    n = len(df)
    rows: list[dict] = []
    for col in numeric.columns:
        s = numeric[col]
        non_null = s.dropna()
        miss = int(s.isna().sum())
        miss_pct = (miss / n * 100) if n else 0.0
        if non_null.empty:
            # All-null column — show structure but no statistics.
            rows.append({
                "Column": col, "Count": 0, "Missing": miss,
                "Missing %": _round2(miss_pct), "Unique": 0,
                "Mean": float("nan"), "Std": float("nan"),
                "Min": float("nan"), "25%": float("nan"),
                "50%": float("nan"), "75%": float("nan"),
                "Max": float("nan"), "Skew": float("nan"),
                "Kurtosis": float("nan"),
            })
            continue
        q = non_null.quantile([0.25, 0.50, 0.75])
        rows.append({
            "Column":    col,
            "Count":     int(non_null.count()),
            "Missing":   miss,
            "Missing %": _round2(miss_pct),
            "Unique":    int(s.nunique(dropna=True)),
            "Mean":      _round2(non_null.mean()),
            "Std":       _round2(non_null.std()),
            "Min":       _round2(non_null.min()),
            "25%":       _round2(q.loc[0.25]),
            "50%":       _round2(q.loc[0.50]),
            "75%":       _round2(q.loc[0.75]),
            "Max":       _round2(non_null.max()),
            "Skew":      _round2(non_null.skew()) if len(non_null) > 2 else float("nan"),
            "Kurtosis":  _round2(non_null.kurt()) if len(non_null) > 3 else float("nan"),
        })
    return pd.DataFrame(rows)


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


@st.cache_data(show_spinner=False)
def column_detail_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One-column drilldown — full statistical profile as a Metric/Value table.

    Numeric columns include describe-style quartiles, skew, kurtosis, plus
    ML-useful extras (range, IQR, zero / negative counts). Non-numeric
    columns include mode, top value, top frequency, and example values.
    Returns an empty frame if ``col`` doesn't exist.
    """
    if col not in df.columns:
        return pd.DataFrame()
    s = df[col]
    n = len(s)
    miss = int(s.isna().sum())
    miss_pct = (miss / n * 100) if n else 0.0
    unique = int(s.nunique(dropna=True))
    non_null = s.dropna()

    rows: list[tuple[str, str]] = [
        ("Dtype",         str(s.dtype)),
        ("Total values",  f"{n:,}"),
        ("Missing",       f"{miss:,}"),
        ("Missing %",     f"{miss_pct:.2f}%"),
        ("Unique values", f"{unique:,}"),
    ]

    if pd.api.types.is_numeric_dtype(s) and not non_null.empty:
        q1, q2, q3 = (
            float(non_null.quantile(0.25)),
            float(non_null.quantile(0.50)),
            float(non_null.quantile(0.75)),
        )
        mn, mx = float(non_null.min()), float(non_null.max())
        rows += [
            ("Mean",     f"{non_null.mean():.2f}"),
            ("Median",   f"{q2:.2f}"),
            ("Std",      f"{non_null.std():.2f}"),
            ("Min",      f"{mn:.2f}"),
            ("25%",      f"{q1:.2f}"),
            ("75%",      f"{q3:.2f}"),
            ("Max",      f"{mx:.2f}"),
            ("Range",    f"{(mx - mn):.2f}"),
            ("IQR",      f"{(q3 - q1):.2f}"),
            ("Skew",     f"{non_null.skew():.2f}" if len(non_null) > 2 else "—"),
            ("Kurtosis", f"{non_null.kurt():.2f}" if len(non_null) > 3 else "—"),
            ("Zeros",    f"{int((non_null == 0).sum()):,}"),
            ("Negatives", f"{int((non_null < 0).sum()):,}"),
        ]
    elif not non_null.empty:
        counts = non_null.astype("string").value_counts()
        top_val = counts.index[0]
        top_freq = int(counts.iloc[0])
        top_pct = top_freq / len(non_null) * 100
        rows += [
            ("Mode",          str(top_val)),
            ("Top frequency", f"{top_freq:,}"),
            ("Top %",         f"{top_pct:.2f}%"),
            ("Examples",      ", ".join(counts.index[:5].tolist())),
        ]

    return pd.DataFrame(rows, columns=["Metric", "Value"])
