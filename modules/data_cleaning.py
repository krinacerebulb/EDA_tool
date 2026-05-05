"""Cleaning-focused analysis: missing values, duplicates, outliers.

This module DOES NOT modify the DataFrame. It only reports findings so the user
can decide what to do.
"""

import numpy as np
import pandas as pd


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and percentage of missing values, sorted descending."""
    missing = df.isna().sum()
    percent = (missing / len(df) * 100).round(2) if len(df) else missing
    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing.values,
        "Missing %": percent.values,
    })
    return summary.sort_values("Missing %", ascending=False).reset_index(drop=True)


def duplicate_summary(df: pd.DataFrame) -> dict:
    """Total and percentage of duplicated rows."""
    total = int(df.duplicated().sum())
    percent = round((total / len(df) * 100), 2) if len(df) else 0.0
    return {"duplicate_rows": total, "duplicate_percent": percent}


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag outliers per numeric column using the IQR rule:
    values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            rows.append({"Column": col, "Outlier Count": 0, "Outlier %": 0.0,
                         "Lower Bound": np.nan, "Upper Bound": np.nan})
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        percent = round((count / len(series) * 100), 2) if len(series) else 0.0
        rows.append({
            "Column": col,
            "Outlier Count": count,
            "Outlier %": percent,
            "Lower Bound": round(lower, 3),
            "Upper Bound": round(upper, 3),
        })
    return pd.DataFrame(rows)
