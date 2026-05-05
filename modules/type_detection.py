"""Smart object-to-numeric type detection and conversion.

For every column with dtype ``object``, attempt a numeric coercion. If at least
``threshold`` of the non-null values parse as numbers, treat the column as
numeric in a *processed copy* of the DataFrame. The original DataFrame is never
mutated.

Public API:
    analyze_object_columns(df, threshold=0.7) -> list[dict]
    apply_smart_conversion(df, threshold=0.7) -> (processed_df, report)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Strings that should be treated as missing rather than as invalid numerics.
_NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?"}


def _clean_string_series(series: pd.Series) -> pd.Series:
    """Strip whitespace and turn common null-like tokens into real NaNs."""
    s = series.astype("string").str.strip()
    lowered = s.str.lower()
    s = s.mask(lowered.isin(_NULL_TOKENS))
    # Strip thousands separators only when the rest looks numeric.
    looks_numeric = s.str.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", na=False)
    s = s.mask(looks_numeric, s.str.replace(",", "", regex=False))
    return s


def analyze_object_columns(
    df: pd.DataFrame,
    threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """Inspect every object column and report on numeric convertibility.

    Returns a list of dicts (one per object column) with:
        column, total_non_null, valid_numeric, invalid_count,
        valid_percent, invalid_percent, convertible, invalid_examples.
    """
    object_cols = df.select_dtypes(include=["object"]).columns
    report: list[dict[str, Any]] = []

    for col in object_cols:
        original = df[col]
        cleaned = _clean_string_series(original)
        total_non_null = int(cleaned.notna().sum())

        if total_non_null == 0:
            report.append({
                "column": col,
                "total_non_null": 0,
                "valid_numeric": 0,
                "invalid_count": 0,
                "valid_percent": 0.0,
                "invalid_percent": 0.0,
                "convertible": False,
                "invalid_examples": [],
            })
            continue

        coerced = pd.to_numeric(cleaned, errors="coerce")
        valid_numeric = int(coerced.notna().sum())
        invalid_mask = cleaned.notna() & coerced.isna()
        invalid_count = int(invalid_mask.sum())
        valid_pct = valid_numeric / total_non_null
        invalid_pct = invalid_count / total_non_null

        invalid_examples = (
            original[invalid_mask]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        report.append({
            "column": col,
            "total_non_null": total_non_null,
            "valid_numeric": valid_numeric,
            "invalid_count": invalid_count,
            "valid_percent": round(valid_pct * 100, 2),
            "invalid_percent": round(invalid_pct * 100, 2),
            "convertible": valid_pct >= threshold,
            "invalid_examples": invalid_examples,
        })

    return report


def apply_smart_conversion(
    df: pd.DataFrame,
    threshold: float = 0.7,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return a processed copy with eligible object columns coerced to numeric.

    The original DataFrame is not modified. Columns where at least ``threshold``
    of non-null values parse as numbers get replaced (in the copy) with the
    coerced numeric series; invalid entries become NaN. All other object columns
    are left as-is.
    """
    report = analyze_object_columns(df, threshold=threshold)
    if not report:
        return df.copy(), report

    processed = df.copy()
    for entry in report:
        if not entry["convertible"]:
            continue
        col = entry["column"]
        cleaned = _clean_string_series(df[col])
        processed[col] = pd.to_numeric(cleaned, errors="coerce")

    return processed, report


def conversion_summary_frame(report: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten the report into a tabular summary for display."""
    if not report:
        return pd.DataFrame(
            columns=[
                "Column", "Converted", "Valid Numeric", "Invalid Count",
                "Valid %", "Invalid %", "Invalid Examples",
            ]
        )
    rows = [
        {
            "Column": e["column"],
            "Converted": "Yes" if e["convertible"] else "No",
            "Valid Numeric": e["valid_numeric"],
            "Invalid Count": e["invalid_count"],
            "Valid %": e["valid_percent"],
            "Invalid %": e["invalid_percent"],
            "Invalid Examples": ", ".join(e["invalid_examples"]) if e["invalid_examples"] else "",
        }
        for e in report
    ]
    return pd.DataFrame(rows)


def converted_columns(report: list[dict[str, Any]]) -> list[str]:
    """Names of columns that were converted to numeric."""
    return [e["column"] for e in report if e["convertible"]]


def had_invalid_coercions(report: list[dict[str, Any]]) -> bool:
    """True if any *converted* column lost values to NaN during coercion."""
    return any(e["convertible"] and e["invalid_count"] > 0 for e in report)
