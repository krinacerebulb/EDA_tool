"""Generate human-readable insights from a DataFrame.

The goal is NOT to be exhaustive — it's to surface a few simple,
actionable observations a non-technical reader can understand.
"""

import numpy as np
import pandas as pd


# Thresholds for what counts as "notable". Tuned for simple, readable output.
HIGH_CORR_THRESHOLD = 0.7
HIGH_MISSING_THRESHOLD = 20.0      # percent
DOMINANT_CATEGORY_THRESHOLD = 60.0  # percent
SKEW_THRESHOLD = 1.0               # absolute skew considered "skewed"


def _missing_insights(df: pd.DataFrame) -> list[str]:
    out = []
    missing_pct = df.isna().mean() * 100
    high = missing_pct[missing_pct >= HIGH_MISSING_THRESHOLD].sort_values(ascending=False)
    for col, pct in high.items():
        out.append(
            f"Column **{col}** has {pct:.1f}% missing values — consider imputing or dropping it."
        )
    total_pct = df.isna().mean().mean() * 100
    if total_pct == 0:
        out.append("No missing values detected across the dataset.")
    return out


def _duplicate_insights(df: pd.DataFrame) -> list[str]:
    dup = int(df.duplicated().sum())
    if dup:
        pct = dup / len(df) * 100 if len(df) else 0
        return [f"Found {dup} duplicate rows ({pct:.1f}% of the dataset)."]
    return ["No duplicate rows detected."]


def _correlation_insights(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []

    corr = numeric.corr().abs()
    # Grab upper triangle to avoid self-pairs and duplicates.
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().sort_values(ascending=False)
    strong = pairs[pairs >= HIGH_CORR_THRESHOLD]

    out = []
    for (col_a, col_b), value in strong.items():
        direction = "positive" if numeric[col_a].corr(numeric[col_b]) > 0 else "negative"
        out.append(
            f"Strong {direction} correlation ({value:.2f}) between **{col_a}** and **{col_b}**."
        )
    if not out and numeric.shape[1] >= 2:
        out.append("No strongly correlated numeric pairs (|r| ≥ 0.7) detected.")
    return out


def _skew_insights(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number])
    out = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 3:
            continue
        skew = series.skew()
        if abs(skew) >= SKEW_THRESHOLD:
            shape = "right-skewed" if skew > 0 else "left-skewed"
            out.append(
                f"**{col}** is {shape} (skew = {skew:.2f}) — consider a log/transform if modelling."
            )
    return out


def _dominant_category_insights(df: pd.DataFrame) -> list[str]:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    out = []
    for col in cat_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        top_value = series.value_counts(normalize=True).iloc[0] * 100
        top_name = series.value_counts().index[0]
        if top_value >= DOMINANT_CATEGORY_THRESHOLD:
            out.append(
                f"Category **{col}** is dominated by '{top_name}' "
                f"({top_value:.1f}% of values) — low variability."
            )
    return out


def _outlier_insights(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number])
    out = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        if count and count / len(series) >= 0.05:
            pct = count / len(series) * 100
            out.append(f"**{col}** has {count} outliers ({pct:.1f}%) by the IQR rule.")
    return out


def generate_insights(df: pd.DataFrame) -> list[str]:
    """Return a flat list of plain-language insight strings."""
    insights: list[str] = []
    insights += _missing_insights(df)
    insights += _duplicate_insights(df)
    insights += _correlation_insights(df)
    insights += _skew_insights(df)
    insights += _dominant_category_insights(df)
    insights += _outlier_insights(df)
    if not insights:
        insights.append("The dataset looks clean — no notable issues surfaced.")
    return insights
