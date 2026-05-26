"""Target-based EDA: relate every feature to a user-chosen target column.

Two regimes are supported:

* **Numerical target** — Pearson correlation + scatter plots.
* **Categorical target** — per-group means for numeric features and
  boxplots; chi-square-style top categorical associations via
  normalised cross-tab variation.

All functions are pure (no Streamlit calls) so they can be reused in
reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------------------------- #
# Type helpers
# --------------------------------------------------------------------- #

def is_numeric_target(df: pd.DataFrame, target: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[target])


# --------------------------------------------------------------------- #
# Target summary cards / detail
# --------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def target_summary(df: pd.DataFrame, target: str, precision: int = 2) -> dict:
    """Structured stats for the selected target column.

    Returns a dict ready to drive Streamlit cards + a detail table. Shape
    depends on whether the target is numeric:

    Numeric target:
        {is_numeric: True, total, missing, missing_pct, unique,
         mean, median, std, min, q25, q75, max, skew, kurtosis,
         detail: pd.DataFrame[Metric, Value]}

    Categorical target:
        {is_numeric: False, total, missing, missing_pct, unique,
         top_value, top_freq, top_pct,
         class_distribution: pd.DataFrame[Class, Count, %],
         detail: pd.DataFrame[Metric, Value]}
    """
    if target not in df.columns:
        return {}

    p = max(0, int(precision))
    s = df[target]
    n = len(s)
    miss = int(s.isna().sum())
    miss_pct = (miss / n * 100) if n else 0.0
    unique = int(s.nunique(dropna=True))
    non_null = s.dropna()
    is_num = pd.api.types.is_numeric_dtype(s)

    out: dict = {
        "is_numeric":  is_num,
        "total":       n,
        "missing":     miss,
        "missing_pct": round(miss_pct, 2),
        "unique":      unique,
    }

    if is_num and not non_null.empty:
        q1 = float(non_null.quantile(0.25))
        q2 = float(non_null.quantile(0.50))
        q3 = float(non_null.quantile(0.75))
        mn, mx = float(non_null.min()), float(non_null.max())
        skew = float(non_null.skew()) if len(non_null) > 2 else float("nan")
        kurt = float(non_null.kurt()) if len(non_null) > 3 else float("nan")
        out.update({
            "mean":     round(float(non_null.mean()), p),
            "median":   round(q2, p),
            "std":      round(float(non_null.std()), p),
            "min":      round(mn, p),
            "q25":      round(q1, p),
            "q75":      round(q3, p),
            "max":      round(mx, p),
            "skew":     round(skew, p) if not np.isnan(skew) else float("nan"),
            "kurtosis": round(kurt, p) if not np.isnan(kurt) else float("nan"),
        })
        out["detail"] = pd.DataFrame([
            ("Count",     f"{int(non_null.count()):,}"),
            ("Missing",   f"{miss:,}"),
            ("Missing %", f"{miss_pct:.{p}f}%"),
            ("Unique",    f"{unique:,}"),
            ("Mean",      f"{out['mean']:.{p}f}"),
            ("Std",       f"{out['std']:.{p}f}"),
            ("Min",       f"{out['min']:.{p}f}"),
            ("25%",       f"{out['q25']:.{p}f}"),
            ("50%",       f"{out['median']:.{p}f}"),
            ("75%",       f"{out['q75']:.{p}f}"),
            ("Max",       f"{out['max']:.{p}f}"),
            ("Range",     f"{(mx - mn):.{p}f}"),
            ("IQR",       f"{(q3 - q1):.{p}f}"),
            ("Skew",      f"{out['skew']:.{p}f}" if not np.isnan(out['skew']) else "—"),
            ("Kurtosis",  f"{out['kurtosis']:.{p}f}" if not np.isnan(out['kurtosis']) else "—"),
        ], columns=["Metric", "Value"])
    elif not non_null.empty:
        counts = non_null.astype("string").value_counts()
        top_val = counts.index[0]
        top_freq = int(counts.iloc[0])
        top_pct = top_freq / len(non_null) * 100
        class_dist = pd.DataFrame({
            "Class": counts.index.astype(str),
            "Count": counts.values,
            "%":     (counts.values / len(non_null) * 100).round(p),
        })
        out.update({
            "top_value":          str(top_val),
            "top_freq":           top_freq,
            "top_pct":            round(top_pct, p),
            "class_distribution": class_dist,
        })
        out["detail"] = pd.DataFrame([
            ("Total values",  f"{n:,}"),
            ("Missing",       f"{miss:,}"),
            ("Missing %",     f"{miss_pct:.{p}f}%"),
            ("Unique classes", f"{unique:,}"),
            ("Most frequent", str(top_val)),
            ("Frequency",     f"{top_freq:,}"),
            ("Frequency %",   f"{top_pct:.{p}f}%"),
        ], columns=["Metric", "Value"])
    else:
        # All-null target — degenerate but worth representing.
        out["detail"] = pd.DataFrame([
            ("Total values", f"{n:,}"),
            ("Missing",      f"{miss:,}"),
            ("Missing %",    f"{miss_pct:.{p}f}%"),
            ("Unique",       f"{unique:,}"),
        ], columns=["Metric", "Value"])

    return out


# --------------------------------------------------------------------- #
# Numeric target
# --------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def correlations_with_target(
    df: pd.DataFrame, target: str, precision: int = 4,
) -> pd.DataFrame:
    """Pearson correlation of every numeric feature with the target.

    Returned frame is sorted by absolute correlation (descending) and
    includes a human-friendly strength label. Correlation values are
    rounded to ``precision`` decimals — defaulting to 4 because
    correlations are inherently small and benefit from extra detail.
    """
    if not is_numeric_target(df, target):
        return pd.DataFrame()

    numeric = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric.columns or numeric.shape[1] < 2:
        return pd.DataFrame()

    p = max(0, int(precision))
    corr = numeric.corr(numeric_only=True)[target].drop(labels=[target])
    out = pd.DataFrame({
        "Feature": corr.index,
        "Correlation": corr.values.round(p),
        "Abs": np.abs(corr.values),
    })
    out["Strength"] = out["Abs"].apply(_strength_label)
    out["Direction"] = np.where(out["Correlation"] >= 0, "positive", "negative")
    out = out.sort_values("Abs", ascending=False).drop(columns="Abs")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------- #
# Categorical target
# --------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def group_means(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Mean of each numeric feature, grouped by the categorical target."""
    if is_numeric_target(df, target):
        return pd.DataFrame()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()

    grouped = df.groupby(target, dropna=False)[numeric_cols].mean(numeric_only=True)
    return grouped.round(3).reset_index()


@st.cache_data(show_spinner=False)
def categorical_importance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Rank numeric features by how much their group means differ across
    target categories.  Uses the coefficient of variation of group means
    (std-dev of group means / overall mean) as a cheap importance score.
    """
    if is_numeric_target(df, target):
        return pd.DataFrame()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()

    rows = []
    for col in numeric_cols:
        series = df[col]
        overall = series.mean()
        if pd.isna(overall) or overall == 0:
            score = float(df.groupby(target)[col].mean().std(ddof=0) or 0)
        else:
            group_std = df.groupby(target)[col].mean().std(ddof=0)
            score = float(abs(group_std / overall)) if pd.notna(group_std) else 0.0
        rows.append({"Feature": col, "Score": round(score, 4)})

    out = pd.DataFrame(rows).sort_values("Score", ascending=False)
    out["Strength"] = out["Score"].apply(_cv_strength_label)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------- #

def generate_target_insights(df: pd.DataFrame, target: str, top_n: int = 5) -> list[str]:
    """Plain-language bullet points describing the strongest relationships."""
    if target not in df.columns:
        return [f"Target column **{target}** not found."]

    lines: list[str] = []

    if is_numeric_target(df, target):
        corr = correlations_with_target(df, target)
        if corr.empty:
            return ["Not enough numeric features to analyse relationships with the target."]

        strong = corr[corr["Correlation"].abs() >= 0.5]
        weak = corr[corr["Correlation"].abs() < 0.1]

        lines.append(
            f"Analysed **{len(corr)}** numeric features against target **{target}**."
        )

        top = corr.head(top_n)
        for _, row in top.iterrows():
            lines.append(
                f"**{row['Feature']}** shows a {row['Strength'].lower()} "
                f"{row['Direction']} correlation "
                f"({row['Correlation']:+.2f}) with **{target}**."
            )

        if not strong.empty:
            names = ", ".join(strong["Feature"].head(5))
            lines.append(f"Strong relationships (|r| ≥ 0.5): {names}.")
        else:
            lines.append("No strong linear relationships (|r| ≥ 0.5) detected.")

        if not weak.empty:
            lines.append(
                f"{len(weak)} feature(s) show negligible correlation "
                "(|r| < 0.1) — likely uninformative on their own."
            )

    else:
        groups = df[target].dropna().nunique()
        lines.append(
            f"Target **{target}** is categorical with **{groups}** distinct groups."
        )

        imp = categorical_importance(df, target)
        if imp.empty:
            lines.append("No numeric features available to compare across groups.")
            return lines

        top = imp.head(top_n)
        for _, row in top.iterrows():
            lines.append(
                f"**{row['Feature']}** varies {row['Strength'].lower()} across "
                f"groups of **{target}** (score {row['Score']:.2f})."
            )

        weak = imp[imp["Score"] < 0.05]
        if not weak.empty:
            lines.append(
                f"{len(weak)} feature(s) show very little variation across groups "
                "and are unlikely to discriminate between categories."
            )

    return lines


# --------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------- #

def _strength_label(abs_corr: float) -> str:
    if abs_corr >= 0.7:
        return "Very Strong"
    if abs_corr >= 0.5:
        return "Strong"
    if abs_corr >= 0.3:
        return "Moderate"
    if abs_corr >= 0.1:
        return "Weak"
    return "Negligible"


def _cv_strength_label(score: float) -> str:
    if score >= 0.5:
        return "Very Strong"
    if score >= 0.2:
        return "Strong"
    if score >= 0.1:
        return "Moderate"
    if score >= 0.05:
        return "Weak"
    return "Negligible"
