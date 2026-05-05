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


# --------------------------------------------------------------------- #
# Type helpers
# --------------------------------------------------------------------- #

def is_numeric_target(df: pd.DataFrame, target: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[target])


# --------------------------------------------------------------------- #
# Numeric target
# --------------------------------------------------------------------- #

def correlations_with_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Pearson correlation of every numeric feature with the target.

    Returned frame is sorted by absolute correlation (descending) and
    includes a human-friendly strength label.
    """
    if not is_numeric_target(df, target):
        return pd.DataFrame()

    numeric = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric.columns or numeric.shape[1] < 2:
        return pd.DataFrame()

    corr = numeric.corr(numeric_only=True)[target].drop(labels=[target])
    out = pd.DataFrame({
        "Feature": corr.index,
        "Correlation": corr.values.round(4),
        "Abs": np.abs(corr.values),
    })
    out["Strength"] = out["Abs"].apply(_strength_label)
    out["Direction"] = np.where(out["Correlation"] >= 0, "positive", "negative")
    out = out.sort_values("Abs", ascending=False).drop(columns="Abs")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------- #
# Categorical target
# --------------------------------------------------------------------- #

def group_means(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Mean of each numeric feature, grouped by the categorical target."""
    if is_numeric_target(df, target):
        return pd.DataFrame()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()

    grouped = df.groupby(target, dropna=False)[numeric_cols].mean(numeric_only=True)
    return grouped.round(3).reset_index()


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
