"""Sidebar-driven dynamic multi-filter system.

Supports:
- Multiple numerical filters, each driven by Min / Max text inputs.
- Multiple categorical filters, each driven by a multi-select.
- All filters combined with AND.
- Returns the filtered DataFrame plus a summary used for the impact KPIs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


# Columns with more unique values than this are treated as free-form
# text and skipped in the filter UI (picking from thousands of options
# is not useful).
_MAX_CATEGORICAL_UNIQUE = 200


def render_sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Render filter widgets in the sidebar and return (filtered_df, summary)."""
    st.sidebar.markdown("## 🔍 Filters")
    st.sidebar.caption("Apply any number of filters — all combined with AND.")

    if df.empty:
        return df, _summary(df, df, [])

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    with st.sidebar.expander("📊 Choose columns to filter", expanded=False):
        chosen_numeric = st.multiselect(
            "Numerical columns",
            numeric_cols,
            default=[],
            key="flt_choose_num",
            help="Each chosen column gets its own Min / Max text input.",
        )

        eligible_cats = [
            c for c in categorical_cols
            if df[c].nunique(dropna=True) <= _MAX_CATEGORICAL_UNIQUE
        ]
        chosen_categorical = st.multiselect(
            "Categorical columns",
            eligible_cats,
            default=[],
            key="flt_choose_cat",
            help=(
                "Each chosen column gets its own multi-select. "
                f"Columns with more than {_MAX_CATEGORICAL_UNIQUE} unique "
                "values are excluded."
            ),
        )

    mask = pd.Series(True, index=df.index)
    applied: list[str] = []

    # -------- Numerical filters: Min / Max text inputs --------
    if chosen_numeric:
        st.sidebar.markdown("### 🔢 Numerical ranges")
        st.sidebar.caption("Leave a field blank to leave that bound open.")

    for col in chosen_numeric:
        series = df[col].dropna()
        if series.empty:
            st.sidebar.caption(f"`{col}`: no non-null values, skipped.")
            continue

        col_min, col_max = float(series.min()), float(series.max())
        st.sidebar.markdown(
            f"**{col}**  \n<span style='color:gray'>data range: "
            f"{col_min:g} to {col_max:g}</span>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.sidebar.columns(2)
        min_str = c1.text_input(
            "Min",
            value="",
            key=f"flt_num_min_{col}",
            placeholder=f"{col_min:g}",
        )
        max_str = c2.text_input(
            "Max",
            value="",
            key=f"flt_num_max_{col}",
            placeholder=f"{col_max:g}",
        )

        min_v = _parse_num(min_str)
        max_v = _parse_num(max_str)

        if min_str.strip() and min_v is None:
            st.sidebar.warning(f"`{col}`: Min value is not a number — ignored.")
        if max_str.strip() and max_v is None:
            st.sidebar.warning(f"`{col}`: Max value is not a number — ignored.")

        effective_min = min_v if min_v is not None else col_min
        effective_max = max_v if max_v is not None else col_max

        if effective_min > effective_max:
            st.sidebar.error(
                f"`{col}`: Min ({effective_min:g}) > Max ({effective_max:g}) — "
                "filter skipped."
            )
            continue

        # Only apply if the user actually narrowed the range.
        if min_v is not None or max_v is not None:
            mask &= df[col].between(effective_min, effective_max)
            applied.append(
                f"`{col}` ∈ [{effective_min:g}, {effective_max:g}]"
            )

    # -------- Categorical filters: multi-select per column --------
    if chosen_categorical:
        st.sidebar.markdown("### 🏷️ Categorical values")
        st.sidebar.caption("Leave empty to keep all values for that column.")

    for col in chosen_categorical:
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(
            col,
            options,
            default=[],
            key=f"flt_cat_{col}",
        )
        if selected:
            mask &= df[col].astype(str).isin(selected)
            applied.append(
                f"`{col}` in {len(selected)} of {len(options)} values"
            )

    filtered = df[mask].copy()

    # Reset button only when something is active.
    if applied:
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 Reset all filters"):
            for key in list(st.session_state.keys()):
                if key.startswith("flt_"):
                    del st.session_state[key]
            st.rerun()

    return filtered, _summary(df, filtered, applied)


def _parse_num(text: str) -> float | None:
    """Parse a numeric text input. Returns None for blank/invalid input."""
    s = (text or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _summary(
    original: pd.DataFrame,
    filtered: pd.DataFrame,
    applied: list[str],
) -> dict[str, Any]:
    total = len(original)
    kept = len(filtered)
    pct = (kept / total * 100) if total else 0.0
    return {
        "original_rows": total,
        "filtered_rows": kept,
        "percent_remaining": round(pct, 2),
        "dropped_rows": total - kept,
        "applied": applied,
        "is_filtered": len(applied) > 0,
    }


def download_button(filtered: pd.DataFrame, label: str = "Download filtered CSV") -> None:
    """Render a CSV download button for the filtered DataFrame."""
    if filtered is None or filtered.empty:
        st.caption("Nothing to download — filtered dataset is empty.")
        return
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name="filtered_dataset.csv",
        mime="text/csv",
    )
