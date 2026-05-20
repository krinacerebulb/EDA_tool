"""Sidebar-driven dynamic multi-filter system.

================================================================================
WHY THIS LIVES IN THE SIDEBAR
================================================================================
The sidebar is the canonical home for filtering controls in analytics SaaS
products (Tableau filters pane, Datadog query rail, Looker filter panel).
Keeping filters there leaves the main page free for visualizations, tables,
and the analysis tabs.

================================================================================
PUBLIC API
================================================================================
    render_filters(df) -> (filtered_df, summary)
        Renders the sidebar "🔍 Filters" panel and returns the filtered
        DataFrame plus a summary dict (used for downstream KPIs).

    render_sidebar_filters(df) -> (filtered_df, summary)
        Back-compat alias for ``render_filters`` — older imports keep working.

    download_button(df, label="…")
        Legacy CSV-only download. Kept for back-compat — new code should
        use ``modules.exporter.render_export_ui`` (multi-format).

================================================================================
DESIGN NOTES
================================================================================
* Renders inside ``with st.sidebar:`` so the whole panel scrolls
  vertically with the rest of the sidebar.
* Two-column layout used for Min / Max numeric inputs (they're short
  and benefit from being side-by-side); single-column elsewhere because
  the sidebar is narrow.
* All widgets persist via ``st.session_state`` keys prefixed ``flt_``,
  so the Reset button can wipe them in one pass.
* Inline impact summary at the bottom (compact text, not metric cards —
  cards stack ugly in narrow columns).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


# Columns with more unique values than this are treated as free-form
# text and skipped in the categorical filter UI.
_MAX_CATEGORICAL_UNIQUE = 200


def render_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Render the filter panel in the sidebar; return (filtered_df, summary)."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        st.caption(
            "Narrow the dataset before EDA. All filters combine with AND."
        )

        if df is None or df.empty:
            st.info("Dataset is empty — no filters to apply.")
            return df, _summary(df, df, [])

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        eligible_cats = [
            c for c in categorical_cols
            if df[c].nunique(dropna=True) <= _MAX_CATEGORICAL_UNIQUE
        ]

        # --- Column picker -----------------------------------------------
        # Single-column layout: the sidebar is too narrow for side-by-side
        # multi-selects (their pills wrap awkwardly).
        chosen_numeric = st.multiselect(
            "Numerical columns to filter",
            numeric_cols,
            default=[],
            key="flt_choose_num",
            help="Each chosen column gets a Min / Max input below.",
            placeholder="Pick numeric columns…",
        )
        chosen_categorical = st.multiselect(
            "Categorical columns to filter",
            eligible_cats,
            default=[],
            key="flt_choose_cat",
            help=(
                "Each chosen column gets a multi-select below. "
                f"Columns with more than {_MAX_CATEGORICAL_UNIQUE} distinct "
                "values are excluded."
            ),
            placeholder="Pick categorical columns…",
        )

        mask = pd.Series(True, index=df.index)
        applied: list[str] = []

        # --- Numerical filters -------------------------------------------
        if chosen_numeric:
            st.markdown("##### Numerical ranges")
            st.caption("Leave a field blank to leave that bound open.")
            for col in chosen_numeric:
                series = df[col].dropna()
                if series.empty:
                    st.caption(f"`{col}`: no non-null values — skipped.")
                    continue
                col_min, col_max = float(series.min()), float(series.max())

                # Column label + data-range hint on its own line, then
                # Min / Max side-by-side underneath.
                st.markdown(
                    f"**{col}**  \n"
                    f"<span style='color:#64748B;font-size:0.82em'>"
                    f"range: {col_min:g} → {col_max:g}</span>",
                    unsafe_allow_html=True,
                )
                cc1, cc2 = st.columns(2)
                with cc1:
                    min_str = st.text_input(
                        "Min",
                        value="",
                        key=f"flt_num_min_{col}",
                        placeholder=f"{col_min:g}",
                    )
                with cc2:
                    max_str = st.text_input(
                        "Max",
                        value="",
                        key=f"flt_num_max_{col}",
                        placeholder=f"{col_max:g}",
                    )

                min_v = _parse_num(min_str)
                max_v = _parse_num(max_str)

                if min_str.strip() and min_v is None:
                    st.warning(f"`{col}`: Min is not a number — ignored.")
                if max_str.strip() and max_v is None:
                    st.warning(f"`{col}`: Max is not a number — ignored.")

                effective_min = min_v if min_v is not None else col_min
                effective_max = max_v if max_v is not None else col_max

                if effective_min > effective_max:
                    st.error(
                        f"`{col}`: Min ({effective_min:g}) > Max "
                        f"({effective_max:g}) — filter skipped."
                    )
                    continue

                if min_v is not None or max_v is not None:
                    mask &= df[col].between(effective_min, effective_max)
                    applied.append(
                        f"`{col}` ∈ [{effective_min:g}, {effective_max:g}]"
                    )

        # --- Categorical filters -----------------------------------------
        if chosen_categorical:
            st.markdown("##### Categorical values")
            st.caption("Leave a multi-select empty to keep all values.")
            for col in chosen_categorical:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                selected = st.multiselect(
                    col,
                    options,
                    default=[],
                    key=f"flt_cat_{col}",
                    placeholder=f"Pick values for {col}…",
                )
                if selected:
                    mask &= df[col].astype(str).isin(selected)
                    applied.append(
                        f"`{col}` in {len(selected)} of {len(options)} values"
                    )

        filtered = df[mask].copy()
        summary = _summary(df, filtered, applied)

        # --- Impact summary + reset --------------------------------------
        st.markdown("---")
        if applied:
            # Compact text summary instead of metric cards — cards stack
            # awkwardly in the narrow sidebar column.
            st.markdown(
                f"**{summary['filtered_rows']:,}** of "
                f"**{summary['original_rows']:,}** rows kept "
                f"({summary['percent_remaining']}%)"
            )
            st.caption(f"{summary['dropped_rows']:,} rows removed by filters")

            if st.button(
                "🔄 Reset all filters",
                key="flt_reset_sidebar",
                use_container_width=True,
                help="Clear every active filter and the column-picker selections.",
            ):
                for key in list(st.session_state.keys()):
                    if key.startswith("flt_"):
                        del st.session_state[key]
                st.rerun()

            with st.expander(
                f"Active filters ({len(applied)})", expanded=False,
            ):
                for rule in applied:
                    st.markdown(f"- {rule}")
        else:
            st.caption(
                f"✅ No filters active — running on full **{len(df):,}**-row dataset."
            )

    return filtered, summary


# Back-compat alias: older code may import ``render_sidebar_filters``.
render_sidebar_filters = render_filters


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
    """Legacy CSV-only download. Prefer ``modules.exporter.render_export_ui``."""
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
