"""Sidebar-driven dynamic multi-filter system, form-gated.

================================================================================
WHY THIS IS FORM-GATED
================================================================================
Filters drive every downstream cached analytic. A single keystroke in a
Min textbox would otherwise:

    1. trigger a full Streamlit rerun,
    2. recompute the filter mask,
    3. invalidate every cached EDA computation that depends on the
       filtered DataFrame (overview, stats, missing, duplicates, outliers,
       insights, all chart paths…),
    4. re-render every visible tab.

On a 200k-row × 100-column industrial dataset that cascade takes several
seconds. The fix: put the per-column filter inputs inside an
``st.form()``. Streamlit batches widget changes inside a form until the
``form_submit_button`` is clicked, so analysis only re-runs when the
user explicitly applies their filters.

Column pickers (which decide WHICH filter widgets to render) sit OUTSIDE
the form so they update the form's content immediately — picking a
column then submitting an empty filter for it is the natural flow.

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
SESSION STATE KEYS
================================================================================
    flt_choose_num    : list[str] — picked numerical columns (outside form)
    flt_choose_cat    : list[str] — picked categorical columns (outside form)
    flt_num_min_{col} : str       — committed Min for that column
    flt_num_max_{col} : str       — committed Max for that column
    flt_cat_{col}     : list[str] — committed selected values

All keys are prefixed ``flt_`` so the Reset button can wipe them in one pass.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


# Columns with more unique values than this are treated as free-form
# text and skipped in the categorical filter UI.
_MAX_CATEGORICAL_UNIQUE = 200


def render_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Render the filter panel in the sidebar; return (filtered_df, summary).

    Behaviour:
        * Column-picker changes show / hide form widgets immediately.
        * Min/Max/multiselect changes inside the form are buffered.
        * Nothing downstream re-runs until the user clicks **Apply Filters**.
        * Reset button (outside the form) wipes all filter state immediately.
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        st.caption(
            "Pick columns, configure values, then click **Apply Filters**. "
            "No analysis re-runs until you apply."
        )

        if df is None or df.empty:
            st.info("Dataset is empty — no filters to apply.")
            return df, _summary(df, df, [])

        # Use per-column dtype probes instead of ``select_dtypes`` here —
        # ``select_dtypes`` copies the matching block, which on a wide
        # industrial dataset (e.g. 285k rows × 12 numeric columns) can
        # blow past available RAM just to enumerate column names.
        numeric_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        categorical_cols = [
            c for c in df.columns
            if (
                pd.api.types.is_object_dtype(df[c])
                or isinstance(df[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(df[c])
            )
        ]
        eligible_cats = [
            c for c in categorical_cols
            if df[c].nunique(dropna=True) <= _MAX_CATEGORICAL_UNIQUE
        ]

        # --- Column pickers (OUTSIDE the form) ----------------------------
        # Outside the form so picking a new column immediately renders its
        # Min/Max or multi-select. These pickers are cheap — they don't
        # touch the dataframe.
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

        # --- Per-column filter widgets (INSIDE the form) ------------------
        # Streamlit batches widget changes inside a form until submit, so
        # nothing downstream re-runs while the user is typing / picking
        # values. The Apply button is what actually triggers the cascade.
        with st.form("flt_apply_form", clear_on_submit=False, border=False):
            if not chosen_numeric and not chosen_categorical:
                st.caption(
                    "Pick one or more columns above to configure filters."
                )

            if chosen_numeric:
                st.markdown("##### Numerical ranges")
                st.caption("Leave a field blank to leave that bound open.")
                for col in chosen_numeric:
                    series = df[col].dropna()
                    if series.empty:
                        st.caption(f"`{col}`: no non-null values — skipped.")
                        continue
                    col_min, col_max = float(series.min()), float(series.max())

                    st.markdown(
                        f"**{col}**  \n"
                        f"<span style='color:#64748B;font-size:0.82em'>"
                        f"range: {col_min:g} → {col_max:g}</span>",
                        unsafe_allow_html=True,
                    )
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.text_input(
                            "Min",
                            value="",
                            key=f"flt_num_min_{col}",
                            placeholder=f"{col_min:g}",
                        )
                    with cc2:
                        st.text_input(
                            "Max",
                            value="",
                            key=f"flt_num_max_{col}",
                            placeholder=f"{col_max:g}",
                        )

            if chosen_categorical:
                st.markdown("##### Categorical values")
                st.caption("Leave a multi-select empty to keep all values.")
                for col in chosen_categorical:
                    options = sorted(
                        df[col].dropna().astype(str).unique().tolist()
                    )
                    st.multiselect(
                        col,
                        options,
                        default=[],
                        key=f"flt_cat_{col}",
                        placeholder=f"Pick values for {col}…",
                    )

            st.form_submit_button(
                "✅ Apply Filters",
                type="primary",
                use_container_width=True,
            )

        # --- Compute mask from the LAST-COMMITTED values ------------------
        # st.session_state holds the last-submitted value for each form
        # widget. Reading from session_state (not from local variables) is
        # what makes the filter "sticky" — the mask reflects the last
        # applied state regardless of in-progress edits.
        mask = pd.Series(True, index=df.index)
        applied: list[str] = []

        for col in chosen_numeric:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            col_min, col_max = float(series.min()), float(series.max())

            min_str = st.session_state.get(f"flt_num_min_{col}", "") or ""
            max_str = st.session_state.get(f"flt_num_max_{col}", "") or ""
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

        for col in chosen_categorical:
            if col not in df.columns:
                continue
            selected = list(st.session_state.get(f"flt_cat_{col}", []) or [])
            if selected:
                # Compute options count so the rule message is informative
                # without scanning the full column twice.
                options_count = df[col].nunique(dropna=True)
                mask &= df[col].astype(str).isin(selected)
                applied.append(
                    f"`{col}` in {len(selected)} of {options_count} values"
                )

        filtered = df[mask].copy()
        summary = _summary(df, filtered, applied)

        # --- Impact summary + reset (outside the form) --------------------
        st.markdown("---")
        if applied:
            # Compact text summary — metric cards stack ugly in a narrow
            # sidebar column.
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
