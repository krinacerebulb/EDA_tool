"""User-driven preprocessing: column drop and manual dtype conversion.

Public API:
    safe_convert(series, target_dtype) -> (series, n_new_na, error)
    apply_preprocessing(df, dropped_cols, manual_dtypes) -> (df, warnings)
    render_preprocessing_ui(df) -> df          # renders Streamlit expander

Session-state keys used:
    dropped_cols      : list[str]
    manual_dtypes     : dict[col -> target_dtype]
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


SUPPORTED_DTYPES = ["int", "float", "string", "category", "datetime", "boolean"]

_TRUE_TOKENS = {"true", "yes", "y", "1", "t"}
_FALSE_TOKENS = {"false", "no", "n", "0", "f"}


def _to_bool(value):
    if pd.isna(value):
        return pd.NA
    s = str(value).strip().lower()
    if s in _TRUE_TOKENS:
        return True
    if s in _FALSE_TOKENS:
        return False
    return pd.NA


def safe_convert(series: pd.Series, target_dtype: str):
    """Convert a Series to ``target_dtype``; return (series, new_na, error).

    ``new_na`` counts values that became missing during the conversion (i.e.
    were non-null before and are null after). If the conversion fails entirely,
    the original series is returned with an error message.
    """
    n_before_na = int(series.isna().sum())
    try:
        if target_dtype == "int":
            converted = pd.to_numeric(series, errors="coerce").astype("Int64")
        elif target_dtype == "float":
            converted = pd.to_numeric(series, errors="coerce").astype(float)
        elif target_dtype == "string":
            converted = series.astype("string")
        elif target_dtype == "category":
            converted = series.astype("category")
        elif target_dtype == "datetime":
            try:
                converted = pd.to_datetime(series, errors="coerce", format="mixed")
            except (TypeError, ValueError):
                converted = pd.to_datetime(series, errors="coerce")
        elif target_dtype == "boolean":
            converted = series.map(_to_bool).astype("boolean")
        else:
            return series, 0, f"Unknown target dtype: {target_dtype}"
    except Exception as exc:
        return series, 0, str(exc)

    n_after_na = int(converted.isna().sum())
    new_na = max(0, n_after_na - n_before_na)
    return converted, new_na, None


@st.cache_data(show_spinner=False)
def apply_preprocessing(
    df: pd.DataFrame,
    dropped_cols: tuple,
    manual_dtypes_items: tuple,
):
    """Apply column drops and dtype conversions; return (df, warnings).

    Args are tuples (rather than ``list`` / ``dict``) so this function can be
    safely memoised by ``st.cache_data`` — tuples are hashable in a stable
    way and behave well as cache keys.
    """
    manual_dtypes = dict(manual_dtypes_items)
    out = df.copy()
    warnings: list[str] = []

    if dropped_cols:
        cols = [c for c in dropped_cols if c in out.columns]
        if cols:
            out = out.drop(columns=cols)

    for col, target in manual_dtypes.items():
        if col not in out.columns:
            continue
        converted, new_na, err = safe_convert(out[col], target)
        if err:
            warnings.append(f"`{col}` → {target} failed: {err}")
            continue
        out[col] = converted
        if new_na > 0:
            warnings.append(
                f"`{col}` → {target}: {new_na} value(s) could not be converted "
                "and became missing values."
            )

    return out, warnings


def _ensure_state():
    # Honour a pending reset BEFORE any widget that binds to these keys
    # instantiates this run — Streamlit forbids assigning to a widget-bound
    # session_state key after the widget has rendered.
    if st.session_state.pop("_ppx_reset_pending", False):
        st.session_state.pop("dropped_cols", None)
        st.session_state.pop("manual_dtypes", None)
    if "dropped_cols" not in st.session_state:
        st.session_state.dropped_cols = []
    if "manual_dtypes" not in st.session_state:
        st.session_state.manual_dtypes = {}


def render_preprocessing_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Render the preprocessing expander and return the preprocessed DataFrame.

    Selections persist in ``st.session_state``, so dropping columns or
    converting dtypes survives reruns and continues to apply on every render.
    """
    _ensure_state()

    with st.expander("⚙️ Preprocessing — drop columns & change dtypes", expanded=False):
        st.caption(
            "Customize the dataset before EDA. Selections persist across "
            "reruns and feed every downstream tab."
        )

        # ---- Column drop ----
        st.markdown("**Drop columns**")
        st.multiselect(
            "Select columns to drop from analysis",
            options=df.columns.tolist(),
            key="dropped_cols",
            help="Selected columns are removed from every tab and visualization.",
        )

        st.markdown("---")

        # ---- Dtype conversion ----
        st.markdown("**Convert column dtype**")
        remaining_cols = [c for c in df.columns if c not in st.session_state.dropped_cols]

        if not remaining_cols:
            st.caption("No columns left to convert — drop selections cover everything.")
        else:
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                col_to_convert = st.selectbox(
                    "Column",
                    options=["(select)"] + remaining_cols,
                    key="ppx_col_picker",
                )
            with c2:
                target_dtype = st.selectbox(
                    "Target dtype",
                    options=SUPPORTED_DTYPES,
                    key="ppx_dtype_picker",
                )
            with c3:
                st.markdown("&nbsp;", unsafe_allow_html=True)  # vertical alignment
                if st.button("Apply", key="ppx_apply"):
                    if col_to_convert != "(select)":
                        st.session_state.manual_dtypes[col_to_convert] = target_dtype

        # ---- Active conversions list ----
        if st.session_state.manual_dtypes:
            st.markdown("**Active conversions**")
            for col, dt in list(st.session_state.manual_dtypes.items()):
                cc1, cc2 = st.columns([5, 1])
                cc1.markdown(f"`{col}` → **{dt}**")
                if cc2.button("Remove", key=f"ppx_rm_{col}"):
                    st.session_state.manual_dtypes.pop(col, None)
                    st.rerun()

        # ---- Reset ----
        st.markdown("---")
        if st.button("Reset all preprocessing", key="ppx_reset"):
            # Defer the actual clear to the next run — see _ensure_state.
            st.session_state["_ppx_reset_pending"] = True
            st.rerun()

    # Apply outside the expander so warnings show inline on the main page.
    # Tuples make the call cacheable across reruns.
    out, warnings = apply_preprocessing(
        df,
        tuple(st.session_state.dropped_cols),
        tuple(sorted(st.session_state.manual_dtypes.items())),
    )
    for msg in warnings:
        st.warning(msg)

    return out
