"""User-driven preprocessing: column drop, manual dtype conversion, custom
datetime formatting (Excel-style format builder + numeric-encoding
conversion + date/time merge), and missing-value handling.

Everything is rendered inside a single ⚙️ Preprocessing expander split
across five tabs so non-technical users see one clean workflow:
  1. Columns       — drop columns from analysis
  2. Data Types    — manually change a column's dtype
  3. Date & Time   — parse strings, decode Excel/Unix timestamps, merge date+time
  4. Missing Values— per-column strategies (drop / mean / median / mode / fill)
  5. Preview       — before/after diff + active-operations summary + reset

Public API:
    safe_convert(series, target_dtype) -> (series, n_new_na, error)
    apply_preprocessing(df, dropped_cols, manual_dtypes, dt_conversions,
                        dt_merges, dt_numeric, missing_value_items)
                                                 -> (df, warnings)
    render_preprocessing_ui(df) -> df

Session-state keys used:
    dropped_cols          : list[str]
    manual_dtypes         : dict[col -> target_dtype]
    dt_conversions        : dict[col -> python_format_str]   ("" = pandas inference)
    dt_merges             : list[dict]                       (new_col, date_col, time_col)
    dt_numeric_conversions: dict[src_col -> {"mode", "target"}]
    missing_value_ops     : dict[col -> {"strategy", "fill_value"}]
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import datetime_formatter as dtfmt


SUPPORTED_DTYPES = ["int", "float", "string", "category", "datetime", "boolean"]

_TRUE_TOKENS = {"true", "yes", "y", "1", "t"}
_FALSE_TOKENS = {"false", "no", "n", "0", "f"}

# Display labels for missing-value strategies. Keys are the persistence form
# (small, stable, machine-friendly); values are what's shown to the user.
MISSING_STRATEGIES: dict[str, str] = {
    "drop_rows": "Drop rows with missing values",
    "mean":      "Fill with column mean (numeric only)",
    "median":    "Fill with column median (numeric only)",
    "mode":      "Fill with most-common value",
    "constant":  "Fill with a custom value",
    "ffill":     "Forward fill (use previous row's value)",
    "bfill":     "Backward fill (use next row's value)",
}


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


# --- Missing-value handling helpers -----------------------------------------


def _coerce_fill_value(value, dtype):
    """Best-effort coercion of a user-supplied fill value to ``dtype``.

    Returns the coerced value if it converts cleanly, otherwise the raw
    string — pandas will broadcast/upcast as needed and we let the warning
    surface any odd cases.
    """
    if value is None or value == "":
        return value
    try:
        if pd.api.types.is_numeric_dtype(dtype):
            return pd.to_numeric(value)
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.to_datetime(value)
        if pd.api.types.is_bool_dtype(dtype):
            return _to_bool(value)
    except (TypeError, ValueError):
        pass
    return value


def _apply_missing_value_ops(
    df: pd.DataFrame,
    items: tuple,
    warnings: list[str],
) -> pd.DataFrame:
    """Apply each (col, strategy, fill_value) rule against ``df``.

    Row-dropping rules accumulate (drop_rows on col A then col B drops the
    union). Fills only touch the named column. Warns when a strategy
    doesn't apply (e.g. mean on a string column).
    """
    out = df
    for col, strategy, fill_value in items:
        if col not in out.columns:
            warnings.append(
                f"`{col}` missing-value rule skipped: column not present."
            )
            continue
        s = out[col]
        n_before = int(s.isna().sum())
        if n_before == 0 and strategy != "drop_rows":
            # Nothing to fill — silently skip (no warning, not a problem).
            continue
        try:
            if strategy == "drop_rows":
                out = out.dropna(subset=[col])
            elif strategy == "mean":
                if not pd.api.types.is_numeric_dtype(s):
                    warnings.append(
                        f"`{col}` mean fill skipped: column is not numeric."
                    )
                    continue
                out = out.assign(**{col: s.fillna(s.mean())})
            elif strategy == "median":
                if not pd.api.types.is_numeric_dtype(s):
                    warnings.append(
                        f"`{col}` median fill skipped: column is not numeric."
                    )
                    continue
                out = out.assign(**{col: s.fillna(s.median())})
            elif strategy == "mode":
                mode = s.mode(dropna=True)
                if mode.empty:
                    warnings.append(
                        f"`{col}` mode fill skipped: no non-null values."
                    )
                    continue
                out = out.assign(**{col: s.fillna(mode.iloc[0])})
            elif strategy == "constant":
                coerced = _coerce_fill_value(fill_value, s.dtype)
                out = out.assign(**{col: s.fillna(coerced)})
            elif strategy == "ffill":
                out = out.assign(**{col: s.ffill()})
            elif strategy == "bfill":
                out = out.assign(**{col: s.bfill()})
            else:
                warnings.append(
                    f"`{col}` missing-value rule skipped: unknown strategy '{strategy}'."
                )
        except Exception as exc:
            warnings.append(f"`{col}` missing-value rule failed: {exc}")
    return out


# --- Cached apply pipeline --------------------------------------------------


@st.cache_data(show_spinner=False)
def apply_preprocessing(
    df: pd.DataFrame,
    dropped_cols: tuple,
    manual_dtypes_items: tuple,
    dt_conversions_items: tuple = (),
    dt_merges_items: tuple = (),
    dt_numeric_items: tuple = (),
    missing_value_items: tuple = (),
):
    """Apply column drops, datetime ops, dtype changes, and missing-value rules.

    Args are tuples (rather than ``list`` / ``dict``) so this function can be
    safely memoised by ``st.cache_data`` — tuples are hashable in a stable
    way and behave well as cache keys.

    ``dt_numeric_items`` is a tuple of ``(src_col, mode, target_col)`` triples.
    ``missing_value_items`` is a tuple of ``(col, strategy, fill_value)`` triples.

    Order:
        1. drop columns
        2. numeric datetime conversions (Excel serial / Unix ts → datetime)
        3. string datetime conversions (custom format, per column)
        4. date+time merges (create new datetime column)
        5. general dtype conversions (skips cols already datetime-handled)
        6. missing-value strategies (last, so mean/median see final dtypes)
    """
    manual_dtypes = dict(manual_dtypes_items)
    dt_conversions = dict(dt_conversions_items)
    out = df.copy()
    warnings: list[str] = []

    if dropped_cols:
        cols = [c for c in dropped_cols if c in out.columns]
        if cols:
            out = out.drop(columns=cols)

    # --- Numeric → datetime conversions (Excel serial, Unix s, Unix ms). ---
    datetime_handled: set[str] = set()
    for src_col, mode, target_col in dt_numeric_items:
        if src_col not in out.columns:
            warnings.append(
                f"`{src_col}` numeric→datetime skipped: column not present."
            )
            continue
        parsed, stats = dtfmt.convert_numeric_datetime(out[src_col], mode)
        out[target_col] = parsed
        if target_col == src_col:
            datetime_handled.add(src_col)
        label = dtfmt.NUMERIC_DT_MODES.get(mode, mode)
        if stats["failed"] > 0:
            warnings.append(
                f"`{target_col}` ({label}): {stats['failed']} value(s) "
                f"became NaT ({stats['percent']}% parsed successfully)."
            )

    # --- Custom string datetime conversions ---
    for col, py_fmt in dt_conversions.items():
        if col not in out.columns:
            continue
        parsed, stats = dtfmt.parse_with_format(out[col], py_fmt or None)
        out[col] = parsed
        datetime_handled.add(col)
        if stats["failed"] > 0:
            warnings.append(
                f"`{col}` → datetime: {stats['failed']} value(s) became NaT "
                f"({stats['percent']}% parsed successfully)."
            )

    # --- Date + Time column merges → new datetime column ---
    for merge_items in dt_merges_items:
        m = dict(merge_items)
        new_col = m.get("new_col")
        date_col = m.get("date_col")
        time_col = m.get("time_col")
        if not new_col or date_col not in out.columns or time_col not in out.columns:
            warnings.append(
                f"Merged column `{new_col}` skipped: source column missing."
            )
            continue
        combined, stats = dtfmt.combine_date_time(out[date_col], out[time_col])
        out[new_col] = combined
        if stats["failed"] > 0:
            warnings.append(
                f"`{new_col}` (combined): {stats['failed']} row(s) could not "
                f"be parsed and became NaT ({stats['percent']}% parsed)."
            )

    # --- General dtype conversions (skip columns already datetime-converted) ---
    for col, target in manual_dtypes.items():
        if col not in out.columns or col in datetime_handled:
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

    # --- Missing-value handling (runs last so it sees final dtypes) ---
    if missing_value_items:
        out = _apply_missing_value_ops(out, missing_value_items, warnings)

    return out, warnings


# --- Session state ----------------------------------------------------------


def _ensure_state():
    # Honour a pending reset BEFORE any widget that binds to these keys
    # instantiates this run — Streamlit forbids assigning to a widget-bound
    # session_state key after the widget has rendered.
    if st.session_state.pop("_ppx_reset_pending", False):
        for k in (
            "dropped_cols", "manual_dtypes", "dt_conversions",
            "dt_merges", "dt_numeric_conversions", "missing_value_ops",
        ):
            st.session_state.pop(k, None)
    st.session_state.setdefault("dropped_cols", [])
    st.session_state.setdefault("manual_dtypes", {})
    st.session_state.setdefault("dt_conversions", {})
    st.session_state.setdefault("dt_merges", [])
    st.session_state.setdefault("dt_numeric_conversions", {})
    st.session_state.setdefault("missing_value_ops", {})


def _datetime_candidates(df: pd.DataFrame) -> list[str]:
    """Columns that could plausibly hold a date/time string."""
    out = []
    dropped = set(st.session_state.dropped_cols)
    converted = set(st.session_state.dt_conversions)
    for c in df.columns:
        if c in dropped or c in converted:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue
        if s.dtype not in ("object", "string", "category"):
            continue
        out.append(c)
    return out


# --- Tab renderers ----------------------------------------------------------


def _render_column_mgmt(df: pd.DataFrame) -> None:
    st.caption(
        "Remove columns you don't need before analysis. Dropped columns "
        "disappear from every downstream tab and the report."
    )
    st.multiselect(
        "Columns to drop",
        options=df.columns.tolist(),
        key="dropped_cols",
    )
    if st.session_state.dropped_cols:
        st.markdown(
            f"**{len(st.session_state.dropped_cols)}** column(s) marked for "
            "removal."
        )


def _render_dtype_section(df: pd.DataFrame) -> None:
    st.caption(
        "Force a column to a specific type — useful when auto-detection "
        "got it wrong (e.g. numeric IDs read as strings)."
    )
    remaining_cols = [
        c for c in df.columns if c not in st.session_state.dropped_cols
    ]
    if not remaining_cols:
        st.caption("No columns left to convert.")
        return

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
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("Apply", key="ppx_apply"):
            if col_to_convert != "(select)":
                st.session_state.manual_dtypes[col_to_convert] = target_dtype
                st.rerun()

    st.caption(
        "Tip: for date/time columns with non-standard formats, use the "
        "**Date & Time** tab instead — it has a format builder, live "
        "preview, and parse statistics."
    )


def _render_dt_string(df: pd.DataFrame) -> None:
    """String → datetime conversion (suggestions, format builder, preview)."""
    candidate_cols = _datetime_candidates(df)
    if not candidate_cols:
        st.caption(
            "No string/object columns available. Already-datetime columns "
            "and active conversions are filtered out."
        )
        return
    target_col = st.selectbox(
        "Datetime-like column",
        options=["(select)"] + candidate_cols,
        key="dtfmt_col",
        help="Pick the column whose date/time values you want to parse.",
    )
    if target_col == "(select)":
        return

    series = df[target_col]
    suggestions = dtfmt.suggest_formats(series, top_n=5)

    if suggestions:
        st.markdown("**Possible detected formats** — click *Use* to load.")
        for i, sug in enumerate(suggestions):
            sc1, sc2, sc3 = st.columns([5, 2, 1])
            sc1.markdown(f"`{sug['label']}`")
            sc2.markdown(f"**{sug['percent']}%** parsed")
            if sc3.button("Use", key=f"dtfmt_use_{target_col}_{i}"):
                st.session_state["dtfmt_builder_input"] = sug["label"]
                st.rerun()
        if suggestions[0]["percent"] < 80:
            st.warning(
                "Could not confidently parse this column. "
                "Please refine the custom format below."
            )
    else:
        st.warning(
            "No preset matched this column. Build a custom format below "
            "using the tokens shown in the caption."
        )

    st.markdown("**Format builder**")
    st.caption(
        "Tokens: `YYYY` year · `MM` month · `DD` day · `HH` hour (24h) · "
        "`mm` minute · `ss` second · `ms` millisecond. Mix with `-`, `/`, "
        "`:`, `.`, space, `T` as separators."
    )
    builder_input = st.text_input(
        "Format string",
        key="dtfmt_builder_input",
        placeholder="YYYY-MM-DD HH:mm:ss",
        help=(
            "Example: `YYYY-MM-DD HH:mm:ss` → matches "
            "`2025-05-12 14:22:45`. Leave empty for pandas auto-inference."
        ),
    )
    python_fmt = dtfmt.builder_to_python(builder_input).strip()
    if not python_fmt:
        st.caption(
            "No format specified — falling back to pandas auto-inference "
            "(less reliable on industrial data)."
        )

    parsed_series, stats = dtfmt.parse_with_format(series, python_fmt or None)
    preview_df = dtfmt.preview_frame(series.dropna(), parsed_series, n=10)
    st.markdown("**Live preview**")
    st.dataframe(preview_df, use_container_width=True)

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Total rows", f"{stats['total']:,}")
    pc2.metric("Parsed", f"{stats['parsed']:,}")
    pc3.metric("Failed (→NaT)", f"{stats['failed']:,}")
    pc4.metric("Success", f"{stats['percent']}%")

    if stats["failed"] > 0:
        st.warning(
            f"{stats['failed']} row(s) could not be parsed and would "
            "become NaT after applying."
        )
        failed_mask = series.notna() & parsed_series.isna()
        failed_rows = series[failed_mask]
        with st.expander(
            f"Show failed values ({stats['failed']:,})", expanded=False,
        ):
            unique_failed = (
                failed_rows.astype("string")
                .value_counts()
                .head(20)
                .rename_axis("Raw Value")
                .reset_index(name="Occurrences")
            )
            st.dataframe(unique_failed, use_container_width=True, hide_index=True)
            st.caption(
                f"Top 20 distinct values out of {failed_rows.nunique():,} "
                "unique unparseable entries."
            )

    if st.button("Apply conversion", key=f"dtfmt_apply_{target_col}"):
        st.session_state.dt_conversions[target_col] = python_fmt or ""
        st.success(
            f"`{target_col}` will be parsed as datetime on every "
            "downstream tab."
        )
        st.rerun()


def _render_dt_numeric(df: pd.DataFrame) -> None:
    """Numeric → datetime (Excel serial / Unix timestamp)."""
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and c not in st.session_state.dropped_cols
    ]
    st.caption(
        "Industrial / IoT exports often store timestamps as numbers: "
        "Excel serial dates (e.g. `46004.04167` → `2025-12-13 01:00`) or "
        "Unix epochs. Convert them here so they show up in Time Series."
    )
    if not numeric_cols:
        st.caption("No numeric columns available to convert.")
        return

    num_target = st.selectbox(
        "Numeric column",
        options=["(select)"] + numeric_cols,
        key="dtfmt_num_col",
    )
    if num_target == "(select)":
        return

    num_series = df[num_target]
    detected_mode, scores = dtfmt.detect_numeric_datetime_mode(num_series)
    if detected_mode:
        st.success(
            f"💡 Detected possible **{dtfmt.NUMERIC_DT_MODES[detected_mode]}** "
            f"format ({scores[detected_mode]:.0f}% of values fall in the "
            "expected range)."
        )
    else:
        st.caption(
            "No encoding crossed the auto-detect threshold. Pick a mode "
            "manually below if you know what the values represent."
        )

    mode_options = list(dtfmt.NUMERIC_DT_MODES.keys())
    default_idx = (
        mode_options.index(detected_mode)
        if detected_mode in mode_options else 0
    )
    chosen_mode = st.selectbox(
        "Convert column as",
        options=mode_options,
        index=default_idx,
        format_func=lambda m: dtfmt.NUMERIC_DT_MODES[m],
        key=f"dtfmt_num_mode_{num_target}",
    )

    overwrite = st.checkbox(
        "Overwrite the original column",
        value=False,
        key=f"dtfmt_num_overwrite_{num_target}",
        help=(
            "Off: create a new `<col>_datetime` column and keep raw "
            "numerics. On: replace the column in place."
        ),
    )
    default_new_name = f"{num_target}_datetime"
    if overwrite:
        new_name = num_target
    else:
        new_name = st.text_input(
            "New column name",
            value=default_new_name,
            key=f"dtfmt_num_newname_{num_target}",
        ).strip() or default_new_name

    parsed_num, num_stats = dtfmt.convert_numeric_datetime(num_series, chosen_mode)
    non_null_idx = num_series.dropna().head(10).index
    preview_num = pd.DataFrame({
        "Raw Value": num_series.loc[non_null_idx].astype("string").values,
        "Converted Datetime": parsed_num.loc[non_null_idx]
            .astype("string").fillna("(unparsed)").values,
    })
    st.markdown("**Live preview**")
    st.dataframe(preview_num, use_container_width=True)

    nc1, nc2, nc3, nc4 = st.columns(4)
    nc1.metric("Total rows", f"{num_stats['total']:,}")
    nc2.metric("Parsed", f"{num_stats['parsed']:,}")
    nc3.metric("Failed (→NaT)", f"{num_stats['failed']:,}")
    nc4.metric("Success", f"{num_stats['percent']}%")

    if num_stats["failed"] > 0:
        st.warning(
            f"{num_stats['failed']} value(s) fell outside the valid range "
            "for this encoding and would become NaT."
        )
        num_failed_mask = num_series.notna() & parsed_num.isna()
        num_failed_rows = num_series[num_failed_mask]
        with st.expander(
            f"Show failed values ({num_stats['failed']:,})", expanded=False,
        ):
            unique_failed_num = (
                num_failed_rows.astype("string")
                .value_counts()
                .head(20)
                .rename_axis("Raw Value")
                .reset_index(name="Occurrences")
            )
            st.dataframe(
                unique_failed_num, use_container_width=True, hide_index=True,
            )

    if st.button("Apply conversion", key=f"dtfmt_num_apply_{num_target}"):
        if not overwrite and new_name in df.columns and new_name != num_target:
            st.error(
                f"Column `{new_name}` already exists — choose a different "
                "name or enable overwrite."
            )
        else:
            st.session_state.dt_numeric_conversions[num_target] = {
                "mode": chosen_mode,
                "target": new_name,
            }
            label = dtfmt.NUMERIC_DT_MODES[chosen_mode]
            if overwrite:
                st.success(f"`{num_target}` will be converted in place ({label}).")
            else:
                st.success(
                    f"`{new_name}` will be created from `{num_target}` "
                    f"({label})."
                )
            st.rerun()


def _render_dt_merge(df: pd.DataFrame) -> None:
    """Combine separate date + time columns into a new datetime column."""
    candidate_cols = _datetime_candidates(df)
    st.caption(
        "Industrial datasets often split date and time into separate "
        "columns. This creates a single combined datetime column without "
        "removing the originals."
    )
    mergeable = candidate_cols + [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
        and c not in st.session_state.dropped_cols
        and c not in candidate_cols
    ]
    if len(mergeable) < 2:
        st.caption("Need at least two date/time-like columns to combine.")
        return

    mc1, mc2 = st.columns(2)
    with mc1:
        merge_date = st.selectbox(
            "Date column",
            options=["(select)"] + mergeable,
            key="dtfmt_merge_date",
        )
    with mc2:
        time_options = ["(select)"] + [c for c in mergeable if c != merge_date]
        merge_time = st.selectbox(
            "Time column",
            options=time_options,
            key="dtfmt_merge_time",
        )
    new_col_name = st.text_input(
        "New column name",
        value=st.session_state.get("dtfmt_merge_name", "combined_datetime"),
        key="dtfmt_merge_name",
    )

    if merge_date == "(select)" or merge_time == "(select)":
        return

    combined, stats = dtfmt.combine_date_time(df[merge_date], df[merge_time])
    preview = pd.DataFrame({
        "Date":     df[merge_date].head(10).astype("string").fillna("(null)").values,
        "Time":     df[merge_time].head(10).astype("string").fillna("(null)").values,
        "Combined": combined.head(10).astype("string").fillna("(unparsed)").values,
    })
    st.markdown("**Live preview**")
    st.dataframe(preview, use_container_width=True)

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Parsed", f"{stats['parsed']:,}")
    sc2.metric("Failed", f"{stats['failed']:,}")
    sc3.metric("Success", f"{stats['percent']}%")

    if st.button("Add merged column", key="dtfmt_merge_apply"):
        name = (new_col_name or "").strip()
        if not name:
            st.error("Choose a name for the new column.")
        elif name in df.columns:
            st.error(
                f"Column `{name}` already exists — choose a different name."
            )
        else:
            existing = next(
                (m for m in st.session_state.dt_merges if m["new_col"] == name),
                None,
            )
            if existing:
                existing.update({"date_col": merge_date, "time_col": merge_time})
            else:
                st.session_state.dt_merges.append({
                    "new_col": name,
                    "date_col": merge_date,
                    "time_col": merge_time,
                })
            st.success(f"`{name}` will be added as a datetime column.")
            st.rerun()


def _render_datetime_section(df: pd.DataFrame) -> None:
    """Container for the three datetime operations, picked via radio."""
    # Auto-detection banner for numeric datetime candidates.
    numeric_candidates = dtfmt.detect_numeric_datetime_columns(df)
    unconfigured = [
        c for c in numeric_candidates
        if c["column"] not in st.session_state.dt_numeric_conversions
    ]
    if unconfigured:
        preview_names = ", ".join(
            f"`{c['column']}` ({dtfmt.NUMERIC_DT_MODES[c['mode']]})"
            for c in unconfigured[:3]
        )
        more = (
            f" and {len(unconfigured) - 3} more"
            if len(unconfigured) > 3 else ""
        )
        st.info(
            f"💡 Detected possible numeric datetime column(s): "
            f"{preview_names}{more}. Switch the operation type to "
            "**From number** to convert them."
        )

    op = st.radio(
        "Operation",
        options=[
            "From text",
            "From number (Excel / Unix)",
            "Combine date + time columns",
        ],
        horizontal=True,
        key="dt_op_type",
    )
    st.markdown("")  # vertical breathing room
    if op == "From text":
        _render_dt_string(df)
    elif op == "From number (Excel / Unix)":
        _render_dt_numeric(df)
    else:
        _render_dt_merge(df)


def _render_missing_section(df: pd.DataFrame) -> None:
    """Per-column missing-value strategies."""
    st.caption(
        "Configure a strategy per column. Rules apply automatically on "
        "every rerun and persist across reruns."
    )
    miss_mask = df.isna().any()
    miss_cols = miss_mask[miss_mask].index.tolist()
    miss_cols = [c for c in miss_cols if c not in st.session_state.dropped_cols]

    if not miss_cols:
        st.success("No remaining columns have missing values.")
        return

    miss_summary = pd.DataFrame({
        "Column":     miss_cols,
        "Missing":    [int(df[c].isna().sum()) for c in miss_cols],
        "Missing %":  [f"{df[c].isna().mean() * 100:.2f}%" for c in miss_cols],
        "Dtype":      [str(df[c].dtype) for c in miss_cols],
    })
    st.dataframe(miss_summary, use_container_width=True, hide_index=True)

    c1, c2 = st.columns([3, 3])
    with c1:
        col = st.selectbox(
            "Column",
            options=["(select)"] + miss_cols,
            key="mv_col",
        )
    with c2:
        if col == "(select)":
            available_keys = list(MISSING_STRATEGIES.keys())
        else:
            # Hide mean/median for non-numeric columns.
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            available_keys = [
                k for k in MISSING_STRATEGIES.keys()
                if is_numeric or k not in {"mean", "median"}
            ]
        strategy = st.selectbox(
            "Strategy",
            options=available_keys,
            format_func=lambda k: MISSING_STRATEGIES[k],
            key="mv_strategy",
        )

    fill_value = ""
    if strategy == "constant":
        fill_value = st.text_input(
            "Fill value",
            value="",
            key="mv_fill_value",
            help="The value to substitute for missing entries.",
        )

    if st.button("Apply missing-value rule", key="mv_apply"):
        if col == "(select)":
            st.error("Pick a column first.")
        elif strategy == "constant" and not fill_value:
            st.error("Provide a fill value for the *constant* strategy.")
        else:
            st.session_state.missing_value_ops[col] = {
                "strategy": strategy,
                "fill_value": fill_value,
            }
            st.success(f"`{col}` → {MISSING_STRATEGIES[strategy]}")
            st.rerun()


def _render_preview_section(df_in: pd.DataFrame, df_out: pd.DataFrame) -> None:
    """Final review: operation summary + before/after diff + reset."""
    op_counts = {
        "Columns dropped":      len(st.session_state.dropped_cols),
        "Dtype conversions":    len(st.session_state.manual_dtypes),
        "Datetime (string)":    len(st.session_state.dt_conversions),
        "Datetime (numeric)":   len(st.session_state.dt_numeric_conversions),
        "Date + Time merges":   len(st.session_state.dt_merges),
        "Missing-value rules":  len(st.session_state.missing_value_ops),
    }
    total = sum(op_counts.values())

    if total == 0:
        st.info(
            "No preprocessing operations configured. Use the other tabs to "
            "drop columns, change types, parse dates, or fill missing values."
        )
        return

    st.markdown(f"**{total} operation(s) active**")
    summary_df = pd.DataFrame(
        [(k, v) for k, v in op_counts.items() if v > 0],
        columns=["Operation", "Count"],
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # --- Before / after diff ---
    in_cols = list(df_in.columns)
    out_cols = list(df_out.columns)
    dropped = [c for c in in_cols if c not in out_cols]
    added   = [c for c in out_cols if c not in in_cols]
    changed_dtype = [
        c for c in in_cols
        if c in out_cols and str(df_in[c].dtype) != str(df_out[c].dtype)
    ]

    if dropped or added or changed_dtype:
        st.markdown("---")
        st.markdown("**Before / after**")

        dtype_rows: list[dict] = []
        for c in changed_dtype:
            dtype_rows.append({
                "Column": c,
                "Before": str(df_in[c].dtype),
                "After":  str(df_out[c].dtype),
            })
        for c in dropped:
            dtype_rows.append({
                "Column": c, "Before": str(df_in[c].dtype), "After": "(dropped)",
            })
        for c in added:
            dtype_rows.append({
                "Column": c, "Before": "(new)", "After": str(df_out[c].dtype),
            })
        if dtype_rows:
            st.dataframe(
                pd.DataFrame(dtype_rows),
                use_container_width=True, hide_index=True,
            )

        # Side-by-side first-5-rows preview of columns whose dtype changed
        # or which were newly added. Dropped columns are not previewed —
        # the dtype table already says "(dropped)".
        sample_cols = changed_dtype + added
        if sample_cols:
            st.markdown("*First 5 rows — changed / new columns*")
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown("Original")
                in_cols_to_show = [c for c in sample_cols if c in df_in.columns]
                if in_cols_to_show:
                    st.dataframe(
                        df_in[in_cols_to_show].head(5),
                        use_container_width=True,
                    )
                else:
                    st.caption("All listed columns are newly created.")
            with pc2:
                st.markdown("Processed")
                st.dataframe(df_out[sample_cols].head(5), use_container_width=True)

    # --- Active rules with Remove buttons ---
    st.markdown("---")
    st.markdown("**Active operations**")
    _render_active_ops_list()

    # --- Reset ---
    st.markdown("---")
    if st.button("🗑 Reset all preprocessing", key="ppx_reset_all"):
        st.session_state["_ppx_reset_pending"] = True
        st.rerun()


def _render_active_ops_list() -> None:
    """All configured operations with per-row Remove buttons."""
    any_rendered = False
    for col, dt in list(st.session_state.manual_dtypes.items()):
        any_rendered = True
        cc1, cc2 = st.columns([5, 1])
        cc1.markdown(f"`{col}` → **{dt}** dtype")
        if cc2.button("Remove", key=f"ppx_rm_dtype_{col}"):
            st.session_state.manual_dtypes.pop(col, None)
            st.rerun()
    for col, fmt in list(st.session_state.dt_conversions.items()):
        any_rendered = True
        rc1, rc2 = st.columns([5, 1])
        rc1.markdown(
            f"`{col}` → datetime "
            f"({'`' + fmt + '`' if fmt else 'auto-detect'})"
        )
        if rc2.button("Remove", key=f"ppx_rm_dt_{col}"):
            st.session_state.dt_conversions.pop(col, None)
            st.rerun()
    for src, info in list(st.session_state.dt_numeric_conversions.items()):
        any_rendered = True
        label = dtfmt.NUMERIC_DT_MODES.get(info["mode"], info["mode"])
        target = info["target"]
        rc1, rc2 = st.columns([5, 1])
        if target == src:
            rc1.markdown(f"`{src}` → datetime ({label}, overwrite)")
        else:
            rc1.markdown(f"`{src}` → `{target}` ({label})")
        if rc2.button("Remove", key=f"ppx_rm_num_{src}"):
            st.session_state.dt_numeric_conversions.pop(src, None)
            st.rerun()
    for i, m in enumerate(list(st.session_state.dt_merges)):
        any_rendered = True
        rc1, rc2 = st.columns([5, 1])
        rc1.markdown(
            f"`{m['date_col']}` + `{m['time_col']}` → `{m['new_col']}`"
        )
        if rc2.button("Remove", key=f"ppx_rm_merge_{i}"):
            st.session_state.dt_merges = [
                x for x in st.session_state.dt_merges
                if x["new_col"] != m["new_col"]
            ]
            st.rerun()
    for col, info in list(st.session_state.missing_value_ops.items()):
        any_rendered = True
        label = MISSING_STRATEGIES.get(info["strategy"], info["strategy"])
        extra = (
            f" → `{info['fill_value']}`"
            if info["strategy"] == "constant" and info["fill_value"] else ""
        )
        rc1, rc2 = st.columns([5, 1])
        rc1.markdown(f"`{col}` · {label}{extra}")
        if rc2.button("Remove", key=f"ppx_rm_mv_{col}"):
            st.session_state.missing_value_ops.pop(col, None)
            st.rerun()
    if not any_rendered:
        st.caption("No per-column rules yet (only column drops above).")


# --- Public entry point -----------------------------------------------------


def render_preprocessing_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Render the single ⚙️ Preprocessing expander and return the processed df.

    Selections persist in ``st.session_state``, so every operation
    (dropping, dtype conversion, datetime parsing, merging, missing-value
    handling) survives reruns and continues to apply on every render.
    Downstream tabs (time-series, filters, viz, report) automatically pick
    up the converted dtypes and added/removed columns.
    """
    _ensure_state()

    with st.expander("⚙️ Preprocessing", expanded=False):
        st.caption(
            "One unified workflow — drop columns, change types, parse "
            "dates, handle missing values. Selections persist across "
            "reruns and feed every downstream tab."
        )
        tabs = st.tabs([
            "Columns",
            "Data Types",
            "Date & Time",
            "Missing Values",
            "Preview",
        ])

        with tabs[0]:
            _render_column_mgmt(df)
        with tabs[1]:
            _render_dtype_section(df)
        with tabs[2]:
            _render_datetime_section(df)
        with tabs[3]:
            _render_missing_section(df)

        # Apply inside the expander so the Preview tab can show the diff.
        dt_merges_items = tuple(
            tuple(sorted(m.items())) for m in st.session_state.dt_merges
        )
        dt_numeric_items = tuple(
            (src, info["mode"], info["target"])
            for src, info in sorted(st.session_state.dt_numeric_conversions.items())
        )
        missing_value_items = tuple(
            (col, info["strategy"], info["fill_value"])
            for col, info in sorted(st.session_state.missing_value_ops.items())
        )
        out, warnings = apply_preprocessing(
            df,
            tuple(st.session_state.dropped_cols),
            tuple(sorted(st.session_state.manual_dtypes.items())),
            tuple(sorted(st.session_state.dt_conversions.items())),
            dt_merges_items,
            dt_numeric_items,
            missing_value_items,
        )

        with tabs[4]:
            _render_preview_section(df, out)

    # Surface warnings on the main page so users don't need the expander open.
    for msg in warnings:
        st.warning(msg)

    return out
