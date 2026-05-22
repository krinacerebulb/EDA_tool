"""Auto EDA — Streamlit front-end.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

LOGO_PATH = Path(__file__).parent / "assets" / "cb-logo-tagline-main.png"
LOGO = str(LOGO_PATH) if LOGO_PATH.exists() else None
FAVICON_PATH = Path(__file__).parent / "assets" / "image.png"
FAVICON = str(FAVICON_PATH) if FAVICON_PATH.exists() else None

from modules import (
    data_cleaning,
    data_loader,
    eda_analysis,
    exporter,
    filters,
    insights as insights_mod,
    interactive_viz as iviz,
    multi_file_loader,
    preprocessing,
    report_generator,
    target_analysis,
    time_series as ts_mod,
    type_detection,
)
from utils.helpers import human_bytes, split_columns


st.set_page_config(
    page_title="Auto EDA Platform",
    page_icon=FAVICON or LOGO or ":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Single light Plotly template — the dark theme has been removed to keep the
# render pipeline lightweight on large industrial datasets.
import plotly.io as pio
pio.templates.default = "plotly_white"

st.markdown(
    """
    <style>
      /* --- Hide non-essential Streamlit chrome (without breaking the
             collapsed-sidebar expand chevron) --- */
      #MainMenu {visibility: hidden;}
      /* `display: none` (not just hidden) so the footer reclaims its space. */
      footer {display: none !important;}
      [data-testid="stDeployButton"] {display: none !important;}
      [data-testid="stToolbarActions"] {visibility: hidden;}
      [data-testid="stStatusWidget"] {display: none !important;}
      [data-testid="stDecoration"] {display: none !important;}

      /* "Made with Streamlit" / "Hosted with Streamlit" viewer badges.
         Class names differ across Streamlit versions, so we match any
         element whose class contains "viewerBadge". */
      [class*="viewerBadge"] {display: none !important;}

      /* The "Made with Streamlit" link in older builds. */
      a[href^="https://streamlit.io"][target="_blank"] {display: none !important;}

      /* Force the sidebar expand control to stay visible and on top. */
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        z-index: 999 !important;
      }

      /* --- Spacing polish --- */
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      [data-testid="stSidebar"] {padding-top: 0.5rem;}
      [data-testid="stSidebar"] [data-testid="stImage"] {margin-bottom: 0.6rem;}

      /* --- Section titles --- */
      h1, h2, h3 {color: #1F2A37; letter-spacing: -0.01em;}
      h1 {font-weight: 600;}

      /* --- Metric cards --- */
      [data-testid="stMetric"] {
        background: #FAFBFC;
        border: 1px solid #E5E7EB;
        padding: 0.75rem 1rem;
        border-radius: 8px;
      }

      /* --- Buttons --- */
      .stButton > button, .stDownloadButton > button {
        border-radius: 6px;
        font-weight: 500;
        padding: 0.45rem 1rem;
      }

      /* --- Plot container breathing room --- */
      [data-testid="stPlotlyChart"] {margin-bottom: 0.75rem;}

      /* --- Sidebar product strip --- */
      .cerebulb-product {
        font-size: 0.78rem;
        font-weight: 600;
        color: #6B7280;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 0.25rem 0 1rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
if LOGO:
    st.sidebar.image(LOGO, width=180)
st.sidebar.markdown(
    "<div class='cerebulb-product'>Auto EDA Platform</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("Upload one or more datasets to get started.")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV / Excel / JSON / Parquet (multiple OK)",
    type=["csv", "xlsx", "xls", "json", "parquet", "pq"],
    accept_multiple_files=True,
    help=(
        "Upload one file for a standard EDA, or several with matching "
        "schemas (e.g. January.csv, February.csv) to merge them. Files are "
        "concatenated with an auto-generated `source_file` column when "
        "two or more files are uploaded. Schema mismatches are aligned to "
        "the column-union and missing cells are filled with NaN — the "
        "platform never crashes on partial mismatches."
    ),
)

# Preview-row count is a fixed product decision (10 rows is the SaaS
# convention — enough to spot patterns, short enough to fit on one
# screen). Automatic type conversion is always on so dirty industrial
# columns ("123", "No Data", "456") are recovered as numeric without
# the user needing to opt in.
preview_rows = 10
auto_convert = True


# ---------- Main ----------
st.title("Auto EDA Platform")
st.caption("Automated exploratory data analysis, visualizations, and insights — a Cerebulb product.")

if not uploaded_files:
    st.info("Upload one or more datasets from the sidebar to begin.")
    st.stop()

# Route through the multi-file loader. It handles a single file just as
# well as N files — the merge step is a no-op when only one frame
# survives. The loader is "never raises" by contract, but we still wrap
# the call so any unexpected exception surfaces as a clean error rather
# than a stack trace.
with st.spinner(
    f"Loading {len(uploaded_files)} file(s) and running sanitization…"
):
    try:
        raw_df, multi_report = multi_file_loader.load_multiple_files(
            uploaded_files,
        )
    except Exception as exc:
        st.error(f"Unexpected error while loading the files: {exc}")
        st.stop()

# Per-file load status — show which files succeeded and which failed.
_files_status = multi_report.get("per_file", {})
_ok_files = [n for n, m in _files_status.items() if m["status"] == "ok"]
_bad_files = [n for n, m in _files_status.items() if m["status"] == "failed"]

if _bad_files:
    with st.sidebar.expander(
        f"⚠ {len(_bad_files)} file(s) failed", expanded=False,
    ):
        for n in _bad_files:
            st.markdown(f"**{n}** — {_files_status[n]['error']}")

if raw_df is None or raw_df.empty:
    if _bad_files and not _ok_files:
        st.error(
            "Every uploaded file failed to load. See the per-file errors "
            "in the sidebar for details."
        )
    else:
        st.warning(
            "Uploaded file(s) loaded successfully but contain no rows."
        )
    st.stop()

# Display name for the upload batch — used in titles, export filenames,
# and the HTML report header.
if len(_ok_files) == 1:
    dataset_label = _ok_files[0]
else:
    dataset_label = f"{len(_ok_files)} merged files"
# Filesystem-safe stem for default export filenames.
dataset_stem = (
    Path(_ok_files[0]).stem if len(_ok_files) == 1 else "merged_dataset"
)

st.success(
    f"Loaded **{dataset_label}** — "
    f"{raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns."
)

# Multi-file merge summary — only shown when there's actually multi-file
# context to report. Single-file uploads with a clean schema produce no
# warnings, so this panel stays out of the way.
if len(_ok_files) > 1 or multi_report.get("schema_warnings"):
    with st.expander(
        f"📂 Multi-file merge summary — {len(_ok_files)} file(s) merged",
        expanded=bool(multi_report.get("schema_warnings")),
    ):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Files merged", len(_ok_files))
        mc2.metric("Files failed", len(_bad_files))
        mc3.metric("Merged rows", f"{multi_report.get('merged_rows', 0):,}")
        mc4.metric("Merged columns", multi_report.get("merged_cols", 0))

        if _files_status:
            per_file_rows = pd.DataFrame([
                {
                    "File": n,
                    "Status": m["status"],
                    "Rows": m["rows"],
                    "Columns": m["cols"],
                    "Note": m["error"] or "",
                }
                for n, m in _files_status.items()
            ])
            st.dataframe(per_file_rows, width="stretch", hide_index=True)

        for msg in multi_report.get("schema_warnings", []):
            st.warning(msg)

        alignment = multi_report.get("schema_alignment", {})
        gaps = {
            n: a for n, a in alignment.items()
            if a.get("missing_in_this_file") or a.get("unique_to_this_file")
        }
        if gaps:
            with st.expander("Per-file column analysis", expanded=False):
                st.dataframe(
                    pd.DataFrame([
                        {
                            "File": n,
                            "Missing in this file": ", ".join(
                                a.get("missing_in_this_file", [])
                            ) or "—",
                            "Unique to this file": ", ".join(
                                a.get("unique_to_this_file", [])
                            ) or "—",
                        }
                        for n, a in gaps.items()
                    ]),
                    width="stretch", hide_index=True,
                )


# ---------- Sanitization summary (industrial dirty-data report) ----------
# ``data_loader._read_bytes`` runs the production sanitization layer on every
# upload and attaches the report via ``df.attrs``. We surface a short banner
# here and a richer breakdown on the Cleaning tab.
sanitization_report = getattr(raw_df, "attrs", {}).get("sanitization_report", {})
_san_changes = (
    int(sanitization_report.get("tokens_replaced_total", 0))
    + len(sanitization_report.get("numeric_conversions", []))
    + len(sanitization_report.get("datetime_conversions", []))
    + len(sanitization_report.get("boolean_conversions", []))
    + len(sanitization_report.get("arrow_unsafe_columns", []))
    + int(sanitization_report.get("infinities_replaced", 0))
)
if _san_changes > 0:
    bits = []
    if sanitization_report.get("tokens_replaced_total"):
        bits.append(
            f"replaced **{sanitization_report['tokens_replaced_total']:,}** "
            "dirty token(s) (e.g. *No Data*, *Bad*, *Sensor Fail*) with NaN"
        )
    if sanitization_report.get("numeric_conversions"):
        bits.append(
            f"recovered **{len(sanitization_report['numeric_conversions'])}** "
            "mixed-type column(s) as numeric"
        )
    if sanitization_report.get("datetime_conversions"):
        bits.append(
            f"parsed **{len(sanitization_report['datetime_conversions'])}** "
            "column(s) as datetime"
        )
    if sanitization_report.get("infinities_replaced"):
        bits.append(
            f"removed **{sanitization_report['infinities_replaced']}** "
            "infinite value(s)"
        )
    if sanitization_report.get("arrow_unsafe_columns"):
        bits.append(
            f"hardened **{len(sanitization_report['arrow_unsafe_columns'])}** "
            "column(s) for PyArrow rendering"
        )
    st.info("🧼 Auto-sanitization: " + "; ".join(bits) + ".")


# ---------- Smart type detection (object → numeric) ----------
# original_df stays untouched; processed_df is what every downstream tab uses.
original_df = raw_df
if auto_convert:
    processed_df, conversion_report = type_detection.apply_smart_conversion(
        original_df, threshold=0.7,
    )
else:
    processed_df = original_df.copy()
    conversion_report = []

converted_cols = type_detection.converted_columns(conversion_report)
if converted_cols:
    st.info(
        f"🔄 Auto-converted {len(converted_cols)} object column(s) to numeric: "
        + ", ".join(f"`{c}`" for c in converted_cols)
        + ". See the **Cleaning** tab for details."
    )
if type_detection.had_invalid_coercions(conversion_report):
    st.warning(
        "Some values were converted to NaN during processing. "
        "Check the **Type Conversion Summary** under the Cleaning tab."
    )


# ---------- User-driven preprocessing (drop columns, change dtypes) ----------
# Sits between auto-conversion and filtering so every downstream tab sees the
# customized dataset. Selections persist in st.session_state across reruns.
preprocessed_df = preprocessing.render_preprocessing_ui(processed_df)


# ---------- Sidebar filters ----------
# Filters render in the sidebar (Tableau / Datadog / Looker convention).
# The panel itself shows its own inline impact summary + reset button,
# so the main column stays focused on tables and visualizations.
df, flt_summary = filters.render_filters(preprocessed_df)

if df.empty:
    st.warning(
        "Current filters exclude every row. Relax them from the sidebar "
        "to continue."
    )
    st.stop()


# ---------- Derived analytics (all on filtered df) ----------
# Every function below is @st.cache_data-wrapped, so this block is heavy only
# on the FIRST run for a given filtered DataFrame. Subsequent reruns (widget
# changes, tab switches) return from cache.
with st.spinner("Computing summary statistics…"):
    overview = eda_analysis.dataset_overview(df)
    dtype_info = data_loader.detect_column_types(df)
    missing = data_cleaning.missing_value_summary(df)
    dup_summary = data_cleaning.duplicate_summary(df)
    outliers = data_cleaning.detect_outliers_iqr(df)
    numeric_stats = eda_analysis.numeric_statistics(df)
    categorical_stats = eda_analysis.categorical_statistics(df)
    numeric_cols, categorical_cols, _ = split_columns(df)


tabs = st.tabs([
    "Overview",
    "Cleaning",
    "Statistics",
    "Visualizations",
    "Time Series",
    "Target EDA",
    "Insights",
    "Report",
])


# ---------- Overview ----------
with tabs[0]:
    # KPI strip lives ABOVE the preview so the headline shape of the
    # dataset is the first thing the user sees — a SaaS-dashboard
    # convention. Preview, column types, and exports follow below.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['rows']:,}")
    c2.metric("Columns", overview["columns"])
    c3.metric("Memory", human_bytes(overview["memory_bytes"]))
    c4.metric("Missing values", f"{overview['total_missing']:,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(preview_rows), width="stretch")

    st.subheader("Column Types")
    st.dataframe(dtype_info, width="stretch")

    # --- Export panel — multi-format download (CSV / Excel / JSON / Parquet) ---
    # Driven by ``exporter.render_export_ui``. When filters are active we
    # offer two tabs: rows kept by the filters, and rows excluded. When no
    # filters are active we offer one panel covering the full dataset.
    st.subheader("Export dataset")
    if flt_summary["is_filtered"]:
        excluded_df = preprocessed_df.loc[
            preprocessed_df.index.difference(df.index)
        ]
        export_tabs = st.tabs([
            f"Filtered rows ({len(df):,})",
            f"Excluded rows ({len(excluded_df):,})",
        ])
        with export_tabs[0]:
            st.caption(
                "Sanitized, preprocessed, and filtered dataset — the exact "
                "data feeding every tab on this page."
            )
            exporter.render_export_ui(
                df,
                base_filename=f"{dataset_stem}_filtered",
                key_prefix="exp_filtered",
                label="Download filtered data",
            )
        with export_tabs[1]:
            if excluded_df.empty:
                st.caption("No excluded rows for the current filters.")
            else:
                st.caption(
                    "Rows that the current sidebar filters removed — useful "
                    "for diff / QC workflows."
                )
                exporter.render_export_ui(
                    excluded_df,
                    base_filename=f"{dataset_stem}_excluded",
                    key_prefix="exp_excluded",
                    label="Download excluded data",
                )
    else:
        st.caption(
            "No sidebar filters are active. The export below contains the "
            "full sanitized + preprocessed dataset."
        )
        exporter.render_export_ui(
            df,
            base_filename=dataset_stem,
            key_prefix="exp_full",
            label="Download dataset",
        )


# ---------- Cleaning ----------
with tabs[1]:
    st.subheader("Data Preprocessing Report")
    st.caption(
        "Production sanitization layer — runs once on load. Cleans industrial "
        "dirty tokens (*No Data*, *Bad*, *Sensor Fail*, Excel errors, ...), "
        "recovers mixed-type columns, and guarantees PyArrow-safe rendering."
    )
    if not sanitization_report:
        st.caption("No sanitization metadata attached to this dataset.")
    elif _san_changes == 0:
        st.success("Dataset was already clean — no industrial tokens or "
                   "mixed-type columns detected.")
    else:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric(
            "Tokens → NaN",
            f"{sanitization_report.get('tokens_replaced_total', 0):,}",
        )
        sc2.metric(
            "Numeric recovered",
            len(sanitization_report.get("numeric_conversions", [])),
        )
        sc3.metric(
            "Datetime recovered",
            len(sanitization_report.get("datetime_conversions", [])),
        )
        sc4.metric(
            "Arrow-hardened",
            len(sanitization_report.get("arrow_unsafe_columns", [])),
        )

        tok_per_col = sanitization_report.get("tokens_replaced_per_column", {})
        if tok_per_col:
            with st.expander(
                f"Dirty tokens replaced — {len(tok_per_col)} column(s)",
                expanded=False,
            ):
                tok_df = pd.DataFrame(
                    [{"Column": c, "Tokens → NaN": n}
                     for c, n in sorted(tok_per_col.items(),
                                        key=lambda kv: -kv[1])]
                )
                st.dataframe(tok_df, width="stretch", hide_index=True)

        num_log = sanitization_report.get("numeric_conversions", [])
        if num_log:
            with st.expander(
                f"Mixed-type → numeric — {len(num_log)} column(s)",
                expanded=False,
            ):
                st.dataframe(
                    pd.DataFrame(num_log),
                    width="stretch", hide_index=True,
                )

        dt_log = sanitization_report.get("datetime_conversions", [])
        if dt_log:
            with st.expander(
                f"Detected datetime columns — {len(dt_log)} column(s)",
                expanded=False,
            ):
                st.dataframe(
                    pd.DataFrame(dt_log),
                    width="stretch", hide_index=True,
                )

        arrow_log = sanitization_report.get("arrow_unsafe_columns", [])
        if arrow_log:
            with st.expander(
                f"PyArrow-hardened columns — {len(arrow_log)} column(s)",
                expanded=False,
            ):
                st.caption(
                    "These columns had mixed Python types that would crash "
                    "Streamlit's PyArrow renderer. They were stringified so "
                    "they display safely."
                )
                st.dataframe(
                    pd.DataFrame(arrow_log),
                    width="stretch", hide_index=True,
                )

        errors = sanitization_report.get("errors", [])
        if errors:
            with st.expander(
                f"Sanitization warnings — {len(errors)}", expanded=False,
            ):
                for msg in errors:
                    st.markdown(f"- {msg}")

    st.markdown("---")
    st.subheader("Type Conversion Summary")
    if not auto_convert:
        st.caption(
            "Automatic type conversion is disabled — enable it from the sidebar "
            "to coerce mostly-numeric object columns."
        )
    elif not conversion_report:
        st.caption("No object columns were found in this dataset.")
    else:
        summary_frame = type_detection.conversion_summary_frame(conversion_report)
        st.dataframe(summary_frame, width="stretch")

        examples = [e for e in conversion_report if e["convertible"] and e["invalid_examples"]]
        if examples:
            with st.expander("Sample invalid values per converted column", expanded=False):
                for entry in examples:
                    st.markdown(
                        f"**{entry['column']}** — invalid examples: "
                        f"{entry['invalid_examples']}"
                    )

        if type_detection.had_invalid_coercions(conversion_report):
            st.warning(
                "Some values were converted to NaN during processing. "
                "They are shown above as 'Invalid' and excluded from numeric stats."
            )

    st.markdown("---")
    st.subheader("Detected Datetime Columns")
    dt_summary = ts_mod.datetime_detection_summary(df)
    if dt_summary.empty:
        st.caption(
            "No datetime-like columns detected. Use **Preprocessing → "
            "Convert column dtype** above to force a column to datetime if "
            "you know it should be parsed that way."
        )
    else:
        st.dataframe(dt_summary, width="stretch")
        st.caption(
            "Confidence is the share of non-null values that parsed as a "
            "valid date/time. Invalid counts are estimated when sampling "
            "is used on large columns."
        )

    st.markdown("---")
    st.subheader("Missing Values")
    st.dataframe(missing, width="stretch")

    st.subheader("Duplicates")
    st.write(
        f"**{dup_summary['duplicate_rows']:,}** duplicate rows "
        f"({dup_summary['duplicate_percent']}% of the dataset)."
    )

    st.subheader("Outliers")
    st.caption(
        "Two complementary detectors. **IQR** is robust to skew (good default "
        "for industrial sensor data); **Z-score** assumes a roughly normal "
        "distribution and flags values more than `k` standard deviations from "
        "the mean."
    )
    if outliers.empty:
        st.info("No numeric columns to check for outliers.")
    else:
        z_threshold = st.number_input(
            "Z-score threshold (k)",
            min_value=2.0, max_value=5.0, value=3.0, step=0.5,
            help=(
                "Values with |(x − mean) / std| > k are flagged as outliers. "
                "k = 3 is the classical convention (~0.27% of values under a "
                "normal distribution). Accepts any value between 2.0 and 5.0."
            ),
            key="outlier_zscore_threshold",
        )
        outliers_z = data_cleaning.detect_outliers_zscore(df, threshold=z_threshold)

        outlier_tabs = st.tabs(["IQR method", f"Z-score (k = {z_threshold:g})"])
        with outlier_tabs[0]:
            st.caption(
                "Flags values below Q1 − 1.5·IQR or above Q3 + 1.5·IQR. "
                "Recommended for skewed or non-normal columns."
            )
            st.dataframe(outliers, width="stretch")
        with outlier_tabs[1]:
            st.caption(
                f"Flags values with |z| > {z_threshold:g}. Columns with zero "
                "variance produce no outliers."
            )
            if outliers_z.empty:
                st.info("No numeric columns available for Z-score analysis.")
            else:
                st.dataframe(outliers_z, width="stretch")

    st.caption("Note: nothing is modified automatically — this is analysis only.")


# ---------- Statistics ----------
with tabs[2]:
    st.subheader("Numeric Columns")
    st.caption(
        "Full `describe()`-style summary plus missing %, unique count, "
        "skewness, and kurtosis. All values rounded to two decimals."
    )
    if numeric_stats.empty:
        st.info("No numeric columns detected.")
    else:
        st.dataframe(numeric_stats, width="stretch", hide_index=True)

    st.subheader("Categorical Columns")
    if not categorical_stats:
        st.info("No categorical columns detected.")
    else:
        for col, info in categorical_stats.items():
            with st.expander(f"{col}  —  {info['unique']} unique, mode: {info['mode']}"):
                st.dataframe(info["top_values"], width="stretch", hide_index=True)

    # ---- Column drilldown ----
    st.markdown("---")
    st.subheader("Column Drilldown")
    st.caption("Pick any column for its full statistical profile.")
    drill_col = st.selectbox(
        "Column",
        options=["(select)"] + list(df.columns),
        key="stats_drill_col",
    )
    if drill_col != "(select)":
        detail_df = eda_analysis.column_detail_stats(df, drill_col)
        if detail_df.empty:
            st.info("No detail available for the selected column.")
        else:
            st.dataframe(detail_df, width="stretch", hide_index=True)


# ---------- Visualizations (interactive Plotly) ----------
# Refactor note: every chart config is wrapped in an ``st.form``. Widget
# changes inside a form don't trigger a rerun — only the **Build chart**
# submit does. The chart bytes are stored in ``st.session_state`` so the
# rendered chart persists across reruns (tab switches, filter changes
# that don't affect the chart) until the user explicitly rebuilds.
with tabs[3]:
    if not numeric_cols and not categorical_cols:
        st.info("Nothing to visualize.")
    else:
        viz_type = st.selectbox(
            "Chart type",
            [
                "Histogram",
                "Boxplot",
                "Scatter plot",
                "Bar chart (categorical)",
                "Correlation heatmap",
                "Multi-line time series",
            ],
        )

        # Each chart type gets its own sub-form so widget changes within a
        # chart don't fire until "Build chart" is clicked. The last-built
        # config per chart type is cached in session_state under a stable
        # key so a tab-switch doesn't lose the chart.

        if viz_type == "Histogram":
            if not numeric_cols:
                st.info("No numeric columns available.")
            else:
                with st.form("viz_hist_form", clear_on_submit=False, border=False):
                    st.selectbox("Column", numeric_cols, key="hist_col")
                    st.number_input(
                        "Bins",
                        min_value=5, max_value=100, value=30, step=1,
                        key="hist_bins",
                        help="Number of histogram buckets (5–100).",
                    )
                    built = st.form_submit_button(
                        "📊 Build histogram",                        use_container_width=True,
                    )
                if built:
                    st.session_state["_viz_hist_cfg"] = (
                        st.session_state["hist_col"],
                        int(st.session_state["hist_bins"]),
                    )
                cfg = st.session_state.get("_viz_hist_cfg")
                if cfg:
                    col, bins = cfg
                    if col in numeric_cols:
                        with st.spinner("Building histogram…"):
                            st.pyplot(
                                iviz.histogram(df, col, bins=bins),
                                clear_figure=True,
                            )
                    else:
                        st.info("Selected column no longer exists — pick another and rebuild.")
                else:
                    st.info("Configure the chart above and click **Build histogram**.")

        elif viz_type == "Boxplot":
            if not numeric_cols:
                st.info("No numeric columns available.")
            else:
                with st.form("viz_box_form", clear_on_submit=False, border=False):
                    st.selectbox("Column", numeric_cols, key="box_col")
                    built = st.form_submit_button(
                        "📊 Build boxplot",                        use_container_width=True,
                    )
                if built:
                    st.session_state["_viz_box_cfg"] = st.session_state["box_col"]
                cfg = st.session_state.get("_viz_box_cfg")
                if cfg and cfg in numeric_cols:
                    with st.spinner("Building boxplot…"):
                        st.pyplot(iviz.boxplot(df, cfg), clear_figure=True)
                elif cfg:
                    st.info("Selected column no longer exists — pick another and rebuild.")
                else:
                    st.info("Configure the chart above and click **Build boxplot**.")

        elif viz_type == "Scatter plot":
            if len(numeric_cols) < 2:
                st.info("Need at least two numeric columns for a scatter plot.")
            else:
                # Pre-form: read current X to filter Y options. Picking X
                # is cheap (it doesn't touch the dataframe), so a rerun
                # here is acceptable to keep Y/X mutually exclusive.
                with st.form("viz_scatter_form", clear_on_submit=False, border=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.selectbox("X axis", numeric_cols, key="scatter_x")
                    with c2:
                        y_options = [
                            c for c in numeric_cols
                            if c != st.session_state.get("scatter_x", numeric_cols[0])
                        ]
                        st.selectbox("Y axis", y_options, key="scatter_y")
                    with c3:
                        color_options = ["(none)"] + categorical_cols + numeric_cols
                        st.selectbox("Colour by", color_options, key="scatter_color")
                    built = st.form_submit_button(
                        "📊 Build scatter plot",                        use_container_width=True,
                    )
                if built:
                    color_v = st.session_state["scatter_color"]
                    st.session_state["_viz_scatter_cfg"] = (
                        st.session_state["scatter_x"],
                        st.session_state["scatter_y"],
                        None if color_v == "(none)" else color_v,
                    )
                cfg = st.session_state.get("_viz_scatter_cfg")
                if cfg:
                    x_col, y_col, color_col = cfg
                    if x_col in numeric_cols and y_col in numeric_cols:
                        with st.spinner("Building scatter plot…"):
                            st.pyplot(
                                iviz.scatter(df, x_col, y_col, color=color_col),
                                clear_figure=True,
                            )
                    else:
                        st.info("Selected columns no longer exist — pick again and rebuild.")
                else:
                    st.info("Configure the chart above and click **Build scatter plot**.")

        elif viz_type == "Bar chart (categorical)":
            if not categorical_cols:
                st.info("No categorical columns available.")
            else:
                with st.form("viz_bar_form", clear_on_submit=False, border=False):
                    st.selectbox("Column", categorical_cols, key="bar_col")
                    st.number_input(
                        "Top N values",
                        min_value=3, max_value=30, value=10, step=1,
                        key="bar_top_n",
                        help="How many of the most frequent categories to show (3–30).",
                    )
                    built = st.form_submit_button(
                        "📊 Build bar chart",                        use_container_width=True,
                    )
                if built:
                    st.session_state["_viz_bar_cfg"] = (
                        st.session_state["bar_col"],
                        int(st.session_state["bar_top_n"]),
                    )
                cfg = st.session_state.get("_viz_bar_cfg")
                if cfg:
                    col, top_n = cfg
                    if col in categorical_cols:
                        with st.spinner("Building bar chart…"):
                            st.pyplot(
                                iviz.bar_chart(df, col, top_n=top_n),
                                clear_figure=True,
                            )
                    else:
                        st.info("Selected column no longer exists — pick another and rebuild.")
                else:
                    st.info("Configure the chart above and click **Build bar chart**.")

        elif viz_type == "Correlation heatmap":
            # No options — single button trigger.
            if st.button("📊 Build correlation heatmap",                         key="viz_corr_build", use_container_width=True):
                st.session_state["_viz_corr_built"] = True
            if st.session_state.get("_viz_corr_built"):
                with st.spinner("Computing correlations…"):
                    fig = iviz.correlation_heatmap(df)
                if fig is None:
                    st.info("Need at least 2 numeric columns for a correlation heatmap.")
                else:
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("Click **Build correlation heatmap** to compute and render.")

        elif viz_type == "Multi-line time series":
            datetime_cols_viz = ts_mod.detect_datetime_columns(df)
            if not datetime_cols_viz:
                st.info(
                    "No datetime columns detected — make sure at least one "
                    "column holds parseable date/time values."
                )
            elif not numeric_cols:
                st.info("No numeric columns available to plot.")
            else:
                viz_freq_labels = list(iviz.TIME_AGG_FREQUENCIES.keys())
                viz_agg_func_labels = list(iviz.TIME_AGG_FUNCTIONS.keys())
                with st.form("viz_ml_form", clear_on_submit=False, border=False):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.selectbox(
                            "Date column", datetime_cols_viz, key="ml_date",
                        )
                    with c2:
                        st.multiselect(
                            "Numerical columns to compare",
                            numeric_cols,
                            default=numeric_cols[: min(2, len(numeric_cols))],
                            key="ml_values",
                            help=(
                                "Pick two or more numeric columns. If their "
                                "scales differ by more than 10×, a secondary "
                                "y-axis is added automatically."
                            ),
                        )
                    c3, c4 = st.columns(2)
                    with c3:
                        st.selectbox(
                            "Aggregation interval",
                            viz_freq_labels,
                            index=0,
                            key="ml_agg",
                            help=(
                                "Resample into wall-clock buckets before "
                                "plotting. Choose ``None`` to plot every raw "
                                "row."
                            ),
                        )
                    with c4:
                        st.selectbox(
                            "Aggregation function",
                            viz_agg_func_labels,
                            index=0,
                            key="ml_agg_func",
                            help=(
                                "Function applied inside each aggregation "
                                "bucket (only used when an interval is set)."
                            ),
                        )
                    built = st.form_submit_button(
                        "📊 Build time-series chart",                        use_container_width=True,
                    )
                if built:
                    st.session_state["_viz_ml_cfg"] = (
                        st.session_state["ml_date"],
                        tuple(st.session_state["ml_values"]),
                        st.session_state["ml_agg"],
                        st.session_state["ml_agg_func"],
                    )
                cfg = st.session_state.get("_viz_ml_cfg")
                if cfg:
                    ml_date, ml_values, ml_agg, ml_agg_func = cfg
                    ml_values = list(ml_values)
                    if not ml_values:
                        st.info("Select at least one numeric column and rebuild.")
                    else:
                        fig = iviz.multi_line_time_series(
                            df,
                            date_col=ml_date,
                            value_cols=ml_values,
                            aggregation=ml_agg,
                            agg_func=ml_agg_func,
                        )
                        if fig is None:
                            st.warning(
                                "No rows left after parsing the date column. "
                                "Check that it contains valid dates."
                            )
                        else:
                            st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Configure the chart above and click **Build time-series chart**.")


# ---------- Time Series ----------
# Refactor note: ``prepare_series``, ``time_series_plot`` and
# ``trend_insights`` are NOT cached and rebuild the Plotly figure on
# every rerun. Wrap the config widgets in a form so the user picks
# everything first and only the submit triggers the expensive prepare
# + plot + insights pass. The last-built config is stashed in
# session_state so the chart survives tab-switches and sidebar reruns.
with tabs[4]:
    st.subheader("Time Series Trend Analysis")
    st.caption(
        "Automatically detects datetime columns, then builds an interactive "
        "Plotly chart with optional rolling-mean smoothing. "
        "Configure the chart below and click **Plot time series**."
    )

    datetime_cols = ts_mod.detect_datetime_columns(df)
    ts_value_candidates = df.select_dtypes(include=["number"]).columns.tolist()

    if not datetime_cols:
        st.info(
            "No datetime columns detected. Make sure at least one column "
            "holds parseable date/time values."
        )
    elif not ts_value_candidates:
        st.info("No numeric columns available to use as the time-series value.")
    else:
        rolling_unit_choices = {
            "minutes": "min",
            "hours":   "h",
            "days":    "D",
        }
        unit_labels = list(rolling_unit_choices.keys())

        with st.form("ts_form", clear_on_submit=False, border=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.selectbox("Date column", datetime_cols, key="ts_date")
            with c2:
                st.selectbox(
                    "Value column (numeric)",
                    ts_value_candidates,
                    key="ts_value",
                )
            with c3:
                st.checkbox(
                    "Rolling window", value=False, key="ts_smooth",
                    help=(
                        "Overlay a time-based rolling mean. The window "
                        "width is a wall-clock interval and tolerates "
                        "irregular sensor timestamps."
                    ),
                )
            c4, c5 = st.columns(2)
            with c4:
                st.number_input(
                    "Rolling window size",
                    min_value=1,
                    max_value=10_000,
                    value=1,
                    step=1,
                    key="ts_window_size",
                    help=(
                        "How many of the unit chosen on the right (e.g. 15 "
                        "minutes, 4 hours, 7 days)."
                    ),
                )
            with c5:
                st.selectbox(
                    "Rolling window unit",
                    unit_labels,
                    index=unit_labels.index("hours"),
                    key="ts_window_unit",
                    help="Wall-clock unit for the rolling window width.",
                )
            built = st.form_submit_button(
                "📊 Plot time series",                use_container_width=True,
            )

        if built:
            st.session_state["_ts_cfg"] = (
                st.session_state["ts_date"],
                st.session_state["ts_value"],
                bool(st.session_state["ts_smooth"]),
                int(st.session_state["ts_window_size"]),
                st.session_state["ts_window_unit"],
            )

        cfg = st.session_state.get("_ts_cfg")
        if cfg:
            date_col, value_col, apply_smoothing, window_size, window_unit = cfg
            offset_alias = rolling_unit_choices.get(window_unit, "h")
            rolling_window = (
                f"{window_size}{offset_alias}" if apply_smoothing else None
            )
            rolling_display_label = (
                f"Rolling mean ({window_size} {window_unit})"
                if apply_smoothing else None
            )

            if date_col not in df.columns or value_col not in df.columns:
                st.info(
                    "Selected columns no longer exist — reconfigure and "
                    "click **Plot time series** again."
                )
            else:
                prepared = ts_mod.prepare_series(df, date_col, value_col)
                if prepared.empty:
                    st.warning(
                        "No rows left after parsing and cleaning — check that "
                        f"`{date_col}` contains dates and `{value_col}` "
                        "contains numbers."
                    )
                else:
                    st.caption(
                        f"Using {len(prepared):,} rows from "
                        f"{prepared[date_col].min():%Y-%m-%d} to "
                        f"{prepared[date_col].max():%Y-%m-%d}."
                    )
                    st.plotly_chart(
                        ts_mod.time_series_plot(
                            prepared, date_col, value_col,
                            rolling_window=rolling_window,
                            rolling_label=rolling_display_label,
                        ),
                        width="stretch",
                    )

                    st.markdown("### 🔎 Trend insights")
                    for line in ts_mod.trend_insights(prepared, date_col, value_col):
                        st.markdown(f"- {line}")
        else:
            st.info("Configure the chart above and click **Plot time series**.")


# ---------- Target-based EDA ----------
# Refactor note: the target selector + Run button are wrapped in a form
# so picking a target alone doesn't fire all the cached lookups +
# Plotly figure builds. Only **Run target analysis** commits the choice.
# The last-committed target is stashed in session_state so the analysis
# survives reruns. Sub-plots (target distribution, target-over-time,
# target-by-category) each have their own small forms below.
with tabs[5]:
    st.subheader("Target-Based EDA")
    st.caption(
        "Pick a target column to see how every other feature relates to it. "
        "Numeric targets use correlation; categorical targets use grouped "
        "comparisons."
    )

    with st.form("target_pick_form", clear_on_submit=False, border=False):
        st.selectbox(
            "Target column",
            ["(none)"] + list(df.columns),
            key="target_col",
        )
        run_target = st.form_submit_button(
            "🎯 Run target analysis",            use_container_width=True,
        )
    if run_target:
        st.session_state["_target_committed"] = st.session_state["target_col"]

    target = st.session_state.get("_target_committed", "(none)")
    # Don't carry over a target that no longer exists (e.g. the user dropped
    # the column via the preprocessing UI).
    if target not in df.columns:
        target = "(none)"

    if target == "(none)":
        st.info(
            "Pick a target column above and click **Run target analysis** "
            "to generate the per-target views."
        )
    else:
        numeric_target = target_analysis.is_numeric_target(df, target)
        st.markdown(
            f"**Target type:** {'numerical' if numeric_target else 'categorical'} "
            f"(`{df[target].dtype}`)"
        )

        # ---- Target summary cards ----
        tgt_summary = target_analysis.target_summary(df, target)
        if tgt_summary:
            tc1, tc2, tc3, tc4, tc5 = st.columns(5)
            tc1.metric("Total values", f"{tgt_summary['total']:,}")
            tc2.metric(
                "Missing",
                f"{tgt_summary['missing']:,} ({tgt_summary['missing_pct']:.2f}%)",
            )
            if numeric_target and tgt_summary.get("mean") is not None:
                tc3.metric("Mean",   f"{tgt_summary['mean']:.2f}")
                tc4.metric("Median", f"{tgt_summary['median']:.2f}")
                tc5.metric("Std",    f"{tgt_summary['std']:.2f}")
            else:
                tc3.metric("Unique classes", f"{tgt_summary['unique']:,}")
                tc4.metric(
                    "Most frequent",
                    str(tgt_summary.get("top_value", "—")),
                )
                tc5.metric(
                    "Top frequency",
                    f"{tgt_summary.get('top_freq', 0):,}",
                    f"{tgt_summary.get('top_pct', 0):.2f}%",
                )
            with st.expander("Full target statistics", expanded=False):
                st.dataframe(
                    tgt_summary["detail"], width="stretch", hide_index=True,
                )
                if not numeric_target and "class_distribution" in tgt_summary:
                    st.markdown("**Class distribution**")
                    st.dataframe(
                        tgt_summary["class_distribution"],
                        width="stretch", hide_index=True,
                    )

        if numeric_target:
            corr_df = target_analysis.correlations_with_target(df, target)
            if not corr_df.empty:
                st.markdown("### Correlation with target")

                def _fmt_corr(v):
                    if pd.isna(v):
                        return ""
                    arrow = "▲" if v >= 0 else "▼"
                    return f"{arrow} {v:+.4f}"

                def _color_corr(v):
                    if pd.isna(v):
                        return ""
                    return (
                        "color: #2ca02c; font-weight: 600;"
                        if v >= 0
                        else "color: #d62728; font-weight: 600;"
                    )

                styled_corr = (
                    corr_df.style
                    .format({"Correlation": _fmt_corr})
                    .map(_color_corr, subset=["Correlation"])
                )
                st.dataframe(styled_corr, width="stretch")

                strongest = corr_df.iloc[0]
                st.markdown(
                    f"Most influential feature: **{strongest['Feature']}** "
                    f"(r = {strongest['Correlation']:+.3f} — "
                    f"{strongest['Strength'].lower()} {strongest['Direction']})."
                )
            else:
                st.info(
                    "Need at least one other numeric feature to correlate "
                    "with the target."
                )

            # --- Distribution of the target (form-gated histogram) ---
            st.markdown("### Distribution of target")
            with st.form("target_hist_form", clear_on_submit=False, border=False):
                st.number_input(
                    "Bins",
                    min_value=5, max_value=100, value=30, step=1,
                    key="target_hist_bins",
                    help="Number of histogram buckets (5–100).",
                )
                hist_built = st.form_submit_button(
                    "📊 Plot distribution",                    use_container_width=True,
                )
            if hist_built:
                st.session_state["_target_hist_cfg"] = (
                    target, int(st.session_state["target_hist_bins"]),
                )
            hist_cfg = st.session_state.get("_target_hist_cfg")
            if hist_cfg and hist_cfg[0] == target:
                st.pyplot(
                    iviz.histogram(df, target, bins=hist_cfg[1]),
                    clear_figure=True,
                )
            else:
                st.caption("Click **Plot distribution** to render the histogram.")

            # --- Target over time (form-gated multi-line chart) ---
            target_datetime_cols = ts_mod.detect_datetime_columns(df)
            if target_datetime_cols:
                st.markdown("### Target over time")
                tgt_agg_func_labels = list(iviz.TIME_AGG_FUNCTIONS.keys())
                # Daily buckets are a sensible default for industrial sensor
                # data — keeps the chart responsive on long series. The user
                # still chooses the aggregation function (Avg / Sum / Count
                # / Std) below.
                tgt_default_interval = "Daily"
                with st.form("target_line_form", clear_on_submit=False, border=False):
                    st.selectbox(
                        "Date column",
                        target_datetime_cols,
                        key="target_line_date",
                    )
                    st.selectbox(
                        "Aggregation function",
                        tgt_agg_func_labels,
                        index=0,
                        key="target_line_agg_func",
                        help=(
                            "Function applied across daily buckets of the "
                            "target column."
                        ),
                    )
                    line_built = st.form_submit_button(
                        "📊 Plot target over time",                        use_container_width=True,
                    )
                if line_built:
                    st.session_state["_target_line_cfg"] = (
                        target,
                        st.session_state["target_line_date"],
                        st.session_state["target_line_agg_func"],
                    )
                line_cfg = st.session_state.get("_target_line_cfg")
                if line_cfg and line_cfg[0] == target:
                    _, ld, la_func = line_cfg
                    line_fig = iviz.multi_line_time_series(
                        df, date_col=ld, value_cols=[target],
                        aggregation=tgt_default_interval, agg_func=la_func,
                    )
                    if line_fig is None:
                        st.info(
                            f"Could not parse `{ld}` as dates — no rows "
                            "left to plot."
                        )
                    else:
                        st.plotly_chart(line_fig, width="stretch")
                else:
                    st.caption(
                        "Click **Plot target over time** to render the chart."
                    )
            else:
                st.caption(
                    "No datetime column detected — skipping target-over-time plot."
                )

            # --- Target distribution across categorical features ---
            cat_options = [c for c in categorical_cols if c != target]
            if cat_options:
                st.markdown("### Target by categorical feature")
                with st.form("target_grpbox_form", clear_on_submit=False, border=False):
                    st.selectbox(
                        "Categorical feature",
                        cat_options,
                        key="target_box_cat",
                    )
                    box_built = st.form_submit_button(
                        "📊 Plot grouped box",                        use_container_width=True,
                    )
                if box_built:
                    st.session_state["_target_box_cfg"] = (
                        target, st.session_state["target_box_cat"],
                    )
                box_cfg = st.session_state.get("_target_box_cfg")
                if box_cfg and box_cfg[0] == target:
                    _, tc = box_cfg
                    if tc in cat_options:
                        st.pyplot(
                            iviz.grouped_box(df, category=tc, numeric=target),
                            clear_figure=True,
                        )
                else:
                    st.caption("Click **Plot grouped box** to render the chart.")
            else:
                st.caption(
                    "No categorical features available for grouped box plot."
                )

        else:
            # --- Counts per category (form-gated bar chart) ---
            st.markdown(f"### Count per **{target}** category")
            with st.form("target_bar_form", clear_on_submit=False, border=False):
                st.number_input(
                    "Top N categories",
                    min_value=3, max_value=30, value=10, step=1,
                    key="target_bar_topn",
                    help="How many of the most frequent categories to show (3–30).",
                )
                bar_built = st.form_submit_button(
                    "📊 Plot category counts",                    use_container_width=True,
                )
            if bar_built:
                st.session_state["_target_bar_cfg"] = (
                    target, int(st.session_state["target_bar_topn"]),
                )
            bar_cfg = st.session_state.get("_target_bar_cfg")
            if bar_cfg and bar_cfg[0] == target:
                _, tn = bar_cfg
                st.pyplot(
                    iviz.bar_chart(df, target, top_n=tn),
                    clear_figure=True,
                )
            else:
                st.caption("Click **Plot category counts** to render the chart.")

            # --- Group means + importance (cached lookups, render directly) ---
            means_df = target_analysis.group_means(df, target)
            if means_df.empty:
                st.info("No numeric features to compare across the target groups.")
            else:
                st.markdown(f"### Mean of numeric features per **{target}**")
                st.dataframe(means_df, width="stretch")

                imp_df = target_analysis.categorical_importance(df, target)
                st.markdown("### Most influential features")
                st.dataframe(imp_df, width="stretch")

                if not imp_df.empty:
                    top_feature = imp_df.iloc[0]["Feature"]
                    st.markdown(
                        f"Top-ranked feature: **{top_feature}** — its mean varies most "
                        f"across the groups of **{target}**."
                    )

                    # --- Boxplots: numeric features vs target groups (form-gated) ---
                    st.markdown("### Numeric features by target group")
                    num_options = imp_df["Feature"].tolist()
                    with st.form("target_numbox_form", clear_on_submit=False, border=False):
                        st.multiselect(
                            "Numeric features",
                            num_options,
                            default=num_options[: min(2, len(num_options))],
                            key="target_box_features",
                        )
                        nbox_built = st.form_submit_button(
                            "📊 Plot grouped boxes",                            use_container_width=True,
                        )
                    if nbox_built:
                        st.session_state["_target_numbox_cfg"] = (
                            target,
                            tuple(st.session_state["target_box_features"]),
                        )
                    nbox_cfg = st.session_state.get("_target_numbox_cfg")
                    if nbox_cfg and nbox_cfg[0] == target:
                        _, features = nbox_cfg
                        for feat in features:
                            if feat in num_options:
                                st.pyplot(
                                    iviz.grouped_box(df, category=target, numeric=feat),
                                    clear_figure=True,
                                )
                    else:
                        st.caption(
                            "Click **Plot grouped boxes** to render the charts."
                        )

        st.markdown("### Textual insights")
        for line in target_analysis.generate_target_insights(df, target):
            st.markdown(f"- {line}")


# ---------- Insights ----------
with tabs[6]:
    st.subheader("Generated Insights")
    with st.spinner("Analysing..."):
        insight_lines = insights_mod.generate_insights(df)
    for line in insight_lines:
        st.markdown(f"- {line}")


# ---------- Report ----------
with tabs[7]:
    st.subheader("Downloadable HTML Report")
    st.write(
        "Generate a clean executive-style HTML report — dataset overview, "
        "column health, time range, basic statistics, and a correlation heatmap."
    )
    st.caption(
        "The report is built from the **currently filtered** data — "
        "adjust filters in the sidebar first if you want a subset. "
        "To save as PDF, open the downloaded HTML in a browser and use "
        "**Print → Save as PDF**."
    )

    with st.expander("Branding (optional)", expanded=False):
        b1, b2 = st.columns(2)
        with b1:
            report_title = st.text_input(
                "Report title",
                value="Auto EDA Report",
                key="report_title",
            )
        with b2:
            company_name = st.text_input(
                "Company / team name",
                value="",
                key="report_company",
                help="Shown above the title. Leave blank to omit.",
            )

    if st.button("Generate report"):
        with st.spinner("Building report..."):
            html = report_generator.build_html_report(
                df=df,
                filename=dataset_label,
                overview=overview,
                dtype_info=dtype_info,
                missing=missing,
                numeric_stats=numeric_stats,
                dup_count=dup_summary["duplicate_rows"],
                company_name=company_name.strip() or None,
                report_title=report_title.strip() or "Auto EDA Report",
                logo_path=LOGO,
            )
        st.success("Report ready.")
        st.download_button(
            "Download HTML report",
            data=html.encode("utf-8"),
            file_name=exporter.make_export_filename(
                f"eda_report_{dataset_stem}", "html",
            ),
            mime="text/html",
        )
