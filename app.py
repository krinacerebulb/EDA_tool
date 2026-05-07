"""Auto EDA — Streamlit front-end.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

LOGO_PATH = Path(__file__).parent / "assets" / "cb-logo-tagline-main.png"
LOGO = str(LOGO_PATH) if LOGO_PATH.exists() else None

from modules import (
    data_cleaning,
    data_loader,
    eda_analysis,
    filters,
    insights as insights_mod,
    interactive_viz as iviz,
    preprocessing,
    report_generator,
    target_analysis,
    time_series as ts_mod,
    type_detection,
)
from utils.helpers import human_bytes, plotly_template, split_columns


st.set_page_config(
    page_title="Auto EDA Platform",
    page_icon=LOGO or ":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme bootstrap ----------
# Reads the theme that was selected on the previous run (default: Light) so
# the CSS injected below already reflects the user's choice. The actual radio
# widget is rendered later in the sidebar; switching it triggers a rerun and
# this block re-evaluates with the new value.
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
_active_theme = st.session_state.theme

# Drive Plotly's *default* template too — chart helpers also read it, but
# setting it here protects any direct px/go usage we may add later.
import plotly.io as pio
pio.templates.default = "plotly_dark" if _active_theme == "Dark" else "plotly_white"

st.markdown(
    """
    <style>
      /* --- Hide non-essential Streamlit chrome (without breaking the
             collapsed-sidebar expand chevron) --- */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      [data-testid="stDeployButton"] {display: none;}
      [data-testid="stToolbarActions"] {visibility: hidden;}
      [data-testid="stStatusWidget"] {visibility: hidden;}

      /* Force the sidebar expand control to stay visible. */
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
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

# ---------- Dark-mode overrides (only injected when theme == "Dark") ----------
if _active_theme == "Dark":
    st.markdown(
        """
        <style>
          .stApp {
            background-color: #0E1117;
            color: #E5E7EB;
          }
          .stApp p, .stApp li, .stApp label, .stApp .stMarkdown,
          .stApp [data-testid="stCaptionContainer"] {
            color: #E5E7EB;
          }
          .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #F1F5F9 !important;
          }
          [data-testid="stSidebar"] {
            background-color: #161B22 !important;
            border-right: 1px solid #21262D;
          }
          [data-testid="stSidebar"] *,
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] .stMarkdown {
            color: #E5E7EB;
          }
          [data-testid="stSidebar"] .cerebulb-product {color: #9CA3AF;}
          [data-testid="stMetric"] {
            background: #161B22 !important;
            border: 1px solid #30363D !important;
          }
          [data-testid="stMetricLabel"] {color: #9CA3AF !important;}
          [data-testid="stMetricValue"] {color: #F1F5F9 !important;}
          .stTabs [data-baseweb="tab-list"] {
            border-bottom: 1px solid #30363D;
          }
          .stTabs [data-baseweb="tab"] {color: #9CA3AF;}
          .stTabs [aria-selected="true"] {color: #F1F5F9 !important;}
          [data-testid="stExpander"] {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 8px;
          }
          [data-testid="stExpander"] summary {color: #E5E7EB;}
          [data-baseweb="input"] input,
          [data-baseweb="select"] > div,
          [data-baseweb="textarea"] textarea {
            background-color: #161B22 !important;
            color: #E5E7EB !important;
            border-color: #30363D !important;
          }
          .stButton > button, .stDownloadButton > button {
            background-color: #21262D;
            color: #F1F5F9;
            border: 1px solid #30363D;
          }
          .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: #30363D;
            border-color: #444C56;
          }
          [data-testid="stPlotlyChart"] {background: transparent;}
          [data-testid="stDataFrame"], [data-testid="stTable"] {
            background-color: #0E1117;
          }
          [data-testid="stAlert"] {
            background-color: #161B22 !important;
            border-color: #30363D !important;
            color: #E5E7EB !important;
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

# Theme toggle — value persists in st.session_state.theme via the key.
st.sidebar.radio(
    "Theme",
    options=["Light", "Dark"],
    horizontal=True,
    key="theme",
    help="Switch between light and dark mode. Plotly charts adapt automatically.",
)

st.sidebar.markdown("Upload a dataset to get started.")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV, Excel, JSON, or Parquet",
    type=["csv", "xlsx", "xls", "json", "parquet", "pq"],
)

st.sidebar.markdown("---")
preview_rows = st.sidebar.slider("Preview rows", 5, 50, 10)

st.sidebar.markdown("---")
auto_convert = st.sidebar.checkbox(
    "Enable automatic type conversion",
    value=True,
    help=(
        "Detect object columns that are mostly numeric (≥ 70% of non-null "
        "values parse as numbers) and convert them so they participate in "
        "numeric EDA, plots, and correlations. Original data is preserved."
    ),
)


# ---------- Main ----------
st.title("Auto EDA Platform")
st.caption("Automated exploratory data analysis, visualizations, and insights — a Cerebulb product.")

if uploaded_file is None:
    st.info("Upload a dataset from the sidebar to begin.")
    st.stop()

# Load with spinner + error handling.
with st.spinner("Loading dataset..."):
    try:
        raw_df = data_loader.load_dataset(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # last-resort guard
        st.error(f"Unexpected error while loading the file: {exc}")
        st.stop()

if raw_df.empty:
    st.warning("The uploaded file loaded successfully but contains no rows.")
    st.stop()

st.success(
    f"Loaded **{uploaded_file.name}** — "
    f"{raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns."
)


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


# ---------- Sidebar: dynamic filters (applied to everything below) ----------
st.sidebar.markdown("---")
df, flt_summary = filters.render_sidebar_filters(preprocessed_df)

# Filter status metrics.
if flt_summary["is_filtered"]:
    st.markdown("### 🎯 Filter impact")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows remaining", f"{flt_summary['filtered_rows']:,}")
    m2.metric("Original rows", f"{flt_summary['original_rows']:,}")
    m3.metric("Rows removed", f"{flt_summary['dropped_rows']:,}")
    m4.metric("% remaining", f"{flt_summary['percent_remaining']}%")

    if flt_summary["applied"]:
        with st.expander(f"Active filters ({len(flt_summary['applied'])})", expanded=False):
            for rule in flt_summary["applied"]:
                st.markdown(f"- {rule}")

if df.empty:
    st.warning(
        "Current filters exclude every row. Relax them from the sidebar to continue."
    )
    st.stop()


# ---------- Derived analytics (all on filtered df) ----------
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
    st.subheader("Dataset Preview")
    st.dataframe(df.head(preview_rows), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['rows']:,}")
    c2.metric("Columns", overview["columns"])
    c3.metric("Memory", human_bytes(overview["memory_bytes"]))
    c4.metric("Missing values", f"{overview['total_missing']:,}")

    st.subheader("Column Types")
    st.dataframe(dtype_info, use_container_width=True)

    st.subheader("Download filtered dataset")
    filters.download_button(df)


# ---------- Cleaning ----------
with tabs[1]:
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
        st.dataframe(summary_frame, use_container_width=True)

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
        st.dataframe(dt_summary, use_container_width=True)
        st.caption(
            "Confidence is the share of non-null values that parsed as a "
            "valid date/time. Invalid counts are estimated when sampling "
            "is used on large columns."
        )

    st.markdown("---")
    st.subheader("Missing Values")
    st.dataframe(missing, use_container_width=True)

    st.subheader("Duplicates")
    st.write(
        f"**{dup_summary['duplicate_rows']:,}** duplicate rows "
        f"({dup_summary['duplicate_percent']}% of the dataset)."
    )

    st.subheader("Outliers (IQR method)")
    if outliers.empty:
        st.info("No numeric columns to check for outliers.")
    else:
        st.dataframe(outliers, use_container_width=True)

    st.caption("Note: nothing is modified automatically — this is analysis only.")


# ---------- Statistics ----------
with tabs[2]:
    st.subheader("Numeric Columns")
    if numeric_stats.empty:
        st.info("No numeric columns detected.")
    else:
        st.dataframe(numeric_stats, use_container_width=True)

    st.subheader("Categorical Columns")
    if not categorical_stats:
        st.info("No categorical columns detected.")
    else:
        for col, info in categorical_stats.items():
            with st.expander(f"{col}  —  {info['unique']} unique, mode: {info['mode']}"):
                st.dataframe(info["top_values"], use_container_width=True)


# ---------- Visualizations (interactive Plotly) ----------
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

        if viz_type == "Histogram":
            if not numeric_cols:
                st.info("No numeric columns available.")
            else:
                col = st.selectbox("Column", numeric_cols, key="hist_col")
                bins = st.slider("Bins", 5, 100, 30)
                st.plotly_chart(
                    iviz.histogram(df, col, bins=bins),
                    use_container_width=True,
                )

        elif viz_type == "Boxplot":
            if not numeric_cols:
                st.info("No numeric columns available.")
            else:
                col = st.selectbox("Column", numeric_cols, key="box_col")
                st.plotly_chart(iviz.boxplot(df, col), use_container_width=True)

        elif viz_type == "Scatter plot":
            if len(numeric_cols) < 2:
                st.info("Need at least two numeric columns for a scatter plot.")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    x_col = st.selectbox("X axis", numeric_cols, key="scatter_x")
                with c2:
                    y_col = st.selectbox(
                        "Y axis",
                        [c for c in numeric_cols if c != x_col],
                        key="scatter_y",
                    )
                with c3:
                    color_options = ["(none)"] + categorical_cols + numeric_cols
                    color = st.selectbox("Colour by", color_options, key="scatter_color")
                color_col = None if color == "(none)" else color
                st.plotly_chart(
                    iviz.scatter(df, x_col, y_col, color=color_col),
                    use_container_width=True,
                )

        elif viz_type == "Bar chart (categorical)":
            if not categorical_cols:
                st.info("No categorical columns available.")
            else:
                col = st.selectbox("Column", categorical_cols, key="bar_col")
                top_n = st.slider("Top N values", 3, 30, 10)
                st.plotly_chart(
                    iviz.bar_chart(df, col, top_n=top_n),
                    use_container_width=True,
                )

        elif viz_type == "Correlation heatmap":
            fig = iviz.correlation_heatmap(df)
            if fig is None:
                st.info("Need at least 2 numeric columns for a correlation heatmap.")
            else:
                st.plotly_chart(fig, use_container_width=True)

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
                c1, c2 = st.columns([1, 2])
                with c1:
                    ml_date = st.selectbox(
                        "Date column", datetime_cols_viz, key="ml_date",
                    )
                with c2:
                    ml_values = st.multiselect(
                        "Numerical columns to compare",
                        numeric_cols,
                        default=numeric_cols[: min(2, len(numeric_cols))],
                        key="ml_values",
                        help=(
                            "Pick two or more numeric columns. If their scales "
                            "differ by more than 10×, a secondary y-axis is "
                            "added automatically."
                        ),
                    )

                c3, c4 = st.columns(2)
                with c3:
                    ml_agg = st.selectbox(
                        "Aggregation",
                        ["None", "Daily", "Monthly"],
                        index=0,
                        key="ml_agg",
                        help="Resample (mean) before plotting.",
                    )
                with c4:
                    ml_sample = st.slider(
                        "Max points",
                        min_value=500,
                        max_value=20000,
                        value=5000,
                        step=500,
                        key="ml_sample",
                        help="Downsample for performance on large datasets.",
                    )

                if not ml_values:
                    st.info("Select at least one numeric column.")
                else:
                    fig = iviz.multi_line_time_series(
                        df,
                        date_col=ml_date,
                        value_cols=ml_values,
                        aggregation=ml_agg,
                        sample_max=ml_sample,
                    )
                    if fig is None:
                        st.warning(
                            "No rows left after parsing the date column. "
                            "Check that it contains valid dates."
                        )
                    else:
                        st.plotly_chart(fig, use_container_width=True)


# ---------- Time Series ----------
with tabs[4]:
    st.subheader("Time Series Trend Analysis")
    st.caption(
        "Automatically detects datetime columns, then builds an interactive "
        "Plotly chart with optional rolling-mean smoothing."
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
        c1, c2, c3 = st.columns(3)
        with c1:
            date_col = st.selectbox("Date column", datetime_cols, key="ts_date")
        with c2:
            value_col = st.selectbox(
                "Value column (numeric)",
                ts_value_candidates,
                key="ts_value",
            )
        with c3:
            apply_smoothing = st.checkbox("Rolling average", value=False, key="ts_smooth")

        rolling_window = None
        if apply_smoothing:
            rolling_window = st.slider(
                "Rolling window (periods)",
                min_value=2,
                max_value=60,
                value=7,
                key="ts_window",
            )

        prepared = ts_mod.prepare_series(df, date_col, value_col)

        if prepared.empty:
            st.warning(
                "No rows left after parsing and cleaning — check that "
                f"`{date_col}` contains dates and `{value_col}` contains numbers."
            )
        else:
            st.caption(
                f"Using {len(prepared):,} rows from "
                f"{prepared[date_col].min():%Y-%m-%d} to "
                f"{prepared[date_col].max():%Y-%m-%d}."
            )
            st.plotly_chart(
                ts_mod.time_series_plot(
                    prepared, date_col, value_col, rolling_window=rolling_window,
                ),
                use_container_width=True,
            )

            st.markdown("### 🔎 Trend insights")
            for line in ts_mod.trend_insights(prepared, date_col, value_col):
                st.markdown(f"- {line}")


# ---------- Target-based EDA ----------
with tabs[5]:
    st.subheader("Target-Based EDA")
    st.caption(
        "Pick a target column to see how every other feature relates to it. "
        "Numeric targets use correlation; categorical targets use grouped comparisons."
    )

    target = st.selectbox(
        "Target column",
        ["(none)"] + list(df.columns),
        key="target_col",
    )

    if target == "(none)":
        st.info("Select a target column above to generate analysis.")
    else:
        numeric_target = target_analysis.is_numeric_target(df, target)
        st.markdown(
            f"**Target type:** {'numerical' if numeric_target else 'categorical'} "
            f"(`{df[target].dtype}`)"
        )

        if numeric_target:
            corr_df = target_analysis.correlations_with_target(df, target)
            if not corr_df.empty:
                st.markdown("### Correlation with target")
                st.dataframe(corr_df, use_container_width=True)

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

            # --- Distribution of the target ---
            st.markdown("### Distribution of target")
            hist_bins = st.slider(
                "Bins", 5, 100, 30, key="target_hist_bins",
            )
            st.plotly_chart(
                iviz.histogram(df, target, bins=hist_bins),
                use_container_width=True,
            )

            # --- Target over time ---
            target_datetime_cols = ts_mod.detect_datetime_columns(df)
            if target_datetime_cols:
                st.markdown("### Target over time")
                tgt_date_col = st.selectbox(
                    "Date column",
                    target_datetime_cols,
                    key="target_line_date",
                )
                tgt_agg = st.selectbox(
                    "Aggregation",
                    ["None", "Daily", "Monthly"],
                    index=1 if len(df) > 500 else 0,
                    key="target_line_agg",
                )
                line_fig = iviz.multi_line_time_series(
                    df,
                    date_col=tgt_date_col,
                    value_cols=[target],
                    aggregation=tgt_agg,
                )
                if line_fig is None:
                    st.info(
                        f"Could not parse `{tgt_date_col}` as dates — "
                        "no rows left to plot."
                    )
                else:
                    st.plotly_chart(line_fig, use_container_width=True)
            else:
                st.caption(
                    "No datetime column detected — skipping target-over-time plot."
                )

            # --- Target distribution across categorical features ---
            cat_options = [c for c in categorical_cols if c != target]
            if cat_options:
                st.markdown("### Target by categorical feature")
                tgt_cat = st.selectbox(
                    "Categorical feature",
                    cat_options,
                    key="target_box_cat",
                )
                st.plotly_chart(
                    iviz.grouped_box(df, category=tgt_cat, numeric=target),
                    use_container_width=True,
                )
            else:
                st.caption(
                    "No categorical features available for grouped box plot."
                )

        else:
            # --- Counts per category ---
            st.markdown(f"### Count per **{target}** category")
            top_n = st.slider(
                "Top N categories", 3, 30, 10, key="target_bar_topn",
            )
            st.plotly_chart(
                iviz.bar_chart(df, target, top_n=top_n),
                use_container_width=True,
            )

            # --- Group means + importance ---
            means_df = target_analysis.group_means(df, target)
            if means_df.empty:
                st.info("No numeric features to compare across the target groups.")
            else:
                st.markdown(f"### Mean of numeric features per **{target}**")
                st.dataframe(means_df, use_container_width=True)

                imp_df = target_analysis.categorical_importance(df, target)
                st.markdown("### Most influential features")
                st.dataframe(imp_df, use_container_width=True)

                if not imp_df.empty:
                    top_feature = imp_df.iloc[0]["Feature"]
                    st.markdown(
                        f"Top-ranked feature: **{top_feature}** — its mean varies most "
                        f"across the groups of **{target}**."
                    )

                    # --- Boxplot: numeric feature vs target groups ---
                    st.markdown("### Numeric features by target group")
                    num_options = imp_df["Feature"].tolist()
                    chosen_features = st.multiselect(
                        "Numeric features",
                        num_options,
                        default=num_options[: min(2, len(num_options))],
                        key="target_box_features",
                    )
                    for feat in chosen_features:
                        st.plotly_chart(
                            iviz.grouped_box(df, category=target, numeric=feat),
                            use_container_width=True,
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
    st.write("Generate a single-file HTML report covering overview, stats, insights, and charts.")
    st.caption(
        "The report is built from the **currently filtered** data — "
        "adjust filters in the sidebar first if you want a subset."
    )

    if st.button("Generate report"):
        with st.spinner("Building report..."):
            html = report_generator.build_html_report(
                df=df,
                filename=uploaded_file.name,
                overview=overview,
                dtype_info=dtype_info,
                missing=missing,
                numeric_stats=numeric_stats,
                insights=insight_lines,
            )
        st.success("Report ready.")
        st.download_button(
            "Download HTML report",
            data=html.encode("utf-8"),
            file_name=f"eda_report_{uploaded_file.name}.html",
            mime="text/html",
        )
