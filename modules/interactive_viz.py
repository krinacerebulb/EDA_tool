"""Chart builders for the Auto EDA tabs.

Every chart except ``multi_line_time_series`` returns a **matplotlib**
``Figure``. Plotly is reserved for the multi-line / dual-axis time series
view where interactive zoom is genuinely useful. Static matplotlib charts
ship to the browser as PNG bytes, which is dramatically lighter than
Plotly's SVG/WebGL renderer on industrial datasets.

**No sampling.** Every chart renders on the complete DataFrame the caller
passes in. For ML-focused EDA / feature engineering / outlier inspection,
full-data accuracy matters more than render speed. Rendering is tuned
instead via small marker sizes, alpha blending, and the matplotlib ``Agg``
backend — none of which discard rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # server-side rendering, no GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px  # noqa: F401 — kept for parity / future use
import plotly.graph_objects as go

from utils.helpers import apply_legend, plotly_template, short_label


# Consistent figure size + base style. Seaborn whitegrid + a slim DPI keeps
# PNG payloads small and modern-looking.
_FIG_W = 8
_FIG_H = 4.5
_DPI = 100

_BASE_COLOR = "#2C7BE5"
_ACCENT_COLOR = "#EF553B"
_BAND_COLOR = "#FFA15A"


def _new_axes(figsize=(_FIG_W, _FIG_H)):
    """Build a fresh (fig, ax) styled consistently across all charts."""
    sns.set_style("whitegrid", {"axes.edgecolor": "#D1D5DB"})
    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
    ax.tick_params(axis="both", labelsize=9, colors="#374151")
    ax.title.set_color("#1F2A37")
    for spine in ax.spines.values():
        spine.set_color("#D1D5DB")
    return fig, ax


# --------------------------------------------------------------------- #
# Static (matplotlib / seaborn) charts
# --------------------------------------------------------------------- #

def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> plt.Figure:
    """Histogram with mean and ±1σ overlays, rendered on the full series."""
    series = pd.to_numeric(df[column], errors="coerce").dropna()

    fig, ax = _new_axes()
    sns.histplot(series, bins=bins, color=_BASE_COLOR, alpha=0.85,
                 edgecolor="white", ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    title = f"Distribution of {column}"
    if not series.empty:
        mean = float(series.mean())
        std = float(series.std()) if len(series) > 1 else float("nan")

        if pd.notna(mean):
            ax.axvline(mean, color=_ACCENT_COLOR, linewidth=2,
                       label=f"μ = {mean:.3g}")
        if pd.notna(std) and std > 0:
            ax.axvline(mean - std, color=_BAND_COLOR, linewidth=1.4,
                       linestyle="--", label=f"−1σ ({mean - std:.3g})")
            ax.axvline(mean + std, color=_BAND_COLOR, linewidth=1.4,
                       linestyle="--", label=f"+1σ ({mean + std:.3g})")
            title = f"Distribution of {column}  —  μ = {mean:.3g}, σ = {std:.3g}"
        elif pd.notna(mean):
            title = f"Distribution of {column}  —  μ = {mean:.3g}"
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    ax.set_title(title, fontsize=11, fontweight="600")
    fig.tight_layout()
    return fig


def boxplot(df: pd.DataFrame, column: str) -> plt.Figure:
    """Vertical boxplot for a single numeric column (full data).

    Vertical orientation gives clearer comparisons across multiple
    industrial features and matches the convention used in grouped
    boxplots elsewhere in the app.
    """
    series = df[column].dropna()
    fig, ax = _new_axes(figsize=(3.6, 5.0))
    sns.boxplot(y=series, color=_BASE_COLOR, fliersize=2.0, ax=ax, width=0.45)
    ax.set_ylabel(column)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_title(f"Boxplot of {column}", fontsize=11, fontweight="600")
    fig.tight_layout()
    return fig


def bar_chart(df: pd.DataFrame, column: str, top_n: int = 10) -> plt.Figure:
    """Horizontal bar chart of the top-N most frequent categories."""
    counts = df[column].dropna().astype(str).value_counts().head(top_n)
    fig, ax = _new_axes(figsize=(_FIG_W, max(3.0, 0.4 * len(counts) + 1.2)))
    ax.barh(counts.index[::-1], counts.values[::-1], color=_BASE_COLOR)
    ax.set_xlabel("Count")
    ax.set_ylabel(column)
    ax.set_title(f"Top {len(counts)} values in {column}",
                 fontsize=11, fontweight="600")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    trendline: bool = False,
) -> plt.Figure:
    """Scatter plot, optionally coloured by a third column. Full-data.

    Marker size and alpha are tuned to keep dense plots readable on
    100k+-row inputs without removing any rows. The matplotlib ``Agg``
    backend rasterises everything server-side, so the browser only ever
    sees a PNG.
    """
    cols = [c for c in [x, y, color] if c and c in df.columns]
    plot_df = df[cols].dropna(subset=[x, y])
    n = len(plot_df)

    # Alpha and marker size scale with point count to preserve density
    # information without discarding rows.
    if n > 200_000:
        alpha, marker_size = 0.18, 3
    elif n > 50_000:
        alpha, marker_size = 0.28, 4
    elif n > 10_000:
        alpha, marker_size = 0.45, 5
    else:
        alpha, marker_size = 0.65, 8

    fig, ax = _new_axes()
    if color and color in plot_df.columns:
        sns.scatterplot(
            data=plot_df, x=x, y=y, hue=color,
            palette="tab10", s=marker_size, alpha=alpha,
            edgecolor="none", linewidth=0, ax=ax,
        )
        ax.legend(loc="best", fontsize=8, frameon=True, title=color)
    else:
        ax.scatter(
            plot_df[x], plot_df[y],
            s=marker_size, alpha=alpha, color=_BASE_COLOR,
            edgecolors="none", linewidths=0,
        )

    if trendline and n >= 2:
        try:
            xs = pd.to_numeric(plot_df[x], errors="coerce")
            ys = pd.to_numeric(plot_df[y], errors="coerce")
            mask = xs.notna() & ys.notna()
            if mask.sum() >= 2:
                slope, intercept = np.polyfit(xs[mask], ys[mask], 1)
                xline = np.linspace(xs[mask].min(), xs[mask].max(), 100)
                ax.plot(xline, slope * xline + intercept,
                        color=_ACCENT_COLOR, linewidth=1.8,
                        label=f"y = {slope:.3g}·x + {intercept:.3g}")
                ax.legend(loc="best", fontsize=8)
        except Exception:
            pass

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}", fontsize=11, fontweight="600")
    fig.tight_layout()
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    precision: int = 2,
) -> go.Figure | None:
    """Interactive Plotly correlation heatmap on every numeric column.

    Plotly is used here (rather than matplotlib) so users can hover any cell
    to read the exact correlation value, zoom into a sub-block of features,
    and pan across wide matrices — useful for ML feature-selection workflows.

    Inline cell annotations are auto-suppressed on wide matrices so the grid
    stays readable; values remain visible on hover regardless of size.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None

    p = max(0, int(precision))
    corr = numeric.corr().round(max(p, 3))
    n = corr.shape[0]

    # Inline annotations off past ~25 columns — values stay on hover anyway.
    text_auto: str | bool = f".{p}f" if n <= 25 else False

    # Height scales with column count so labels don't crush together.
    height = max(420, min(1100, 28 * n + 160))

    fig = px.imshow(
        corr,
        text_auto=text_auto,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title=f"Correlation Heatmap  ({n} numeric columns)",
        template=plotly_template(),
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="corr"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=height,
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def outlier_scatter(
    df: pd.DataFrame,
    column: str,
    x_col: str | None = None,
    method: str = "iqr",
    k: float = 1.5,
    z_threshold: float = 3.0,
    color_by: str | None = None,
    precision: int = 2,
    start_dt: pd.Timestamp | None = None,
    end_dt: pd.Timestamp | None = None,
) -> go.Figure | None:
    """Interactive Plotly scatter that paints outliers in a distinct colour.

    Useful in industrial EDA for spotting sensor drift, bad reads, or
    extreme operational events at a glance. Normal points render in
    muted blue; outliers render in bright red with horizontal dashed
    lines marking the lower / upper bounds.

    Parameters
    ----------
    column
        Numeric column to inspect.
    x_col
        Optional column for the X axis. Datetime or numeric columns are
        used as-is; anything else falls back to row index. ``None`` →
        row index.
    method
        ``"iqr"`` (default) flags points outside [Q1 − k·IQR, Q3 + k·IQR].
        ``"zscore"`` flags points with |z| > ``z_threshold``.
    color_by
        Optional categorical column. When set, INLIER markers are coloured
        by this column so per-group context (e.g. machine ID, shift) is
        visible alongside outlier flags. Outliers stay red regardless.
    start_dt, end_dt
        Optional time window. Only applied when ``x_col`` is a datetime
        column; restricts both the outlier statistics and the plot to
        the chosen operational window (e.g. one shift).
    """
    if column not in df.columns:
        return None

    df_in = df
    if (
        (start_dt is not None or end_dt is not None)
        and x_col
        and x_col in df.columns
    ):
        parsed_dt = pd.to_datetime(df[x_col], errors="coerce")
        mask = pd.Series(True, index=df.index)
        if start_dt is not None:
            mask &= parsed_dt >= pd.Timestamp(start_dt)
        if end_dt is not None:
            mask &= parsed_dt <= pd.Timestamp(end_dt)
        df_in = df.loc[mask]
        if df_in.empty:
            return None

    s = pd.to_numeric(df_in[column], errors="coerce")
    mask_valid = s.notna()
    if not mask_valid.any():
        return None

    s_clean = s[mask_valid]

    # --- Outlier mask -----------------------------------------------------
    if method == "zscore":
        mean = float(s_clean.mean())
        std = float(s_clean.std(ddof=0))
        if std == 0 or np.isnan(std):
            is_outlier = pd.Series(False, index=s_clean.index)
            lower = upper = float("nan")
        else:
            lower = mean - z_threshold * std
            upper = mean + z_threshold * std
            is_outlier = (s_clean < lower) | (s_clean > upper)
        bounds_label = f"Z-score (k = {z_threshold:g})"
    else:
        q1 = float(s_clean.quantile(0.25))
        q3 = float(s_clean.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        is_outlier = (s_clean < lower) | (s_clean > upper)
        bounds_label = f"IQR (k = {k:g})"

    # --- X axis -----------------------------------------------------------
    if x_col and x_col in df_in.columns:
        x_series = df_in.loc[mask_valid, x_col]
        if pd.api.types.is_datetime64_any_dtype(df_in[x_col]) or pd.api.types.is_numeric_dtype(df_in[x_col]):
            x_label = x_col
        else:
            x_series = pd.to_datetime(x_series, errors="coerce")
            x_label = x_col
    else:
        x_series = pd.Series(range(len(s_clean)), index=s_clean.index)
        x_label = "Row index"

    p = max(0, int(precision))
    n_total = int(mask_valid.sum())
    n_out = int(is_outlier.sum())
    out_pct = (n_out / n_total * 100) if n_total else 0.0

    fig = go.Figure()

    # --- Inliers ---------------------------------------------------------
    inlier_idx = s_clean.index[~is_outlier]
    if color_by and color_by in df_in.columns and len(inlier_idx) > 0:
        # Group-coloured inliers using a tab10-style palette.
        groups = df_in.loc[inlier_idx, color_by].astype(str)
        palette = [
            "#2C7BE5", "#19C37D", "#AB63FA", "#19D3F3",
            "#FFA15A", "#B6E880", "#FF97FF", "#FECB52",
        ]
        for i, (grp, idxs) in enumerate(groups.groupby(groups).groups.items()):
            fig.add_trace(go.Scatter(
                x=x_series.loc[idxs],
                y=s_clean.loc[idxs],
                mode="markers",
                name=f"{grp}",
                marker=dict(
                    color=palette[i % len(palette)],
                    size=5, opacity=0.6,
                ),
                hovertemplate=(
                    f"{x_label}: %{{x}}<br>"
                    f"{column}: %{{y:.{p}f}}<br>"
                    f"{color_by}: {grp}<extra></extra>"
                ),
            ))
    elif len(inlier_idx) > 0:
        fig.add_trace(go.Scatter(
            x=x_series.loc[inlier_idx],
            y=s_clean.loc[inlier_idx],
            mode="markers",
            name=f"Normal ({n_total - n_out:,})",
            marker=dict(color="#2C7BE5", size=5, opacity=0.55),
            hovertemplate=(
                f"{x_label}: %{{x}}<br>"
                f"{column}: %{{y:.{p}f}}<extra></extra>"
            ),
        ))

    # --- Outliers (always red, on top) -----------------------------------
    outlier_idx = s_clean.index[is_outlier]
    if len(outlier_idx) > 0:
        fig.add_trace(go.Scatter(
            x=x_series.loc[outlier_idx],
            y=s_clean.loc[outlier_idx],
            mode="markers",
            name=f"Outlier ({n_out:,})",
            marker=dict(
                color="#EF553B",
                size=9,
                opacity=0.95,
                symbol="diamond",
                line=dict(color="#7F1D1D", width=1),
            ),
            hovertemplate=(
                f"<b>OUTLIER</b><br>"
                f"{x_label}: %{{x}}<br>"
                f"{column}: %{{y:.{p}f}}<extra></extra>"
            ),
        ))

    # --- Bound lines -----------------------------------------------------
    if not (np.isnan(upper) or np.isnan(lower)):
        fig.add_hline(
            y=upper,
            line=dict(color="#EF553B", dash="dash", width=1),
            annotation_text=f"Upper {upper:.{p}f}",
            annotation_position="top right",
            annotation_font=dict(color="#7F1D1D", size=10),
        )
        fig.add_hline(
            y=lower,
            line=dict(color="#EF553B", dash="dash", width=1),
            annotation_text=f"Lower {lower:.{p}f}",
            annotation_position="bottom right",
            annotation_font=dict(color="#7F1D1D", size=10),
        )

    fig.update_layout(
        title=(
            f"Outlier detection — {column} · {bounds_label} · "
            f"{n_out:,} of {n_total:,} flagged ({out_pct:.{p}f}%)"
        ),
        xaxis_title=x_label,
        yaxis_title=column,
        template=plotly_template(),
        hovermode="closest",
        margin=dict(l=10, r=10, t=70, b=10),
    )
    # Size + place the legend from the actual series so it's always visible.
    apply_legend(fig)
    if precision is not None:
        fig.update_yaxes(tickformat=f".{p}f")
    return fig


def grouped_box(df: pd.DataFrame, category: str, numeric: str) -> plt.Figure:
    """Boxplot of a numeric column grouped by a categorical column. Full data.

    Every group present in the data is plotted. Figure width scales with the
    group count so labels stay readable even on high-cardinality categories.
    """
    sub = df[[category, numeric]].dropna()
    n_groups = sub[category].astype(str).nunique()
    width = max(_FIG_W, min(24, 0.4 * n_groups + 4))

    fig, ax = _new_axes(figsize=(width, 4.5))
    sns.boxplot(
        data=sub, x=category, y=numeric,
        color=_BASE_COLOR, fliersize=2.0, ax=ax,
    )
    ax.set_title(f"{numeric} by {category}", fontsize=11, fontweight="600")
    ax.tick_params(axis="x", labelrotation=30)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# Multi-line / dual-axis time series — kept as Plotly for interactivity
# --------------------------------------------------------------------- #

_LINE_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

_DUAL_AXIS_RATIO = 10.0

# Aggregation frequency aliases exposed in the UI. Values are pandas offset
# aliases — passed straight into ``DataFrame.resample(...)``.
TIME_AGG_FREQUENCIES: dict[str, str | None] = {
    "None":       None,
    "5 minutes":  "5min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "Hourly":     "1h",
    "Daily":      "D",
    "Monthly":    "MS",
}

# Aggregation functions exposed in the UI. Maps user-facing label → pandas
# method name accepted by ``Resampler.agg``. "None" means *do not aggregate*
# — plot every raw row regardless of the chosen interval.
TIME_AGG_FUNCTIONS: dict[str, str | None] = {
    "None":               None,
    "Average":            "mean",
    "Sum":                "sum",
    "Count":              "count",
    "Standard Deviation": "std",
}


def _aggregate_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    freq: str | None,
    agg_func: str | None = "mean",
) -> pd.DataFrame:
    work = df[[date_col, *value_cols]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    for col in value_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)

    # Both ``freq`` and ``agg_func`` must be set to actually resample.
    # Picking "None" in either dropdown short-circuits to raw rows so the
    # user can see every reading.
    if freq and agg_func:
        work = (
            work.set_index(date_col)
            .resample(freq)[value_cols]
            .agg(agg_func)
            .reset_index()
        )
    return work


def _split_axis_groups(
    work: pd.DataFrame,
    value_cols: list[str],
) -> tuple[list[str], list[str]]:
    scales: dict[str, float] = {}
    for col in value_cols:
        s = work[col].dropna().abs()
        if s.empty:
            continue
        med = float(s.median())
        if med > 0:
            scales[col] = med

    if len(scales) < 2:
        return list(value_cols), []

    smallest = min(scales.values())
    largest = max(scales.values())
    if largest / smallest <= _DUAL_AXIS_RATIO:
        return list(value_cols), []

    threshold = (smallest * largest) ** 0.5
    primary = [c for c in value_cols if scales.get(c, threshold) <= threshold]
    secondary = [c for c in value_cols if c not in primary]
    if not secondary or not primary:
        return list(value_cols), []
    return primary, secondary


def multi_line_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    aggregation: str = "None",
    agg_func: str = "Average",
    start_dt: pd.Timestamp | None = None,
    end_dt: pd.Timestamp | None = None,
    precision: int | None = None,
) -> go.Figure | None:
    """Interactive multi-line time series with optional dual y-axis.

    Stays on Plotly so users can zoom and brush across long industrial time
    series. ``aggregation`` selects the time bucket (e.g. "5 minutes",
    "Hourly", "Daily", "Monthly"); ``agg_func`` selects the reduction
    applied inside each bucket (Average / Sum / Count / Standard Deviation).
    This is *aggregation*, not random sampling, and is fully user-controlled.
    With ``aggregation="None"`` every raw row is plotted.

    ``start_dt`` / ``end_dt`` clip the plot to a specific operational window
    (e.g. zoom into a single shift). They run BEFORE aggregation so the
    aggregation buckets reflect the filtered range only.
    """
    if not value_cols:
        return None

    freq = TIME_AGG_FREQUENCIES.get(aggregation)
    fn = TIME_AGG_FUNCTIONS.get(agg_func, "mean")

    # Range filter must happen before aggregation; copy + parse so it works
    # whether the caller's date_col is already datetime or still a string.
    df_in = df
    if start_dt is not None or end_dt is not None:
        df_in = df.copy()
        parsed_dt = pd.to_datetime(df_in[date_col], errors="coerce")
        mask = pd.Series(True, index=df_in.index)
        if start_dt is not None:
            mask &= parsed_dt >= pd.Timestamp(start_dt)
        if end_dt is not None:
            mask &= parsed_dt <= pd.Timestamp(end_dt)
        df_in = df_in.loc[mask]
        if df_in.empty:
            return None

    work = _aggregate_time_series(df_in, date_col, value_cols, freq, agg_func=fn)
    if work.empty:
        return None

    primary, secondary = _split_axis_groups(work, value_cols)

    fig = go.Figure()
    color_iter = iter(_LINE_COLORS * ((len(value_cols) // len(_LINE_COLORS)) + 1))

    for col in primary:
        fig.add_trace(
            go.Scatter(
                x=work[date_col],
                y=work[col],
                mode="lines",
                # Legend shows a shortened tag; the full name stays in hover.
                name=short_label(col),
                line=dict(color=next(color_iter), width=1.8),
                hovertemplate=f"{col}<br>{date_col}: %{{x}}<br>%{{y}}<extra></extra>",
            )
        )

    for col in secondary:
        fig.add_trace(
            go.Scatter(
                x=work[date_col],
                y=work[col],
                mode="lines",
                name=f"{short_label(col)} (right axis)",
                line=dict(color=next(color_iter), width=1.8, dash="dot"),
                yaxis="y2",
                hovertemplate=f"{col} (right axis)<br>{date_col}: %{{x}}<br>%{{y}}<extra></extra>",
            )
        )

    # Title: with long industrial tags, joining every column name overflows
    # the header, so collapse to a count when there's more than one series.
    if len(value_cols) == 1:
        title = f"{short_label(value_cols[0])} over {date_col}"
    else:
        title = f"{len(value_cols)} series over {date_col}"

    # Axis titles are intentionally omitted: each series is already named in
    # the legend, and printing the full tag(s) vertically on the y-axis (and
    # again on the right axis) produced giant, unreadable side labels.
    # x-axis title is omitted on purpose — the title already says "over
    # {date_col}", and a redundant axis label only competes with the legend
    # for the bottom strip.
    layout: dict = dict(
        title=title,
        template=plotly_template(),
        hovermode="x unified",
    )
    if secondary:
        layout["yaxis2"] = dict(
            overlaying="y",
            side="right",
            showgrid=False,
        )

    fig.update_layout(**layout)
    # Legend goes below the plot — clear of the range-selector buttons at the
    # top (and of the secondary right axis when present).
    apply_legend(fig, below=True, secondary_axis=bool(secondary))
    fig.update_xaxes(
        # No range-slider: the mini-plot collided with the legend below it,
        # and the range-selector buttons already provide the time filtering.
        rangeselector=dict(
            buttons=[
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
    )
    if precision is not None:
        p = max(0, int(precision))
        fig.update_yaxes(tickformat=f".{p}f")
        if secondary:
            fig.update_layout(yaxis2=dict(
                **(layout.get("yaxis2") or {}),
                tickformat=f".{p}f",
            ))
    return fig
