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

from utils.helpers import plotly_template


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
    """Horizontal boxplot for a single numeric column (full data)."""
    series = df[column].dropna()
    fig, ax = _new_axes(figsize=(_FIG_W, 2.6))
    sns.boxplot(x=series, color=_BASE_COLOR, fliersize=2.0, ax=ax)
    ax.set_xlabel(column)
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


def correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
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

    corr = numeric.corr().round(3)
    n = corr.shape[0]

    # Inline annotations off past ~25 columns — values stay on hover anyway.
    text_auto: str | bool = ".2f" if n <= 25 else False

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


def _aggregate_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    freq: str | None,
) -> pd.DataFrame:
    work = df[[date_col, *value_cols]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    for col in value_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)

    if freq:
        work = (
            work.set_index(date_col)
            .resample(freq)[value_cols]
            .mean()
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
) -> go.Figure | None:
    """Interactive multi-line time series with optional dual y-axis.

    Stays on Plotly so users can zoom and brush across long industrial time
    series. ``aggregation`` ("Daily" / "Monthly") deterministically resamples
    by mean — this is *aggregation*, not random sampling, and is fully
    user-controlled. With ``aggregation="None"`` every raw row is plotted.
    """
    if not value_cols:
        return None

    freq_map = {"None": None, "Daily": "D", "Monthly": "MS"}
    freq = freq_map.get(aggregation)

    work = _aggregate_time_series(df, date_col, value_cols, freq)
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
                name=col,
                line=dict(color=next(color_iter), width=1.8),
                hovertemplate=f"{date_col}: %{{x}}<br>{col}: %{{y}}<extra></extra>",
            )
        )

    for col in secondary:
        fig.add_trace(
            go.Scatter(
                x=work[date_col],
                y=work[col],
                mode="lines",
                name=f"{col} (right axis)",
                line=dict(color=next(color_iter), width=1.8, dash="dot"),
                yaxis="y2",
                hovertemplate=f"{date_col}: %{{x}}<br>{col}: %{{y}}<extra></extra>",
            )
        )

    title_cols = ", ".join(value_cols)
    layout: dict = dict(
        title=f"{title_cols} over {date_col}",
        xaxis_title=date_col,
        template=plotly_template(),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    if secondary:
        layout["yaxis"] = dict(title=", ".join(primary))
        layout["yaxis2"] = dict(
            title=", ".join(secondary),
            overlaying="y",
            side="right",
            showgrid=False,
        )
    else:
        layout["yaxis_title"] = ", ".join(primary)

    fig.update_layout(**layout)
    fig.update_xaxes(
        rangeslider_visible=True,
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
    return fig
