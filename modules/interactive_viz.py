"""Interactive Plotly-based chart builders.

Each function returns a plotly.graph_objects.Figure that can be rendered
with st.plotly_chart(fig, use_container_width=True).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import plotly_template


def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    """Interactive histogram for a numeric column with mean and ±1σ overlays."""
    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        title=f"Distribution of {column}",
        marginal="box",
        opacity=0.85,
        template=plotly_template(),
    )
    fig.update_layout(bargap=0.05, yaxis_title="Frequency")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if not series.empty:
        mean = float(series.mean())
        std = float(series.std()) if len(series) > 1 else float("nan")

        if pd.notna(mean):
            fig.add_vline(
                x=mean,
                line_color="#EF553B",
                line_width=2,
                annotation_text=f"μ = {mean:.3g}",
                annotation_position="top",
                annotation_font_color="#EF553B",
            )

        if pd.notna(std) and std > 0:
            fig.add_vline(
                x=mean - std,
                line_color="#FFA15A",
                line_dash="dash",
                line_width=1.5,
                annotation_text="−1σ",
                annotation_position="top",
                annotation_font_color="#FFA15A",
            )
            fig.add_vline(
                x=mean + std,
                line_color="#FFA15A",
                line_dash="dash",
                line_width=1.5,
                annotation_text="+1σ",
                annotation_position="top",
                annotation_font_color="#FFA15A",
            )
            fig.update_layout(
                title=f"Distribution of {column}  —  μ = {mean:.3g}, σ = {std:.3g}"
            )
        elif pd.notna(mean):
            fig.update_layout(
                title=f"Distribution of {column}  —  μ = {mean:.3g}"
            )

    return fig


def boxplot(df: pd.DataFrame, column: str) -> go.Figure:
    """Interactive boxplot for a single numeric column."""
    fig = px.box(
        df,
        x=column,
        points="outliers",
        title=f"Boxplot of {column}",
        template=plotly_template(),
    )
    return fig


def bar_chart(df: pd.DataFrame, column: str, top_n: int = 10) -> go.Figure:
    """Bar chart of the top-N most frequent categories."""
    counts = df[column].dropna().astype(str).value_counts().head(top_n)
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        title=f"Top {len(counts)} values in {column}",
        labels={"x": "Count", "y": column},
        template=plotly_template(),
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    trendline: bool = False,
) -> go.Figure:
    """Interactive scatter plot, optionally coloured by a third column."""
    kwargs = dict(
        data_frame=df,
        x=x,
        y=y,
        title=f"{y} vs {x}",
        template=plotly_template(),
        opacity=0.7,
    )
    if color and color in df.columns:
        kwargs["color"] = color
    if trendline:
        kwargs["trendline"] = "ols"
    try:
        return px.scatter(**kwargs)
    except Exception:
        # statsmodels (needed for trendline) might not be installed.
        kwargs.pop("trendline", None)
        return px.scatter(**kwargs)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """Interactive correlation heatmap of numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None

    corr = numeric.corr().round(3)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Heatmap",
        template=plotly_template(),
    )
    fig.update_layout(coloraxis_colorbar=dict(title="corr"))
    return fig


def grouped_box(df: pd.DataFrame, category: str, numeric: str) -> go.Figure:
    """Boxplot of a numeric column grouped by a categorical column."""
    fig = px.box(
        df,
        x=category,
        y=numeric,
        points="outliers",
        title=f"{numeric} by {category}",
        template=plotly_template(),
    )
    fig.update_layout(xaxis_tickangle=-30)
    return fig


# --------------------------------------------------------------------- #
# Multi-line / dual-axis time series
# --------------------------------------------------------------------- #

# Colour palette for multi-line traces (Plotly default sequence).
_LINE_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# Threshold for engaging the secondary y-axis. If max(scale)/min(scale)
# across selected columns exceeds this, the smaller-scale columns get
# pushed onto a secondary axis so all lines remain visible.
_DUAL_AXIS_RATIO = 10.0


def _aggregate_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    freq: str | None,
) -> pd.DataFrame:
    """Aggregate (mean) by the given pandas offset freq, or return as-is.

    ``freq`` is ``"D"`` for daily, ``"MS"`` for month-start, or ``None`` to
    keep the raw rows.
    """
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
    """Decide which columns belong on primary vs secondary y-axis.

    A column's "scale" is the median absolute value. If max/min across
    columns exceeds ``_DUAL_AXIS_RATIO``, columns at or below the geometric
    mean go on the primary axis and the rest on the secondary axis.
    """
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
    # Edge case: all columns landed on one side (e.g. ties at threshold).
    if not secondary or not primary:
        return list(value_cols), []
    return primary, secondary


def multi_line_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    aggregation: str = "None",
    sample_max: int = 10000,
) -> go.Figure | None:
    """Plot multiple numeric columns over time, with auto dual-axis.

    Parameters
    ----------
    aggregation : "None" | "Daily" | "Monthly"
        Resample to daily (``"D"``) or month-start (``"MS"``) means before
        plotting. ``"None"`` keeps raw rows.
    sample_max : int
        If the prepared frame still has more rows than this, evenly sample
        ``sample_max`` rows so plotting stays responsive.
    """
    if not value_cols:
        return None

    freq_map = {"None": None, "Daily": "D", "Monthly": "MS"}
    freq = freq_map.get(aggregation)

    work = _aggregate_time_series(df, date_col, value_cols, freq)
    if work.empty:
        return None

    if len(work) > sample_max:
        idx = np.linspace(0, len(work) - 1, sample_max).astype(int)
        work = work.iloc[idx].reset_index(drop=True)

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
