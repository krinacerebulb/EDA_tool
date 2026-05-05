"""Time-series trend analysis utilities.

Covers:
- Detecting datetime columns (native dtype OR parseable object columns).
- Preparing a sorted (date, value) frame for plotting.
- Building an interactive Plotly line chart with rolling mean, range
  slider, and range-selector buttons.
- Producing short textual trend insights (direction, peaks/troughs,
  biggest step change, basic autocorrelation-based seasonality hint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


_DATETIME_PARSE_THRESHOLD = 0.8  # % of sampled values that must parse


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that are datetime or look parseable as datetime.

    A column qualifies if either:
    - pandas already considers its dtype datetime-like, or
    - it is an object column whose first sampled non-null values parse
      into datetime with a success rate >= ``_DATETIME_PARSE_THRESHOLD``.
    """
    found: list[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            found.append(col)
            continue
        if s.dtype != "object":
            continue
        sample = s.dropna().head(50)
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
        except Exception:
            continue
        if parsed.notna().mean() >= _DATETIME_PARSE_THRESHOLD:
            found.append(col)
    return found


def prepare_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Return a (date_col, value_col) frame with parsed, sorted, clean rows."""
    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col, value_col]).sort_values(date_col)
    out = out.reset_index(drop=True)
    return out


def time_series_plot(
    prepared: pd.DataFrame,
    date_col: str,
    value_col: str,
    rolling_window: int | None = None,
) -> go.Figure:
    """Interactive line chart with zoom, hover, range slider, and smoothing."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prepared[date_col],
            y=prepared[value_col],
            mode="lines+markers",
            name=value_col,
            line=dict(color="#636EFA", width=1.5),
            marker=dict(size=4),
            opacity=0.75,
            hovertemplate=f"{date_col}: %{{x}}<br>{value_col}: %{{y}}<extra></extra>",
        )
    )

    if rolling_window and rolling_window > 1 and len(prepared) >= rolling_window:
        smoothed = (
            prepared[value_col]
            .rolling(window=rolling_window, min_periods=1)
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=prepared[date_col],
                y=smoothed,
                mode="lines",
                name=f"Rolling mean ({rolling_window})",
                line=dict(color="#EF553B", width=3),
            )
        )

    fig.update_layout(
        title=f"{value_col} over {date_col}",
        xaxis_title=date_col,
        yaxis_title=value_col,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
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


def trend_insights(
    prepared: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> list[str]:
    """Generate short textual insights about trend, peaks, drops, seasonality."""
    if len(prepared) < 3:
        return ["Not enough data points for trend analysis (need at least 3)."]

    vals = prepared[value_col].to_numpy(dtype=float)
    dates = prepared[date_col]
    insights: list[str] = []

    # --- Overall direction (linear fit slope) ---
    x = np.arange(len(vals))
    slope, _ = np.polyfit(x, vals, 1)
    net_change = vals[-1] - vals[0]
    pct_change = (net_change / vals[0] * 100) if vals[0] != 0 else float("nan")

    if abs(slope) < 1e-12:
        direction = "essentially flat"
    elif slope > 0:
        direction = "**increasing** 📈"
    else:
        direction = "**decreasing** 📉"

    pct_str = f"{pct_change:+.1f}%" if np.isfinite(pct_change) else "n/a"
    insights.append(
        f"Overall trend is {direction} (net change = {net_change:+.2f}, "
        f"{pct_str} vs. start)."
    )

    # --- Peak and trough ---
    peak_idx = int(np.argmax(vals))
    trough_idx = int(np.argmin(vals))
    peak_date = pd.Timestamp(dates.iloc[peak_idx]).strftime("%Y-%m-%d")
    trough_date = pd.Timestamp(dates.iloc[trough_idx]).strftime("%Y-%m-%d")
    insights.append(f"Peak value **{vals[peak_idx]:.2f}** on **{peak_date}**.")
    insights.append(f"Lowest value **{vals[trough_idx]:.2f}** on **{trough_date}**.")

    # --- Biggest step change ---
    diffs = np.diff(vals)
    if diffs.size:
        up_idx = int(np.argmax(diffs))
        down_idx = int(np.argmin(diffs))
        if diffs[up_idx] > 0:
            up_date = pd.Timestamp(dates.iloc[up_idx + 1]).strftime("%Y-%m-%d")
            insights.append(
                f"Largest jump: **+{diffs[up_idx]:.2f}** on {up_date}."
            )
        if diffs[down_idx] < 0:
            down_date = pd.Timestamp(dates.iloc[down_idx + 1]).strftime("%Y-%m-%d")
            insights.append(
                f"Largest drop: **{diffs[down_idx]:.2f}** on {down_date}."
            )

    # --- Basic seasonality hint from autocorrelation ---
    try:
        s = pd.Series(vals)
        n = len(s)
        candidate_lags = [7, 12, 24, 30]
        lags = [lag for lag in candidate_lags if lag < n - 1]
        best = None
        for lag in lags:
            ac = s.autocorr(lag=lag)
            if ac is None or np.isnan(ac):
                continue
            if best is None or abs(ac) > abs(best[1]):
                best = (lag, ac)
        if best is not None and abs(best[1]) >= 0.3:
            insights.append(
                f"Possible seasonality — autocorrelation at lag "
                f"**{best[0]}** is **{best[1]:+.2f}**."
            )
    except Exception:
        pass

    return insights
