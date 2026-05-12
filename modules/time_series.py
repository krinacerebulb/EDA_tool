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
import streamlit as st

from utils.helpers import plotly_template


_DATETIME_PARSE_THRESHOLD = 0.7  # % of sampled values that must parse
_SAMPLE_LIMIT = 500              # cap for detection to keep things responsive
_QUICK_CHECK_SIZE = 8            # values tested in the cheap pre-filter

# Strategies tried in order; we keep the parse with the highest success rate.
# Covers: ISO datetimes, datetimes with milliseconds, time-only strings,
# day-first formats (e.g. "31-12-2024 12:45:30").
_PARSE_STRATEGIES: list[dict] = [
    {"format": "mixed"},
    {},                           # pandas default (handles ISO and many ISO-like)
    {"dayfirst": True},
    {"format": "%H:%M:%S"},
    {"format": "%H:%M:%S.%f"},
    {"format": "%d-%m-%Y %H:%M:%S"},
    {"format": "%d/%m/%Y %H:%M:%S"},
]


def _looks_like_datetime(series: pd.Series) -> bool:
    """Cheap heuristic: do any of the first few values parse as a datetime?

    Used as a pre-filter so wide datasets full of free-text columns don't pay
    the cost of the full multi-strategy parse on columns that obviously
    aren't dates (names, addresses, IDs, etc.).
    """
    sample = series.dropna().head(_QUICK_CHECK_SIZE)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
    except Exception:
        return False
    return bool(parsed.notna().any())


def _best_parse(series: pd.Series):
    """Try multiple parsing strategies; return (parsed, success_rate).

    ``parsed`` is None if every strategy raises before producing a result.
    ``success_rate`` is the fraction of non-null inputs that parsed.
    """
    s = series.dropna()
    if s.empty:
        return None, 0.0

    if len(s) > _SAMPLE_LIMIT:
        s = s.sample(_SAMPLE_LIMIT, random_state=42)

    best_parsed = None
    best_rate = 0.0
    for strat in _PARSE_STRATEGIES:
        try:
            parsed = pd.to_datetime(s, errors="coerce", **strat)
        except (TypeError, ValueError):
            continue
        except Exception:
            continue
        rate = float(parsed.notna().mean()) if len(parsed) else 0.0
        if rate > best_rate:
            best_rate = rate
            best_parsed = parsed
    return best_parsed, best_rate


@st.cache_data(show_spinner=False)
def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that are datetime or parseable as datetime.

    A column qualifies if either:
    - pandas already considers its dtype datetime-like, or
    - any parsing strategy in :data:`_PARSE_STRATEGIES` reaches a success rate
      of at least :data:`_DATETIME_PARSE_THRESHOLD` on a sample of its values.

    Handles date-only, time-only, datetime-with-milliseconds, and day-first
    formats.
    """
    found: list[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            found.append(col)
            continue
        # Numeric columns are skipped — they would otherwise be reinterpreted
        # as Unix epochs and pollute the detection.
        if pd.api.types.is_numeric_dtype(s):
            continue
        if s.dtype not in ("object", "string"):
            continue
        # Cheap pre-filter to skip clearly-non-datetime free-text columns.
        if not _looks_like_datetime(s):
            continue
        _, rate = _best_parse(s)
        if rate >= _DATETIME_PARSE_THRESHOLD:
            found.append(col)
    return found


@st.cache_data(show_spinner=False)
def datetime_detection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabular summary of datetime-like columns.

    Columns: ``Column``, ``Detected Type``, ``Confidence``, ``Invalid``.
    Includes any column that parses with at least 50% confidence so users
    can spot near-misses worth a manual dtype conversion.
    """
    rows: list[dict] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            rows.append({
                "Column": col,
                "Detected Type": "datetime (native dtype)",
                "Confidence": "100.0%",
                "Invalid": 0,
            })
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue
        if s.dtype not in ("object", "string"):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        if not _looks_like_datetime(s):
            continue
        parsed, rate = _best_parse(s)
        if parsed is None or rate < 0.5:
            continue
        # Approximate invalid count by extrapolating sample failure rate.
        invalid_in_sample = int(parsed.isna().sum())
        sample_size = len(parsed)
        scale = len(non_null) / sample_size if sample_size else 1.0
        invalid_estimate = int(round(invalid_in_sample * scale))
        rows.append({
            "Column": col,
            "Detected Type": "datetime (parseable)",
            "Confidence": f"{rate * 100:.1f}%",
            "Invalid": invalid_estimate,
        })
    return pd.DataFrame(
        rows, columns=["Column", "Detected Type", "Confidence", "Invalid"]
    )


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
    """Interactive line chart with zoom, hover, range slider, and smoothing.

    Plots every row of ``prepared`` — no downsampling. ``mode="lines"`` (no
    per-point markers) keeps Plotly responsive on long industrial series
    without discarding any data.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prepared[date_col],
            y=prepared[value_col],
            mode="lines",
            name=value_col,
            line=dict(color="#636EFA", width=1.5),
            opacity=0.85,
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
        template=plotly_template(),
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
