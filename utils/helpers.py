"""Small helper utilities shared across modules."""

import base64
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string (for HTML report)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def split_columns(df: pd.DataFrame):
    """Return (numeric_cols, categorical_cols, datetime_cols) based on dtypes.

    Uses per-column dtype probes rather than ``select_dtypes`` because the
    latter copies the matching block — wasteful when all the caller needs
    are column names, and a real OOM risk on wide industrial datasets.
    """
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(c)
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric_cols.append(c)
        elif (
            pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(s)
        ):
            categorical_cols.append(c)
    return numeric_cols, categorical_cols, datetime_cols


def human_bytes(num: int) -> str:
    """Format a byte count into a human-readable string (always 2 decimals)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TB"


def plotly_template() -> str:
    """Return the active Plotly template. Light theme only."""
    return "plotly_white"


def horizontal_legend(below: bool = False) -> dict:
    """Plotly ``legend=`` config: horizontal and centered.

    A horizontal, centered legend keeps every series label visible when a
    chart has many traces (multi-sensor time-series, grouped outliers). The
    default vertical right-side legend and a left-anchored top legend both
    clip with long names or high series counts.

    Parameters
    ----------
    below
        ``True``  → place the legend *below* the plot (``y=-0.2``). Use this
                    for charts with NO range-slider; leave bottom margin room
                    (e.g. ``margin=dict(b=80)``) so it isn't clipped.
        ``False`` → place it just *above* the plot (``y=1.02``). Safe for
                    charts that already have a range-slider / range-selector
                    occupying the area below the x-axis, where a below-legend
                    would collide with them.
    """
    if below:
        return dict(
            orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top",
        )
    return dict(
        orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom",
    )


def side_legend(font_size: int = 10) -> dict:
    """Plotly ``legend=`` config: vertical, anchored to the right of the plot.

    Each series gets its own line, so nothing clips horizontally no matter
    how many traces there are — the best option for many-series or long-name
    charts (multi-sensor time-series). Sitting *outside* the plot (``x=1.02``)
    keeps it clear of the data and of a bottom range-slider.

    Pair with a wide right margin so the legend isn't cut off — e.g.
    ``margin=dict(r=200)``. Avoid on charts with a secondary *right* y-axis,
    where the legend and the axis title would overlap; use
    :func:`horizontal_legend` (top) there instead.
    """
    return dict(
        orientation="v",
        x=1.02,
        y=1,
        xanchor="left",
        yanchor="top",
        font=dict(size=font_size),
    )


def short_label(text, max_len: int = 30) -> str:
    """Shorten a long label, keeping its END with a leading ellipsis.

    Industrial sensor tags (e.g.
    ``HIL_ALU_HKD_SMLTR_85_..._POT_002_BATH_TEMP``) share a long common prefix
    and differ only in the last few segments, so the *tail* is what
    distinguishes them. Keeping the tail (not the head) means two near-
    identical tags still read as different in a legend / title. Full text
    stays available in hover tooltips.
    """
    s = str(text)
    if len(s) <= max_len:
        return s
    return "…" + s[-(max_len - 1):]


def _legend_below(fig, font_size: int):
    """Place a horizontal, centered legend BELOW the plot (wraps onto rows).

    Used for charts whose top edge is occupied by range-selector buttons
    (``1w 1m 3m 6m 1y All``) — a top legend would sit under those buttons.
    The horizontal orientation wraps across multiple rows, so even many
    series stay fully visible. The bottom margin reserves room for it beneath
    the x-axis date labels.
    """
    fig.update_layout(
        legend=dict(
            orientation="h", x=0.5, xanchor="center",
            y=-0.30, yanchor="top", font=dict(size=font_size),
        ),
        margin_b=100,
    )
    return fig


def apply_legend(
    fig, *, secondary_axis: bool = False, below: bool = False,
    font_size: int = 10,
):
    """Attach a fully-visible, correctly-sized legend to a Plotly figure.

    This is the single entry point every Plotly chart should use so legends
    look consistent and never clip. It inspects the figure's own traces and
    adapts placement + margin to the chart:

    * **Few / no legend traces** (0–1 series): Plotly hides the legend for a
      single trace anyway, so nothing is changed — no wasted margin.
    * **``below=True``** (charts with range-selector buttons / a range-slider,
      i.e. time-series): a horizontal legend *below* the plot, wrapping onto
      rows. This is the placement that clears both the ``1w 1m … All`` buttons
      at the top and the range-slider at the bottom. Many series stay visible
      because the horizontal legend wraps.
    * **Several series, single y-axis** (the common non-time-series case): a
      *vertical* legend just right of the plot, one entry per line. The right
      margin is sized from the longest label so long sensor names show in full
      without wasting space on short ones.
    * **Many series** (> ~18) without ``below``: falls back to the horizontal
      below-plot legend — a tall vertical list would run off the figure.
    * **Secondary (right) y-axis present** (and not ``below``): a right-side
      legend would overlap the axis title, so a horizontal legend goes *above*
      the plot instead.

    Call this AFTER all traces have been added.
    """
    names = [
        str(t.name)
        for t in fig.data
        if getattr(t, "name", None)
        and getattr(t, "showlegend", None) is not False
    ]
    n = len(names)

    # Single (or zero) series — Plotly won't show a legend; leave layout be.
    if n < 2:
        return fig

    # Time-series-style charts: the top is taken by range buttons and the
    # bottom by a range-slider, so the legend goes below (clear of both).
    # Also the fallback for very many series, where a vertical list overflows
    # (that fallback has no range-slider, so it needs less clearance).
    if below or n > 18:
        return _legend_below(fig, font_size)

    # A right-side legend collides with a right y-axis → horizontal on top.
    if secondary_axis:
        fig.update_layout(
            legend=dict(
                orientation="h", x=0.5, xanchor="center",
                y=1.02, yanchor="bottom", font=dict(size=font_size),
            ),
        )
        return fig

    # Common case: vertical legend to the right. Size the right margin from
    # the longest label (≈7 px/char at size 10) and clamp to a sane range so
    # it neither clips long names nor wastes space on short ones.
    longest = max(len(s) for s in names)
    right = int(min(420, max(140, longest * 7 + 60)))
    fig.update_layout(legend=side_legend(font_size), margin_r=right)
    return fig


_DEFAULT_DECIMAL_PRECISION = 2


def get_decimal_precision(default: int = _DEFAULT_DECIMAL_PRECISION) -> int:
    """Return the user's display precision (sidebar setting).

    Falls back to ``default`` when the setting hasn't been initialised yet
    (first paint, headless tests, etc.). The value only affects how
    numbers are rendered — it never touches stored data.
    """
    try:
        import streamlit as st
        return int(st.session_state.get("decimal_precision", default))
    except Exception:
        return int(default)


def fmt_num(value, precision: int | None = None, na_rep: str = "—") -> str:
    """Format a single number for display using the active precision.

    Integers are printed with thousands separators and no decimals; floats
    use the chosen precision. ``NaN`` / ``None`` map to ``na_rep``.
    """
    if value is None:
        return na_rep
    try:
        if isinstance(value, float) and pd.isna(value):
            return na_rep
    except Exception:
        pass
    p = get_decimal_precision() if precision is None else int(precision)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(f):
        return na_rep
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    return f"{f:,.{p}f}"


def safe_dataframe(df, **kwargs):
    """PyArrow-safe ``st.dataframe`` wrapper.

    Why this exists: ``st.dataframe`` serialises through PyArrow. If the
    frame still has a mixed-type object column (e.g. an on-the-fly summary
    frame built outside the loader's sanitization path), PyArrow raises
    ``ArrowInvalid`` and the whole Streamlit page errors. This wrapper runs
    the input through ``make_arrow_safe`` first, so rendering is guaranteed
    to succeed.

    Use ``safe_dataframe(df, width="stretch")`` anywhere you would have
    called ``st.dataframe(df, width="stretch")`` — especially for frames
    built from heterogeneous sources (preview tables, conversion summaries,
    user-built previews).
    """
    import streamlit as st
    try:
        from modules.data_sanitization import make_arrow_safe
        safe = make_arrow_safe(df)
        return st.dataframe(safe, **kwargs)
    except Exception as exc:
        # Absolute last resort — full stringification.
        try:
            from modules.data_sanitization import force_stringify
            return st.dataframe(force_stringify(df), **kwargs)
        except Exception:
            st.warning(f"Could not render DataFrame: {exc}")
            return None


def compress_strings_to_category(
    df: pd.DataFrame,
    max_unique_ratio: float = 0.5,
    min_rows: int = 1000,
) -> pd.DataFrame:
    """Convert low-cardinality object columns to ``category`` in-place-by-return.

    Big win for memory on industrial datasets: a 500k-row column with 100
    unique strings drops from ~25 MB to a few hundred KB. Downstream code
    that uses ``select_dtypes(include=["object", "category", "bool"])`` still
    sees the column, so behaviour is preserved.

    Skipped automatically for small DataFrames (below ``min_rows``) where the
    overhead wouldn't pay off.
    """
    if len(df) < min_rows:
        return df
    object_cols = df.select_dtypes(include=["object"]).columns
    if object_cols.empty:
        return df
    for col in object_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique == 0:
            continue
        if nunique / len(df) <= max_unique_ratio:
            df[col] = df[col].astype("category")
    return df
