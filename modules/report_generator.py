"""Build a single self-contained HTML report from the EDA results.

The report is intentionally minimal: an executive-style summary aimed at
managers / clients / ops teams. It carries the dataset overview, column
health, time range, basic statistics, and a single correlation heatmap.
Per-column histograms / boxplots / bar charts are deliberately omitted —
they belong in the interactive UI, not in a printed report.
"""

from __future__ import annotations

import base64
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from modules import visualization
from utils.helpers import fig_to_base64, human_bytes, split_columns


_CSS = """
  :root {
    --ink: #1F2A37;
    --ink-soft: #4B5563;
    --muted: #6B7280;
    --line: #E5E7EB;
    --bg-soft: #F9FAFB;
    --accent: #2C3E50;
    --good: #047857;
    --warn: #B45309;
    --bad:  #B91C1C;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: var(--ink); background: #fff;
    margin: 0; padding: 40px 56px 80px;
    max-width: 1180px; margin-left: auto; margin-right: auto;
    line-height: 1.5;
  }
  header.report-header {
    display: flex; align-items: center; justify-content: space-between;
    gap: 24px; padding-bottom: 20px; border-bottom: 2px solid var(--accent);
  }
  header .brand { display: flex; align-items: center; gap: 16px; }
  header .brand img { max-height: 56px; max-width: 180px; }
  header .brand .org { font-size: 13px; color: var(--muted);
                       text-transform: uppercase; letter-spacing: 0.08em; }
  header h1 { margin: 0; font-size: 22px; color: var(--accent); font-weight: 600; }
  header .meta { text-align: right; font-size: 12px; color: var(--muted);
                 line-height: 1.6; }
  header .meta b { color: var(--ink); font-weight: 600; }
  section { margin-top: 36px; }
  section h2 {
    font-size: 15px; color: var(--accent); text-transform: uppercase;
    letter-spacing: 0.06em; margin: 0 0 14px; padding-bottom: 6px;
    border-bottom: 1px solid var(--line); font-weight: 700;
  }
  p.lede { color: var(--ink-soft); font-size: 14px; margin: 0 0 12px; }
  .cards { display: grid;
           grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 12px; margin: 14px 0 4px; }
  .card { background: var(--bg-soft); border: 1px solid var(--line);
          border-left: 4px solid var(--line);
          border-radius: 8px; padding: 14px 16px; }
  .card.ok   { border-left-color: var(--good); }
  .card.warn { border-left-color: var(--warn);
               background: linear-gradient(0deg, #FEF3C7 0%, var(--bg-soft) 100%); }
  .card.bad  { border-left-color: var(--bad);
               background: linear-gradient(0deg, #FEE2E2 0%, var(--bg-soft) 100%); }
  .card .label { font-size: 11px; color: var(--muted);
                 text-transform: uppercase; letter-spacing: 0.06em; }
  .card .value { font-size: 20px; font-weight: 600; margin-top: 4px;
                 color: var(--ink); }
  .card.warn .value { color: var(--warn); }
  .card.bad  .value { color: var(--bad); }
  .card .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 11px; font-weight: 600;
           background: var(--bg-soft); color: var(--ink-soft); }
  .badge.strong   { background: #DCFCE7; color: var(--good); }
  .badge.moderate { background: #FEF3C7; color: var(--warn); }
  .badge.weak     { background: #F3F4F6; color: var(--muted); }
  table { border-collapse: collapse; width: 100%; font-size: 13px;
          margin: 6px 0 0; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--line); }
  th { background: var(--bg-soft); font-weight: 600; color: var(--ink-soft);
       font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
  tr:last-child td { border-bottom: 0; }
  td.numeric { text-align: right; font-variant-numeric: tabular-nums; }
  .status.ok    { color: var(--good); font-weight: 600; }
  .status.warn  { color: var(--warn); font-weight: 600; }
  .status.bad   { color: var(--bad);  font-weight: 600; }
  .chart { margin: 8px 0; }
  .chart img { width: 100%; border: 1px solid var(--line); border-radius: 6px; }
  footer { margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--line);
           font-size: 11px; color: var(--muted); text-align: center; }
  @media print {
    body { padding: 24px; }
    .cards { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
    section { page-break-inside: avoid; }
  }
"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{report_title} — {filename}</title>
<style>{css}</style>
</head>
<body>
{header_html}

<section>
  <h2>Executive Summary</h2>
  <p class="lede">A snapshot of the dataset's shape, completeness, and coverage.</p>
  {summary_cards_html}
</section>

<section>
  <h2>Dataset Profile</h2>
  {profile_html}
</section>

{time_range_section}

<section>
  <h2>Column Health</h2>
  <p class="lede">Every column with its data type, completeness, and uniqueness — sorted by missing % so problem columns surface first.</p>
  {column_health_html}
</section>

<section>
  <h2>Numeric Summary</h2>
  <p class="lede">Mean, spread, and range for each numeric measure.</p>
  {numeric_html}
</section>

{correlation_section}

<footer>
  Generated by Auto EDA Platform · {timestamp}
</footer>
</body>
</html>
"""


# ----- helpers --------------------------------------------------------------


def _fmt_cell(v, precision: int = 2) -> str:
    """Pretty-print a single value for a report table cell.

    - integers get thousand separators
    - floats render with ``precision`` decimals (default 2 for back-compat)
    - NaN / NaT show as an em dash so empty cells read cleanly
    """
    if pd.isna(v):
        return "—"
    if isinstance(v, bool):  # bool is a subclass of int, must check first
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{int(precision)}f}"
    return escape(str(v))


def _df_to_html(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    precision: int = 2,
) -> str:
    """Render a DataFrame as an HTML table.

    Numeric-typed cells (and any column listed in ``numeric_cols``) get the
    ``numeric`` CSS class for right-alignment and tabular figures, plus the
    two-decimal float formatting from :func:`_fmt_cell`.
    """
    if df is None or df.empty:
        return "<p style='color:#6B7280;font-size:13px;'>No data.</p>"
    explicit_numeric = set(numeric_cols or [])
    # Auto-detect numeric columns so callers don't have to keep the list in sync.
    auto_numeric = {
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    }
    is_numeric = explicit_numeric | auto_numeric

    cols_html = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows_html_parts: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in df.columns:
            v = row[col]
            cls = " class='numeric'" if col in is_numeric else ""
            cells.append(f"<td{cls}>{_fmt_cell(v, precision=precision)}</td>")
        rows_html_parts.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{cols_html}</tr></thead><tbody>{''.join(rows_html_parts)}</tbody></table>"


def _card(label: str, value: str, sub: str = "", status: str = "") -> str:
    """Render a summary card. ``status`` ∈ {"", "ok", "warn", "bad"} adds a
    colored accent strip + tinted background so problem metrics stand out."""
    cls = f"card {status}".strip()
    sub_html = f"<div class='sub'>{escape(sub)}</div>" if sub else ""
    return (
        f"<div class='{cls}'><div class='label'>{escape(label)}</div>"
        f"<div class='value'>{escape(value)}</div>{sub_html}</div>"
    )


def _logo_to_data_uri(path: str | Path) -> str | None:
    """Inline an image as a base64 data URI so the HTML stays self-contained."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        ext = p.suffix.lower().lstrip(".") or "png"
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "svg": "svg+xml"}.get(ext, "png")
        with open(p, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
        return f"data:image/{mime};base64,{data}"
    except Exception:
        return None


def _format_duration(span: pd.Timedelta) -> str:
    """Render a Timedelta as `Xd Yh Zm` — never raw seconds."""
    if pd.isna(span) or span == pd.Timedelta(0):
        return "0"
    total_seconds = int(span.total_seconds())
    days, rem = divmod(total_seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes and not days:
        parts.append(f"{minutes} min")
    return " ".join(parts) or "< 1 min"


def _detect_primary_datetime(df: pd.DataFrame) -> str | None:
    """Pick the datetime column with the widest span — the report's 'main' axis."""
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns
    if dt_cols.empty:
        return None
    best_col, best_span = None, pd.Timedelta(0)
    for col in dt_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        span = s.max() - s.min()
        if span > best_span:
            best_col, best_span = col, span
    return best_col


def _time_range_section_html(df: pd.DataFrame) -> str:
    col = _detect_primary_datetime(df)
    if not col:
        return ""
    s = df[col].dropna()
    if s.empty:
        return ""
    start, end = s.min(), s.max()
    duration = _format_duration(end - start)
    cards = "".join([
        _card("Start", start.strftime("%Y-%m-%d %H:%M:%S")),
        _card("End",   end.strftime("%Y-%m-%d %H:%M:%S")),
        _card("Duration", duration),
        _card("Datetime column", col, f"{len(s):,} non-null timestamps"),
    ])
    return (
        "<section><h2>Time Range</h2>"
        f"<p class='lede'>Coverage of the primary datetime column.</p>"
        f"<div class='cards'>{cards}</div></section>"
    )


def _column_health_html(df: pd.DataFrame, precision: int = 2) -> str:
    """One-row-per-column table: dtype, completeness, uniqueness, status."""
    if df.empty:
        return _df_to_html(pd.DataFrame())
    n = len(df)
    rows: list[dict] = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        missing_pct = (missing / n * 100) if n else 0.0
        unique = int(s.nunique(dropna=True))

        if missing_pct >= 50:
            status = "<span class='status bad'>High missing</span>"
        elif missing_pct >= 10:
            status = "<span class='status warn'>Some missing</span>"
        elif unique == 1 and n > 1:
            status = "<span class='status warn'>Constant</span>"
        elif unique == n and n > 0:
            status = "<span class='status ok'>Unique key</span>"
        else:
            status = "<span class='status ok'>OK</span>"

        rows.append({
            "Column": col,
            "Type": str(s.dtype),
            "Missing": f"{missing_pct:.{int(precision)}f}%",
            "Unique": f"{unique:,}",
            "Status": status,
        })
    health_df = pd.DataFrame(rows)

    # Sort by missing % desc — problem columns first.
    sort_key = (
        health_df["Missing"].str.rstrip("%").astype(float)
    )
    health_df = health_df.iloc[sort_key.argsort()[::-1]].reset_index(drop=True)

    cols_html = "".join(f"<th>{escape(c)}</th>" for c in health_df.columns)
    body_parts: list[str] = []
    for _, row in health_df.iterrows():
        body_parts.append(
            "<tr>"
            f"<td>{escape(str(row['Column']))}</td>"
            f"<td>{escape(str(row['Type']))}</td>"
            f"<td class='numeric'>{escape(str(row['Missing']))}</td>"
            f"<td class='numeric'>{escape(str(row['Unique']))}</td>"
            f"<td>{row['Status']}</td>"
            "</tr>"
        )
    return (
        f"<table><thead><tr>{cols_html}</tr></thead>"
        f"<tbody>{''.join(body_parts)}</tbody></table>"
    )


def _summary_cards_html(
    df: pd.DataFrame,
    overview: dict,
    dup_count: int,
    precision: int = 2,
) -> str:
    """Four headline metrics with subtle health badges.

    Thresholds:
      - missing %  ≥ 30 → bad, ≥ 10 → warn, else ok
      - duplicate% ≥ 20 → bad, ≥  5 → warn, else ok
      - memory     ≥ 200MB → warn (informational; never bad)
    """
    n_rows = overview["rows"]
    n_cols = overview["columns"]
    total_missing = overview["total_missing"]
    cell_count = n_rows * n_cols if n_rows and n_cols else 0
    missing_pct = (total_missing / cell_count * 100) if cell_count else 0.0
    dup_pct = (dup_count / n_rows * 100) if n_rows else 0.0
    mem_bytes = overview.get("memory_bytes", 0)
    mem_mb = mem_bytes / (1024 ** 2) if mem_bytes else 0.0

    # Only highlight problem values — a neutral card means "nothing to worry about".
    missing_status = "bad" if missing_pct >= 30 else "warn" if missing_pct >= 10 else ""
    dup_status = "bad" if dup_pct >= 20 else "warn" if dup_pct >= 5 else ""
    mem_status = "warn" if mem_mb >= 200 else ""

    cards = [
        _card("Rows", f"{n_rows:,}"),
        _card("Columns", f"{n_cols:,}"),
        _card(
            "Missing values",
            f"{total_missing:,}",
            f"{missing_pct:.{int(precision)}f}% of cells",
            status=missing_status,
        ),
        _card(
            "Duplicate rows",
            f"{dup_count:,}",
            f"{dup_pct:.{int(precision)}f}% of rows",
            status=dup_status,
        ),
        _card("Memory", human_bytes(mem_bytes), status=mem_status),
    ]
    return f"<div class='cards'>{''.join(cards)}</div>"


def _profile_html(df: pd.DataFrame, overview: dict) -> str:
    numeric_cols, cat_cols, dt_cols = split_columns(df)
    mem = human_bytes(overview["memory_bytes"])
    rows = [
        ("Numeric columns",     f"{len(numeric_cols):,}"),
        ("Categorical columns", f"{len(cat_cols):,}"),
        ("Datetime columns",    f"{len(dt_cols):,}"),
        ("Memory in memory",    mem),
    ]
    return _df_to_html(
        pd.DataFrame(rows, columns=["Metric", "Value"]),
        numeric_cols=["Value"],
    )


def _top_correlations_html(df: pd.DataFrame, top_n: int = 5, precision: int = 2) -> str:
    """Highlight the strongest numeric relationships as a small ranked table.

    Adds business-readable interpretation (Strong / Moderate / Weak +
    positive / negative) so non-technical readers can scan it without
    knowing what a correlation coefficient is.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return ""
    try:
        corr = numeric.corr()
    except Exception:
        return ""
    cols = list(corr.columns)
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r):
                pairs.append((cols[i], cols[j], float(r)))
    if not pairs:
        return ""
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    pairs = pairs[:top_n]

    rows_html: list[str] = []
    for a, b, r in pairs:
        mag = abs(r)
        if mag >= 0.7:
            badge_cls, label = "strong", "Strong"
        elif mag >= 0.4:
            badge_cls, label = "moderate", "Moderate"
        else:
            badge_cls, label = "weak", "Weak"
        direction = "positive" if r > 0 else "negative" if r < 0 else "neutral"
        rows_html.append(
            "<tr>"
            f"<td>{escape(a)} ↔ {escape(b)}</td>"
            f"<td class='numeric'>{r:+.{int(precision)}f}</td>"
            f"<td><span class='badge {badge_cls}'>{label}</span> {direction}</td>"
            "</tr>"
        )
    return (
        "<h3 style='font-size:13px;margin:18px 0 6px;color:var(--ink-soft);"
        "text-transform:uppercase;letter-spacing:0.06em;'>Strongest Relationships</h3>"
        "<table><thead><tr>"
        "<th>Variables</th><th>Correlation</th><th>Interpretation</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _correlation_section_html(df: pd.DataFrame, precision: int = 2) -> str:
    fig = visualization.correlation_heatmap(df)
    if fig is None:
        return ""
    img = fig_to_base64(fig)
    top_table = _top_correlations_html(df, top_n=5, precision=precision)
    return (
        "<section><h2>Correlation Heatmap</h2>"
        "<p class='lede'>How strongly numeric measures move together. "
        "Values near +1 / -1 indicate strong relationships; near 0 means little linear association.</p>"
        f"<div class='chart'><img src='data:image/png;base64,{img}' alt='Correlation heatmap' /></div>"
        f"{top_table}"
        "</section>"
    )


def _header_html(
    report_title: str,
    filename: str,
    timestamp: str,
    company_name: str | None,
    logo_path: str | Path | None,
) -> str:
    logo_uri = _logo_to_data_uri(logo_path) if logo_path else None
    logo_html = f"<img src='{logo_uri}' alt='logo' />" if logo_uri else ""
    org_html = f"<div class='org'>{escape(company_name)}</div>" if company_name else ""
    return (
        "<header class='report-header'>"
        "<div class='brand'>"
        f"{logo_html}"
        f"<div>{org_html}<h1>{escape(report_title)}</h1></div>"
        "</div>"
        "<div class='meta'>"
        f"<div>Dataset: <b>{escape(filename)}</b></div>"
        f"<div>Generated: {escape(timestamp)}</div>"
        "</div>"
        "</header>"
    )


# ----- public API -----------------------------------------------------------


def build_html_report(
    df: pd.DataFrame,
    filename: str,
    overview: dict,
    dtype_info: pd.DataFrame,  # kept for API compatibility; not rendered
    missing: pd.DataFrame,      # kept for API compatibility; not rendered
    numeric_stats: pd.DataFrame,
    *,
    dup_count: int = 0,
    company_name: str | None = None,
    report_title: str = "Auto EDA Report",
    logo_path: str | Path | None = None,
    precision: int = 2,
) -> str:
    """Assemble a clean executive-style HTML report.

    ``dtype_info`` and ``missing`` are accepted for backwards compatibility
    with the previous signature; the new layout derives them on the fly from
    ``df`` (column health table). The single chart is the correlation
    heatmap. No insights, no per-column distribution plots.
    """
    # _ = dtype_info, missing  # intentionally unused — kept for caller compat

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return _HTML_TEMPLATE.format(
        css=_CSS,
        report_title=escape(report_title),
        filename=escape(filename),
        timestamp=escape(timestamp),
        header_html=_header_html(
            report_title=report_title,
            filename=filename,
            timestamp=timestamp,
            company_name=company_name,
            logo_path=logo_path,
        ),
        summary_cards_html=_summary_cards_html(
            df, overview, dup_count, precision=precision,
        ),
        profile_html=_profile_html(df, overview),
        time_range_section=_time_range_section_html(df),
        column_health_html=_column_health_html(df, precision=precision),
        numeric_html=_df_to_html(
            numeric_stats if numeric_stats is not None else pd.DataFrame(),
            numeric_cols=(
                [c for c in numeric_stats.columns if c != "Column"]
                if numeric_stats is not None and not numeric_stats.empty
                else []
            ),
            precision=precision,
        ),
        correlation_section=_correlation_section_html(df, precision=precision),
    )
