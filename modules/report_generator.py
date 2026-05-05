"""Build a single self-contained HTML report from the EDA results."""

from datetime import datetime
from html import escape

import pandas as pd

from modules import visualization
from utils.helpers import fig_to_base64, human_bytes, split_columns


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Auto EDA Report - {filename}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          margin: 30px; color: #222; max-width: 1100px; }}
  h1 {{ color: #2C3E50; border-bottom: 2px solid #2C3E50; padding-bottom: 6px; }}
  h2 {{ color: #34495E; margin-top: 32px; }}
  table {{ border-collapse: collapse; margin: 10px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  th {{ background: #f4f6f8; }}
  .meta {{ color: #555; font-size: 13px; }}
  .insight {{ background: #eef6ff; border-left: 4px solid #4C72B0;
              padding: 8px 12px; margin: 6px 0; border-radius: 4px; }}
  .chart {{ margin: 14px 0; }}
  .chart img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>Auto EDA Report</h1>
  <p class="meta">File: <b>{filename}</b> &nbsp;|&nbsp; Generated: {timestamp}</p>

  <h2>1. Dataset Overview</h2>
  {overview_html}

  <h2>2. Column Types</h2>
  {dtypes_html}

  <h2>3. Missing Values</h2>
  {missing_html}

  <h2>4. Numeric Summary</h2>
  {numeric_html}

  <h2>5. Key Insights</h2>
  {insights_html}

  <h2>6. Visualizations</h2>
  {charts_html}
</body>
</html>
"""


def _df_to_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p><i>No data.</i></p>"
    return df.to_html(index=False, border=0, classes="data-table", escape=True)


def _insights_to_html(insights: list[str]) -> str:
    if not insights:
        return "<p><i>No insights generated.</i></p>"
    items = "".join(f'<div class="insight">{escape(line)}</div>' for line in insights)
    return items


def _build_charts_html(df: pd.DataFrame, max_numeric: int = 3, max_categorical: int = 3) -> str:
    """Embed a handful of the most useful charts as base64 images."""
    chunks: list[str] = []
    numeric_cols, cat_cols, _ = split_columns(df)

    # Correlation heatmap first — most informative summary.
    fig = visualization.correlation_heatmap(df)
    if fig is not None:
        chunks.append(
            f'<div class="chart"><h3>Correlation Heatmap</h3>'
            f'<img src="data:image/png;base64,{fig_to_base64(fig)}" /></div>'
        )

    for col in numeric_cols[:max_numeric]:
        fig = visualization.histogram(df, col)
        chunks.append(
            f'<div class="chart"><h3>Distribution: {escape(col)}</h3>'
            f'<img src="data:image/png;base64,{fig_to_base64(fig)}" /></div>'
        )
        fig = visualization.boxplot(df, col)
        chunks.append(
            f'<div class="chart"><h3>Boxplot: {escape(col)}</h3>'
            f'<img src="data:image/png;base64,{fig_to_base64(fig)}" /></div>'
        )

    for col in cat_cols[:max_categorical]:
        fig = visualization.bar_chart(df, col)
        chunks.append(
            f'<div class="chart"><h3>Top values: {escape(col)}</h3>'
            f'<img src="data:image/png;base64,{fig_to_base64(fig)}" /></div>'
        )

    return "\n".join(chunks) if chunks else "<p><i>No charts available.</i></p>"


def build_html_report(
    df: pd.DataFrame,
    filename: str,
    overview: dict,
    dtype_info: pd.DataFrame,
    missing: pd.DataFrame,
    numeric_stats: pd.DataFrame,
    insights: list[str],
) -> str:
    """Assemble all pieces into a single HTML string ready to be downloaded."""
    overview_rows = [
        ("Rows", overview["rows"]),
        ("Columns", overview["columns"]),
        ("Memory usage", human_bytes(overview["memory_bytes"])),
        ("Total missing values", overview["total_missing"]),
    ]
    overview_html = pd.DataFrame(overview_rows, columns=["Metric", "Value"]).to_html(
        index=False, border=0, escape=True
    )

    return _HTML_TEMPLATE.format(
        filename=escape(filename),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        overview_html=overview_html,
        dtypes_html=_df_to_html(dtype_info),
        missing_html=_df_to_html(missing),
        numeric_html=_df_to_html(numeric_stats),
        insights_html=_insights_to_html(insights),
        charts_html=_build_charts_html(df),
    )
