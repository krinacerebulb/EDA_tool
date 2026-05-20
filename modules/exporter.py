"""Multi-format data export utilities.

================================================================================
WHY THIS MODULE EXISTS
================================================================================
After users have filtered, sanitized, and preprocessed their data they need
to take it OUT of the platform — into Excel for reporting, into Parquet for
downstream pipelines, into JSON for an API, into CSV for sharing.

A naive ``df.to_csv()`` misses critical industrial concerns:

    * **Unicode** — without a UTF-8 BOM, Excel on Windows mangles °C, μm,
      and any non-ASCII column names.
    * **Excel row/column limits** — XLSX is hard-capped at 1,048,576 rows
      and 16,384 columns. Large datasets need a pre-flight check or they
      fail mid-write.
    * **Datetime preservation** — JSON loses tz info, Excel coerces to
      strings unless dtype is preserved. Parquet handles both natively.
    * **Memory** — building the full byte buffer once is fine for < 100 MB
      frames, but we use ``BytesIO`` consistently so callers can swap in
      streaming writers if they need to scale further.
    * **Filename hygiene** — sanitize user-supplied filenames so weird
      characters don't break the OS's save dialog.

================================================================================
PUBLIC API
================================================================================
    export_csv(df)              -> (bytes, mime)
    export_excel(df)            -> (bytes, mime)
    export_json(df)             -> (bytes, mime)
    export_parquet(df)          -> (bytes, mime)

    export_dataframe(df, fmt)   -> (bytes, mime, ext)   # dispatcher

    make_export_filename(base, ext) -> "base_YYYYMMDD_HHMMSS.ext"
    render_export_ui(df, ...)       -> Streamlit panel (selector + download)

================================================================================
DESIGN NOTES
================================================================================
* Every encoder accepts a DataFrame and returns ``(bytes, mime_type)`` so
  the UI layer is format-agnostic.
* ``export_dataframe`` is the single dispatcher — add a new format by
  extending the ``EXPORT_FORMATS`` table and writing one new ``export_xxx``.
* Pre-checks (Excel row limit, etc.) live in the individual encoders so
  the UI layer doesn't need to know about them.
* Filenames are always timestamped to prevent accidental overwrites on the
  user's machine when they export the same dataset twice.
* The reusable ``render_export_ui`` Streamlit panel returns a small audit
  log dict; surface or store this if you need an export trail.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ============================================================================
# FORMAT REGISTRY
# ----------------------------------------------------------------------------
# Display label → file extension (without leading dot).
# Order is preserved when rendering the format dropdown.
# ============================================================================
EXPORT_FORMATS: dict[str, str] = {
    "CSV (.csv)":         "csv",
    "Excel (.xlsx)":      "xlsx",
    "JSON (.json)":       "json",
    "Parquet (.parquet)": "parquet",
}


# Hard limits for XLSX (Microsoft Excel specifications).
EXCEL_ROW_LIMIT = 1_048_575
EXCEL_COL_LIMIT = 16_383


# ============================================================================
# FILENAME
# ============================================================================
def make_export_filename(base: str = "dataset", ext: str = "csv") -> str:
    """Generate a timestamped, OS-safe filename.

    Example: ``make_export_filename("filtered", "parquet")``
        → ``"filtered_20260519_103045.parquet"``

    Why timestamps?
        Browsers download to a single folder by default; without a unique
        suffix each export overwrites the last. Windows users hit this
        constantly.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep only safe filename characters; replace everything else with "_".
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(base or "dataset")).strip("_")
    if not safe:
        safe = "dataset"
    ext = (ext or "csv").lstrip(".").lower()
    return f"{safe}_{ts}.{ext}"


# ============================================================================
# ENCODERS
# ----------------------------------------------------------------------------
# Each encoder is independent — call directly if you want raw bytes for a
# specific format. The UI uses ``export_dataframe`` as the dispatcher.
# ============================================================================
def export_csv(df: pd.DataFrame) -> tuple[bytes, str]:
    """Encode as UTF-8 CSV with a BOM prefix for Excel compatibility.

    The BOM (``\\xef\\xbb\\xbf``) tells Excel-on-Windows the file is UTF-8;
    without it, Excel guesses the encoding column-by-column and frequently
    mangles °C, μm, German umlauts, Cyrillic plant names, etc.
    """
    buf = BytesIO()
    buf.write(b"\xef\xbb\xbf")
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue(), "text/csv"


def export_excel(df: pd.DataFrame) -> tuple[bytes, str]:
    """Encode as XLSX. Raises ValueError if row/col limits are exceeded.

    XLSX is the ONLY format with hard size limits; we check before writing
    so the user gets a clear message instead of a corrupted file.
    """
    if len(df) > EXCEL_ROW_LIMIT:
        raise ValueError(
            f"Excel cannot hold {len(df):,} rows (limit: "
            f"{EXCEL_ROW_LIMIT:,}). Use CSV or Parquet for this dataset."
        )
    if df.shape[1] > EXCEL_COL_LIMIT:
        raise ValueError(
            f"Excel cannot hold {df.shape[1]:,} columns (limit: "
            f"{EXCEL_COL_LIMIT:,})."
        )
    buf = BytesIO()
    # openpyxl is bundled with the standard Streamlit deployment.
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return (
        buf.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def export_json(
    df: pd.DataFrame, *, orient: str = "records",
) -> tuple[bytes, str]:
    """Encode as JSON; datetime columns become ISO 8601 strings.

    ``orient="records"`` produces the most consumer-friendly shape:
        ``[{"col1": v, "col2": v}, ...]``

    ``default_handler=str`` ensures exotic Python objects (Decimal, UUID,
    np.int64, pd.Timestamp, custom classes) never cause ``TypeError`` —
    they're stringified instead.
    """
    json_str = df.to_json(
        orient=orient,
        date_format="iso",
        default_handler=str,
        force_ascii=False,  # preserve Unicode (°C, μm, etc.)
    )
    return json_str.encode("utf-8"), "application/json"


def export_parquet(df: pd.DataFrame) -> tuple[bytes, str]:
    """Encode as Parquet with snappy compression.

    Parquet is the recommended format for industrial-scale data because:
        * preserves every pandas dtype exactly (datetime, nullable Int,
          category)
        * snappy compression typically beats CSV by 5–10×
        * the file is column-oriented, so downstream tools can read
          subsets without parsing everything

    Requires ``pyarrow``, which is already a hard dependency of the app
    (Streamlit uses it for ``st.dataframe``).
    """
    buf = BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    return buf.getvalue(), "application/octet-stream"


# ============================================================================
# DISPATCHER
# ============================================================================
def export_dataframe(
    df: pd.DataFrame, *, fmt: str,
) -> tuple[bytes, str, str]:
    """Encode a DataFrame in the requested format.

    Returns ``(bytes, mime_type, extension)``.

    ``fmt`` accepts either the display label (``"CSV (.csv)"``) or the bare
    extension (``"csv"``, ``"xlsx"``, ``"json"``, ``"parquet"``).
    """
    if fmt in EXPORT_FORMATS:
        ext = EXPORT_FORMATS[fmt]
    else:
        ext = (fmt or "").lstrip(".").lower()
    if ext == "csv":
        b, mime = export_csv(df)
    elif ext == "xlsx":
        b, mime = export_excel(df)
    elif ext == "json":
        b, mime = export_json(df)
    elif ext == "parquet":
        b, mime = export_parquet(df)
    else:
        raise ValueError(f"Unsupported export format: {fmt!r}")
    return b, mime, ext


# ============================================================================
# REUSABLE STREAMLIT UI
# ============================================================================
def render_export_ui(
    df: pd.DataFrame,
    *,
    base_filename: str = "dataset",
    key_prefix: str = "export",
    label: str = "Download",
    show_size: bool = True,
) -> dict[str, Any] | None:
    """Reusable export panel: format selector → filename → download button.

    Use this anywhere the user might want to take data out:
        * the Overview tab (filtered dataset)
        * the Cleaning tab (sanitized dataset)
        * the Report tab (compact summary frame)

    Returns a small dict ``{format, filename, size_bytes, rows, cols}`` on
    successful render, or ``None`` if the DataFrame is empty / format is
    unsupported. Use the return value to log exports if you need an audit
    trail.
    """
    if df is None or df.empty:
        st.caption("Nothing to export — DataFrame is empty.")
        return None

    fc1, fc2 = st.columns([1, 2])
    with fc1:
        fmt_label = st.selectbox(
            "Format",
            options=list(EXPORT_FORMATS.keys()),
            key=f"{key_prefix}_fmt",
            help=(
                "**CSV** — universal, but loses dtypes.\n\n"
                "**Excel** — best for non-technical consumers (max 1 M rows).\n\n"
                "**JSON** — best for APIs / web apps.\n\n"
                "**Parquet** — best for analytics pipelines (preserves dtypes)."
            ),
        )
    ext = EXPORT_FORMATS[fmt_label]

    with fc2:
        custom_name = st.text_input(
            "Filename (without extension)",
            value=base_filename,
            key=f"{key_prefix}_name",
            help="A timestamp is automatically appended.",
        )

    final_filename = make_export_filename(custom_name or base_filename, ext)

    # Pre-flight Excel row limit so we surface a clear error before
    # spending CPU on a doomed encode.
    if ext == "xlsx" and len(df) > EXCEL_ROW_LIMIT:
        st.error(
            f"Excel format cannot hold {len(df):,} rows "
            f"(limit: {EXCEL_ROW_LIMIT:,}). Use CSV or Parquet for this "
            "dataset."
        )
        return None

    # Encode now (Streamlit's download_button needs the bytes upfront).
    try:
        data, mime, _ = export_dataframe(df, fmt=ext)
    except Exception as exc:
        logger.exception("export failed")
        st.error(f"Export failed: {exc}")
        return None

    size_bytes = len(data)
    st.download_button(
        label=f"⬇  {label} ({fmt_label})",
        data=data,
        file_name=final_filename,
        mime=mime,
        key=f"{key_prefix}_dl",
        use_container_width=True,
    )
    if show_size:
        st.caption(
            f"Will save as `{final_filename}` · "
            f"{_human_size(size_bytes)} · "
            f"{len(df):,} rows × {df.shape[1]} cols"
        )

    return {
        "format": ext,
        "filename": final_filename,
        "size_bytes": size_bytes,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
    }


# ============================================================================
# INTERNAL
# ============================================================================
def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"
