"""Load user-uploaded datasets (CSV / Excel / JSON / Parquet) into pandas.

Performance strategy:
- Bytes-based, ``@st.cache_data``-memoised reader so the same uploaded file
  is parsed only once per session.
- **Polars** is tried first for CSV and Parquet (5–10× faster than pandas at
  CSV parsing, lower peak RAM). The parsed result is converted to pandas so
  the rest of the codebase is unaffected. Falls back to pandas if Polars is
  unavailable or fails on a given file.
- Excel / JSON stay on pandas (Polars' support for these is limited).
- After loading, low-cardinality object columns are compressed to ``category``
  — a ~50× memory reduction on string-heavy industrial datasets.
"""

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from .data_sanitization import preprocess_dynamic_dataset


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Polars-first CSV reader with a pandas fallback."""
    try:
        import polars as pl
        df_pl = pl.read_csv(BytesIO(file_bytes), try_parse_dates=False)
        return df_pl.to_pandas()
    except Exception:
        return pd.read_csv(BytesIO(file_bytes))


def _read_parquet(file_bytes: bytes) -> pd.DataFrame:
    """Polars-first Parquet reader with a pandas/pyarrow fallback."""
    try:
        import polars as pl
        df_pl = pl.read_parquet(BytesIO(file_bytes))
        return df_pl.to_pandas()
    except Exception:
        return pd.read_parquet(BytesIO(file_bytes))


# ``show_spinner=False`` because callers display their own, more
# informative spinner (``multi_file_loader.load_multiple_files`` / the
# outer block in ``app.py``). Leaving the default cache-spinner on would
# stack a second "Loading dataset…" overlay on top of the caller's
# spinner — visible to the user as the UI flashing twice on upload.
@st.cache_data(show_spinner=False)
def _read_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame. Cached on (bytes_hash, filename).

    Pipeline:
      1. Parse with the format-appropriate engine (Polars-first for CSV /
         Parquet, pandas for Excel / JSON).
      2. Run ``preprocess_dynamic_dataset`` — this is the production-grade
         sanitization layer that:
           * replaces industrial dirty tokens ("No Data", "Bad", "Sensor
             Fail", "-", Excel errors, ...) with NaN,
           * promotes majority-numeric / majority-date string columns to
             their proper dtype,
           * compresses low-cardinality strings to ``category``,
           * forces the result to be PyArrow-renderable.

    The sanitization report is attached to ``df.attrs["sanitization_report"]``
    so the UI can surface a summary banner without re-running the pipeline.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Please upload one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if suffix == ".csv":
            df = _read_csv(file_bytes)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(BytesIO(file_bytes))
        elif suffix == ".json":
            df = pd.read_json(BytesIO(file_bytes))
        elif suffix in {".parquet", ".pq"}:
            df = _read_parquet(file_bytes)
        else:
            raise ValueError(f"Unhandled file type: {suffix}")
    except ImportError as exc:
        raise ValueError(
            f"Reading '{suffix}' requires an additional engine. "
            f"Install with: pip install pyarrow  ({exc})"
        ) from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse file '{filename}': {exc}") from exc

    # Sanitize. ``preprocess_dynamic_dataset`` is guaranteed not to raise —
    # on internal failure it returns the input unchanged plus an error in
    # the report. We attach the report via ``df.attrs`` so the cached return
    # value is a plain DataFrame (Streamlit cache hashes the *content*; the
    # attrs dict rides along through pickle).
    clean_df, report = preprocess_dynamic_dataset(df)
    try:
        clean_df.attrs["sanitization_report"] = report
    except Exception:
        # Some pandas builds restrict ``attrs`` assignment; not critical.
        pass
    return clean_df


def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Read a file-like object from Streamlit's uploader into a DataFrame.

    Supported formats: CSV, Excel (.xlsx / .xls), JSON, Parquet (.parquet / .pq).

    Raises ValueError if the extension is unsupported or the file can't be
    parsed. Implementation delegates to a bytes-based cached reader so the
    same uploaded file isn't re-parsed across reruns.
    """
    if uploaded_file is None:
        raise ValueError("No file was provided.")
    return _read_bytes(uploaded_file.getvalue(), uploaded_file.name)


def apply_header_overrides(
    df: pd.DataFrame,
    drop_first_n: int = 0,
    promote_row_to_header: bool = False,
) -> pd.DataFrame:
    """Drop leading rows and optionally promote the next row to the header.

    Industrial sensor exports often start with metadata, machine info, or
    blank rows before the real header. This helper lets the UI fix those
    files without re-uploading.

    Order is: drop ``drop_first_n`` rows from the top, then (if requested)
    take what is now the first row and use its values as the new column
    names — that row is consumed in the process. Numeric-looking cells are
    re-coerced via ``pd.to_numeric`` since the previously-parsed dtypes
    were based on the wrong header.
    """
    drop_first_n = max(0, int(drop_first_n))
    if drop_first_n == 0 and not promote_row_to_header:
        return df

    out = df
    # Preserve the upstream sanitization metadata so downstream banners
    # still describe what was cleaned on load.
    original_attrs = dict(getattr(df, "attrs", {}) or {})
    if drop_first_n > 0:
        out = out.iloc[drop_first_n:].reset_index(drop=True)

    if promote_row_to_header and len(out) > 0:
        # Categorical columns can't accept arbitrary new values when we
        # re-coerce later; demote them to object first so the row-promotion
        # and subsequent numeric retry are unconstrained.
        for c in out.columns:
            if isinstance(out[c].dtype, pd.CategoricalDtype):
                out[c] = out[c].astype(object)

        new_header_row = out.iloc[0]
        new_header: list[str] = []
        seen: dict[str, int] = {}
        for i, val in enumerate(new_header_row.tolist()):
            name = "" if pd.isna(val) else str(val).strip()
            if not name:
                name = f"Unnamed_{i}"
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
            new_header.append(name)

        out = out.iloc[1:].reset_index(drop=True)
        out.columns = new_header

        # Reading-with-wrong-header would have left numeric sensor columns
        # parsed as object. Try a column-wise numeric coercion now that the
        # real header is in place; leave anything that doesn't cleanly
        # parse as-is for the downstream smart-conversion + sanitization
        # passes to handle.
        for c in out.columns:
            if out[c].dtype == object:
                coerced = pd.to_numeric(out[c], errors="coerce")
                if coerced.notna().sum() == out[c].notna().sum() and coerced.notna().any():
                    out[c] = coerced

    try:
        out.attrs.update(original_attrs)
    except Exception:
        pass
    return out


@st.cache_data(show_spinner=False)
def detect_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small DataFrame describing each column's detected type and non-null count."""
    info = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(df[c].dtype) for c in df.columns],
        "Non-Null Count": [df[c].notna().sum() for c in df.columns],
        "Unique Values": [df[c].nunique(dropna=True) for c in df.columns],
    })
    return info.reset_index(drop=True)
