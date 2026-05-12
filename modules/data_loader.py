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

from utils.helpers import compress_strings_to_category


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


@st.cache_data(show_spinner="Loading dataset…")
def _read_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame. Cached on (bytes_hash, filename).

    Applies category compression at the end so all downstream tabs see the
    memory-optimised DataFrame.
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

    return compress_strings_to_category(df)


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
