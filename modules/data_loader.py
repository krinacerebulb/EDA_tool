"""Load user-uploaded datasets (CSV / Excel / JSON / Parquet) into pandas DataFrames."""

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}


def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Read a file-like object from Streamlit's uploader into a DataFrame.

    Supported formats: CSV, Excel (.xlsx / .xls), JSON, Parquet (.parquet / .pq).

    Raises ValueError if the extension is unsupported or the file can't be parsed.
    """
    if uploaded_file is None:
        raise ValueError("No file was provided.")

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Please upload one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if suffix == ".csv":
            return pd.read_csv(uploaded_file)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(uploaded_file)
        if suffix == ".json":
            return pd.read_json(uploaded_file)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(uploaded_file)
    except ImportError as exc:
        raise ValueError(
            f"Reading '{suffix}' requires an additional engine. "
            f"Install with: pip install pyarrow  ({exc})"
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not parse file '{uploaded_file.name}': {exc}") from exc

    raise ValueError(f"Unhandled file type: {suffix}")


def detect_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small DataFrame describing each column's detected type and non-null count."""
    info = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(df[c].dtype) for c in df.columns],
        "Non-Null Count": [df[c].notna().sum() for c in df.columns],
        "Unique Values": [df[c].nunique(dropna=True) for c in df.columns],
    })
    return info.reset_index(drop=True)
