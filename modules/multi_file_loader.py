"""Multi-file upload, schema validation, and intelligent merging.

================================================================================
WHY THIS MODULE EXISTS
================================================================================
Industrial users rarely have a single, clean dataset. They have:

    January.csv, February.csv, March.csv     (monthly partitions)
    plant_A.xlsx, plant_B.xlsx               (per-site exports)
    sensors_raw.csv, sensors_qc.csv          (raw vs quality-checked)

A useful EDA platform must let them upload ALL of these together and produce
a single, analysis-ready frame — even if the files disagree about:

    * exact column names (typos, spacing, casing)
    * column dtypes (one file has "Temperature" as float, another as string)
    * column membership (file A has columns missing in file B)
    * encoding (UTF-8 vs Latin-1 vs UTF-16 BOM)

This module enforces graceful, well-documented behaviour for all of these.

================================================================================
PUBLIC API
================================================================================
    load_multiple_files(uploaded_files) -> (merged_df, report)
        End-to-end entry point. Loads, sanitizes, validates, aligns, merges.
        Never raises. Files that fail are skipped and recorded in the report.

    validate_schema(frames) -> {warnings, alignment, dtype_clashes}
        Cross-file consistency check. Reports columns present/missing per
        file and dtype mismatches on shared columns.

    normalize_column_structure(frames) -> {filename: aligned_df}
        Aligns every frame to the UNION of all columns (preserving the
        upload-order of columns — never alphabetised). Missing columns are
        filled with ``NaN``.

    merge_uploaded_files(frames, add_source=True) -> df
        Concatenates aligned frames. Adds a ``source_file`` column so users
        can group / filter by origin.

================================================================================
DESIGN NOTES
================================================================================
* Per-file sanitization is delegated to ``data_loader._read_bytes`` (which
  already runs ``preprocess_dynamic_dataset``). Each file is therefore
  PyArrow-safe BEFORE merging — so the only post-merge concern is dtype
  reconciliation, which we handle with a final ``make_arrow_safe`` pass.
* Row order is preserved across the concat (upload order).
* ``source_file`` is a ``category`` for memory efficiency, and is ONLY
  added when 2+ files were successfully loaded — a single-file upload
  shouldn't be cluttered with a redundant column where every row has the
  same value.
* Column order is preserved from the FIRST file (no alphabetisation);
  columns unique to later files are appended in first-seen order.
* Empty or fully-failed batches return ``(empty_df, report)`` instead of
  raising. The UI is expected to handle the empty case.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .data_loader import load_dataset
from .data_sanitization import make_arrow_safe

logger = logging.getLogger(__name__)


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================
def load_multiple_files(uploaded_files) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load N files, sanitize each, validate schemas, merge into one frame.

    Parameters
    ----------
    uploaded_files
        Iterable of Streamlit ``UploadedFile`` objects (anything with ``.name``
        and ``.getvalue()``). Pass a single file in a list to use single-file
        mode — the merge step is a no-op when only one file succeeds and the
        ``source_file`` column is intentionally omitted.

    Returns
    -------
    (merged_df, report)
        ``merged_df`` is a sanitized, aligned, concatenated DataFrame.
        Empty if every file failed.
        ``report`` documents the load + validation + merge — surface this
        to the user via a Streamlit panel so schema mismatches are visible.

    Guarantees
    ----------
    * Never raises. A corrupted file is recorded with ``status='failed'``
      and skipped; the rest of the batch is processed normally.
    """
    report: dict[str, Any] = _empty_report()

    if not uploaded_files:
        return pd.DataFrame(), report

    # --- Pass 1: load every file independently. ---
    # ``load_dataset`` already runs the full sanitization pipeline via the
    # bytes-based cached reader, so each ``frames[name]`` is individually
    # PyArrow-safe by the time it reaches us.
    frames: dict[str, pd.DataFrame] = {}
    for uf in uploaded_files:
        name = getattr(uf, "name", "<unknown>")
        try:
            df = load_dataset(uf)
            if df is None or df.empty:
                report["per_file"][name] = {
                    "rows": 0,
                    "cols": int(df.shape[1]) if df is not None else 0,
                    "status": "empty",
                    "error": "File parsed successfully but contained no rows.",
                }
                continue
            frames[name] = df
            report["per_file"][name] = {
                "rows": int(len(df)),
                "cols": int(df.shape[1]),
                "status": "ok",
                "error": None,
            }
        except Exception as exc:
            # Corrupted file / unsupported encoding / parser failure — log,
            # record, move on. The remaining files still get processed.
            logger.exception("Failed to load %s", name)
            report["per_file"][name] = {
                "rows": 0, "cols": 0,
                "status": "failed", "error": str(exc),
            }

    if not frames:
        report["schema_warnings"].append(
            "No files could be loaded successfully. See the per-file "
            "errors above for details."
        )
        return pd.DataFrame(), report

    # --- Pass 2: cross-file schema validation. ---
    validation = validate_schema(frames)
    report["schema_warnings"].extend(validation["warnings"])
    report["schema_alignment"] = validation["alignment"]
    report["dtype_clashes"] = validation["dtype_clashes"]

    # --- Pass 3: align columns. ---
    try:
        aligned = normalize_column_structure(frames)
    except Exception as exc:
        # Should not happen in practice — column-set arithmetic is bulletproof.
        logger.exception("normalize_column_structure failed")
        report["schema_warnings"].append(f"Column alignment failed: {exc}")
        aligned = frames  # fall back to raw frames

    # --- Pass 4: concatenate. ---
    # ``source_file`` is only useful when ≥ 2 files were merged. On a
    # single-file upload every row would have the same value, which is
    # redundant noise — so we omit it.
    add_source = len(frames) > 1
    try:
        merged = merge_uploaded_files(aligned, add_source=add_source)
    except Exception as exc:
        logger.exception("merge_uploaded_files failed")
        report["schema_warnings"].append(
            f"Final concat failed ({exc}); returning the first file only."
        )
        # Last-resort: return the first frame so the app doesn't crash.
        merged = next(iter(aligned.values())).copy()
        if add_source:
            merged.insert(
                0, "source_file",
                pd.Categorical([next(iter(aligned.keys()))] * len(merged)),
            )

    report["merged_rows"] = int(len(merged))
    report["merged_cols"] = int(merged.shape[1])
    report["source_files"] = list(frames.keys())
    return merged, report


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================
def validate_schema(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compare schemas across loaded frames.

    Detects:
        * columns present in some files but not others (the most common
          industrial pattern: a sensor was added in February).
        * shared columns whose dtypes differ between files (one file has it
          as ``float64``, another as ``object`` because of a dirty token).

    Returns ``{warnings, alignment, dtype_clashes}``.
    """
    if not frames:
        return {"warnings": [], "alignment": {}, "dtype_clashes": []}

    all_col_sets = [set(df.columns) for df in frames.values()]
    union = set().union(*all_col_sets)
    intersection = set.intersection(*all_col_sets) if all_col_sets else set()

    warnings: list[str] = []
    alignment: dict[str, dict] = {}

    # Single-file batch: nothing to compare.
    if len(frames) == 1:
        only = next(iter(frames))
        alignment[only] = {"missing_in_this_file": [], "unique_to_this_file": []}
        return {"warnings": warnings, "alignment": alignment, "dtype_clashes": []}

    # Per-file gap analysis.
    for name, df in frames.items():
        cols = set(df.columns)
        missing = union - cols              # columns OTHER files have
        unique = cols - intersection        # columns only some files have
        alignment[name] = {
            "missing_in_this_file": sorted(missing),
            "unique_to_this_file": sorted(unique - missing),
        }
        if missing:
            preview = ", ".join(f"`{c}`" for c in sorted(missing)[:5])
            extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            warnings.append(
                f"`{name}` is missing **{len(missing)}** column(s) that other "
                f"files have: {preview}{extra}. They will be filled with NaN."
            )

    # Cross-file dtype consistency on shared columns.
    dtype_clashes: list[dict] = []
    for col in sorted(intersection):
        dtypes = {name: str(df[col].dtype) for name, df in frames.items()}
        if len(set(dtypes.values())) > 1:
            dtype_clashes.append({"column": col, "dtypes": dtypes})
    if dtype_clashes:
        warnings.append(
            f"**{len(dtype_clashes)}** shared column(s) have inconsistent "
            "dtypes across files. Pandas will upcast to a common type on "
            "merge (typically to ``object``)."
        )

    return {
        "warnings": warnings,
        "alignment": alignment,
        "dtype_clashes": dtype_clashes,
    }


# ============================================================================
# COLUMN ALIGNMENT
# ============================================================================
def normalize_column_structure(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Align every frame to the UNION of columns; fill missing with NaN.

    Why a union (not intersection)?
        * Intersection would silently drop columns. Imagine February.csv
          gained a "Pressure" column — taking the intersection would erase
          it from the merged frame entirely.
        * Union preserves every signal. Months where a column didn't exist
          simply show up as NaN, which is the truthful representation.

    All frames are returned with their columns in identical order, driven
    by **upload order** (NOT alphabetised):

        * the first file's column order anchors the layout,
        * any columns that exist only in later files are appended in their
          first-seen order.

    Single-file uploads therefore keep the user's exact original column
    arrangement — sorting would silently reorder their data.
    """
    if not frames:
        return {}

    # Build an ORDERED union (preserve order of first appearance), not the
    # alphabetic ``sorted(set())`` — sorting scrambles the user's layout.
    union: list[str] = []
    seen: set = set()
    for df in frames.values():
        for col in df.columns:
            if col not in seen:
                seen.add(col)
                union.append(col)

    out: dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        aligned = df.copy()
        # Add any columns this frame is missing (filled with NaN).
        for c in union:
            if c not in aligned.columns:
                aligned[c] = pd.NA
        # Identical column order across all frames — driven by upload order.
        aligned = aligned.reindex(columns=union)
        out[name] = aligned
    return out


# ============================================================================
# MERGE
# ============================================================================
def merge_uploaded_files(
    frames: dict[str, pd.DataFrame],
    *,
    add_source: bool = True,
) -> pd.DataFrame:
    """Concatenate aligned frames into a single DataFrame.

    Parameters
    ----------
    frames
        Output of ``normalize_column_structure`` — every frame has identical
        column order.
    add_source
        When True (default), prepend a ``source_file`` column tagging each
        row with the originating filename. The column is stored as
        ``category`` for memory efficiency.

    Why a final ``make_arrow_safe`` pass?
        Concatenation can promote dtypes (e.g. float64 + object → object on
        a shared column). That's a fresh opportunity to introduce a mixed-
        type column. Running the arrow-safety guarantee one more time is
        cheap and prevents downstream Streamlit crashes.
    """
    if not frames:
        return pd.DataFrame()

    if add_source:
        tagged = []
        for name, df in frames.items():
            d = df.copy()
            d["source_file"] = name
            tagged.append(d)
        merged = pd.concat(tagged, axis=0, ignore_index=True, sort=False)
        merged["source_file"] = merged["source_file"].astype("category")
        # Move ``source_file`` to the front — it's the most-used filter
        # dimension on multi-file uploads.
        cols = ["source_file"] + [c for c in merged.columns if c != "source_file"]
        merged = merged[cols]
    else:
        merged = pd.concat(
            list(frames.values()), axis=0, ignore_index=True, sort=False,
        )

    # Final safety net — see docstring note above.
    return make_arrow_safe(merged)


# ============================================================================
# INTERNAL
# ============================================================================
def _empty_report() -> dict[str, Any]:
    return {
        "per_file": {},
        "schema_warnings": [],
        "schema_alignment": {},
        "dtype_clashes": [],
        "merged_rows": 0,
        "merged_cols": 0,
        "source_files": [],
    }
