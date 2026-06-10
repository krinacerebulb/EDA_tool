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
TWO MERGE STRATEGIES — auto-selected per upload batch
================================================================================
* **Stack mode** (current default). All files share the same column set →
  vertical concatenation (rows from each file appended). This is what
  industrial users expect for monthly partitions or per-site exports.

* **Time-align mode** (new). Files disagree on column membership AND every
  file has a detectable datetime column → use the densest file as the
  *anchor* and merge sparser files onto it with ``pd.merge_asof(direction=
  "backward")``. This implements **forward-fill until the next reading**:
  a 6-hourly lab measurement holds for every 1-minute sensor row until the
  next lab measurement arrives. The classic industrial use case is mixing
  high-frequency sensor data with shift-based lab assays.

  If schemas differ but some file has no datetime column, we fall back to
  stack mode and emit a warning — alignment is impossible without a shared
  time axis.

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
def load_multiple_files(
    uploaded_files, sheet_selection: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load N files, sanitize each, validate schemas, merge into one frame.

    Parameters
    ----------
    uploaded_files
        Iterable of Streamlit ``UploadedFile`` objects (anything with ``.name``
        and ``.getvalue()``). Pass a single file in a list to use single-file
        mode — the merge step is a no-op when only one file succeeds and the
        ``source_file`` column is intentionally omitted.
    sheet_selection
        Optional ``{filename: [sheet_name, ...]}`` map telling the loader which
        sheet(s) to read from a multi-sheet Excel workbook. A single value
        loads that one sheet under the original filename; several values load
        each sheet as its own frame (labelled ``"file.xlsx [Sheet]"``) and
        merge them like multiple files. Filenames not in the map (and all
        non-Excel files) fall back to the first sheet. A plain ``str`` value is
        accepted as shorthand for a one-element list.

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
    sheet_selection = sheet_selection or {}

    if not uploaded_files:
        return pd.DataFrame(), report

    # --- Pass 1: load every file independently. ---
    # ``load_dataset`` already runs the full sanitization pipeline via the
    # bytes-based cached reader, so each ``frames[name]`` is individually
    # PyArrow-safe by the time it reaches us.
    frames: dict[str, pd.DataFrame] = {}
    for uf in uploaded_files:
        name = getattr(uf, "name", "<unknown>")
        # Expand each upload into one-or-more (label, sheet) load tasks. A
        # workbook with several selected sheets yields one task per sheet,
        # each labelled "file.xlsx [Sheet]" so it merges like a separate file.
        for load_label, sheet in _expand_sheet_tasks(name, sheet_selection.get(name)):
            try:
                df = load_dataset(uf, sheet_name=sheet)
                if df is None or df.empty:
                    report["per_file"][load_label] = {
                        "rows": 0,
                        "cols": int(df.shape[1]) if df is not None else 0,
                        "status": "empty",
                        "error": "File parsed successfully but contained no rows.",
                    }
                    continue
                frames[load_label] = df
                report["per_file"][load_label] = {
                    "rows": int(len(df)),
                    "cols": int(df.shape[1]),
                    "status": "ok",
                    "error": None,
                }
            except Exception as exc:
                # Corrupted file / unsupported encoding / parser failure — log,
                # record, move on. The remaining files still get processed.
                logger.exception("Failed to load %s", load_label)
                report["per_file"][load_label] = {
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

    # --- Pass 3: pick a merge strategy. ---
    # If schemas match → stack rows (current behaviour). If schemas differ
    # AND every file has a datetime column → switch to time-align mode so
    # the user gets a wide, time-indexed frame with forward-filled sparse
    # readings. See module docstring for the full contract.
    strategy, time_cols = _decide_merge_strategy(frames)
    report["merge_strategy"] = strategy

    if strategy == "time_align":
        try:
            merged = _time_align_merge(frames, time_cols)
            time_col_summary = ", ".join(
                f"`{n}` → `{tc}`" for n, tc in time_cols.items()
            )
            report["schema_warnings"].append(
                "Schemas differ across files — switched to **time-align "
                "mode**. Used the densest file as the anchor and "
                "forward-filled sparser values until each next reading. "
                f"Datetime columns used: {time_col_summary}."
            )
            report["time_align_keys"] = dict(time_cols)
        except Exception as exc:
            logger.exception("Time-align merge failed; falling back to stack")
            report["schema_warnings"].append(
                f"Time-align failed ({exc}); falling back to row-stack "
                "with NaN-filled missing columns."
            )
            strategy = "stack"

    if strategy == "stack":
        # --- Pass 3 (stack): align columns. ---
        try:
            aligned = normalize_column_structure(frames)
        except Exception as exc:
            # Should not happen — column-set arithmetic is bulletproof.
            logger.exception("normalize_column_structure failed")
            report["schema_warnings"].append(f"Column alignment failed: {exc}")
            aligned = frames  # fall back to raw frames

        # --- Pass 4 (stack): concatenate. ---
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
# TIME-ALIGN MERGE (different schemas, joined on date/time)
# ============================================================================
def _detect_first_datetime(df: pd.DataFrame) -> str | None:
    """Best-guess datetime column to use as the time axis.

    Preference order:
      1. Any column already typed as datetime64 — these came out of the
         sanitization pass with high confidence.
      2. First column accepted by ``time_series.detect_datetime_columns``
         (heuristic parse of string columns).
      3. First numeric column that looks like a datetime encoding (Excel serial,
         Unix epoch) according to ``datetime_formatter.detect_numeric_datetime_columns``.

    Returns ``None`` if the frame has no usable time axis — the caller
    should fall back to row-stacking in that case.
    """
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    try:
        # Lazy import — avoids a hard dependency cycle if time_series ever
        # grows to import from this module.
        from .time_series import detect_datetime_columns
        dt_cols = detect_datetime_columns(df)
        if dt_cols:
            return dt_cols[0]
    except Exception:
        logger.exception("Datetime detection failed")

    try:
        from .datetime_formatter import detect_numeric_datetime_columns
        num_dt_cols = detect_numeric_datetime_columns(df)
        if num_dt_cols:
            return num_dt_cols[0]["column"]
    except Exception:
        logger.exception("Numeric datetime detection failed")

    return None


def _decide_merge_strategy(
    frames: dict[str, pd.DataFrame],
) -> tuple[str, dict[str, str | None]]:
    """Pick ``"stack"`` or ``"time_align"`` based on schema overlap.

    If every file has a usable datetime column, prefer ``time_align`` so the
    files can be joined on the time axis. This covers both:

    * matching schemas with different time coverage, and
    * mismatched schemas that need to be aligned horizontally by time.

    If at least one file lacks a datetime column, fall back to ``stack``.
    """
    time_cols: dict[str, str | None] = {
        n: _detect_first_datetime(df) for n, df in frames.items()
    }
    if len(frames) <= 1:
        return "stack", time_cols

    if all(time_cols.values()):
        return "time_align", time_cols
    return "stack", time_cols


def _time_align_merge(
    frames: dict[str, pd.DataFrame],
    time_cols: dict[str, str],
) -> pd.DataFrame:
    """Group files by schema, stack within group, then merge_asof on time.

    Algorithm
    ---------
    1. Group frames by their non-datetime column signature so files that
       share a schema (e.g. three monthly partitions of sensor data) stack
       vertically into one wide frame.
    2. Within each group, canonicalize the datetime column to the first
       file's name and stack.
    3. Pick the densest group (most rows) as the **anchor** — this is
       typically the 1-minute sensor stream.
    4. ``pd.merge_asof(..., direction="backward")`` each other group onto
       the anchor. ``direction="backward"`` means "for each anchor row,
       attach the most recent value from the sparser stream" — which is
       exactly forward-fill-until-next-reading semantics.

    The anchor's datetime column is preserved; the joined groups' datetime
    columns are dropped from the result (they're redundant — the time
    axis is the anchor's).
    """
    # --- Step 1: group by non-time column signature ---
    sig_to_files: dict[frozenset, list[str]] = {}
    for name in frames:
        sig = frozenset(
            c for c in frames[name].columns if c != time_cols[name]
        )
        sig_to_files.setdefault(sig, []).append(name)

    # --- Step 2: build one DataFrame per signature group ---
    groups: list[tuple[str, pd.DataFrame, str]] = []  # (label, df, time_col)
    for sig, names in sig_to_files.items():
        canonical_time = time_cols[names[0]]
        bits = []
        for n in names:
            d = frames[n].copy()
            # Different files in the same group may have different time
            # column NAMES (e.g. "Timestamp" vs "DateTime"). Canonicalize
            # to the first file's name so the stack works cleanly.
            if time_cols[n] != canonical_time:
                d = d.rename(columns={time_cols[n]: canonical_time})
            if pd.api.types.is_numeric_dtype(d[canonical_time]):
                from .datetime_formatter import detect_numeric_datetime_mode, convert_numeric_datetime
                num_mode, _ = detect_numeric_datetime_mode(d[canonical_time])
                if num_mode:
                    parsed_dt, _ = convert_numeric_datetime(d[canonical_time], num_mode)
                    d[canonical_time] = parsed_dt
                else:
                    d[canonical_time] = pd.to_datetime(d[canonical_time], errors="coerce")
            else:
                d[canonical_time] = pd.to_datetime(d[canonical_time], errors="coerce")
            # Strip timezone info — merge_asof rejects a tz-aware key
            # joined against a tz-naive key. Industrial datasets rarely
            # carry consistent tz metadata anyway.
            if (
                pd.api.types.is_datetime64_any_dtype(d[canonical_time])
                and getattr(d[canonical_time].dt, "tz", None) is not None
            ):
                d[canonical_time] = d[canonical_time].dt.tz_localize(None)
            d["__source_file__"] = n
            bits.append(d)
        group_df = pd.concat(bits, axis=0, ignore_index=True, sort=False)
        # merge_asof requires both sides sorted by the join key and free
        # of NaT in that key.
        group_df = (
            group_df.dropna(subset=[canonical_time])
            .sort_values(canonical_time)
            .reset_index(drop=True)
        )
        if group_df.empty:
            continue
        label = " + ".join(names)
        groups.append((label, group_df, canonical_time))

    if not groups:
        return pd.DataFrame()

    # --- Step 3 + 4: anchor on the densest group, merge_asof the rest ---
    groups.sort(key=lambda t: -len(t[1]))
    anchor_label, anchor_df, anchor_time = groups[0]
    # Promote the anchor's per-row source tag to the public name.
    result = anchor_df.rename(columns={"__source_file__": "source_file"})

    for label, other_df, other_time in groups[1:]:
        # Rename the other group's time column to match the anchor's so
        # merge_asof can use a single ``on=`` key. The other group's
        # internal source tag becomes its own column so provenance is
        # preserved without colliding with the anchor's source_file.
        renames = {"__source_file__": f"source_file__{label}"}
        if other_time != anchor_time:
            renames[other_time] = anchor_time
        other_df = other_df.rename(columns=renames)
        # Both sides must be sorted by the merge key.
        result = result.sort_values(anchor_time).reset_index(drop=True)
        other_df = other_df.sort_values(anchor_time).reset_index(drop=True)
        result = pd.merge_asof(
            result,
            other_df,
            on=anchor_time,
            direction="backward",
            allow_exact_matches=True,
        )

    # source_file is the anchor's per-file tag; keep it categorical for
    # the same memory-efficiency reasons as stack mode.
    if "source_file" in result.columns:
        result["source_file"] = result["source_file"].astype("category")
        cols = ["source_file"] + [c for c in result.columns if c != "source_file"]
        result = result[cols]

    return make_arrow_safe(result)


# ============================================================================
# INTERNAL
# ============================================================================
def _expand_sheet_tasks(
    name: str, sheets,
) -> list[tuple[str, str | None]]:
    """Turn a file's sheet selection into ``(load_label, sheet_name)`` tasks.

    * No selection (``None``/empty) or a non-Excel file → a single task that
      reads the default sheet under the original filename.
    * One selected sheet → a single task under the original filename (kept
      clean — no ``[Sheet]`` suffix when there's nothing to disambiguate).
    * Several selected sheets → one task per sheet, each labelled
      ``"file.xlsx [Sheet]"`` so downstream merge/source tagging treats each
      sheet as its own file.
    """
    if not sheets:
        return [(name, None)]
    if isinstance(sheets, str):
        sheets = [sheets]
    if len(sheets) == 1:
        return [(name, sheets[0])]
    return [(f"{name} [{s}]", s) for s in sheets]


def _empty_report() -> dict[str, Any]:
    return {
        "per_file": {},
        "schema_warnings": [],
        "schema_alignment": {},
        "dtype_clashes": [],
        "merged_rows": 0,
        "merged_cols": 0,
        "source_files": [],
        "merge_strategy": "stack",
        "time_align_keys": {},
    }
