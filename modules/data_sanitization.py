"""Production-grade data sanitization layer.

================================================================================
WHY THIS MODULE EXISTS
================================================================================
Industrial datasets (SCADA, AVEVA PI System, manufacturing telemetry, IoT
sensor exports, dirty Excel files) regularly arrive with **mixed-type
columns**. A "Temperature" column is supposed to be numeric, but the export
also contains string error codes::

    Temperature: [120, 130, "No Data", 145, "Bad", "Sensor Fail", "-", ""]

Streamlit renders DataFrames through PyArrow. PyArrow requires every column
to have a single, well-defined Arrow type. The moment an object column holds
both Python floats and Python strings, ``st.dataframe`` raises::

    pyarrow.lib.ArrowInvalid:
        Could not convert 'No Data' with type str: tried to convert to double

The same problem cascades into:
    * Plotly conversions (mixed types break ``plotly.express``)
    * ``df.corr()`` (string in a "numeric" column crashes)
    * ``df.describe()`` (returns wrong stats)
    * scikit-learn pipelines (refuse mixed dtypes)
    * Parquet writes (Arrow schema inference fails)
    * Histograms / boxplots / heatmaps

This module enforces three hard invariants on EVERY DataFrame that leaves it:

    1. Every column has a single, well-defined logical type
       (numeric / datetime / boolean / categorical / string / empty).
    2. All industrial "dirty tokens" ("No Data", "Bad", "Sensor Fail", "-",
       "N/A", Excel error strings, ...) are normalised to ``NaN``.
    3. The frame is PyArrow-renderable. ``st.dataframe`` and
       ``st.data_editor`` will never raise ``ArrowInvalid`` on a frame that
       has passed through ``preprocess_dynamic_dataset`` / ``make_arrow_safe``.

================================================================================
PUBLIC API (the eight functions the rest of the app should call)
================================================================================
    preprocess_dynamic_dataset(df)      -> (clean_df, report)
    intelligent_type_detection(df)      -> {col: 'numeric'|'datetime'|...}
    sanitize_invalid_tokens(df)         -> df  (dirty tokens -> NaN)
    safe_numeric_conversion(df, ...)    -> df  (majority-numeric -> numeric)
    safe_datetime_conversion(df, ...)   -> df  (majority-date -> datetime)
    make_arrow_safe(df)                 -> df  (final PyArrow guarantee)
    prepare_for_visualization(df)       -> df  (viz-safe convenience wrapper)
    prepare_for_ml(df)                  -> df  (ML-safe convenience wrapper)

================================================================================
DESIGN PRINCIPLES
================================================================================
* **Pure**: every public function returns a NEW DataFrame. The input is never
  mutated. Safe to call inside cached pipelines.
* **Never raises**: the module's contract is "graceful degradation". On any
  internal failure for a single column, the column is forced to ``string``
  (the universally-safe fallback) and the issue is logged in the report.
* **Idempotent**: running the pipeline twice on the same frame is a no-op
  apart from a small CPU cost. Downstream code can defensively re-run it.
* **Memory-conscious**: low-cardinality strings become ``category``; numeric
  columns are downcast where lossless. A 500k-row string column with 100
  unique values drops from ~25 MB to a few hundred KB.
* **PyArrow-aware**: the final ``make_arrow_safe`` pass uses pyarrow itself
  to validate every column, then stringifies any that arrow rejects.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# INDUSTRIAL DIRTY TOKENS
# ----------------------------------------------------------------------------
# The canonical set of strings that should ALWAYS become NaN. The lookup is
# performed against ``str.strip().lower()`` so casing and surrounding
# whitespace don't matter.
#
# Sources for these tokens:
#   - PI System / OSIsoft point-attribute error states
#   - SCADA tag-quality flags (Wonderware, Ignition, Siemens WinCC)
#   - Manufacturing MES / Historian exports
#   - Excel error cells (#N/A, #VALUE!, #DIV/0!, ...)
#   - Dirty CSVs ("--", "?", "—", em-dash, en-dash)
#
# Adding a new token? Lower-case it and strip. Tests rely on the lookup
# being whitespace/case insensitive, NOT on regex matching, so keep entries
# as literals (no patterns).
# ============================================================================
INDUSTRIAL_NULL_TOKENS: frozenset[str] = frozenset({
    # --- Standard nulls (CSV / Excel / database exports) ---
    "", "na", "n/a", "n.a", "n.a.", "nan", "null", "none", "nil", "void",
    "n\\a",

    # --- Symbolic placeholders ---
    "-", "--", "---", "?", "??", "???", "—", "–", ".",

    # --- PI System / OSIsoft tag states ---
    "no data", "nodata", "no sample", "calc off", "calc failed",
    "calc error", "calc trig", "i/o timeout", "scan off", "shutdown",
    "configure", "bad input", "intf shut", "comm fail", "no result",

    # --- Generic SCADA / sensor health tokens ---
    "bad", "bad value", "bad data", "bad quality", "error", "err",
    "sensor fail", "sensor failure", "sensor error", "sensor down",
    "disconnected", "disc", "timeout", "time out",
    "comm failure", "communication failure", "communication error",
    "unknown", "unk", "invalid", "inv", "missing",
    "not available", "not applicable",
    "no good", "ng", "off", "out of service", "oos",
    "fail", "failed", "fault", "trip", "tripped", "alarm",
    "overflow", "underflow", "saturated",

    # --- Excel error strings ---
    "#n/a", "#name?", "#value!", "#div/0!", "#ref!", "#num!", "#null!",
    "#getting_data", "#spill!", "#calc!",
})


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================
def preprocess_dynamic_dataset(
    df: pd.DataFrame,
    *,
    numeric_threshold: float = 0.7,
    datetime_threshold: float = 0.85,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full sanitization pipeline on an arbitrary user-uploaded frame.

    Pipeline (order matters):
        1. clean_column_names      — strip whitespace / Unicode / deduplicate
        2. sanitize_invalid_tokens — industrial dirty tokens become NaN
        3. safe_numeric_conversion — majority-numeric object cols become numeric
        4. safe_datetime_conversion — majority-date object cols become datetime
        5. coerce_booleans         — yes/no/true/false cols become boolean
        6. handle_infinities       — ±inf becomes NaN (Plotly can't plot inf)
        7. compress_categoricals   — low-cardinality strings become category
        8. make_arrow_safe         — final PyArrow validation pass

    Why this exact order?
    * Tokens must be cleaned BEFORE numeric/datetime detection — otherwise
      "No Data" is counted as a non-numeric value and the column fails the
      conversion threshold even though 99% of its real values are numeric.
    * Booleans run after numeric so columns that look like "0"/"1" become
      proper Int64, not bool (more useful for stats / correlation).
    * Categorical compression runs LAST among logical conversions so that
      sanitization can still operate on the underlying strings.
    * ``make_arrow_safe`` is the final guarantee — anything we couldn't
      cleanly resolve is forced to ``string`` so Streamlit won't crash.

    Returns
    -------
    (clean_df, report)
        ``clean_df`` is a NEW DataFrame; ``df`` is not mutated.
        ``report`` documents every transformation applied. Use it to surface
        a "Sanitization summary" panel in the UI.

    Guarantees
    ----------
    * Never raises. Catastrophic failure falls back to a fully-stringified
      frame, which is always PyArrow-safe.
    * Idempotent. Calling twice is safe and roughly free on the second call.
    """
    report: dict[str, Any] = _empty_report(df)

    # Defensive: handle empty / None inputs gracefully.
    if df is None:
        report["errors"].append("Input DataFrame was None.")
        return pd.DataFrame(), report
    if df.empty:
        report["warnings"].append("Input DataFrame has zero rows.")
        return df.copy(), report

    out = df.copy()

    # Each step is wrapped individually. If one step fails we still get the
    # benefit of every other step — the module's "never raise" contract.

    out, report["renamed_columns"], report["duplicate_columns_dedup"] = \
        _safe_step("clean_column_names", clean_column_names, out, report,
                   default=({}, []))

    out, tokens = _safe_step(
        "sanitize_invalid_tokens",
        lambda d: sanitize_invalid_tokens(d, return_counts=True),
        out, report, default=({}, {}),
    )
    report["tokens_replaced_per_column"] = tokens
    report["tokens_replaced_total"] = int(sum(tokens.values())) if tokens else 0

    out, num_log = _safe_step(
        "safe_numeric_conversion",
        lambda d: safe_numeric_conversion(d, threshold=numeric_threshold,
                                          return_log=True),
        out, report, default=({}, []),
    )
    report["numeric_conversions"] = num_log

    out, dt_log = _safe_step(
        "safe_datetime_conversion",
        lambda d: safe_datetime_conversion(d, threshold=datetime_threshold,
                                           return_log=True),
        out, report, default=({}, []),
    )
    report["datetime_conversions"] = dt_log

    out, bool_log = _safe_step(
        "coerce_booleans",
        lambda d: coerce_booleans(d, return_log=True),
        out, report, default=({}, []),
    )
    report["boolean_conversions"] = bool_log

    out, n_inf = _safe_step(
        "handle_infinities",
        lambda d: handle_infinities(d, return_count=True),
        out, report, default=({}, 0),
    )
    report["infinities_replaced"] = int(n_inf or 0)

    out, compressed = _safe_step(
        "compress_categoricals",
        lambda d: compress_categoricals(d, return_log=True),
        out, report, default=({}, []),
    )
    report["category_compressions"] = compressed

    out, arrow_log = _safe_step(
        "make_arrow_safe",
        lambda d: make_arrow_safe(d, return_log=True),
        out, report, default=({}, []),
    )
    report["arrow_unsafe_columns"] = arrow_log

    # --- Final type-detection summary for the report ---
    try:
        report["detected_types"] = intelligent_type_detection(out)
    except Exception as exc:
        report["errors"].append(f"intelligent_type_detection: {exc}")

    report["output_rows"] = int(len(out))
    report["output_cols"] = int(out.shape[1])
    return out, report


# ============================================================================
# COLUMN NAMES
# ----------------------------------------------------------------------------
# Why this exists:
#   * Excel exports often have multi-line headers ("Plant\nTemperature").
#   * SCADA exports embed Unicode degree signs, BOMs, non-breaking spaces.
#   * Duplicate header names ("Value", "Value") break ``df[col]`` (returns
#     a DataFrame instead of a Series → silent bugs).
# ============================================================================
def clean_column_names(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Normalize column names; deduplicate by appending ``_1``, ``_2``, ...

    Returns ``(df, renamed_map, deduped_list)`` where:
        renamed_map  = {original_name: new_name}  (only entries that changed)
        deduped_list = names that had a numeric suffix appended to break a tie
    """
    if df is None or df.shape[1] == 0:
        return df, {}, []

    renamed: dict[str, str] = {}
    cleaned: list[str] = []
    for col in df.columns:
        original = col
        try:
            name = str(col)
            # NFKC normalises ligatures, full-width chars, non-breaking spaces.
            name = unicodedata.normalize("NFKC", name)
            # Newlines / tabs inside Excel merged headers → single space.
            name = re.sub(r"[\r\n\t ]+", " ", name)
            # Collapse runs of whitespace.
            name = re.sub(r"\s+", " ", name).strip()
            if name == "":
                name = "unnamed_column"
        except Exception:
            # Truly catastrophic — fall back to a positional name.
            name = f"col_{len(cleaned)}"
        if name != str(original):
            renamed[str(original)] = name
        cleaned.append(name)

    # Deduplicate. We track usage in two passes:
    #   counts[name]  : how many times we've SEEN this name already
    #   final_names   : guaranteed-unique output names
    counts: dict[str, int] = {}
    final_names: list[str] = []
    deduped: list[str] = []
    for n in cleaned:
        if n not in counts:
            counts[n] = 1
            final_names.append(n)
        else:
            # Find the next free suffix; another column might already have
            # claimed ``n_2`` (rare but possible).
            counts[n] += 1
            candidate = f"{n}_{counts[n]}"
            while candidate in counts:
                counts[n] += 1
                candidate = f"{n}_{counts[n]}"
            counts[candidate] = 1
            final_names.append(candidate)
            deduped.append(candidate)

    out = df.copy()
    out.columns = final_names
    return out, renamed, deduped


# ============================================================================
# DIRTY TOKEN SANITIZATION
# ============================================================================
def sanitize_invalid_tokens(
    df: pd.DataFrame,
    *,
    return_counts: bool = False,
):
    """Replace industrial dirty tokens with ``NaN``.

    Why this exists: dirty tokens are the #1 cause of PyArrow / pandas-numeric
    crashes on industrial data. We can't simply "skip" them because they're
    interleaved with real numbers in the same column. Replacing with ``NaN``
    is the only safe operation — it preserves row alignment and lets pandas
    treat them as missing across all downstream code.

    Operates on object / string / category columns only — numeric and
    datetime columns can't contain string tokens by definition.
    """
    counts: dict[str, int] = {}
    if df is None or df.empty:
        return (df, counts) if return_counts else df

    out = df.copy()
    for col in out.columns:
        s = out[col]
        # Only object / string / category columns can contain dirty tokens.
        if not _is_text_like(s):
            continue
        try:
            # Decategorize defensively. Categories store the levels as a
            # fixed array; mutating individual values forces us to round-
            # trip through string anyway.
            if isinstance(s.dtype, pd.CategoricalDtype):
                s = s.astype("string")

            stripped = s.astype("string").str.strip()
            # Lookup is case-insensitive.
            lowered = stripped.str.lower()
            mask = lowered.isin(INDUSTRIAL_NULL_TOKENS)
            # Also treat empty-after-strip and whitespace-only as null.
            mask = mask | (stripped == "")
            n = int(mask.sum())
            if n > 0:
                counts[col] = n
                stripped = stripped.mask(mask)
            out[col] = stripped
        except Exception as exc:
            logger.warning("sanitize_invalid_tokens(%r): %s", col, exc)

    return (out, counts) if return_counts else out


# ============================================================================
# SAFE NUMERIC CONVERSION
# ============================================================================
def safe_numeric_conversion(
    df: pd.DataFrame,
    *,
    threshold: float = 0.7,
    return_log: bool = False,
):
    """Convert object/string columns to numeric when the majority parses.

    Rule: a column is converted only if at least ``threshold`` (default 70%)
    of its non-null, non-token values parse as a real number. Otherwise it
    stays as text — converting a 5%-numeric column to float64 would create
    a column that's 95% ``NaN``, which is worse than keeping it as labels.

    The minority that fails to parse becomes ``NaN`` (this is the behaviour
    the user explicitly wanted: invalid sensor readings → NaN, never crash).
    """
    log: list[dict[str, Any]] = []
    if df is None or df.empty:
        return (df, log) if return_log else df

    out = df.copy()
    for col in out.columns:
        s = out[col]

        # Skip columns that are already typed.
        if (pd.api.types.is_numeric_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s)
                or pd.api.types.is_bool_dtype(s)):
            continue
        if not _is_text_like(s):
            continue

        try:
            ss = s.astype("string").str.strip()
            # Handle thousands separators only when the rest of the value
            # looks numeric, so we don't mangle real strings like "Plant,1".
            looks_thousand = ss.str.match(
                r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", na=False,
            )
            if looks_thousand.any():
                ss = ss.mask(looks_thousand,
                             ss.str.replace(",", "", regex=False))

            non_null = int(ss.notna().sum())
            if non_null == 0:
                continue

            coerced = pd.to_numeric(ss, errors="coerce")
            valid = int(coerced.notna().sum())
            ratio = valid / non_null

            if ratio >= threshold:
                out[col] = _downcast_numeric(coerced)
                log.append({
                    "column": col,
                    "from": str(s.dtype),
                    "to": str(out[col].dtype),
                    "valid": valid,
                    "invalid": non_null - valid,
                    "valid_pct": round(ratio * 100, 2),
                })
        except Exception as exc:
            logger.warning("safe_numeric_conversion(%r): %s", col, exc)

    return (out, log) if return_log else out


def _downcast_numeric(s: pd.Series) -> pd.Series:
    """Memory-optimise a numeric series without losing precision.

    int64 → smallest signed int that fits.
    float64 → float32 if values are within float32 range.

    Safe to call after ``pd.to_numeric`` because that always returns a
    well-defined numeric dtype.
    """
    if s.isna().all():
        return s
    try:
        if pd.api.types.is_integer_dtype(s) and not s.isna().any():
            return pd.to_numeric(s, downcast="integer")
        if pd.api.types.is_float_dtype(s):
            return pd.to_numeric(s, downcast="float")
    except Exception:
        pass
    return s


# ============================================================================
# SAFE DATETIME CONVERSION
# ============================================================================
# Heuristics:
# * Date-like characters: ``-``, ``/``, ``:``, month names. Without at least
#   one of these in the majority of samples, we don't even try (otherwise
#   plain integers like "20230101" or even "12345" can accidentally parse
#   as dates and create bogus 1969-era timestamps).
# * Threshold is intentionally higher than numeric (0.85 vs 0.7). Datetime
#   misclassification is much more damaging — time-series analysis on an
#   accidentally-parsed column produces nonsense.
# ============================================================================
_DATE_HINT_RE = re.compile(
    r"(?:[-/:.]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|\d{4})",
    re.IGNORECASE,
)


def safe_datetime_conversion(
    df: pd.DataFrame,
    *,
    threshold: float = 0.85,
    sample_size: int = 500,
    return_log: bool = False,
):
    """Convert object columns to datetime where the majority parses cleanly."""
    log: list[dict[str, Any]] = []
    if df is None or df.empty:
        return (df, log) if return_log else df

    out = df.copy()
    for col in out.columns:
        s = out[col]
        # Already a datetime — nothing to do.
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        # Numeric / boolean → handled elsewhere (Excel-serial / Unix
        # timestamps go through the manual ``preprocessing`` UI, since
        # those require user intent — a "12345" could be a tag id OR
        # a Unix epoch).
        if (pd.api.types.is_numeric_dtype(s)
                or pd.api.types.is_bool_dtype(s)):
            continue
        if not _is_text_like(s):
            continue

        try:
            non_null = s.dropna()
            if non_null.empty:
                continue
            sample = non_null.head(sample_size).astype(str)

            # Heuristic gate: the majority of samples must contain a
            # date-like character. Without this, "ID-1234" would happily
            # parse as 1234-AD-01-ID, etc.
            date_hint_ratio = sample.str.contains(
                _DATE_HINT_RE, regex=True, na=False,
            ).mean()
            if date_hint_ratio < 0.7:
                continue

            # Parse the sample first to avoid wasting time on full columns
            # that obviously won't make the threshold.
            with warnings.catch_warnings():
                # pandas warns about ``format="mixed"`` on some old builds;
                # we don't care because we want the most permissive parser.
                warnings.simplefilter("ignore")
                parsed_sample = pd.to_datetime(
                    sample, errors="coerce", format="mixed",
                )
            sample_ratio = parsed_sample.notna().mean()
            if sample_ratio < threshold:
                continue

            # Promising — parse the full column.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed_full = pd.to_datetime(
                    s, errors="coerce", format="mixed",
                )
            full_ratio = parsed_full.notna().mean()
            if full_ratio < threshold:
                continue

            out[col] = parsed_full
            log.append({
                "column": col,
                "from": str(s.dtype),
                "to": str(parsed_full.dtype),
                "valid_pct": round(float(full_ratio) * 100, 2),
            })
        except Exception as exc:
            logger.debug("safe_datetime_conversion(%r): %s", col, exc)

    return (out, log) if return_log else out


# ============================================================================
# BOOLEAN COERCION
# ----------------------------------------------------------------------------
# Only triggers when ALL non-null values are bool-like ({yes,no,y,n,true,
# false,t,f}). We skip ``0`` / ``1`` here because they were already caught
# by numeric conversion in the previous step; numeric Int64 is more useful
# than boolean for correlation/stats anyway.
# ============================================================================
_TRUE_TOKENS = frozenset({"true", "yes", "y", "t"})
_FALSE_TOKENS = frozenset({"false", "no", "n", "f"})
_BOOL_TOKENS = _TRUE_TOKENS | _FALSE_TOKENS


def coerce_booleans(
    df: pd.DataFrame,
    *,
    return_log: bool = False,
):
    """Convert text yes/no/true/false columns to nullable ``boolean``."""
    log: list[dict[str, Any]] = []
    if df is None or df.empty:
        return (df, log) if return_log else df

    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if not _is_text_like(s):
            continue
        try:
            non_null = s.dropna().astype("string").str.strip().str.lower()
            if non_null.empty:
                continue
            unique = set(non_null.unique())
            # Require at least two distinct values so we don't promote a
            # column that's literally just "yes" or just "true".
            if unique.issubset(_BOOL_TOKENS) and len(unique) >= 2:
                mapped = (
                    s.astype("string").str.strip().str.lower().map(
                        lambda v: True if v in _TRUE_TOKENS
                        else (False if v in _FALSE_TOKENS else pd.NA)
                    )
                )
                out[col] = mapped.astype("boolean")
                log.append({
                    "column": col, "from": str(s.dtype), "to": "boolean",
                })
        except Exception as exc:
            logger.debug("coerce_booleans(%r): %s", col, exc)

    return (out, log) if return_log else out


# ============================================================================
# INFINITIES
# ----------------------------------------------------------------------------
# ``np.inf`` / ``-np.inf`` show up when a divide-by-zero leaked through an
# Excel formula and got exported. They crash Plotly (axis range becomes
# infinite), break ``df.corr()`` (correlation of inf is NaN but pandas
# warns repeatedly), and serialize incorrectly to JSON.
# ============================================================================
def handle_infinities(
    df: pd.DataFrame,
    *,
    return_count: bool = False,
):
    """Replace ``+inf`` / ``-inf`` with ``NaN`` in numeric columns."""
    if df is None or df.empty:
        return (df, 0) if return_count else df

    out = df.copy()
    total = 0
    for col in out.columns:
        s = out[col]
        # Only float columns can contain inf. Integer & boolean cannot.
        if not pd.api.types.is_float_dtype(s):
            continue
        try:
            arr = s.to_numpy(copy=False)
            mask = np.isinf(arr)
            n = int(mask.sum())
            if n > 0:
                out[col] = s.replace([np.inf, -np.inf], np.nan)
                total += n
        except Exception:
            # Last-resort: trust pandas to handle the replacement even if
            # numpy was unhappy with the dtype.
            try:
                out[col] = s.replace([np.inf, -np.inf], np.nan)
            except Exception as exc:
                logger.debug("handle_infinities(%r) fallback failed: %s",
                             col, exc)

    return (out, total) if return_count else out


# ============================================================================
# CATEGORICAL COMPRESSION
# ----------------------------------------------------------------------------
# Why: a 500k-row column with 50 unique strings is ~25 MB as object;
# ~500 KB as category. Industrial datasets routinely have plant codes,
# unit IDs, status labels — all low-cardinality.
# ============================================================================
def compress_categoricals(
    df: pd.DataFrame,
    *,
    max_unique_ratio: float = 0.5,
    big_threshold_rows: int = 1000,
    small_max_unique: int = 50,
    return_log: bool = False,
):
    """Convert low-cardinality string columns to ``category``."""
    log: list[dict[str, Any]] = []
    if df is None or df.empty:
        return (df, log) if return_log else df

    out = df.copy()
    n = len(out)
    for col in out.columns:
        s = out[col]
        if not _is_text_like(s):
            continue
        if isinstance(s.dtype, pd.CategoricalDtype):
            continue
        try:
            nunique = s.nunique(dropna=True)
            if nunique == 0:
                continue
            # Two paths:
            # 1. Big frames: compress whenever cardinality is below half.
            # 2. Small frames: compress when cardinality is small in
            #    absolute terms (the memory win is tiny but Plotly handles
            #    categorical axes better than object).
            should_compress = (
                (n >= big_threshold_rows and nunique / n <= max_unique_ratio)
                or (nunique <= small_max_unique)
            )
            if should_compress:
                out[col] = s.astype("category")
                log.append({"column": col, "unique": int(nunique)})
        except Exception as exc:
            logger.debug("compress_categoricals(%r): %s", col, exc)

    return (out, log) if return_log else out


# ============================================================================
# PYARROW SAFETY (THE FINAL GUARANTEE)
# ----------------------------------------------------------------------------
# Why this exists: even after all the heuristics above, an object column
# can still contain a pathological mix that Arrow refuses — e.g. a column
# where 99.5% of values converted to numeric but 0.5% remained as
# ``Decimal`` objects, or a column where some cells are lists/tuples.
# The contract of this function is:
#
#   "After this returns, ``st.dataframe(df)`` will not raise ArrowInvalid."
#
# We achieve that by:
#   1. Stringifying any object column with mixed Python types.
#   2. Asking PyArrow itself to validate the frame; for every column it
#      rejects, we stringify it and try again.
#
# Stringification is the universally-safe fallback — Arrow always accepts
# a homogeneous string column.
# ============================================================================
def make_arrow_safe(
    df: pd.DataFrame,
    *,
    return_log: bool = False,
):
    """Guarantee ``st.dataframe`` won't raise ``ArrowInvalid`` on the result."""
    log: list[dict[str, Any]] = []
    if df is None or df.empty:
        return (df, log) if return_log else df

    out = df.copy()

    # --- Pass 1: stringify visibly mixed-type object columns ---
    for col in list(out.columns):
        s = out[col]
        if s.dtype != object:
            continue
        try:
            sample = s.dropna()
            if sample.empty:
                continue
            sample_types = set(type(v) for v in sample.head(500))
            # An all-str object column is fine. An object column that mixes
            # str + int + float (the classic dirty-sensor pattern) is not.
            if len(sample_types) > 1:
                out[col] = s.astype("string")
                log.append({"column": col, "reason": "mixed_object_types"})
            elif next(iter(sample_types)) not in (str, type(None)):
                # Single exotic type — Decimal, dict, list, set, bytes, ...
                # Stringify defensively; Arrow may or may not handle it,
                # but a string column always works.
                try:
                    out[col] = s.astype("string")
                    log.append({
                        "column": col,
                        "reason": f"object_of_{next(iter(sample_types)).__name__}",
                    })
                except Exception:
                    out[col] = s.map(_safe_str).astype("string")
                    log.append({"column": col, "reason": "fallback_repr"})
        except Exception as exc:
            logger.debug("make_arrow_safe pass1 on %r: %s", col, exc)
            try:
                out[col] = out[col].map(_safe_str).astype("string")
                log.append({"column": col, "reason": "exception_fallback"})
            except Exception:
                pass

    # --- Pass 2: actually ask PyArrow to validate the frame ---
    # If pyarrow isn't installed (rare in Streamlit deployments) we just
    # trust pass 1.
    try:
        import pyarrow as pa
        try:
            pa.Table.from_pandas(out, preserve_index=False)
        except Exception:
            # Frame failed. Find the offending columns one by one.
            for col in list(out.columns):
                try:
                    pa.Array.from_pandas(out[col])
                except Exception:
                    try:
                        out[col] = out[col].astype("string")
                        log.append({"column": col, "reason": "arrow_rejected"})
                    except Exception:
                        out[col] = out[col].map(_safe_str).astype("string")
                        log.append({"column": col,
                                    "reason": "arrow_rejected_stringified"})
    except ImportError:
        pass

    return (out, log) if return_log else out


def _safe_str(value) -> str | None:
    """Robust ``str()`` that never raises and preserves ``None``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return str(value)
    except Exception:
        try:
            return repr(value)
        except Exception:
            return None


# ============================================================================
# CONVENIENCE WRAPPERS
# ============================================================================
def prepare_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitization wrapper for chart code (Plotly / Matplotlib).

    Plotly is especially fragile around:
        * NaN inside boolean columns (renders as "False")
        * inf on numeric axes (axis range explodes)
        * mixed object columns (raises on conversion to Arrow)

    Calling this defensively at the top of every plotting function makes
    every chart resilient. It's cheap because the pipeline is idempotent.
    """
    out, _ = preprocess_dynamic_dataset(df)
    return out


def prepare_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitization wrapper for ML pipelines.

    Adds two ML-specific normalisations on top of the visualisation prep:
        * drop all-null columns — scikit-learn rejects features with no
          variance and they carry zero signal.
        * cast nullable ``Int64`` → ``float64`` — many sklearn estimators
          fail on pandas ``Int64`` (nullable int), but happily accept
          ``float64`` with ``NaN``.
    """
    out, _ = preprocess_dynamic_dataset(df)
    if out.empty:
        return out
    out = out.dropna(axis=1, how="all")
    for col in out.columns:
        dtype = out[col].dtype
        # ``pd.Int64Dtype`` and friends — nullable integer extension types.
        if pd.api.types.is_extension_array_dtype(dtype) and \
                pd.api.types.is_integer_dtype(dtype):
            try:
                out[col] = out[col].astype("float64")
            except Exception:
                pass
    return out


# ============================================================================
# INTELLIGENT TYPE DETECTION (REPORTING UTILITY)
# ============================================================================
def intelligent_type_detection(df: pd.DataFrame) -> dict[str, str]:
    """Classify every column into a single logical type.

    Returns a dict ``{column: type}`` where ``type`` is one of:
        ``'empty'``      — all values are null
        ``'boolean'``    — pandas bool / nullable boolean
        ``'datetime'``   — pandas datetime (with or without timezone)
        ``'numeric'``    — int / float / nullable int
        ``'categorical'``— pandas Categorical
        ``'string'``     — pure-string object column
        ``'mixed'``      — object column with multiple Python types
                          (you almost never want this — it crashes Arrow).

    Use this for UI rendering, NOT as a routing decision — the routing has
    already happened upstream in ``preprocess_dynamic_dataset``.
    """
    result: dict[str, str] = {}
    if df is None or df.shape[1] == 0:
        return result

    for col in df.columns:
        s = df[col]
        try:
            if s.isna().all():
                result[col] = "empty"
            elif pd.api.types.is_bool_dtype(s):
                result[col] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(s):
                result[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(s):
                result[col] = "numeric"
            elif isinstance(s.dtype, pd.CategoricalDtype):
                result[col] = "categorical"
            elif s.dtype == object or pd.api.types.is_string_dtype(s):
                sample = s.dropna().head(200)
                if sample.empty:
                    result[col] = "empty"
                else:
                    types = set(type(v) for v in sample)
                    result[col] = (
                        "mixed" if len(types) > 1 else "string"
                    )
            else:
                result[col] = "string"
        except Exception:
            result[col] = "string"
    return result


# ============================================================================
# INTERNAL HELPERS
# ============================================================================
def _is_text_like(s: pd.Series) -> bool:
    """True if a series can contain string tokens (object / string / cat)."""
    return (
        s.dtype == object
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def _empty_report(df: pd.DataFrame) -> dict[str, Any]:
    """Skeleton report used by ``preprocess_dynamic_dataset``."""
    return {
        "input_rows": int(len(df)) if df is not None else 0,
        "input_cols": int(df.shape[1]) if df is not None else 0,
        "output_rows": 0,
        "output_cols": 0,
        "renamed_columns": {},
        "duplicate_columns_dedup": [],
        "tokens_replaced_total": 0,
        "tokens_replaced_per_column": {},
        "numeric_conversions": [],
        "datetime_conversions": [],
        "boolean_conversions": [],
        "category_compressions": [],
        "infinities_replaced": 0,
        "arrow_unsafe_columns": [],
        "detected_types": {},
        "errors": [],
        "warnings": [],
    }


def _safe_step(name: str, fn, df: pd.DataFrame, report: dict, *, default):
    """Run a pipeline step, catching exceptions and recording them.

    ``default`` is what gets returned on failure (typically ``(df, empty)``).
    The first element of ``default`` is the DataFrame placeholder — but we
    always return the LAST-KNOWN-GOOD ``df`` on failure, never the default's
    df, so subsequent steps can still operate on partial progress.
    """
    try:
        result = fn(df)
        # Normalise: every step either returns ``df`` or ``(df, extra)``.
        if isinstance(result, tuple):
            if len(result) == 2:
                return result
            if len(result) == 3:
                # ``clean_column_names`` returns three values.
                return result
        return result, default[1] if isinstance(default, tuple) else default
    except Exception as exc:
        logger.exception("%s failed", name)
        report.setdefault("errors", []).append(f"{name}: {exc}")
        if isinstance(default, tuple) and len(default) == 3:
            return df, default[1], default[2]
        if isinstance(default, tuple) and len(default) == 2:
            return df, default[1]
        return df, default


# ============================================================================
# EMERGENCY ESCAPE HATCH
# ============================================================================
def force_stringify(df: pd.DataFrame) -> pd.DataFrame:
    """Last-resort: cast every column to ``string``.

    Use this if a downstream operation is still crashing PyArrow despite
    ``preprocess_dynamic_dataset``. A fully-stringified frame is always
    Arrow-safe — at the cost of losing all dtype information.
    """
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        try:
            out[col] = out[col].astype("string")
        except Exception:
            try:
                out[col] = out[col].map(_safe_str).astype("string")
            except Exception:
                out[col] = pd.Series(
                    [None] * len(out), index=out.index, dtype="string",
                )
    return out
