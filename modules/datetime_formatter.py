"""Custom datetime formatter for industrial / IoT sensor datasets.

The default ``pd.to_datetime`` inference is flaky on real-world sensor data:
mixed separators, compact strings like ``20250512142245``, milliseconds with
``.`` vs ``,``, day-first European formats, etc. This module gives the user
an Excel-style format builder plus suggestion + preview helpers so they can
disambiguate manually when auto-detection is not confident.

Also covers **numeric datetime encodings** that show up in industrial /
SCADA / IoT exports: Excel serial dates (origin 1899-12-30, fractional days),
Unix seconds, and Unix milliseconds. Auto-detection picks the most plausible
encoding based on the value range.

Public API:
    builder_to_python(builder_format)            -> python strptime format
    parse_with_format(series, py_format)         -> (parsed_series, stats)
    suggest_formats(series, top_n)               -> ranked list of format dicts
    auto_detect_format(series)                   -> best suggestion or None
    combine_date_time(date_s, time_s, ...)       -> (combined_series, stats)
    preview_frame(original, parsed, n)           -> DataFrame for live preview
    detect_numeric_datetime_mode(series)         -> (mode, scores)
    detect_numeric_datetime_columns(df)          -> list of candidate dicts
    convert_numeric_datetime(series, mode)       -> (parsed_series, stats)

Token convention (case is meaningful — same as moment.js / Java SimpleDateFormat):
    YYYY  4-digit year      %Y          mm   minute        %M
    YY    2-digit year      %y          ss   second        %S
    MM    month             %m          ms   microsecond   %f
    DD    day               %d
    HH    hour 24h          %H
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


# Order matters: longer tokens must be substituted before their prefixes
# (otherwise YY would eat the front of YYYY). Case is significant — MM is
# month, mm is minute, in line with the common builder convention.
TOKEN_MAP: list[tuple[str, str]] = [
    ("YYYY", "%Y"),
    ("YY", "%y"),
    ("MM", "%m"),
    ("DD", "%d"),
    ("HH", "%H"),
    ("hh", "%H"),
    ("mm", "%M"),
    ("ss", "%S"),
    ("ms", "%f"),
]

# Surfaced as picker chips in the UI.
BUILDER_TOKENS = ["YYYY", "MM", "DD", "HH", "mm", "ss", "ms"]
BUILDER_SEPARATORS = ["-", "/", " ", ":", ".", "T", "_"]

# Industrial / IoT presets ordered roughly by how often they show up in real
# datasets. Label is the *builder* form so it round-trips with the format
# builder text input.
FORMAT_PRESETS: list[tuple[str, str]] = [
    ("YYYY-MM-DD HH:mm:ss",    "%Y-%m-%d %H:%M:%S"),
    ("YYYY-MM-DD HH:mm:ss.ms", "%Y-%m-%d %H:%M:%S.%f"),
    ("DD-MM-YYYY HH:mm:ss",    "%d-%m-%Y %H:%M:%S"),
    ("DD-MM-YYYY HH.mm.ss",    "%d-%m-%Y %H.%M.%S"),
    ("DD-MM-YYYY HH:mm:ss.ms", "%d-%m-%Y %H:%M:%S.%f"),
    ("DD/MM/YYYY HH:mm:ss",    "%d/%m/%Y %H:%M:%S"),
    ("YYYY/MM/DD HH-mm-ss",    "%Y/%m/%d %H-%M-%S"),
    ("YYYY/MM/DD HH:mm:ss",    "%Y/%m/%d %H:%M:%S"),
    ("MM/DD/YYYY HH:mm:ss",    "%m/%d/%Y %H:%M:%S"),
    ("YYYY-MM-DDTHH:mm:ss",    "%Y-%m-%dT%H:%M:%S"),
    ("YYYY-MM-DD",             "%Y-%m-%d"),
    ("DD-MM-YYYY",             "%d-%m-%Y"),
    ("DD/MM/YYYY",             "%d/%m/%Y"),
    ("MM/DD/YYYY",             "%m/%d/%Y"),
    ("HH:mm:ss",               "%H:%M:%S"),
    ("HH:mm:ss.ms",            "%H:%M:%S.%f"),
    ("YYYYMMDDHHmmss",         "%Y%m%d%H%M%S"),
    ("YYYYMMDDHHmm",           "%Y%m%d%H%M"),
    ("YYYYMMDD",               "%Y%m%d"),
]

_NULL_STRINGS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "nat"}


def builder_to_python(builder_format: str) -> str:
    """Translate ``YYYY-MM-DD HH:mm:ss`` → ``%Y-%m-%d %H:%M:%S``.

    Anything not matching a token in :data:`TOKEN_MAP` passes through
    unchanged, so users can freely mix tokens with their own separators.
    """
    if not builder_format:
        return ""
    out = builder_format
    for token, code in TOKEN_MAP:
        out = out.replace(token, code)
    return out


def _to_clean_strings(series: pd.Series) -> pd.Series:
    """Return a stripped string series with common null tokens replaced by NA."""
    if pd.api.types.is_datetime64_any_dtype(series):
        # Already datetime — return ISO-formatted strings so callers can
        # always reason about strings downstream.
        return series.astype("string")
    s = series.astype("string").str.strip()
    lowered = s.str.lower()
    return s.mask(lowered.isin(_NULL_STRINGS))


def parse_with_format(
    series: pd.Series,
    python_format: str | None,
) -> tuple[pd.Series, dict]:
    """Parse ``series`` with ``python_format``; ``None``/empty → pandas inference.

    Returns the parsed datetime series plus a stats dict:
        total      : total rows in the input series
        non_null   : non-null inputs (denominator for success %)
        parsed     : rows that produced a valid Timestamp
        failed     : non-null inputs that became NaT
        percent    : parsed / non_null * 100, rounded to one decimal
    """
    cleaned = _to_clean_strings(series)
    total = int(len(cleaned))
    non_null = int(cleaned.notna().sum())

    try:
        if python_format:
            parsed = pd.to_datetime(cleaned, errors="coerce", format=python_format)
        else:
            try:
                parsed = pd.to_datetime(cleaned, errors="coerce", format="mixed")
            except (TypeError, ValueError):
                parsed = pd.to_datetime(cleaned, errors="coerce")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(cleaned, errors="coerce")

    parsed_count = int(parsed.notna().sum())
    failed = max(0, non_null - parsed_count)
    percent = (parsed_count / non_null * 100) if non_null else 0.0
    return parsed, {
        "total": total,
        "non_null": non_null,
        "parsed": parsed_count,
        "failed": failed,
        "percent": round(percent, 1),
    }


# Patterns pandas does not natively pick up — surfaced into the suggestion
# list ahead of the regular presets when every sample value matches.
_COMPACT_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^\d{14}$"), "YYYYMMDDHHmmss", "%Y%m%d%H%M%S"),
    (re.compile(r"^\d{12}$"), "YYYYMMDDHHmm",   "%Y%m%d%H%M"),
    (re.compile(r"^\d{8}$"),  "YYYYMMDD",       "%Y%m%d"),
]


def _compact_format_hint(sample: Iterable[str]) -> tuple[str, str] | None:
    """Return ``(label, python_format)`` if every sample is a compact digit run."""
    values = [str(v).strip() for v in sample if v is not None and str(v).strip()]
    if not values:
        return None
    for pattern, label, fmt in _COMPACT_PATTERNS:
        if all(pattern.match(v) for v in values):
            return label, fmt
    return None


_SUGGEST_SAMPLE_SIZE = 250
_SUGGEST_THRESHOLD = 0.5  # keep formats that parse at least 50% of the sample


def suggest_formats(series: pd.Series, top_n: int = 5) -> list[dict]:
    """Rank :data:`FORMAT_PRESETS` by parse success rate on a sample.

    Each entry: ``{"label", "python", "percent", "parsed", "non_null"}``.
    Empty list if the column is entirely null. Compact digit-string formats
    are surfaced ahead of the presets when applicable.
    """
    cleaned = _to_clean_strings(series).dropna()
    if cleaned.empty:
        return []

    if len(cleaned) > _SUGGEST_SAMPLE_SIZE:
        cleaned = cleaned.sample(_SUGGEST_SAMPLE_SIZE, random_state=42)
    non_null = int(len(cleaned))

    presets = list(FORMAT_PRESETS)
    compact = _compact_format_hint(cleaned.tolist())
    if compact and compact not in presets:
        presets.insert(0, compact)

    results: list[dict] = []
    seen_fmts: set[str] = set()
    for label, fmt in presets:
        if fmt in seen_fmts:
            continue
        seen_fmts.add(fmt)
        try:
            parsed = pd.to_datetime(cleaned, errors="coerce", format=fmt)
        except (TypeError, ValueError):
            continue
        parsed_count = int(parsed.notna().sum())
        rate = parsed_count / non_null if non_null else 0.0
        if rate >= _SUGGEST_THRESHOLD:
            results.append({
                "label": label,
                "python": fmt,
                "percent": round(rate * 100, 1),
                "parsed": parsed_count,
                "non_null": non_null,
            })
    results.sort(key=lambda r: r["percent"], reverse=True)
    return results[:top_n]


def auto_detect_format(series: pd.Series) -> dict | None:
    """Best single-format guess. Returns the top suggestion or ``None``."""
    suggestions = suggest_formats(series, top_n=1)
    return suggestions[0] if suggestions else None


def combine_date_time(
    date_series: pd.Series,
    time_series: pd.Series,
    date_format: str | None = None,
    time_format: str | None = None,
) -> tuple[pd.Series, dict]:
    """Merge a separate date column + time column into one datetime series.

    Behaviour:
    - If the date column is already datetime, its date component is kept and
      the time column is parsed as a timedelta (so it can be added).
    - Otherwise both are coerced to strings, concatenated with a space, and
      parsed with the combined format (or pandas inference if either is None).
    - Missing time values default to midnight rather than dropping the row.
    """
    if pd.api.types.is_datetime64_any_dtype(date_series):
        date_part = date_series.dt.normalize()
        time_strs_raw = _to_clean_strings(time_series)
        # Non-null time values that fail to parse as a timedelta count as
        # failures — without this, every unparseable time silently becomes
        # midnight and the user has no way of knowing the merge was lossy.
        time_td = pd.to_timedelta(time_strs_raw, errors="coerce")
        time_failed_mask = time_strs_raw.notna() & time_td.isna()
        combined = date_part + time_td.fillna(pd.Timedelta(0))
        combined = combined.where(date_series.notna() & ~time_failed_mask)
        non_null = int(date_series.notna().sum())
        parsed_count = int(combined.notna().sum())
        failed = max(0, non_null - parsed_count)
        pct = (parsed_count / non_null * 100) if non_null else 0.0
        return combined, {
            "total": int(len(date_series)),
            "non_null": non_null,
            "parsed": parsed_count,
            "failed": failed,
            "percent": round(pct, 1),
        }

    date_strs = _to_clean_strings(date_series)
    time_strs = _to_clean_strings(time_series).fillna("00:00:00")
    joined = (date_strs + " " + time_strs).where(date_strs.notna())

    if date_format and time_format:
        fmt = f"{date_format} {time_format}"
    else:
        fmt = None
    return parse_with_format(joined, fmt)


def preview_frame(
    original: pd.Series,
    parsed: pd.Series,
    n: int = 10,
) -> pd.DataFrame:
    """Side-by-side ``Raw Value`` / ``Parsed Datetime`` table for the UI."""
    raw = original.head(n).astype("string").fillna("(null)")
    parsed_head = parsed.head(n)
    # NaT renders as <NA> when cast to string — replace with a friendlier label.
    parsed_str = parsed_head.astype("string").fillna("(unparsed)")
    return pd.DataFrame({"Raw Value": raw.values, "Parsed Datetime": parsed_str.values})


# ---------------------------------------------------------------------------
# Numeric datetime encodings (Excel serial, Unix seconds, Unix milliseconds)
# ---------------------------------------------------------------------------
#
# Industrial datasets routinely ship date/time columns as numbers — Excel
# exports use fractional days since 1899-12-30, IoT brokers use Unix epochs.
# Pandas treats them as plain numeric columns, so datetime detection silently
# misses them. We score each numeric column against the plausible range for
# every encoding and surface the strongest match.

# Excel serial bounds: ~1927-05-19 to ~2118-12-15. Wide enough to catch every
# industrial dataset, narrow enough to avoid false positives on counters that
# happen to drift into the tens of thousands.
_EXCEL_RANGE = (10_000.0, 80_000.0)

# Unix seconds: 2000-01-01 to 2100-01-01 inclusive. Anything in this band
# looks far more like a timestamp than like sensor readings.
_UNIX_S_RANGE = (946_684_800.0, 4_102_444_800.0)

# Unix milliseconds: same bounds scaled by 1000.
_UNIX_MS_RANGE = (946_684_800_000.0, 4_102_444_800_000.0)

# Share of non-null values that must fall in the plausible range before a
# mode is flagged as a likely candidate.
_NUMERIC_DT_THRESHOLD = 0.8

NUMERIC_DT_MODES: dict[str, str] = {
    "excel_serial": "Excel Serial Datetime",
    "unix_s":       "Unix Timestamp (seconds)",
    "unix_ms":      "Unix Timestamp (milliseconds)",
}


def _share_in_range(series: pd.Series, low: float, high: float) -> float:
    if series.empty:
        return 0.0
    return float(((series >= low) & (series <= high)).mean())


def detect_numeric_datetime_mode(
    series: pd.Series,
) -> tuple[str | None, dict[str, float]]:
    """Guess the encoding of a numeric datetime column.

    Returns ``(mode, scores)``:
      - ``mode`` is one of the keys of :data:`NUMERIC_DT_MODES`, or ``None``
        if no encoding crosses :data:`_NUMERIC_DT_THRESHOLD`.
      - ``scores`` maps every candidate mode to the percentage of non-null
        values that fall in its plausible range. Always populated so the UI
        can show the user *why* a mode was (or wasn't) suggested.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    scores = {
        "excel_serial": round(_share_in_range(numeric, *_EXCEL_RANGE) * 100, 1),
        "unix_s":       round(_share_in_range(numeric, *_UNIX_S_RANGE) * 100, 1),
        "unix_ms":      round(_share_in_range(numeric, *_UNIX_MS_RANGE) * 100, 1),
    }
    # Need a few real values to make any judgement.
    if numeric.empty or len(numeric) < 3:
        return None, scores
    best_mode = max(scores, key=scores.get)
    if scores[best_mode] >= _NUMERIC_DT_THRESHOLD * 100:
        return best_mode, scores
    return None, scores


def detect_numeric_datetime_columns(df: pd.DataFrame) -> list[dict]:
    """Scan numeric columns for likely datetime encodings.

    Each entry: ``{"column", "mode", "scores"}``. Boolean columns are
    skipped — they are numeric in pandas but never timestamps. Empty list
    if nothing crosses the threshold.
    """
    out: list[dict] = []
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            continue
        mode, scores = detect_numeric_datetime_mode(s)
        if mode is None:
            continue
        out.append({"column": col, "mode": mode, "scores": scores})
    return out


def convert_numeric_datetime(
    series: pd.Series,
    mode: str,
) -> tuple[pd.Series, dict]:
    """Vectorized Numeric → datetime conversion. Returns (parsed, stats).

    Modes:
      - ``excel_serial`` : days since 1899-12-30 (Excel epoch with the
        documented 1900 leap-year offset baked in via the origin shift).
      - ``unix_s``       : seconds since 1970-01-01.
      - ``unix_ms``      : milliseconds since 1970-01-01.
      - anything else    : falls back to pandas inference.

    All conversion paths use ``errors="coerce"`` so out-of-range or malformed
    values become NaT rather than raising. Stats follow the same shape as
    :func:`parse_with_format`.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    total = int(len(numeric))
    non_null = int(numeric.notna().sum())
    try:
        if mode == "excel_serial":
            parsed = pd.to_datetime(
                numeric, unit="D", origin="1899-12-30", errors="coerce",
            )
        elif mode == "unix_s":
            parsed = pd.to_datetime(numeric, unit="s", errors="coerce")
        elif mode == "unix_ms":
            parsed = pd.to_datetime(numeric, unit="ms", errors="coerce")
        else:
            parsed = pd.to_datetime(numeric, errors="coerce")
    except (TypeError, ValueError, OverflowError):
        parsed = pd.Series(pd.NaT, index=numeric.index, dtype="datetime64[ns]")

    parsed_count = int(parsed.notna().sum())
    failed = max(0, non_null - parsed_count)
    pct = (parsed_count / non_null * 100) if non_null else 0.0
    return parsed, {
        "total": total,
        "non_null": non_null,
        "parsed": parsed_count,
        "failed": failed,
        "percent": round(pct, 1),
    }
