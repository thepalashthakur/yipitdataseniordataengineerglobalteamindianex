"""Reusable source-field cleaning functions."""

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Optional, Tuple

from .config import (
    CATEGORY_MAPPING,
    CURRENCY_TO_USD,
    INDUSTRY_MAPPING,
    MISSING_MARKERS,
    UNDISCLOSED_MARKERS,
)
from .schemas import ArrParseResult, DateParseResult


_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
_SCALE_MULTIPLIERS = {
    "k": Decimal("1000"),
    "thousand": Decimal("1000"),
    "m": Decimal("1000000"),
    "million": Decimal("1000000"),
    "b": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
}
_AMOUNT_PATTERN = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?\s*$",
    re.IGNORECASE,
)


def _clean_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _detect_currency(text: str) -> Tuple[Optional[str], Optional[str]]:
    currencies = set()
    for symbol, currency in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            currencies.add(currency)
    for code in CURRENCY_TO_USD:
        if re.search(r"\b{}\b".format(code), text, flags=re.IGNORECASE):
            currencies.add(code)
    if len(currencies) > 1:
        return None, "multiple currency markers found"
    return (next(iter(currencies)) if currencies else "USD"), None


def _strip_currency(text: str) -> str:
    stripped = text
    for symbol in _CURRENCY_SYMBOLS:
        stripped = stripped.replace(symbol, "")
    stripped = re.sub(r"\b(?:USD|EUR|GBP|JPY)\b", "", stripped, flags=re.IGNORECASE)
    return stripped.replace(",", "").strip()


def _parse_amount_component(text: str) -> Tuple[Decimal, Optional[str]]:
    match = _AMOUNT_PATTERN.fullmatch(_strip_currency(text))
    if not match:
        raise ValueError("amount does not match a supported format")
    try:
        number = Decimal(match.group(1))
    except InvalidOperation as exc:
        raise ValueError("amount is not numeric") from exc
    if number < 0:
        raise ValueError("negative amounts are not valid ARR")
    scale = match.group(2).lower() if match.group(2) else None
    return number, scale


def parse_arr(value: Any) -> ArrParseResult:
    """Parse one messy revenue value into an integer USD ARR observation."""

    raw = _clean_optional_string(value)
    marker = "" if raw is None else raw.casefold()
    if marker in MISSING_MARKERS:
        return ArrParseResult(raw, None, None, None, None, "MISSING", "missing revenue")
    if marker in UNDISCLOSED_MARKERS:
        return ArrParseResult(raw, None, None, None, None, "UNDISCLOSED", "revenue not disclosed")

    assert raw is not None
    currency, currency_error = _detect_currency(raw)
    if currency_error:
        return ArrParseResult(raw, None, None, None, None, "INVALID_FORMAT", currency_error)

    parts = re.split(r"\s+-\s+", raw.strip())
    if len(parts) > 2:
        return ArrParseResult(raw, None, currency, None, None, "INVALID_FORMAT", "range has more than two endpoints")

    try:
        parsed = [_parse_amount_component(part) for part in parts]
        scales = [scale for _, scale in parsed if scale]
        inherited_scale = scales[0] if scales else None
        if len(set(scales)) > 1:
            raise ValueError("range endpoints use different scales")
        values = [number * _SCALE_MULTIPLIERS.get(scale or inherited_scale, Decimal("1")) for number, scale in parsed]
        lower = values[0]
        upper = values[-1]
        if len(values) == 2 and lower > upper:
            raise ValueError("range lower bound exceeds upper bound")
        source_amount = (lower + upper) / Decimal("2") if len(values) == 2 else lower
        converted = source_amount * CURRENCY_TO_USD[currency or "USD"]
        arr_usd = int(converted.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return ArrParseResult(
            raw,
            arr_usd,
            currency,
            str(lower),
            str(upper),
            "PARSED",
            None,
        )
    except (ValueError, InvalidOperation) as exc:
        return ArrParseResult(raw, None, currency, None, None, "INVALID_FORMAT", str(exc))


_DATE_FORMATS = (
    ("%Y-%m-%d", "ISO_DATE"),
    ("%Y-%m-%dT%H:%M:%SZ", "ISO_UTC_TIMESTAMP"),
    ("%m/%d/%Y", "US_SLASH"),
    ("%d-%m-%Y", "EU_DASH"),
    ("%d %b %Y", "DAY_ABBREVIATED_MONTH"),
    ("%B %d, %Y", "MONTH_NAME_DAY"),
)


def parse_published_date(value: Any) -> DateParseResult:
    """Parse supported source date shapes with a deterministic ambiguity policy."""

    raw = _clean_optional_string(value)
    if raw is None or raw.casefold() in MISSING_MARKERS:
        return DateParseResult(raw, None, None, None, None, "MISSING", "missing published date")
    for date_format, format_name in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(raw, date_format).date()
            reason = None
            if format_name == "US_SLASH":
                first, second, _ = raw.split("/")
                if int(first) <= 12 and int(second) <= 12:
                    reason = "ambiguous numeric date interpreted as MM/DD/YYYY"
            return DateParseResult(
                raw,
                parsed,
                parsed.year,
                ((parsed.month - 1) // 3) + 1,
                parsed.month,
                "PARSED",
                reason,
            )
        except ValueError:
            continue
    return DateParseResult(raw, None, None, None, None, "INVALID", "unsupported or invalid date")


def standardize_category(value: Any) -> str:
    raw = _clean_optional_string(value)
    if raw is None:
        return "UNMAPPED"
    return CATEGORY_MAPPING.get(raw.casefold(), "UNMAPPED")


def standardize_industry(value: Any) -> str:
    raw = _clean_optional_string(value)
    if raw is None:
        return "UNMAPPED"
    return INDUSTRY_MAPPING.get(raw.casefold(), "UNMAPPED")


def company_size_category(employee_count: Any) -> str:
    if employee_count is None:
        return "UNKNOWN"
    try:
        employees = int(employee_count)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if employees < 0:
        return "UNKNOWN"
    if employees < 10_000:
        return "SMALL"
    if employees <= 30_000:
        return "MEDIUM"
    return "LARGE"


def calculate_company_age(founded_year: Any, published_year: Any) -> Optional[int]:
    if founded_year is None or published_year is None:
        return None
    try:
        age = int(published_year) - int(founded_year)
    except (TypeError, ValueError):
        return None
    return age if age >= 0 else None

