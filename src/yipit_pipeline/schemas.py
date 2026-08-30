"""Small typed result objects shared by cleaning and matching functions."""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class ArrParseResult:
    revenue_raw: Optional[str]
    arr_usd: Optional[int]
    source_currency: Optional[str]
    lower_bound_source: Optional[str]
    upper_bound_source: Optional[str]
    arr_parse_status: str
    arr_parse_reason: Optional[str]


@dataclass(frozen=True)
class DateParseResult:
    published_date_raw: Optional[str]
    published_date: Optional[date]
    published_year: Optional[int]
    published_quarter: Optional[int]
    published_month: Optional[int]
    date_parse_status: str
    date_parse_reason: Optional[str]


@dataclass(frozen=True)
class CompanyResolution:
    company_name_raw: str
    company_id: str
    company_name_canonical: Optional[str]
    company_entity_name: str
    company_match_status: str
    company_match_method: str
    company_match_confidence: float
    has_company_metadata: bool

