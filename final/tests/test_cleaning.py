from datetime import date

import pytest

from yipit_pipeline.cleaning import (
    calculate_company_age,
    company_size_category,
    parse_arr,
    parse_published_date,
    standardize_category,
    standardize_industry,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5.2B", 5_200_000_000),
        ("$5,200,000,000", 5_200_000_000),
        ("5.2 billion", 5_200_000_000),
        ("500M USD", 500_000_000),
        ("$10M - $20M", 15_000_000),
        ("€2.1B", 2_310_000_000),
        ("£100M", 127_000_000),
        ("¥15B", 100_000_000),
        ("75000.0M USD", 75_000_000_000),
    ],
)
def test_parse_arr_supported_formats(raw, expected):
    result = parse_arr(raw)
    assert result.arr_parse_status == "PARSED"
    assert result.arr_usd == expected
    assert isinstance(result.arr_usd, int)


@pytest.mark.parametrize(
    ("raw", "status"),
    [(None, "MISSING"), ("", "MISSING"), ("N/A", "MISSING"), ("Not disclosed", "UNDISCLOSED")],
)
def test_parse_arr_non_observations(raw, status):
    result = parse_arr(raw)
    assert result.arr_parse_status == status
    assert result.arr_usd is None


@pytest.mark.parametrize("raw", ["garbage", "$-5M", "$20M - $10M", "€10M USD"])
def test_parse_arr_rejects_invalid_values(raw):
    assert parse_arr(raw).arr_parse_status == "INVALID_FORMAT"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2023-05-12", date(2023, 5, 12)),
        ("2023-05-12T00:00:00Z", date(2023, 5, 12)),
        ("05/12/2023", date(2023, 5, 12)),
        ("12-05-2023", date(2023, 5, 12)),
        ("12 May 2023", date(2023, 5, 12)),
        ("May 12, 2023", date(2023, 5, 12)),
    ],
)
def test_parse_date_supported_formats(raw, expected):
    result = parse_published_date(raw)
    assert result.date_parse_status == "PARSED"
    assert result.published_date == expected
    assert result.published_year == 2023
    assert result.published_quarter == 2
    assert result.published_month == 5


def test_ambiguous_numeric_date_uses_us_policy():
    result = parse_published_date("04/05/2023")
    assert result.published_date == date(2023, 4, 5)
    assert "MM/DD/YYYY" in result.date_parse_reason


@pytest.mark.parametrize("raw", [None, "", "2023-02-30", "31/12/2023"])
def test_invalid_or_missing_dates_are_null(raw):
    assert parse_published_date(raw).published_date is None


def test_taxonomy_and_derived_company_fields():
    assert standardize_category("Artificial Intelligence") == "AI_ML"
    assert standardize_category("Cloud Services") == "CLOUD"
    assert standardize_category("new category") == "UNMAPPED"
    assert standardize_industry("AI/ML") == "AI_ML"
    assert company_size_category(9_999) == "SMALL"
    assert company_size_category(10_000) == "MEDIUM"
    assert company_size_category(30_000) == "MEDIUM"
    assert company_size_category(30_001) == "LARGE"
    assert calculate_company_age(2010, 2023) == 13
    assert calculate_company_age(2024, 2023) is None

