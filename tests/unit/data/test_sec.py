import pytest

from equity_research.core.exceptions import RequiredSourceUnavailableError, UnsupportedTickerError
from equity_research.data.sec import SecClient


def test_sec_client_normalizes_revenue_fact(load_fixture, fixture_http):
    client = SecClient(http=fixture_http(load_fixture("sec/companyfacts_msft.json")))
    fact = client.latest_annual_fact("0000789019", "Revenues")
    assert fact.unit == "USD"
    assert fact.fiscal_year == 2024
    assert fact.form == "10-K"
    assert fact.value == 245122000000


def test_latest_annual_fact_ignores_quarterly_observations(load_fixture, fixture_http):
    # The fixture also contains a Q2 10-Q entry for FY2024; it must not win
    # over the FY2023 annual entry if a newer annual fact didn't exist.
    client = SecClient(http=fixture_http(load_fixture("sec/companyfacts_msft.json")))
    fact = client.latest_annual_fact("0000789019", "Revenues")
    assert fact.fiscal_period == "FY"


def test_annual_facts_returns_most_recent_first(load_fixture, fixture_http):
    client = SecClient(http=fixture_http(load_fixture("sec/companyfacts_msft.json")))
    facts = client.annual_facts("0000789019", "Revenues", limit=2)
    assert [f.fiscal_year for f in facts] == [2024, 2023]
    assert facts[1].value == 211915000000


def test_annual_facts_limit_one_matches_latest_annual_fact(load_fixture, fixture_http):
    client = SecClient(http=fixture_http(load_fixture("sec/companyfacts_msft.json")))
    assert client.annual_facts("0000789019", "Revenues", limit=1) == [
        client.latest_annual_fact("0000789019", "Revenues")
    ]


def test_latest_annual_fact_raises_for_missing_tag(load_fixture, fixture_http):
    client = SecClient(http=fixture_http(load_fixture("sec/companyfacts_msft.json")))
    with pytest.raises(RequiredSourceUnavailableError):
        client.latest_annual_fact("0000789019", "NetIncomeLoss")


def test_resolve_cik_finds_ticker(load_fixture, fixture_http):
    client = SecClient(http=fixture_http(load_fixture("sec/company_tickers.json")))
    assert client.resolve_cik("msft") == "0000789019"


def test_resolve_cik_raises_for_unknown_ticker(load_fixture, fixture_http):
    client = SecClient(http=fixture_http(load_fixture("sec/company_tickers.json")))
    with pytest.raises(UnsupportedTickerError):
        client.resolve_cik("ZZZZ")


def test_filings_builds_archive_urls(load_fixture, fixture_http):
    client = SecClient(http=fixture_http(load_fixture("sec/submissions_msft.json")))
    filings = client.filings("0000789019")
    assert filings[0].form == "10-K"
    assert filings[0].accession == "0000789019-24-000012"
    assert filings[0].url == (
        "https://www.sec.gov/Archives/edgar/data/789019/000078901924000012/msft-20240630.htm"
    )
