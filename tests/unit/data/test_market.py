from datetime import UTC, date, datetime
from decimal import Decimal

from equity_research.data.market import MarketDataClient


def test_market_data_client_parses_daily_history(load_fixture, fixture_http):
    client = MarketDataClient(http=fixture_http(load_fixture("market/msft_daily.json")))
    series = client.history("MSFT", start=date(2024, 1, 1), end=date(2024, 1, 5))

    assert series.ticker == "MSFT"
    assert series.source == "yahoo_finance_delayed"
    assert len(series.points) == 2

    first = series.points[0]
    assert first.session_date == datetime.fromtimestamp(1704153600, tz=UTC).date()
    assert first.close == first.adjusted_close == Decimal("373.9")
    assert first.volume == 21500000


def test_market_data_client_never_marks_prices_as_real_time(load_fixture, fixture_http):
    client = MarketDataClient(http=fixture_http(load_fixture("market/msft_daily.json")))
    series = client.history("MSFT", start=date(2024, 1, 1), end=date(2024, 1, 5))
    assert "delayed" in series.source
