"""Market-data adapter: explicitly delayed/end-of-day daily price history.

Prices are never presented as real-time. Each `PriceSeries` carries its
source and retrieval timestamp separately from each `PricePoint`'s market
session date, so a report can show them distinctly per the design spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Protocol

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


class HttpResponse(Protocol):
    status_code: int

    def json(self) -> dict: ...


class HttpClient(Protocol):
    def get(
        self,
        url: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class PricePoint:
    session_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adjusted_close: Decimal
    volume: int


@dataclass(frozen=True)
class PriceSeries:
    ticker: str
    points: list[PricePoint]
    source: str
    retrieved_at: datetime


class MarketDataClient:
    """Daily adjusted price history, sourced from Yahoo Finance's chart API."""

    def __init__(self, http: HttpClient, source_label: str = "yahoo_finance_delayed"):
        self._http = http
        self._source_label = source_label

    def history(self, ticker: str, start: date, end: date) -> PriceSeries:
        params = {
            "period1": _to_epoch(start),
            "period2": _to_epoch(end),
            "interval": "1d",
        }
        url = CHART_URL.format(ticker=ticker)
        payload = self._http.get(url, params=params).json()
        result = payload["chart"]["result"][0]

        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])

        points = [
            PricePoint(
                session_date=datetime.fromtimestamp(ts, tz=UTC).date(),
                open=Decimal(str(quote["open"][i])),
                high=Decimal(str(quote["high"][i])),
                low=Decimal(str(quote["low"][i])),
                close=Decimal(str(quote["close"][i])),
                adjusted_close=Decimal(str(adjclose[i])),
                volume=int(quote["volume"][i]),
            )
            for i, ts in enumerate(timestamps)
        ]
        return PriceSeries(
            ticker=ticker,
            points=points,
            source=self._source_label,
            retrieved_at=datetime.now(UTC),
        )


def _to_epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())
