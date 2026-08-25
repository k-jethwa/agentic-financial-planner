"""Fake, network-free implementations of every research-graph I/O boundary.

Shared by unit tests (`tests/unit/agents/test_graph.py`), integration tests
(`tests/integration/api/test_runs.py`), and, later, the recorded evaluation
harness — one canned MSFT-shaped dataset instead of three copies of it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from equity_research.agents.graph import GraphDependencies
from equity_research.core.exceptions import RequiredSourceUnavailableError, UnsupportedTickerError
from equity_research.data.market import PricePoint, PriceSeries
from equity_research.data.news import NewsItem
from equity_research.data.sec import Filing, NormalizedFact
from equity_research.retrieval.vector_store import InMemoryVectorStore

CIK = "0000789019"

ANNUAL_FACTS = {
    "Revenues": [("245122000000", 2024), ("211915000000", 2023)],
    "NetCashProvidedByUsedInOperatingActivities": [("118548000000", 2024)],
    "PaymentsToAcquirePropertyPlantAndEquipment": [("44477000000", 2024)],
}

SAMPLE_FILING_HTML = """
<html><body>
<h1>Item 1A. Risk Factors</h1>
<p>Supply chain disruption could materially affect delivery timelines.</p>
</body></html>
"""


class FakeSecClient:
    def __init__(self, *, resolves: bool = True):
        self._resolves = resolves

    def resolve_cik(self, ticker: str) -> str:
        if not self._resolves:
            raise UnsupportedTickerError(ticker, "not found in fake index")
        return CIK

    def filings(self, cik: str) -> list[Filing]:
        return [
            Filing(
                accession="0000789019-24-000012",
                form="10-K",
                filing_date=date(2024, 7, 30),
                url="https://www.sec.gov/Archives/edgar/data/789019/msft-20240630.htm",
            )
        ]

    def annual_facts(self, cik: str, tag: str, taxonomy: str = "us-gaap", *, limit: int = 1):
        return [self._fact(tag, value, fy) for value, fy in ANNUAL_FACTS.get(tag, [])[:limit]]

    def latest_annual_fact(self, cik: str, tag: str, taxonomy: str = "us-gaap") -> NormalizedFact:
        facts = self.annual_facts(cik, tag, limit=1)
        if not facts:
            raise RequiredSourceUnavailableError(f"companyfacts:{cik}", f"missing tag {tag}")
        return facts[0]

    @staticmethod
    def _fact(tag: str, value: str, fiscal_year: int) -> NormalizedFact:
        return NormalizedFact(
            tag=tag,
            taxonomy="us-gaap",
            unit="USD",
            value=Decimal(value),
            fiscal_year=fiscal_year,
            fiscal_period="FY",
            period_start=date(fiscal_year - 1, 7, 1),
            period_end=date(fiscal_year, 6, 30),
            accession=f"0000789019-{fiscal_year}-000012",
            filed=date(fiscal_year, 7, 30),
            form="10-K",
        )


class FakeMarketDataClient:
    def history(self, ticker: str, start: date, end: date) -> PriceSeries:
        points = [
            PricePoint(
                session_date=date(2024, 1, day),
                open=Decimal("400"),
                high=Decimal("410"),
                low=Decimal("395"),
                close=Decimal("405"),
                adjusted_close=Decimal(str(400 + day)),
                volume=1_000_000,
            )
            for day in range(1, 6)
        ]
        return PriceSeries(
            ticker=ticker, points=points, source="fake", retrieved_at=datetime.now(UTC)
        )


class FakeNewsClient:
    def recent_items(self, ticker: str, max_items: int = 8) -> list[NewsItem]:
        return [
            NewsItem(
                title=f"{ticker} announces quarterly results",
                url="https://news.example.com/a",
                published_at=datetime(2024, 8, 1, tzinfo=UTC),
                source="fake-feed",
                summary="Results summary.",
            )
        ]


def fake_html_fetcher(filing: Filing) -> str:
    return SAMPLE_FILING_HTML


def build_fake_deps(*, resolves: bool = True) -> GraphDependencies:
    return GraphDependencies(
        sec_client=FakeSecClient(resolves=resolves),
        market_client=FakeMarketDataClient(),
        vector_store=InMemoryVectorStore(),
        filing_html_fetcher=fake_html_fetcher,
        news_client=FakeNewsClient(),
    )
