"""Fake, network-free implementations of every research-graph I/O boundary.

Shared by unit tests (`tests/unit/agents/test_graph.py`), integration tests
(`tests/integration/api/test_runs.py`), and the recorded evaluation harness
(`evals/run_evals.py`) — one configurable set of fakes instead of three
copies of it. Every fake accepts overrides so a caller can build anything
from a plain successful large-cap run to an unresolvable ticker, a missing
XBRL tag, or an outaged required source, without touching the graph or any
real adapter.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from equity_research.agents.graph import GraphDependencies
from equity_research.agents.valuation import DcfAssumptions
from equity_research.core.exceptions import RequiredSourceUnavailableError, UnsupportedTickerError
from equity_research.data.market import PricePoint, PriceSeries
from equity_research.data.news import NewsItem
from equity_research.data.sec import Filing, NormalizedFact
from equity_research.retrieval.vector_store import InMemoryVectorStore, VectorStore

CIK = "0000789019"

# A large-cap-shaped default dataset (MSFT-like magnitudes); any scenario
# that doesn't care about specific numbers can just use this as-is.
ANNUAL_FACTS: dict[str, list[tuple[str, int]]] = {
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

DEFAULT_FILING = Filing(
    accession="0000789019-24-000012",
    form="10-K",
    filing_date=date(2024, 7, 30),
    url="https://www.sec.gov/Archives/edgar/data/789019/msft-20240630.htm",
)


class FakeSecClient:
    def __init__(
        self,
        *,
        resolves: bool = True,
        cik: str = CIK,
        annual_facts: dict[str, list[tuple[str, int]]] | None = None,
        filings_result: list[Filing] | None = None,
        raise_on_filings: bool = False,
    ):
        self._resolves = resolves
        self._cik = cik
        self._annual_facts = ANNUAL_FACTS if annual_facts is None else annual_facts
        self._filings_result = [DEFAULT_FILING] if filings_result is None else filings_result
        self._raise_on_filings = raise_on_filings

    def resolve_cik(self, ticker: str) -> str:
        if not self._resolves:
            raise UnsupportedTickerError(ticker, "not found in fake index")
        return self._cik

    def filings(self, cik: str) -> list[Filing]:
        if self._raise_on_filings:
            raise RequiredSourceUnavailableError(f"submissions:{cik}", "simulated outage")
        return self._filings_result

    def annual_facts(self, cik: str, tag: str, taxonomy: str = "us-gaap", *, limit: int = 1):
        entries = self._annual_facts.get(tag)
        if not entries:
            raise RequiredSourceUnavailableError(
                f"companyfacts:{cik}", f"missing tag {taxonomy}:{tag}"
            )
        return [self._fact(tag, value, fy) for value, fy in entries[:limit]]

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
    def __init__(self, *, points: list[PricePoint] | None = None):
        self._points = points if points is not None else _default_price_points()

    def history(self, ticker: str, start: date, end: date) -> PriceSeries:
        return PriceSeries(
            ticker=ticker, points=self._points, source="fake", retrieved_at=datetime.now(UTC)
        )


def _default_price_points() -> list[PricePoint]:
    return [
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


class FakeNewsClient:
    def __init__(self, *, items: list[NewsItem] | None = None, raises: bool = False):
        self._items = items
        self._raises = raises

    def recent_items(self, ticker: str, max_items: int = 8) -> list[NewsItem]:
        if self._raises:
            raise RequiredSourceUnavailableError("news:feed", "simulated feed outage")
        if self._items is not None:
            return self._items[:max_items]
        return [
            NewsItem(
                title=f"{ticker} announces quarterly results",
                url="https://news.example.com/a",
                published_at=datetime(2024, 8, 1, tzinfo=UTC),
                source="fake-feed",
                summary="Results summary.",
            )
        ]


def make_html_fetcher(html: str = SAMPLE_FILING_HTML):
    def _fetcher(filing: Filing) -> str:
        return html

    return _fetcher


fake_html_fetcher = make_html_fetcher()


def build_fake_deps(
    *,
    resolves: bool = True,
    cik: str = CIK,
    annual_facts: dict[str, list[tuple[str, int]]] | None = None,
    filings_result: list[Filing] | None = None,
    raise_on_filings: bool = False,
    price_points: list[PricePoint] | None = None,
    news_items: list[NewsItem] | None = None,
    news_raises: bool = False,
    filing_html: str = SAMPLE_FILING_HTML,
    vector_store: VectorStore | None = None,
    dcf_assumptions: DcfAssumptions | None = None,
) -> GraphDependencies:
    kwargs: dict = {}
    if dcf_assumptions is not None:
        kwargs["dcf_assumptions"] = dcf_assumptions
    return GraphDependencies(
        sec_client=FakeSecClient(
            resolves=resolves,
            cik=cik,
            annual_facts=annual_facts,
            filings_result=filings_result,
            raise_on_filings=raise_on_filings,
        ),
        market_client=FakeMarketDataClient(points=price_points),
        vector_store=vector_store or InMemoryVectorStore(),
        filing_html_fetcher=make_html_fetcher(filing_html),
        news_client=FakeNewsClient(items=news_items, raises=news_raises),
        **kwargs,
    )
