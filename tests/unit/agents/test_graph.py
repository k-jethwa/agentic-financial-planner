"""Graph orchestration tests.

Every I/O boundary (SEC, market data, filing HTML, news) is a fake, so the
graph runs deterministically with no network access. These tests exercise
the full sequential workflow and its fatal-error short-circuit.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from equity_research.agents.graph import GraphDependencies, build_research_graph
from equity_research.core.exceptions import RequiredSourceUnavailableError, UnsupportedTickerError
from equity_research.core.models import ReportMode, ResearchRequest, ResearchState
from equity_research.data.market import PricePoint, PriceSeries
from equity_research.data.news import NewsItem
from equity_research.data.sec import Filing, NormalizedFact
from equity_research.retrieval.vector_store import InMemoryVectorStore

CIK = "0000789019"

_ANNUAL_FACTS = {
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
        return [self._fact(tag, value, fy) for value, fy in _ANNUAL_FACTS.get(tag, [])[:limit]]

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


def _fake_html_fetcher(filing: Filing) -> str:
    return SAMPLE_FILING_HTML


def _build_deps(*, resolves: bool = True) -> GraphDependencies:
    return GraphDependencies(
        sec_client=FakeSecClient(resolves=resolves),
        market_client=FakeMarketDataClient(),
        vector_store=InMemoryVectorStore(),
        filing_html_fetcher=_fake_html_fetcher,
        news_client=FakeNewsClient(),
    )


def initial_state(
    ticker: str = "MSFT", question: str = "Assess supply chain risk"
) -> ResearchState:
    request = ResearchRequest(ticker=ticker, question=question, report_mode=ReportMode.FULL)
    return ResearchState(request=request)


def test_full_run_reaches_critic_with_evidence_from_every_branch():
    graph = build_research_graph(_build_deps())
    result = graph.invoke(initial_state())

    trace_nodes = [event.node for event in result["trace"]]
    assert trace_nodes == [
        "planner",
        "market_data",
        "fundamentals",
        "filings",
        "news",
        "valuation",
        "critic",
    ]
    assert result["trace"][-1].status == "completed"

    evidence_source_types = {e.source_type for e in result["evidence"]}
    assert evidence_source_types == {"market_data", "sec_xbrl", "sec_filing", "news"}
    assert all(e.evidence_id for e in result["evidence"])
    assert all(e.source_url for e in result["evidence"])

    metric_names = {m.name for m in result["derived_metrics"]}
    assert "period_return" in metric_names
    assert "revenue_growth_yoy" in metric_names
    assert "valuation" in result["analyses"]

    report_payload = result["report"]["json"]
    assert "not investment advice" in report_payload["disclaimer"].lower()
    assert report_payload["sources"]
    claim_groups = ("bull_thesis", "bear_thesis", "risks", "catalysts")
    assert all(
        claim["evidence_ids"] for group in claim_groups for claim in report_payload[group]
    )
    assert "not investment advice" in result["report"]["markdown"].lower()


def test_unsupported_ticker_routes_straight_to_critic():
    graph = build_research_graph(_build_deps(resolves=False))
    result = graph.invoke(initial_state())

    trace_nodes = [event.node for event in result["trace"]]
    assert trace_nodes == ["planner", "critic"]
    assert result["trace"][-1].node == "critic"
    assert result["trace"][-1].status == "failed"
    assert any(error.fatal for error in result["errors"])
    assert result["evidence"] == []
    assert result.get("report") is None  # no evidence to synthesize a report from


def test_missing_filing_html_is_recorded_but_does_not_stop_the_run():
    def failing_fetcher(filing: Filing) -> str:
        raise RequiredSourceUnavailableError(filing.url, "timeout")

    deps = _build_deps()
    deps.filing_html_fetcher = failing_fetcher

    graph = build_research_graph(deps)
    result = graph.invoke(initial_state())

    trace_nodes = [event.node for event in result["trace"]]
    assert trace_nodes[-1] == "critic"
    filings_errors = [e for e in result["errors"] if e.node == "filings"]
    assert filings_errors
    assert not any(e.fatal for e in filings_errors)
    # sec_filing evidence is dropped, but the other branches still ran.
    assert "sec_filing" not in {e.source_type for e in result["evidence"]}
    assert "market_data" in {e.source_type for e in result["evidence"]}


def test_missing_xbrl_tag_is_non_fatal_and_omits_dependent_metrics():
    class MissingCapexSecClient(FakeSecClient):
        def annual_facts(self, cik: str, tag: str, taxonomy: str = "us-gaap", *, limit: int = 1):
            if tag == "PaymentsToAcquirePropertyPlantAndEquipment":
                return []
            return super().annual_facts(cik, tag, taxonomy, limit=limit)

    deps = _build_deps()
    deps.sec_client = MissingCapexSecClient()

    graph = build_research_graph(deps)
    result = graph.invoke(initial_state())

    assert result["trace"][-1].node == "critic"
    assert not any(error.fatal for error in result["errors"])
    assert "fcf_margin" not in {m.name for m in result["derived_metrics"]}
    valuation = result["analyses"]["valuation"]
    assert valuation.__class__.__name__ == "ValuationUnavailable"
