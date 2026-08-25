"""Graph orchestration tests.

Every I/O boundary (SEC, market data, filing HTML, news) is a fake, so the
graph runs deterministically with no network access. These tests exercise
the full sequential workflow and its fatal-error short-circuit.
"""

from __future__ import annotations

from support.graph_fakes import FakeSecClient, build_fake_deps

from equity_research.agents.graph import build_research_graph
from equity_research.core.exceptions import RequiredSourceUnavailableError
from equity_research.core.models import ReportMode, ResearchRequest, ResearchState
from equity_research.data.sec import Filing


def initial_state(
    ticker: str = "MSFT", question: str = "Assess supply chain risk"
) -> ResearchState:
    request = ResearchRequest(ticker=ticker, question=question, report_mode=ReportMode.FULL)
    return ResearchState(request=request)


def test_full_run_reaches_critic_with_evidence_from_every_branch():
    graph = build_research_graph(build_fake_deps())
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
    graph = build_research_graph(build_fake_deps(resolves=False))
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

    deps = build_fake_deps()
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

    deps = build_fake_deps()
    deps.sec_client = MissingCapexSecClient()

    graph = build_research_graph(deps)
    result = graph.invoke(initial_state())

    assert result["trace"][-1].node == "critic"
    assert not any(error.fatal for error in result["errors"])
    assert "fcf_margin" not in {m.name for m in result["derived_metrics"]}
    valuation = result["analyses"]["valuation"]
    assert valuation.__class__.__name__ == "ValuationUnavailable"
