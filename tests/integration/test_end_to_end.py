"""End-to-end acceptance test: a full recorded run, from request to a
rendered, evidence-linked report -- the release-gate assertion from the
design spec (100% of numeric report claims carry evidence IDs and every
report carries its disclaimer).
"""

from __future__ import annotations

import pytest
from support.graph_fakes import build_fake_deps

from equity_research.agents.graph import build_research_graph
from equity_research.core.models import ReportMode, ResearchRequest, ResearchState
from equity_research.reports.models import InvestmentReport


def initial_state(ticker: str = "MSFT") -> ResearchState:
    request = ResearchRequest(
        ticker=ticker,
        question="Assess the investment thesis and supply chain risk factors",
        report_mode=ReportMode.FULL,
    )
    return ResearchState(request=request)


@pytest.fixture
def recorded_graph():
    return build_research_graph(build_fake_deps())


def test_recorded_full_run_has_cited_numeric_claims(recorded_graph):
    result = recorded_graph.invoke(initial_state("MSFT"))
    report = InvestmentReport.model_validate(result["report"]["json"])

    assert all(claim.evidence_ids for claim in report.numeric_claims)
    assert report.disclaimer
    assert "not investment advice" in report.disclaimer.lower()


def test_recorded_full_run_every_source_has_a_resolvable_url(recorded_graph):
    result = recorded_graph.invoke(initial_state("MSFT"))
    report = InvestmentReport.model_validate(result["report"]["json"])

    assert report.sources
    assert all(source.source_url.startswith("https://") for source in report.sources)


def test_recorded_run_records_all_node_transitions_for_a_completed_run(recorded_graph):
    result = recorded_graph.invoke(initial_state("MSFT"))
    nodes = [event.node for event in result["trace"]]
    assert nodes == [
        "planner",
        "market_data",
        "fundamentals",
        "filings",
        "news",
        "valuation",
        "critic",
    ]
    assert result["trace"][-1].status == "completed"


def test_recorded_run_records_all_node_transitions_for_a_failed_run(recorded_graph):
    deps = build_fake_deps(resolves=False)
    graph = build_research_graph(deps)
    result = graph.invoke(initial_state("ZZZZ"))

    nodes = [event.node for event in result["trace"]]
    assert nodes == ["planner", "critic"]
    assert result["trace"][-1].status == "failed"
    assert result.get("report") is None  # planner-fatal path never sets it
