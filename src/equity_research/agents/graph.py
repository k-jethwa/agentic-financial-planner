"""Assembles the LangGraph research workflow.

Every node reads and writes the single typed `ResearchState`; cross-node
facts travel only as `Evidence`/`DerivedMetric` records, never as free-form
chat. The MVP runs workers sequentially, in a fixed order, for
debuggability (per design spec) rather than dynamically branching off the
planner's task plan.

The graph node registered as `"critic"` (`synthesize_and_critique`) is the
sole route into a final report: it drafts a memo from ledger evidence
(`agents.synthesis`), runs the full citation-coverage / freshness /
contradiction / overconfidence critic (`agents.critic`), repairs at most
once by dropping any claim that fails citation coverage, and renders the
result. `critic_gate` is a separate, minimal fallback used only when the
planner never resolved a CIK — there is no evidence to synthesize a report
from, so it just records the terminal trace event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from equity_research.agents.critic import critique, repair
from equity_research.agents.filings import FilingHtmlFetcher, research_filings
from equity_research.agents.fundamentals import build_snapshot, calculate_growth
from equity_research.agents.market_data import research_market_data
from equity_research.agents.news import research_news
from equity_research.agents.planner import plan_research
from equity_research.agents.synthesis import (
    PROMPTS_VERSION,
    SynthesisModel,
    TemplateSynthesisModel,
)
from equity_research.agents.valuation import (
    DcfAssumptions,
    ValuationResult,
    ValuationUnavailable,
    build_dcf,
)
from equity_research.core.evidence import Evidence
from equity_research.core.exceptions import RequiredSourceUnavailableError
from equity_research.core.models import ResearchError, ResearchState, TraceEvent
from equity_research.data.market import MarketDataClient
from equity_research.data.news import NewsClient
from equity_research.data.sec import NormalizedFact, SecClient
from equity_research.reports.models import SourceLedgerEntry, ValuationSummary, build_report
from equity_research.reports.renderer import render_report
from equity_research.retrieval.vector_store import VectorStore

DEFAULT_DCF_ASSUMPTIONS = DcfAssumptions(wacc=Decimal("0.09"), terminal_growth=Decimal("0.025"))

try:
    SOFTWARE_VERSION = version("equity-research")
except PackageNotFoundError:  # pragma: no cover - only when not installed at all
    SOFTWARE_VERSION = "0.0.0-dev"

_FUNDAMENTALS_TAGS = {
    "net_income": "NetIncomeLoss",
    "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
    "capital_expenditures": "PaymentsToAcquirePropertyPlantAndEquipment",
}


@dataclass
class GraphDependencies:
    """Everything a compiled research graph needs, injected at build time.

    Kept as one bundle so tests can swap every I/O boundary (SEC client,
    market data, vector store, filing fetcher, news client) for a fixture-
    backed fake, the same pattern used by the data adapters.
    """

    sec_client: SecClient
    market_client: MarketDataClient
    vector_store: VectorStore
    filing_html_fetcher: FilingHtmlFetcher
    news_client: NewsClient
    dcf_assumptions: DcfAssumptions = field(default_factory=lambda: DEFAULT_DCF_ASSUMPTIONS)
    synthesis_model: SynthesisModel = field(default_factory=TemplateSynthesisModel)
    max_filings: int = 2
    top_k_filing_evidence: int = 6
    max_news_items: int = 8


def research_fundamentals(state: ResearchState, *, sec_client: SecClient) -> dict:
    """Fundamentals node: fetch SEC XBRL facts and normalize them.

    A tag that fails to fetch (missing or unavailable) is recorded as a
    non-fatal warning and simply left out of the snapshot — one missing
    XBRL tag degrades a metric, it does not fail the run.
    """
    cik = state.financial_facts.get("cik")
    if not cik:
        return {
            "errors": [
                *state.errors,
                ResearchError(
                    node="fundamentals", message="no CIK resolved by planner", fatal=True
                ),
            ],
            "trace": [
                *state.trace,
                TraceEvent(node="fundamentals", status="failed", detail="missing cik"),
            ],
        }

    errors: list[ResearchError] = []

    def fetch_pair(tag: str) -> tuple[NormalizedFact | None, NormalizedFact | None]:
        try:
            facts = sec_client.annual_facts(cik, tag, limit=2)
        except RequiredSourceUnavailableError as exc:
            errors.append(ResearchError(node="fundamentals", message=str(exc), fatal=False))
            return None, None
        current = facts[0] if facts else None
        prior = facts[1] if len(facts) > 1 else None
        return current, prior

    def fetch_one(tag: str) -> NormalizedFact | None:
        try:
            return sec_client.latest_annual_fact(cik, tag)
        except RequiredSourceUnavailableError as exc:
            errors.append(ResearchError(node="fundamentals", message=str(exc), fatal=False))
            return None

    revenue, prior_revenue = fetch_pair("Revenues")
    net_income = fetch_one(_FUNDAMENTALS_TAGS["net_income"])
    operating_cash_flow = fetch_one(_FUNDAMENTALS_TAGS["operating_cash_flow"])
    capital_expenditures = fetch_one(_FUNDAMENTALS_TAGS["capital_expenditures"])

    snapshot = build_snapshot(
        ticker=state.request.ticker,
        cik=cik,
        revenue=revenue,
        prior_revenue=prior_revenue,
        net_income=net_income,
        operating_cash_flow=operating_cash_flow,
        capital_expenditures=capital_expenditures,
    )
    line_items = [
        item
        for item in (
            snapshot.revenue,
            snapshot.prior_revenue,
            snapshot.net_income,
            snapshot.operating_cash_flow,
            snapshot.capital_expenditures,
        )
        if item is not None
    ]

    return {
        "financial_facts": {**state.financial_facts, "fundamentals_snapshot": snapshot},
        "evidence": [*state.evidence, *(item.evidence for item in line_items)],
        "derived_metrics": [*state.derived_metrics, *calculate_growth(snapshot)],
        "errors": [*state.errors, *errors],
        "trace": [
            *state.trace,
            TraceEvent(node="fundamentals", detail=f"{len(line_items)} facts resolved"),
        ],
    }


def research_valuation(state: ResearchState, *, assumptions: DcfAssumptions) -> dict:
    """Valuation node: a transparent DCF, or a recorded reason it is unavailable."""
    snapshot = state.financial_facts.get("fundamentals_snapshot")
    if snapshot is None:
        return {
            "errors": [
                *state.errors,
                ResearchError(
                    node="valuation", message="no fundamentals snapshot available", fatal=False
                ),
            ],
            "trace": [
                *state.trace,
                TraceEvent(node="valuation", status="failed", detail="missing snapshot"),
            ],
        }

    result = build_dcf(snapshot, assumptions)
    errors: list[ResearchError] = []
    if isinstance(result, ValuationUnavailable):
        errors.append(
            ResearchError(
                node="valuation",
                message=f"valuation unavailable: missing {result.missing_inputs}",
                fatal=False,
            )
        )

    return {
        "analyses": {**state.analyses, "valuation": result},
        "errors": [*state.errors, *errors],
        "trace": [*state.trace, TraceEvent(node="valuation", detail=type(result).__name__)],
    }


def synthesize_and_critique(state: ResearchState, *, synthesis_model: SynthesisModel) -> dict:
    """Draft a memo from ledger evidence, critique it, repair once, and render it.

    This is the only path that ever sets `state.report`. Non-fatal warnings
    accumulated by earlier nodes (missing tags, a failed filing download, an
    empty news feed, ...) are carried into the rendered report's warnings so
    a partial report is always clearly marked as partial, per design spec.
    """
    as_of = state.request.as_of_date or datetime.now(UTC).date()
    valuation_analysis = state.analyses.get("valuation")

    draft = synthesis_model.draft(
        request=state.request,
        evidence=state.evidence,
        derived_metrics=state.derived_metrics,
        valuation=valuation_analysis,
    )

    result = critique(draft, state.evidence, as_of=as_of)
    if result.blocking:
        draft = repair(draft, result)
        result = critique(draft, state.evidence, as_of=as_of)
        final_status = "unresolved" if result.blocking else "repaired"
        result = result.model_copy(update={"status": final_status})

    report = build_report(
        ticker=state.request.ticker,
        question=state.request.question,
        report_mode=state.request.report_mode.value,
        as_of=as_of,
        generated_at=datetime.now(UTC),
        draft=draft,
        valuation=_valuation_summary(valuation_analysis),
        derived_metrics=state.derived_metrics,
        sources=[_source_ledger_entry(item) for item in state.evidence],
        critic=result,
        warnings=[error.message for error in state.errors] + [f.message for f in result.warnings],
        model_name=synthesis_model.model_name,
        prompts_version=PROMPTS_VERSION,
        software_version=SOFTWARE_VERSION,
    )
    markdown, payload = render_report(report)

    status = "failed" if result.blocking else "completed"
    return {
        "report": {"markdown": markdown, "json": payload},
        "trace": [*state.trace, TraceEvent(node="critic", status=status, detail=result.status)],
    }


def critic_gate(state: ResearchState) -> dict:
    """Fatal-only fallback: used when the planner never resolved a CIK, so
    there is no evidence to synthesize a report from. Still records the
    terminal "critic" trace event, matching "the critic is the only route
    into final synthesis" even on the failure path.
    """
    has_fatal = any(error.fatal for error in state.errors)
    status = "failed" if has_fatal else "completed"
    detail = "fatal errors present; no report produced" if has_fatal else "no fatal errors"
    return {"trace": [*state.trace, TraceEvent(node="critic", status=status, detail=detail)]}


def _valuation_summary(
    valuation: ValuationResult | ValuationUnavailable | None,
) -> ValuationSummary:
    if isinstance(valuation, ValuationResult):
        return ValuationSummary(
            available=True,
            formula=valuation.formula,
            wacc=str(valuation.assumptions.wacc),
            terminal_growth=str(valuation.assumptions.terminal_growth),
            equity_value=str(valuation.equity_value),
            value_per_share=(
                str(valuation.value_per_share) if valuation.value_per_share is not None else None
            ),
            sensitivity={key: str(value) for key, value in valuation.sensitivity.items()},
        )
    if isinstance(valuation, ValuationUnavailable):
        return ValuationSummary(available=False, missing_inputs=valuation.missing_inputs)
    return ValuationSummary(available=False, missing_inputs=["valuation_not_attempted"])


def _source_ledger_entry(evidence: Evidence) -> SourceLedgerEntry:
    return SourceLedgerEntry(
        evidence_id=evidence.evidence_id,
        source_type=evidence.source_type,
        source_url=str(evidence.source_url),
        locator=evidence.locator,
        retrieved_at=evidence.retrieved_at,
        published_at=evidence.published_at,
        confidence=evidence.confidence,
    )


def route_after_planner(state: ResearchState) -> str:
    """Skip every specialist and go straight to the fatal-only critic
    fallback on a fatal planner error (e.g. an unresolvable ticker) —
    nothing downstream can run without a CIK."""
    if any(error.fatal for error in state.errors):
        return "critic_fatal"
    return "market_data"


def build_research_graph(deps: GraphDependencies) -> CompiledStateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("planner", lambda s: plan_research(s, sec_client=deps.sec_client))
    graph.add_node(
        "market_data", lambda s: research_market_data(s, market_client=deps.market_client)
    )
    graph.add_node(
        "fundamentals", lambda s: research_fundamentals(s, sec_client=deps.sec_client)
    )
    graph.add_node(
        "filings",
        lambda s: research_filings(
            s,
            sec_client=deps.sec_client,
            vector_store=deps.vector_store,
            html_fetcher=deps.filing_html_fetcher,
            max_filings=deps.max_filings,
            top_k=deps.top_k_filing_evidence,
        ),
    )
    graph.add_node(
        "news",
        lambda s: research_news(s, news_client=deps.news_client, max_items=deps.max_news_items),
    )
    graph.add_node(
        "valuation", lambda s: research_valuation(s, assumptions=deps.dcf_assumptions)
    )
    graph.add_node(
        "critic", lambda s: synthesize_and_critique(s, synthesis_model=deps.synthesis_model)
    )
    graph.add_node("critic_fatal", critic_gate)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"market_data": "market_data", "critic_fatal": "critic_fatal"},
    )
    graph.add_edge("market_data", "fundamentals")
    graph.add_edge("fundamentals", "filings")
    graph.add_edge("filings", "news")
    graph.add_edge("news", "valuation")
    graph.add_edge("valuation", "critic")
    graph.add_edge("critic", END)
    graph.add_edge("critic_fatal", END)

    return graph.compile()
