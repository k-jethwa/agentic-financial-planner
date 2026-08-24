"""Intake/Planner: validate the ticker, resolve its CIK, and build a
deterministic, schema-validated task plan.

The plan is data (a list of `PlannedTask`), never executable model text: no
graph node lets a model choose which node runs or with what parameters
outside this fixed, code-defined mapping. `ExecutionPlan.validate_bounds`
rejects an unbounded plan or a task naming an unsupported node before it
can reach the graph.
"""

from __future__ import annotations

from equity_research.core.exceptions import UnsupportedTickerError
from equity_research.core.models import (
    ExecutionPlan,
    PlannedTask,
    ReportMode,
    ResearchError,
    ResearchState,
    TraceEvent,
)
from equity_research.data.sec import SecClient

ALLOWED_NODES = {"market_data", "fundamentals", "filings", "news", "valuation", "critic"}

# The MVP plan is a fixed sequence per report mode; "classification" only
# selects which fixed subset of nodes is relevant to the request. The graph
# itself (Task 6, `graph.py`) still runs every node sequentially for
# debuggability — the plan is currently a validated, inspectable record of
# intent rather than something that drives conditional graph routing.
FULL_REPORT_NODES = ["market_data", "fundamentals", "filings", "news", "valuation"]
QUESTION_NODES = ["fundamentals", "filings"]


def plan_research(state: ResearchState, *, sec_client: SecClient) -> dict:
    """Resolve the ticker's CIK and build the bounded task plan.

    Returns a partial state update. A ticker that cannot be resolved to a
    CIK is a fatal error: no specialist can run without it (fail closed).
    """
    try:
        cik = sec_client.resolve_cik(state.request.ticker)
    except UnsupportedTickerError as exc:
        return _failure(state, str(exc))

    node_names = (
        FULL_REPORT_NODES if state.request.report_mode == ReportMode.FULL else QUESTION_NODES
    )
    plan = ExecutionPlan(tasks=[PlannedTask(node=name, params={"cik": cik}) for name in node_names])
    problems = plan.validate_bounds(ALLOWED_NODES)
    if problems:
        return _failure(state, f"invalid research plan: {'; '.join(problems)}")

    return {
        "plan": plan,
        "financial_facts": {**state.financial_facts, "cik": cik},
        "trace": [*state.trace, TraceEvent(node="planner", detail=f"cik={cik}")],
    }


def _failure(state: ResearchState, message: str) -> dict:
    return {
        "errors": [*state.errors, ResearchError(node="planner", message=message, fatal=True)],
        "trace": [*state.trace, TraceEvent(node="planner", status="failed", detail=message)],
    }
