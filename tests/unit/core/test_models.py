import pytest
from pydantic import ValidationError

from equity_research.core.models import (
    ExecutionPlan,
    PlannedTask,
    ReportMode,
    ResearchRequest,
    ResearchState,
    RunStatus,
    is_valid_transition,
)


def test_research_request_defaults_to_full_report_and_generates_run_id():
    request = ResearchRequest(ticker="MSFT", question="Assess risks")
    assert request.report_mode is ReportMode.FULL
    assert request.run_id is not None


def test_research_request_rejects_malformed_ticker():
    with pytest.raises(ValidationError):
        ResearchRequest(ticker="msft!", question="Assess risks")


def test_research_state_defaults_are_empty_but_typed():
    request = ResearchRequest(ticker="MSFT", question="Assess risks")
    state = ResearchState(request=request)
    assert state.evidence == []
    assert state.errors == []
    assert state.report is None


def test_execution_plan_rejects_unsupported_node():
    plan = ExecutionPlan(tasks=[PlannedTask(node="market_data"), PlannedTask(node="trade")])
    problems = plan.validate_bounds(allowed_nodes={"market_data", "fundamentals"})
    assert any("trade" in problem for problem in problems)


def test_execution_plan_rejects_unbounded_plan():
    plan = ExecutionPlan(tasks=[PlannedTask(node="market_data") for _ in range(20)], max_tasks=16)
    problems = plan.validate_bounds(allowed_nodes={"market_data"})
    assert any("max_tasks" in problem for problem in problems)


@pytest.mark.parametrize(
    ("current", "requested", "expected"),
    [
        (RunStatus.PENDING, RunStatus.RUNNING, True),
        (RunStatus.RUNNING, RunStatus.COMPLETED, True),
        (RunStatus.COMPLETED, RunStatus.RUNNING, False),
        (RunStatus.FAILED, RunStatus.RUNNING, False),
    ],
)
def test_run_status_transitions(current, requested, expected):
    assert is_valid_transition(current, requested) is expected
