import pytest

from equity_research.core.exceptions import InvalidRunTransitionError
from equity_research.core.models import ResearchRequest, RunStatus, TraceEvent
from equity_research.storage.repositories import ResearchRunNotFoundError, ResearchRunRepository


def test_run_persists_state_transitions(session):
    repo = ResearchRunRepository(session)
    run_id = repo.create(ResearchRequest(ticker="MSFT", question="Assess risks"))
    repo.transition(run_id, RunStatus.RUNNING)
    assert repo.get(run_id).status is RunStatus.RUNNING


def test_new_run_starts_pending(session):
    repo = ResearchRunRepository(session)
    run_id = repo.create(ResearchRequest(ticker="MSFT", question="Assess risks"))
    assert repo.get(run_id).status is RunStatus.PENDING


def test_invalid_transition_is_rejected(session):
    repo = ResearchRunRepository(session)
    run_id = repo.create(ResearchRequest(ticker="MSFT", question="Assess risks"))
    repo.transition(run_id, RunStatus.RUNNING)
    repo.transition(run_id, RunStatus.COMPLETED)
    with pytest.raises(InvalidRunTransitionError):
        repo.transition(run_id, RunStatus.RUNNING)


def test_get_unknown_run_raises(session):
    repo = ResearchRunRepository(session)
    with pytest.raises(ResearchRunNotFoundError):
        repo.get(__import__("uuid").uuid4())


def test_trace_events_are_appended_in_order(session):
    repo = ResearchRunRepository(session)
    run_id = repo.create(ResearchRequest(ticker="MSFT", question="Assess risks"))
    repo.append_trace(run_id, TraceEvent(node="planner", status="started"))
    repo.append_trace(run_id, TraceEvent(node="planner", status="completed"))
    repo.append_trace(run_id, TraceEvent(node="market_data", status="started"))

    trace = repo.get(run_id).trace
    assert [event.node for event in trace] == ["planner", "planner", "market_data"]
    assert [event.status for event in trace] == ["started", "completed", "started"]


def test_report_round_trips_through_the_repository(session):
    repo = ResearchRunRepository(session)
    run_id = repo.create(ResearchRequest(ticker="MSFT", question="Assess risks"))
    repo.set_report(run_id, {"summary": "bull case", "sources": []})
    assert repo.get(run_id).report == {"summary": "bull case", "sources": []}
