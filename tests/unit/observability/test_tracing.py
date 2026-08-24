import pytest

from equity_research.core.models import ResearchRequest
from equity_research.observability.tracing import traced_node
from equity_research.storage.database import create_sqlite_engine, init_db, session_factory
from equity_research.storage.repositories import ResearchRunRepository


@pytest.fixture
def repo():
    engine = create_sqlite_engine(":memory:")
    init_db(engine)
    make_session = session_factory(engine)
    with make_session() as session:
        yield ResearchRunRepository(session)


def test_traced_node_records_started_then_completed(repo):
    run_id = repo.create(ResearchRequest(ticker="MSFT", question="Assess risks"))
    with traced_node(repo, run_id, "planner"):
        pass
    trace = repo.get(run_id).trace
    assert [(e.node, e.status) for e in trace] == [("planner", "started"), ("planner", "completed")]


def test_traced_node_records_failure_and_reraises(repo):
    run_id = repo.create(ResearchRequest(ticker="MSFT", question="Assess risks"))
    with pytest.raises(ValueError):
        with traced_node(repo, run_id, "market_data"):
            raise ValueError("boom")
    trace = repo.get(run_id).trace
    assert [(e.node, e.status) for e in trace] == [
        ("market_data", "started"),
        ("market_data", "failed"),
    ]
    assert trace[-1].detail == "boom"
