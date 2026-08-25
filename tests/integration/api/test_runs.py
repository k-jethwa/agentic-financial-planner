"""API integration tests: a real FastAPI app, an in-memory SQLite DB, and a
graph built entirely from fakes (`support.graph_fakes`) -- no network, no
secrets, matching the design spec's CI constraint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from support.graph_fakes import build_fake_deps

from equity_research.agents.graph import build_research_graph
from equity_research.api.app import create_app
from equity_research.core.config import Settings


def _client(*, resolves: bool = True) -> TestClient:
    settings = Settings(db_path=Path(":memory:"), test_mode=True)
    graph = build_research_graph(build_fake_deps(resolves=resolves))
    app = create_app(settings=settings, graph=graph)
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return _client()


def test_create_then_read_run(client: TestClient):
    created = client.post("/v1/runs", json={"ticker": "MSFT", "question": "Analyze valuation"})
    assert created.status_code == 202
    run_id = created.json()["run_id"]

    read = client.get(f"/v1/runs/{run_id}")
    assert read.status_code == 200
    body = read.json()
    assert body["ticker"] == "MSFT"
    assert body["status"] == "completed"


def test_create_run_rejects_invalid_ticker(client: TestClient):
    response = client.post("/v1/runs", json={"ticker": "not-a-ticker!", "question": "Assess risk"})
    assert response.status_code == 422


def test_create_run_rejects_blank_question(client: TestClient):
    response = client.post("/v1/runs", json={"ticker": "MSFT", "question": ""})
    assert response.status_code == 422


def test_report_endpoint_returns_disclaimer_and_sources(client: TestClient):
    created = client.post("/v1/runs", json={"ticker": "MSFT", "question": "Assess risk"})
    run_id = created.json()["run_id"]

    report = client.get(f"/v1/runs/{run_id}/report")
    assert report.status_code == 200
    body = report.json()
    assert body["status"] == "completed"
    assert "not investment advice" in body["report"]["disclaimer"].lower()
    assert body["report"]["sources"]
    assert "not investment advice" in body["report_markdown"].lower()


def test_trace_endpoint_shows_every_node_transition(client: TestClient):
    created = client.post("/v1/runs", json={"ticker": "MSFT", "question": "Assess risk"})
    run_id = created.json()["run_id"]

    trace = client.get(f"/v1/runs/{run_id}/trace")
    assert trace.status_code == 200
    nodes = [event["node"] for event in trace.json()["trace"]]
    assert nodes == [
        "planner",
        "market_data",
        "fundamentals",
        "filings",
        "news",
        "valuation",
        "critic",
    ]


def test_unsupported_ticker_yields_a_failed_run_with_a_useful_error():
    client = _client(resolves=False)
    created = client.post("/v1/runs", json={"ticker": "ZZZZ", "question": "Assess risk"})
    run_id = created.json()["run_id"]

    read = client.get(f"/v1/runs/{run_id}")
    assert read.json()["status"] == "failed"

    trace = client.get(f"/v1/runs/{run_id}/trace")
    assert trace.json()["trace"][-1]["status"] == "failed"

    report = client.get(f"/v1/runs/{run_id}/report")
    assert report.json()["report"] is None


def test_get_unknown_run_returns_404(client: TestClient):
    response = client.get("/v1/runs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_health_endpoint_shows_disclaimer(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert "not investment advice" in response.json()["disclaimer"].lower()
