"""FastAPI application factory: wires the research run API to storage and
the compiled research graph.

`create_app` builds real production dependencies from `Settings` by
default. Tests pass `graph=` explicitly (a graph built from fakes, the
same pattern `tests/unit/agents/test_graph.py` uses) so nothing ever
touches a live network or a secret in CI, and typically also point
`settings.db_path` at `:memory:` so no on-disk database is created.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph

from equity_research.agents.dependencies import build_production_graph_dependencies
from equity_research.agents.graph import build_research_graph
from equity_research.api.routes import runs
from equity_research.core.config import Settings
from equity_research.storage.database import create_sqlite_engine, init_db, session_factory
from equity_research.storage.repositories import ResearchRunRepository

DISCLAIMER = "Research and educational use only — not investment advice."


def create_app(
    settings: Settings | None = None,
    *,
    graph: CompiledStateGraph | None = None,
) -> FastAPI:
    settings = settings or Settings()
    engine = create_sqlite_engine(settings.db_path)
    init_db(engine)
    make_session = session_factory(engine)
    compiled_graph = graph or build_research_graph(build_production_graph_dependencies(settings))

    def get_repository() -> Iterator[ResearchRunRepository]:
        with make_session() as session:
            yield ResearchRunRepository(session)

    def get_graph() -> CompiledStateGraph:
        return compiled_graph

    app = FastAPI(
        title="Equity Research Assistant API",
        version="0.1.0",
        description=DISCLAIMER,
    )
    app.dependency_overrides[runs.get_repository] = get_repository
    app.dependency_overrides[runs.get_graph] = get_graph
    app.include_router(runs.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "disclaimer": DISCLAIMER}

    return app
