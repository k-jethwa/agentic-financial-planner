"""POST/GET endpoints for research runs.

`create_run` validates the request payload via `CreateRunRequest`, creates
the run, and executes the compiled research graph synchronously before
responding — there is no background worker/queue in this MVP, so a client
can immediately GET a terminal-status run right after POST returns. A
future async worker could replace `_execute`'s synchronous call without
changing this route's contract. `get_repository`/`get_graph` are
placeholders `api.app.create_app` overrides with the real per-request
repository and the compiled graph (or, in tests, with fakes).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from equity_research.api.schemas import (
    CreateRunRequest,
    CreateRunResponse,
    RunReportResponse,
    RunSummaryResponse,
    RunTraceResponse,
    TraceEventResponse,
)
from equity_research.core.models import ResearchRequest, ResearchState, RunStatus, TraceEvent
from equity_research.storage.repositories import (
    ResearchRun,
    ResearchRunNotFoundError,
    ResearchRunRepository,
)

router = APIRouter(prefix="/v1/runs", tags=["runs"])


def get_repository() -> Iterator[ResearchRunRepository]:  # pragma: no cover - overridden below
    raise NotImplementedError("repository dependency not configured; use api.app.create_app")


def get_graph() -> Any:  # pragma: no cover - overridden by create_app
    raise NotImplementedError("graph dependency not configured; use api.app.create_app")


RepositoryDep = Annotated[ResearchRunRepository, Depends(get_repository)]
GraphDep = Annotated[Any, Depends(get_graph)]


@router.post("", status_code=202, response_model=CreateRunResponse)
def create_run(
    payload: CreateRunRequest,
    repo: RepositoryDep,
    graph: GraphDep,
) -> CreateRunResponse:
    request = ResearchRequest(
        ticker=payload.ticker,
        question=payload.question,
        report_mode=payload.report_mode,
        as_of_date=payload.as_of_date,
    )
    run_id = request.run_id
    repo.create(request)
    _execute(repo, run_id, request, graph)
    final = repo.get(run_id)
    return CreateRunResponse(run_id=final.run_id, status=final.status)


@router.get("/{run_id}", response_model=RunSummaryResponse)
def get_run(run_id: UUID, repo: RepositoryDep) -> RunSummaryResponse:
    run = _get_or_404(repo, run_id)
    return RunSummaryResponse(
        run_id=run.run_id,
        ticker=run.request.ticker,
        question=run.request.question,
        report_mode=run.request.report_mode,
        status=run.status,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@router.get("/{run_id}/report", response_model=RunReportResponse)
def get_run_report(run_id: UUID, repo: RepositoryDep) -> RunReportResponse:
    run = _get_or_404(repo, run_id)
    return RunReportResponse(
        run_id=run.run_id,
        status=run.status,
        report=run.report["json"] if run.report else None,
        report_markdown=run.report["markdown"] if run.report else None,
    )


@router.get("/{run_id}/trace", response_model=RunTraceResponse)
def get_run_trace(run_id: UUID, repo: RepositoryDep) -> RunTraceResponse:
    run = _get_or_404(repo, run_id)
    return RunTraceResponse(
        run_id=run.run_id,
        status=run.status,
        trace=[TraceEventResponse(**event.model_dump()) for event in run.trace],
    )


def _get_or_404(repo: ResearchRunRepository, run_id: UUID) -> ResearchRun:
    try:
        return repo.get(run_id)
    except ResearchRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _execute(repo: ResearchRunRepository, run_id: UUID, request: ResearchRequest, graph) -> None:
    repo.transition(run_id, RunStatus.RUNNING)
    try:
        result = graph.invoke(ResearchState(request=request))
    except Exception as exc:
        # Last-resort safety net: a node bug must still leave the run in a
        # terminal, inspectable state rather than corrupting it or 500-ing
        # the request with no trace of what happened.
        repo.append_trace(run_id, TraceEvent(node="graph", status="failed", detail=str(exc)))
        repo.transition(run_id, RunStatus.FAILED)
        return

    for event in result["trace"]:
        repo.append_trace(run_id, event)

    report = result.get("report")
    if report is not None:
        repo.set_report(run_id, report)

    has_fatal = any(error.fatal for error in result.get("errors", []))
    final_status = RunStatus.FAILED if has_fatal or report is None else RunStatus.COMPLETED
    repo.transition(run_id, final_status)
