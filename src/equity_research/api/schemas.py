"""HTTP request/response schemas for the research run API.

Kept separate from the domain models in `core.models` so the wire format
can evolve independently of the internal `ResearchRequest`/`ResearchState`
contracts. Request payloads are validated here before a run is ever
created — an invalid ticker or a missing question never reaches the graph.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from equity_research.core.models import ReportMode, RunStatus

TickerField = Annotated[str, StringConstraints(pattern=r"^[A-Z.]{1,10}$")]


class CreateRunRequest(BaseModel):
    ticker: TickerField
    question: str = Field(min_length=1, max_length=2000)
    report_mode: ReportMode = ReportMode.FULL
    as_of_date: date | None = None


class CreateRunResponse(BaseModel):
    run_id: UUID
    status: RunStatus


class TraceEventResponse(BaseModel):
    node: str
    status: str
    occurred_at: datetime
    detail: str | None = None


class RunSummaryResponse(BaseModel):
    run_id: UUID
    ticker: str
    question: str
    report_mode: ReportMode
    status: RunStatus
    created_at: datetime
    updated_at: datetime


class RunTraceResponse(BaseModel):
    run_id: UUID
    status: RunStatus
    trace: list[TraceEventResponse]


class RunReportResponse(BaseModel):
    run_id: UUID
    status: RunStatus
    report: dict | None = None
    report_markdown: str | None = None
