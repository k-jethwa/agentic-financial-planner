"""Repository for durable research runs.

This is the only write path for run status and trace events: graph nodes
and the API layer go through `ResearchRunRepository` rather than touching
ORM rows directly, so status-transition rules stay enforced in one place.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from equity_research.core.exceptions import InvalidRunTransitionError
from equity_research.core.models import (
    ReportMode,
    ResearchRequest,
    RunStatus,
    TraceEvent,
    is_valid_transition,
)
from equity_research.storage.database import ResearchRunORM, TraceEventORM


class ResearchRunNotFoundError(LookupError):
    def __init__(self, run_id: UUID):
        self.run_id = run_id
        super().__init__(f"No research run with id {run_id}")


class ResearchRun(BaseModel):
    """The persisted, read-facing view of a research run."""

    run_id: UUID
    request: ResearchRequest
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    trace: list[TraceEvent] = Field(default_factory=list)
    report: dict | None = None


class ResearchRunRepository:
    """SQLAlchemy-backed persistence for research runs and their traces."""

    def __init__(self, session: Session):
        self._session = session

    def create(self, request: ResearchRequest) -> UUID:
        now = datetime.now(UTC)
        row = ResearchRunORM(
            run_id=str(request.run_id),
            ticker=request.ticker,
            question=request.question,
            report_mode=ReportMode(request.report_mode).value,
            as_of_date=request.as_of_date.isoformat() if request.as_of_date else None,
            status=RunStatus.PENDING.value,
            request_json=request.model_dump_json(),
            report_json=None,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )
        self._session.add(row)
        self._session.commit()
        return request.run_id

    def transition(self, run_id: UUID, status: RunStatus) -> None:
        row = self._get_row(run_id)
        current = RunStatus(row.status)
        if not is_valid_transition(current, status):
            raise InvalidRunTransitionError(current.value, status.value)
        row.status = status.value
        row.updated_at = datetime.now(UTC).isoformat()
        self._session.commit()

    def append_trace(self, run_id: UUID, event: TraceEvent) -> None:
        self._get_row(run_id)  # ensure the run exists before recording its trace
        trace_row = TraceEventORM(
            run_id=str(run_id),
            node=event.node,
            status=event.status,
            occurred_at=event.occurred_at.isoformat(),
            detail=event.detail,
        )
        self._session.add(trace_row)
        self._session.commit()

    def set_report(self, run_id: UUID, report: dict) -> None:
        row = self._get_row(run_id)
        row.report_json = json.dumps(report)
        row.updated_at = datetime.now(UTC).isoformat()
        self._session.commit()

    def get(self, run_id: UUID) -> ResearchRun:
        row = self._get_row(run_id)
        trace_rows = self._session.scalars(
            select(TraceEventORM)
            .where(TraceEventORM.run_id == str(run_id))
            .order_by(TraceEventORM.id)
        ).all()
        trace = [
            TraceEvent(
                node=t.node,
                status=t.status,
                occurred_at=datetime.fromisoformat(t.occurred_at),
                detail=t.detail,
            )
            for t in trace_rows
        ]
        return ResearchRun(
            run_id=UUID(row.run_id),
            request=ResearchRequest.model_validate_json(row.request_json),
            status=RunStatus(row.status),
            created_at=datetime.fromisoformat(row.created_at),
            updated_at=datetime.fromisoformat(row.updated_at),
            trace=trace,
            report=json.loads(row.report_json) if row.report_json else None,
        )

    def _get_row(self, run_id: UUID) -> ResearchRunORM:
        row = self._session.get(ResearchRunORM, str(run_id))
        if row is None:
            raise ResearchRunNotFoundError(run_id)
        return row
