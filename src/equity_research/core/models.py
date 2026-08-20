"""Request, plan, and run-state contracts shared across the research graph.

`ResearchState` is the single typed object every graph node reads and
writes. Cross-agent facts never travel as free-form chat; they travel as
`Evidence` (see `equity_research.core.evidence`) attached to this state.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

from equity_research.core.evidence import DerivedMetric, Evidence, SourceType

TickerSymbol = Annotated[str, StringConstraints(pattern=r"^[A-Z.]{1,10}$")]


class ReportMode(StrEnum):
    FULL = "full"
    QUESTION = "question"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED)


# Transitions allowed out of each status. Anything not listed here (e.g.
# COMPLETED -> RUNNING) is invalid and must be rejected by the repository.
_ALLOWED_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.FAILED},
    RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
}


def is_valid_transition(current: RunStatus, requested: RunStatus) -> bool:
    return requested in _ALLOWED_TRANSITIONS.get(current, set())


class ResearchRequest(BaseModel):
    """The validated, user-facing research ask."""

    ticker: TickerSymbol
    question: str
    report_mode: ReportMode = ReportMode.FULL
    as_of_date: date | None = None
    run_id: uuid.UUID = Field(default_factory=uuid.uuid4)


class PlannedTask(BaseModel):
    """One unit of work in the planner's deterministic task plan."""

    node: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """The planner's bounded, schema-validated task list.

    An unbounded plan or a task naming an unsupported node/tool must be
    rejected before it reaches the graph.
    """

    tasks: list[PlannedTask] = Field(default_factory=list)
    max_tasks: int = 16

    def validate_bounds(self, allowed_nodes: set[str]) -> list[str]:
        """Return a list of validation problems (empty means valid)."""
        problems: list[str] = []
        if len(self.tasks) > self.max_tasks:
            problems.append(f"plan has {len(self.tasks)} tasks, exceeds max_tasks={self.max_tasks}")
        for task in self.tasks:
            if task.node not in allowed_nodes:
                problems.append(f"unsupported node in plan: {task.node!r}")
        return problems


class SourceRecord(BaseModel):
    """A raw source fetch attempt, prior to evidence extraction."""

    source_id: str
    source_type: SourceType
    url: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ok: bool = True
    error: str | None = None


class ResearchError(BaseModel):
    """A recoverable or terminal error raised by a graph node."""

    node: str
    message: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    fatal: bool = False


class TraceEvent(BaseModel):
    """One node transition, for the trace viewer and audit surface."""

    node: str
    status: Literal["started", "completed", "failed"] = "completed"
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    detail: str | None = None


class ResearchState(BaseModel):
    """The complete, typed state threaded through every graph node."""

    request: ResearchRequest
    plan: ExecutionPlan = Field(default_factory=ExecutionPlan)
    sources: list[SourceRecord] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    financial_facts: dict[str, Any] = Field(default_factory=dict)
    derived_metrics: list[DerivedMetric] = Field(default_factory=list)
    analyses: dict[str, Any] = Field(default_factory=dict)
    errors: list[ResearchError] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    report: dict[str, Any] | None = None
