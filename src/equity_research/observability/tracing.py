"""Trace-event helpers so every graph node emits a start/complete/fail event.

The design spec requires the trace viewer to show all node transitions for
both a completed and a failed run. Nodes should wrap their work in
`traced_node` rather than calling `append_trace` directly, so the
started/completed/failed bookkeeping can't be forgotten in a new worker.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from equity_research.core.models import TraceEvent
from equity_research.storage.repositories import ResearchRunRepository


@contextmanager
def traced_node(repo: ResearchRunRepository, run_id: UUID, node: str) -> Iterator[None]:
    """Record started/completed/failed trace events around a node's work.

    Exceptions are recorded (with their message as `detail`) and re-raised
    unchanged; this function never swallows a node failure.
    """
    repo.append_trace(run_id, TraceEvent(node=node, status="started"))
    try:
        yield
    except Exception as exc:
        repo.append_trace(run_id, TraceEvent(node=node, status="failed", detail=str(exc)))
        raise
    else:
        repo.append_trace(run_id, TraceEvent(node=node, status="completed"))
