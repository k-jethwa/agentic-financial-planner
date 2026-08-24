"""News/Catalyst node: a bounded, deduplicated set of recent public news.

Every item becomes `Evidence` labeled `news`, carrying only its reported
headline/link/summary — this node records reported facts, never an impact
hypothesis, as fact. News is optional: a fetch failure or an empty result
is a non-fatal warning, never a run failure (per design spec).
"""

from __future__ import annotations

from datetime import UTC, datetime

from equity_research.core.evidence import Evidence
from equity_research.core.exceptions import EquityResearchError
from equity_research.core.models import ResearchError, ResearchState, TraceEvent
from equity_research.data.news import NewsClient


def research_news(state: ResearchState, *, news_client: NewsClient, max_items: int = 8) -> dict:
    try:
        items = news_client.recent_items(state.request.ticker, max_items=max_items)
    except EquityResearchError as exc:
        return {
            "errors": [*state.errors, ResearchError(node="news", message=str(exc), fatal=False)],
            "trace": [*state.trace, TraceEvent(node="news", status="failed", detail=str(exc))],
        }

    retrieved_at = datetime.now(UTC)
    evidence = [
        Evidence(
            evidence_id=f"news:{item.item_id}",
            claim=item.title,
            source_type="news",
            source_url=item.url,
            retrieved_at=retrieved_at,
            published_at=item.published_at.date() if item.published_at else None,
            locator=item.source,
            excerpt=item.summary,
            confidence="medium",
        )
        for item in items
    ]

    errors = (
        []
        if items
        else [ResearchError(node="news", message="no news items found", fatal=False)]
    )
    return {
        "evidence": [*state.evidence, *evidence],
        "errors": [*state.errors, *errors],
        "trace": [*state.trace, TraceEvent(node="news", detail=f"{len(items)} items")],
    }
