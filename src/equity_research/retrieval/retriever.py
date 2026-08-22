"""Citation-preserving filing retrieval.

`FilingRetriever.search` is the only way a downstream worker reads filing
text: it never returns raw chunk text without turning it into an `Evidence`
record carrying the chunk's source URL, locator, and content hash, so a
retrieved passage always traces back to one SEC filing location.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from equity_research.core.evidence import Confidence, Evidence
from equity_research.retrieval.vector_store import VectorStore

MAX_TOP_K = 20
EXCERPT_CHARS = 500

_HIGH_CONFIDENCE_SCORE = 0.6
_MEDIUM_CONFIDENCE_SCORE = 0.3


class FilingRetriever:
    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store

    def search(self, ticker: str, query: str, top_k: int = 6) -> list[Evidence]:
        hits = self._vector_store.query(ticker, query, top_k=min(top_k, MAX_TOP_K))
        retrieved_at = datetime.now(UTC)
        return [
            Evidence(
                evidence_id=f"filing:{hit.record.chunk_id}",
                claim=query,
                source_type="sec_filing",
                source_url=hit.record.metadata["source_url"],
                retrieved_at=retrieved_at,
                published_at=_parse_date(hit.record.metadata.get("filing_date")),
                locator=hit.record.metadata.get("locator", ""),
                excerpt=hit.record.text[:EXCERPT_CHARS],
                confidence=_confidence_for_score(hit.score),
            )
            for hit in hits
        ]


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _confidence_for_score(score: float) -> Confidence:
    if score >= _HIGH_CONFIDENCE_SCORE:
        return "high"
    if score >= _MEDIUM_CONFIDENCE_SCORE:
        return "medium"
    return "low"
