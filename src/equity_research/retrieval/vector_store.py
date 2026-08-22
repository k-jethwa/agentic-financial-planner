"""Vector storage for filing chunks: an in-memory fake plus a Pinecone adapter.

Every record retains its citation metadata (ticker, CIK, form, accession,
filing date, source URL, locator, content hash) so a retrieved chunk can
become an `Evidence` record without losing its citation trail. Tests, and
any run without Pinecone credentials configured, use `InMemoryVectorStore`;
`PineconeVectorStore` is only constructed when a Pinecone API key is
configured, and only imports the `pinecone` package at that point, so it
stays an optional dependency for the rest of the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VectorRecord:
    """One embedded (or embeddable) filing chunk, with full citation metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class ScoredRecord:
    record: VectorRecord
    score: float


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None: ...

    def query(self, ticker: str, query_text: str, top_k: int = 6) -> list[ScoredRecord]: ...


class InMemoryVectorStore:
    """A dependency-free vector-store fake.

    Scores by query/document token overlap rather than a real embedding
    similarity. Sufficient for unit/integration tests and local development
    without Pinecone credentials configured; never a substitute for ranked
    semantic retrieval quality.
    """

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            self._records[record.chunk_id] = record

    def query(self, ticker: str, query_text: str, top_k: int = 6) -> list[ScoredRecord]:
        query_terms = _tokenize(query_text)
        scored = [
            ScoredRecord(record=record, score=score)
            for record in self._records.values()
            if record.metadata.get("ticker") == ticker
            if (score := _overlap_score(query_terms, _tokenize(record.text))) > 0
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]


def _tokenize(text: str) -> set[str]:
    return {stripped.lower() for word in text.split() if (stripped := word.strip(".,;:()[]\"'"))}


def _overlap_score(query_terms: set[str], doc_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & doc_terms) / len(query_terms)


class PineconeVectorStore:
    """Production vector store: a Pinecone serverless index with integrated
    (server-side) text embedding, so no separate embedding provider is
    required for v1.

    Construct this only when `Settings.has_pinecone_credentials` is true.
    """

    TEXT_FIELD = "chunk_text"

    def __init__(self, api_key: str, index_name: str, *, namespace: str = "filings"):
        try:
            from pinecone import Pinecone
        except ImportError as exc:  # pragma: no cover - exercised only when misconfigured
            raise ImportError(
                "the 'pinecone' package is required for PineconeVectorStore; install it "
                "only in environments where PINECONE_API_KEY is configured"
            ) from exc
        self._index = Pinecone(api_key=api_key).Index(index_name)
        self._namespace = namespace

    def upsert(self, records: list[VectorRecord]) -> None:
        payload = [
            {"_id": record.chunk_id, self.TEXT_FIELD: record.text, **record.metadata}
            for record in records
        ]
        self._index.upsert_records(self._namespace, payload)

    def query(self, ticker: str, query_text: str, top_k: int = 6) -> list[ScoredRecord]:
        result = self._index.search(
            namespace=self._namespace,
            query={
                "top_k": top_k,
                "inputs": {"text": query_text},
                "filter": {"ticker": ticker},
            },
        )
        hits = result.get("result", {}).get("hits", [])
        scored: list[ScoredRecord] = []
        for hit in hits:
            fields = dict(hit.get("fields", {}))
            text = fields.pop(self.TEXT_FIELD, "")
            metadata = {key: str(value) for key, value in fields.items()}
            scored.append(
                ScoredRecord(
                    record=VectorRecord(chunk_id=hit["_id"], text=text, metadata=metadata),
                    score=float(hit.get("_score", 0.0)),
                )
            )
        return scored
