"""Filing ingestion: normalize SEC filing HTML into citation-preserving chunks.

Filing HTML is untrusted, potentially large, source content. This module
turns it into `FilingChunk` records, each carrying enough metadata (ticker,
CIK, form, accession, filing date, source URL, locator, content hash) to
become an `Evidence` record later. A chunk with no locator or source URL
must never enter the vector store.

Section normalization uses heading tags as break points (`normalize_html_to_
sections`); within-section chunking is pluggable via the `Splitter`
protocol, defaulting to LlamaIndex's sentence-aware `SentenceSplitter`
(`LlamaIndexSplitter`). `FixedSplitter` is a dependency-free alternative
used by tests and available for constrained environments.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from bs4 import BeautifulSoup

from equity_research.data.sec import Filing
from equity_research.retrieval.vector_store import VectorRecord, VectorStore

_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
_SKIP_PARENTS = {"script", "style"}


@dataclass(frozen=True)
class FetchedFiling:
    """A filing whose primary document HTML has already been downloaded."""

    filing: Filing
    html: str
    ticker: str
    cik: str


@dataclass(frozen=True)
class FilingSection:
    heading: str
    text: str


@dataclass(frozen=True)
class FilingChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str]


class Splitter(Protocol):
    def split(self, text: str) -> list[str]: ...


class FixedSplitter:
    """A deterministic, dependency-free character-window splitter."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self._chunk_size = chunk_size
        self._step = chunk_size - overlap

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        windows = (text[i : i + self._chunk_size] for i in range(0, len(text), self._step))
        return [window for window in windows if window.strip()]


class LlamaIndexSplitter:
    """Production splitter: LlamaIndex's sentence-aware, token-sized chunker.

    Chunks respect sentence boundaries and are sized in tokens rather than
    characters, avoiding mid-word/mid-sentence cuts within a filing section.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        from llama_index.core.node_parser import SentenceSplitter

        self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, text: str) -> list[str]:
        from llama_index.core import Document

        if not text.strip():
            return []
        nodes = self._splitter.get_nodes_from_documents([Document(text=text)])
        return [node.get_content() for node in nodes]


def normalize_html_to_sections(html: str) -> list[FilingSection]:
    """Split filing HTML into headed sections using heading tags as breaks.

    Falls back to a single "Document" section when no headings are found,
    so plain-text or unusually structured filings still ingest.
    """
    soup = BeautifulSoup(html, "html.parser")
    sections: list[FilingSection] = []
    heading = "Document"
    parts: list[str] = []

    def flush() -> None:
        text = " ".join(" ".join(parts).split())
        if text:
            sections.append(FilingSection(heading=heading, text=text))

    for element in (soup.body or soup).descendants:
        name = getattr(element, "name", None)
        if name in _HEADING_TAGS:
            flush()
            heading = element.get_text(strip=True) or heading
            parts = []
        elif (
            name is None
            and (parent := element.parent) is not None
            and parent.name not in _SKIP_PARENTS
        ):
            stripped = str(element).strip()
            if stripped:
                parts.append(stripped)
    flush()

    if not sections:
        text = " ".join(soup.get_text(separator=" ").split())
        if text:
            sections = [FilingSection(heading="Document", text=text)]
    return sections


class FilingIngestor:
    """Turns one fetched filing into locator-tagged, hashed chunks and,
    when a vector store is configured, upserts them.
    """

    def __init__(self, splitter: Splitter | None = None, vector_store: VectorStore | None = None):
        self._splitter = splitter or LlamaIndexSplitter()
        self._vector_store = vector_store

    def chunks(self, fetched: FetchedFiling) -> list[FilingChunk]:
        sections = normalize_html_to_sections(fetched.html)
        chunks: list[FilingChunk] = []
        for section_index, section in enumerate(sections):
            pieces = self._splitter.split(section.text)
            for piece_index, piece in enumerate(pieces):
                locator = (
                    f"{section.heading} (part {piece_index + 1}/{len(pieces)})"
                    if len(pieces) > 1
                    else section.heading
                )
                chunks.append(
                    FilingChunk(
                        chunk_id=f"{fetched.filing.accession}:{section_index}:{piece_index}",
                        text=piece,
                        metadata={
                            "ticker": fetched.ticker,
                            "cik": fetched.cik,
                            "form": fetched.filing.form,
                            "accession": fetched.filing.accession,
                            "filing_date": fetched.filing.filing_date.isoformat(),
                            "source_url": fetched.filing.url,
                            "locator": locator,
                            "content_hash": hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                        },
                    )
                )
        return chunks

    def ingest(self, fetched: FetchedFiling) -> int:
        chunks = self.chunks(fetched)
        if self._vector_store is not None and chunks:
            records = [
                VectorRecord(chunk_id=c.chunk_id, text=c.text, metadata=c.metadata) for c in chunks
            ]
            self._vector_store.upsert(records)
        return len(chunks)
