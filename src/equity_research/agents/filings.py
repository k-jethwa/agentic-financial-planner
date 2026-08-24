"""Document Research node: ingest recent filings and retrieve cited passages.

Only ever emits `Evidence` records carrying a source URL and locator: raw
filing text never reaches downstream state without first being turned into
a locator-tagged, hashed chunk (`equity_research.retrieval.ingest`) and
then into cited evidence (`equity_research.retrieval.retriever`). A
missing filing list is a fatal, fail-closed error — filing-dependent claims
cannot be made without it; an individual filing download failure is
recorded and skipped so the other filings can still be ingested.
"""

from __future__ import annotations

from typing import Protocol

from equity_research.core.exceptions import RequiredSourceUnavailableError
from equity_research.core.models import ResearchError, ResearchState, SourceRecord, TraceEvent
from equity_research.data.sec import Filing, SecClient
from equity_research.retrieval.ingest import FetchedFiling, FilingIngestor
from equity_research.retrieval.retriever import FilingRetriever
from equity_research.retrieval.vector_store import VectorStore

RELEVANT_FORMS = ("10-K", "10-Q")


class FilingHtmlFetcher(Protocol):
    def __call__(self, filing: Filing) -> str: ...


def research_filings(
    state: ResearchState,
    *,
    sec_client: SecClient,
    vector_store: VectorStore,
    html_fetcher: FilingHtmlFetcher,
    max_filings: int = 2,
    top_k: int = 6,
) -> dict:
    cik = state.financial_facts.get("cik")
    if not cik:
        return _fatal(state, "no CIK resolved by planner")

    try:
        filings = sec_client.filings(cik)
    except RequiredSourceUnavailableError as exc:
        return _fatal(state, str(exc))

    candidates = [f for f in filings if f.form in RELEVANT_FORMS][:max_filings]
    ingestor = FilingIngestor(vector_store=vector_store)
    sources = list(state.sources)
    errors: list[ResearchError] = []
    ingested_any = False

    for filing in candidates:
        try:
            html = html_fetcher(filing)
        except RequiredSourceUnavailableError as exc:
            errors.append(
                ResearchError(node="filings", message=f"{filing.accession}: {exc}", fatal=False)
            )
            sources.append(
                SourceRecord(
                    source_id=filing.accession,
                    source_type="sec_filing",
                    url=filing.url,
                    ok=False,
                    error=str(exc),
                )
            )
            continue

        ingestor.ingest(
            FetchedFiling(filing=filing, html=html, ticker=state.request.ticker, cik=cik)
        )
        sources.append(
            SourceRecord(source_id=filing.accession, source_type="sec_filing", url=filing.url)
        )
        ingested_any = True

    evidence = list(state.evidence)
    if ingested_any:
        retriever = FilingRetriever(vector_store)
        evidence.extend(
            retriever.search(state.request.ticker, state.request.question, top_k=top_k)
        )
    elif not errors:
        errors.append(
            ResearchError(node="filings", message="no 10-K/10-Q filings found", fatal=False)
        )

    status = "completed" if ingested_any else "failed"
    return {
        "sources": sources,
        "evidence": evidence,
        "errors": [*state.errors, *errors],
        "trace": [
            *state.trace,
            TraceEvent(
                node="filings", status=status, detail=f"{len(candidates)} filings considered"
            ),
        ],
    }


def _fatal(state: ResearchState, message: str) -> dict:
    return {
        "errors": [*state.errors, ResearchError(node="filings", message=message, fatal=True)],
        "trace": [*state.trace, TraceEvent(node="filings", status="failed", detail=message)],
    }
