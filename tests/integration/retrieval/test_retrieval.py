"""Integration test: ingest a sample filing end-to-end and retrieve cited evidence."""

from datetime import date

from equity_research.data.sec import Filing
from equity_research.retrieval.ingest import FetchedFiling, FilingIngestor, FixedSplitter
from equity_research.retrieval.retriever import FilingRetriever
from equity_research.retrieval.vector_store import InMemoryVectorStore

SAMPLE_HTML = """
<html><body>
<h1>Item 1A. Risk Factors</h1>
<p>Supply chain disruption could materially affect the company's ability to deliver
product on time and may increase input costs across our manufacturing base.</p>
<h1>Item 7. Management's Discussion and Analysis</h1>
<p>Revenue grew year over year driven by cloud services demand and pricing actions.</p>
</body></html>
"""


def _fetched_filing() -> FetchedFiling:
    return FetchedFiling(
        filing=Filing(
            accession="0000789019-24-000012",
            form="10-K",
            filing_date=date(2024, 7, 30),
            url="https://www.sec.gov/Archives/edgar/data/789019/000078901924000012/msft-20240630.htm",
        ),
        html=SAMPLE_HTML,
        ticker="MSFT",
        cik="0000789019",
    )


def _build_retriever() -> FilingRetriever:
    store = InMemoryVectorStore()
    FilingIngestor(splitter=FixedSplitter(2000), vector_store=store).ingest(_fetched_filing())
    return FilingRetriever(store)


def test_search_returns_evidence_with_locator_and_source_url():
    results = _build_retriever().search("MSFT", "supply chain disruption risk", top_k=3)

    assert results
    top = results[0]
    assert top.source_type == "sec_filing"
    assert str(top.source_url) == _fetched_filing().filing.url
    assert "Risk Factors" in top.locator
    assert top.evidence_id
    assert top.published_at == date(2024, 7, 30)
    assert top.excerpt


def test_search_is_scoped_to_requested_ticker():
    assert _build_retriever().search("AAPL", "supply chain disruption risk") == []


def test_search_finds_the_relevant_section_for_each_query():
    retriever = _build_retriever()

    risk_hit = retriever.search("MSFT", "supply chain disruption", top_k=1)[0]
    mdna_hit = retriever.search("MSFT", "revenue grew cloud services demand", top_k=1)[0]

    assert "Risk Factors" in risk_hit.locator
    assert "Management's Discussion" in mdna_hit.locator
