from datetime import date

import pytest

from equity_research.data.sec import Filing
from equity_research.retrieval.ingest import FetchedFiling, FilingIngestor, FixedSplitter
from equity_research.retrieval.vector_store import InMemoryVectorStore

SAMPLE_HTML = """
<html><body>
<h1>Item 1. Business</h1>
<p>Contoso designs, manufactures, and sells widgets across North America and Europe.</p>
<h1>Item 1A. Risk Factors</h1>
<p>Supply chain disruption could materially affect the company's ability to deliver
product on time.</p>
</body></html>
"""


def sample_filing() -> FetchedFiling:
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


def test_chunk_metadata_keeps_sec_locator():
    chunks = FilingIngestor(splitter=FixedSplitter(1000)).chunks(sample_filing())
    assert chunks
    assert chunks[0].metadata["source_url"] == sample_filing().filing.url
    assert chunks[0].metadata["locator"]
    assert chunks[0].metadata["accession"] == "0000789019-24-000012"
    assert chunks[0].metadata["content_hash"]


def test_chunks_split_sections_by_heading():
    chunks = FilingIngestor(splitter=FixedSplitter(1000)).chunks(sample_filing())
    headings = {c.metadata["locator"] for c in chunks}
    assert any("Business" in h for h in headings)
    assert any("Risk Factors" in h for h in headings)


def test_chunks_are_split_when_section_exceeds_chunk_size():
    long_text = " ".join(["word"] * 400)
    fetched = FetchedFiling(
        filing=Filing(
            accession="acc-1",
            form="10-K",
            filing_date=date(2024, 1, 1),
            url="https://example.com/f.htm",
        ),
        html=f"<html><body><h1>Item 7. MD&amp;A</h1><p>{long_text}</p></body></html>",
        ticker="TEST",
        cik="0000000001",
    )
    chunks = FilingIngestor(splitter=FixedSplitter(100, overlap=10)).chunks(fetched)
    assert len(chunks) > 1
    assert all(c.metadata["content_hash"] for c in chunks)
    assert "part 1/" in chunks[0].metadata["locator"]


def test_chunks_falls_back_to_document_section_without_headings():
    fetched = FetchedFiling(
        filing=Filing(
            accession="acc-2",
            form="8-K",
            filing_date=date(2024, 3, 1),
            url="https://example.com/g.htm",
        ),
        html="<html><body><p>No headings here, just plain filing text.</p></body></html>",
        ticker="TEST",
        cik="0000000001",
    )
    chunks = FilingIngestor(splitter=FixedSplitter(1000)).chunks(fetched)
    assert len(chunks) == 1
    assert chunks[0].metadata["locator"] == "Document"


def test_chunk_text_never_repeats_its_own_heading():
    # Regression: BeautifulSoup.descendants walks *into* a heading tag right
    # after yielding it, so the heading's own text (even wrapped in inline
    # tags) must never leak into the section body it introduces.
    fetched = FetchedFiling(
        filing=Filing(
            accession="acc-3",
            form="10-K",
            filing_date=date(2024, 1, 1),
            url="https://example.com/h.htm",
        ),
        html=(
            "<html><body>"
            "<h1><b>Item 1A.</b> Risk Factors</h1>"
            "<p>Supply chain disruption is a risk.</p>"
            "</body></html>"
        ),
        ticker="TEST",
        cik="0000000001",
    )
    chunks = FilingIngestor(splitter=FixedSplitter(1000)).chunks(fetched)
    assert chunks[0].metadata["locator"] == "Item 1A. Risk Factors"
    assert chunks[0].text == "Supply chain disruption is a risk."


def test_ingest_upserts_into_vector_store_and_returns_chunk_count():
    store = InMemoryVectorStore()
    ingestor = FilingIngestor(splitter=FixedSplitter(1000), vector_store=store)

    count = ingestor.ingest(sample_filing())

    assert count == 2
    assert len(store.query("MSFT", "widgets", top_k=5)) >= 1


def test_ingest_without_vector_store_still_counts_chunks():
    ingestor = FilingIngestor(splitter=FixedSplitter(1000))
    assert ingestor.ingest(sample_filing()) == 2


def test_fixed_splitter_rejects_bad_bounds():
    with pytest.raises(ValueError):
        FixedSplitter(chunk_size=0)
    with pytest.raises(ValueError):
        FixedSplitter(chunk_size=100, overlap=100)


def test_fixed_splitter_empty_text_yields_no_chunks():
    assert FixedSplitter().split("   ") == []
