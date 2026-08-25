from equity_research.retrieval.vector_store import InMemoryVectorStore, VectorRecord


def _record(chunk_id: str, ticker: str, text: str) -> VectorRecord:
    return VectorRecord(chunk_id=chunk_id, text=text, metadata={"ticker": ticker})


def test_query_only_returns_matching_ticker():
    store = InMemoryVectorStore()
    store.upsert(
        [
            _record("a1", "MSFT", "cloud revenue grew strongly this quarter"),
            _record("b1", "AAPL", "cloud revenue grew strongly this quarter"),
        ]
    )
    results = store.query("MSFT", "cloud revenue", top_k=5)
    assert [hit.record.chunk_id for hit in results] == ["a1"]


def test_query_ranks_by_term_overlap():
    store = InMemoryVectorStore()
    store.upsert(
        [
            _record("low", "MSFT", "unrelated commentary about office furniture"),
            _record("high", "MSFT", "cloud revenue grew due to Azure demand"),
        ]
    )
    results = store.query("MSFT", "cloud revenue Azure", top_k=5)
    assert results[0].record.chunk_id == "high"


def test_upsert_is_idempotent_by_chunk_id():
    store = InMemoryVectorStore()
    store.upsert([_record("a1", "MSFT", "first version of the text")])
    store.upsert([_record("a1", "MSFT", "revised version of the text")])
    results = store.query("MSFT", "revised", top_k=5)
    assert len(results) == 1
    assert results[0].record.text == "revised version of the text"


def test_query_with_no_overlap_returns_empty():
    store = InMemoryVectorStore()
    store.upsert([_record("a1", "MSFT", "cloud revenue grew")])
    assert store.query("MSFT", "zzz unrelated qqq", top_k=5) == []


def test_query_matches_against_the_section_heading_too():
    # A chunk's heading (e.g. "Item 1A. Risk Factors") carries strong topical
    # signal a query may share even when the body text does not repeat it.
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(
                chunk_id="a1",
                text="Supply chain disruption could materially affect delivery timelines.",
                metadata={"ticker": "MSFT", "locator": "Item 1A. Risk Factors"},
            )
        ]
    )
    results = store.query("MSFT", "risk factors", top_k=5)
    assert [hit.record.chunk_id for hit in results] == ["a1"]
