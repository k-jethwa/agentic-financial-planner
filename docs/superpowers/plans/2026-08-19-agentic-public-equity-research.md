# Agentic Public Equity Research Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-grounded, inspectable public-equity research assistant that produces structured, non-advisory investment research reports.

**Architecture:** FastAPI owns durable runs and APIs; LangGraph orchestrates typed specialist workers. LangChain provides model/tool abstractions, LlamaIndex manages filing retrieval, Pinecone stores document embeddings, and Groq provides free-tier inference. Evidence is the sole cross-agent fact contract.

**Tech Stack:** Python 3.12, FastAPI, Streamlit, LangGraph, LangChain, LlamaIndex, Pinecone, Groq, Pydantic v2, SQLAlchemy/SQLite, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-19-public-equity-research-design.md`

## Global Constraints

- Restrict v1 to US-listed common stocks and public/free sources; do not add trading or personalized advice.
- Use Python 3.12 and Pydantic v2; every public model and tool result is typed.
- Persist `as_of` and `retrieved_at` for every numeric fact; all material report claims cite ledger evidence IDs.
- Read secrets only from environment variables: `GROQ_API_KEY`, `PINECONE_API_KEY`, and `SEC_USER_AGENT`.
- Test external integrations using recorded fixtures; CI must not depend on a live API or a secret.
- Use a named SEC user agent, rate limit requests, and fail closed for required SEC evidence.

---

## Planned file structure

```text
src/equity_research/
  api/{app.py,routes/runs.py,schemas.py}
  agents/{graph.py,planner.py,market_data.py,fundamentals.py,filings.py,news.py,valuation.py,synthesis.py,critic.py}
  core/{config.py,models.py,evidence.py,exceptions.py}
  data/{sec.py,market.py,news.py,cache.py}
  retrieval/{ingest.py,retriever.py,vector_store.py}
  reports/{models.py,renderer.py}
  storage/{database.py,repositories.py}
  observability/{tracing.py}
ui/streamlit_app.py
tests/{unit,integration,fixtures}/
```

### Task 1: Bootstrap the project and typed contracts

**Files:**
- Create: `pyproject.toml`, `.env.example`, `.gitignore`, `src/equity_research/core/{config.py,models.py,evidence.py,exceptions.py}`, `tests/unit/core/test_evidence.py`

**Interfaces:**
- Produces: `ResearchRequest`, `ResearchState`, `Evidence`, `DerivedMetric`, `ReportMode`, `Settings`, and `UnsupportedTickerError`.

- [ ] **Step 1: Write the failing evidence-validation tests**

```python
def test_evidence_requires_url_and_retrieval_time():
    with pytest.raises(ValidationError):
        Evidence(evidence_id="e1", claim="Revenue rose", source_type="sec_xbrl")

def test_derived_metric_keeps_input_evidence_ids():
    metric = DerivedMetric(name="fcf_margin", value=Decimal("0.12"), formula="fcf/revenue", input_evidence_ids=["e1", "e2"])
    assert metric.input_evidence_ids == ["e1", "e2"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/core/test_evidence.py -v`

Expected: FAIL because the contract module does not exist.

- [ ] **Step 3: Implement models and settings**

```python
class ReportMode(StrEnum):
    FULL = "full"
    QUESTION = "question"

class ResearchRequest(BaseModel):
    ticker: Annotated[str, StringConstraints(pattern=r"^[A-Z.]{1,10}$")]
    question: str
    report_mode: ReportMode = ReportMode.FULL
    as_of_date: date | None = None
```

Define `Evidence` exactly as specified and use `pydantic-settings` for `Settings`.

- [ ] **Step 4: Run quality checks**

Run: `ruff check src tests && pytest tests/unit/core/test_evidence.py -v`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example .gitignore src tests/unit/core/test_evidence.py
git commit -m "feat: add research domain contracts"
```

### Task 2: Add durable research runs and trace events

**Files:**
- Create: `src/equity_research/storage/{database.py,repositories.py}`, `src/equity_research/observability/tracing.py`, `tests/unit/storage/test_research_runs.py`

**Interfaces:**
- Consumes: `ResearchRequest`, `ResearchState`.
- Produces: `ResearchRunRepository.create(request) -> UUID`, `transition(run_id, status)`, `append_trace(event)`, and `get(run_id) -> ResearchRun`.

- [ ] **Step 1: Write failing repository tests**

```python
def test_run_persists_state_transitions(session):
    repo = ResearchRunRepository(session)
    run_id = repo.create(ResearchRequest(ticker="MSFT", question="Assess risks"))
    repo.transition(run_id, RunStatus.RUNNING)
    assert repo.get(run_id).status is RunStatus.RUNNING
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/storage/test_research_runs.py -v`

Expected: FAIL with missing repository/database modules.

- [ ] **Step 3: Implement SQLite persistence**

Create SQLAlchemy tables for `research_runs` and append-only `trace_events`; use ISO UTC timestamps and JSON columns for request/state snapshots. Reject invalid transitions such as `COMPLETED -> RUNNING`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/unit/storage/test_research_runs.py -v`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/equity_research/storage src/equity_research/observability tests/unit/storage
git commit -m "feat: persist research runs and traces"
```

### Task 3: Implement SEC and market-data adapters with recorded fixtures

**Files:**
- Create: `src/equity_research/data/{sec.py,market.py,cache.py}`, `tests/unit/data/{test_sec.py,test_market.py}`, `tests/fixtures/sec/companyfacts_msft.json`

**Interfaces:**
- Produces: `SecClient.resolve_cik(ticker) -> str`, `company_facts(cik) -> dict`, `filings(cik) -> list[Filing]`, `MarketDataClient.history(ticker, start, end) -> PriceSeries`.

- [ ] **Step 1: Write failing adapter tests against fixtures**

```python
def test_sec_client_normalizes_revenue_fact(load_fixture):
    client = SecClient(http=FixtureHttp(load_fixture("companyfacts_msft.json")))
    fact = client.latest_annual_fact("0000789019", "Revenues")
    assert fact.unit == "USD"
    assert fact.fiscal_year == 2024
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `pytest tests/unit/data/test_sec.py tests/unit/data/test_market.py -v`

Expected: FAIL because the adapters are missing.

- [ ] **Step 3: Implement bounded clients**

Use `httpx`, a token-bucket limiter, retry only timeout/429/5xx responses, and a disk cache keyed by URL hash. Require `SEC_USER_AGENT`; attach it to every SEC request. Make market prices explicitly daily and include session date/source timestamp.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/data/test_sec.py tests/unit/data/test_market.py -v`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/equity_research/data tests/unit/data tests/fixtures/sec
git commit -m "feat: add cached SEC and market data adapters"
```

### Task 4: Build filing ingestion and citation-preserving retrieval

**Files:**
- Create: `src/equity_research/retrieval/{ingest.py,retriever.py,vector_store.py}`, `tests/unit/retrieval/test_ingest.py`, `tests/integration/retrieval/test_retrieval.py`

**Interfaces:**
- Consumes: `Filing(accession, form, url, filing_date)`.
- Produces: `FilingIngestor.ingest(filing) -> int` and `FilingRetriever.search(ticker, query, top_k=6) -> list[Evidence]`.

- [ ] **Step 1: Write failing ingestion and retrieval tests**

```python
def test_chunk_metadata_keeps_sec_locator(sample_filing):
    chunks = FilingIngestor(splitter=FixedSplitter(100)).chunks(sample_filing)
    assert chunks[0].metadata["source_url"] == sample_filing.url
    assert "locator" in chunks[0].metadata
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/retrieval/test_ingest.py tests/integration/retrieval/test_retrieval.py -v`

Expected: FAIL with missing retrieval modules.

- [ ] **Step 3: Implement LlamaIndex pipeline and Pinecone adapter**

Normalize filing HTML to sections before chunking. Store ticker, CIK, form, accession, filing date, URL, locator, and content hash in every vector record. Use an in-memory vector-store fake in tests; only construct `PineconeVectorStore` when the API key is configured.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/retrieval/test_ingest.py tests/integration/retrieval/test_retrieval.py -v`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/equity_research/retrieval tests/unit/retrieval tests/integration/retrieval
git commit -m "feat: add citation-preserving filing retrieval"
```

### Task 5: Implement financial normalization and transparent valuation

**Files:**
- Create: `src/equity_research/agents/{fundamentals.py,valuation.py}`, `tests/unit/agents/{test_fundamentals.py,test_valuation.py}`

**Interfaces:**
- Consumes: SEC facts and `PriceSeries`.
- Produces: `FundamentalSnapshot`, `calculate_growth(snapshot) -> list[DerivedMetric]`, `build_dcf(snapshot, assumptions) -> ValuationResult`.

- [ ] **Step 1: Write failing calculation tests**

```python
def test_dcf_exposes_all_inputs_and_sensitivity():
    result = build_dcf(snapshot_with_cash_flows(), DcfAssumptions(wacc=Decimal("0.10"), terminal_growth=Decimal("0.025")))
    assert result.formula_input_evidence_ids
    assert result.sensitivity["wacc=0.09,g=0.025"] > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/agents/test_fundamentals.py tests/unit/agents/test_valuation.py -v`

Expected: FAIL because the financial and valuation modules are absent.

- [ ] **Step 3: Implement deterministic financial functions**

Use `Decimal`, never float, for money/ratios. Return `ValuationUnavailable` with enumerated missing inputs rather than inventing values. Include a 3x3 WACC/terminal-growth sensitivity table and source IDs for every input.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/agents/test_fundamentals.py tests/unit/agents/test_valuation.py -v`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/equity_research/agents/fundamentals.py src/equity_research/agents/valuation.py tests/unit/agents
git commit -m "feat: add evidence-linked fundamentals and valuation"
```

### Task 6: Assemble validated LangGraph research workflow

**Files:**
- Create: `src/equity_research/agents/{graph.py,llm.py,planner.py,market_data.py,filings.py,news.py}`, `tests/unit/agents/{test_graph.py,test_llm.py}`

**Interfaces:**
- Consumes: `ResearchState`.
- Produces: `create_research_llm(settings) -> BaseChatModel`, `build_research_graph(deps) -> CompiledStateGraph`, and worker functions returning validated partial state updates.

- [ ] **Step 1: Write failing graph tests**

```python
def test_unsupported_claim_routes_to_critic(fake_dependencies):
    graph = build_research_graph(fake_dependencies)
    result = graph.invoke(initial_state("MSFT"))
    assert result["trace"][-1]["node"] == "critic"
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/agents/test_graph.py -v`

Expected: FAIL because the graph is not implemented.

- [ ] **Step 3: Implement the Groq LangChain adapter and graph nodes**

Use `langchain-groq` with `ChatGroq(model=settings.groq_model, temperature=0)` behind `create_research_llm`; reject startup when no `GROQ_API_KEY` is present outside test mode. Define one node per specialist and restrict all tool calls by configured source count, timeout, and retry budget. Store planner output as structured tasks—not executable model text. Delimit retrieved source text as untrusted data and instruct the model never to treat it as tool or system instructions. Make the critic the only route into final synthesis and cap repair loops at one.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/agents/test_graph.py tests/unit/agents/test_llm.py -v`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/equity_research/agents tests/unit/agents/test_graph.py
git commit -m "feat: orchestrate research workers with LangGraph"
```

### Task 7: Add synthesis, critic gate, and deterministic report rendering

**Files:**
- Create: `src/equity_research/agents/{synthesis.py,critic.py}`, `src/equity_research/reports/{models.py,renderer.py}`, `tests/unit/{agents/test_critic.py,reports/test_renderer.py}`

**Interfaces:**
- Consumes: validated `ResearchState`.
- Produces: `CriticResult`, `InvestmentReport`, and `render_report(report) -> tuple[str, dict]`.

- [ ] **Step 1: Write failing critic and renderer tests**

```python
def test_critic_blocks_claim_without_evidence(state_with_unsupported_claim):
    result = critic(state_with_unsupported_claim)
    assert result.status == CriticStatus.REPAIR_REQUIRED

def test_report_contains_disclaimer_and_source_ledger(complete_report):
    markdown, payload = render_report(complete_report)
    assert "not investment advice" in markdown.lower()
    assert payload["sources"][0]["evidence_id"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/agents/test_critic.py tests/unit/reports/test_renderer.py -v`

Expected: FAIL with missing critic/report modules.

- [ ] **Step 3: Implement claim gate and renderer**

The critic must reject unsupported numeric claims, stale required sources, citation URL mismatches, contradictory values, and false certainty. Render the fixed report sections from the design spec and include model/prompt/source timestamps plus all unresolved warnings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/agents/test_critic.py tests/unit/reports/test_renderer.py -v`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/equity_research/agents src/equity_research/reports tests/unit/agents/test_critic.py tests/unit/reports
git commit -m "feat: add audited investment report rendering"
```

### Task 8: Expose FastAPI endpoints and Streamlit run viewer

**Files:**
- Create: `src/equity_research/api/{app.py,schemas.py,routes/runs.py}`, `ui/streamlit_app.py`, `tests/integration/api/test_runs.py`

**Interfaces:**
- Produces: `POST /v1/runs`, `GET /v1/runs/{run_id}`, `GET /v1/runs/{run_id}/report`, `GET /v1/runs/{run_id}/trace`.

- [ ] **Step 1: Write failing API tests**

```python
def test_create_then_read_run(client):
    created = client.post("/v1/runs", json={"ticker": "MSFT", "question": "Analyze valuation"})
    assert created.status_code == 202
    assert client.get(f"/v1/runs/{created.json()['run_id']}").status_code == 200
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/integration/api/test_runs.py -v`

Expected: FAIL because the API application is absent.

- [ ] **Step 3: Implement API/UI**

Validate request payloads before scheduling a run; return typed status/result payloads. Streamlit must show disclaimer, run status, report sections, sources, warnings, and chronological graph trace. Do not render untrusted source HTML.

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/api/test_runs.py -v`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/equity_research/api ui tests/integration/api
git commit -m "feat: add research run API and viewer"
```

### Task 9: Create evaluation suite, documentation, and CI

**Files:**
- Create: `evals/cases.json`, `evals/run_evals.py`, `tests/integration/test_end_to_end.py`, `.github/workflows/ci.yml`, `README.md`

**Interfaces:**
- Consumes: `ResearchRequest` fixtures and `build_research_graph` with fake/recorded adapters.
- Produces: evaluation JSON summarizing citation coverage, critic behavior, and scenario results.

- [ ] **Step 1: Write failing end-to-end acceptance test**

```python
def test_recorded_full_run_has_cited_numeric_claims(recorded_graph):
    report = recorded_graph.invoke(initial_state("MSFT"))["report"]
    assert all(claim.evidence_ids for claim in report.numeric_claims)
    assert report.disclaimer
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/integration/test_end_to_end.py -v`

Expected: FAIL until the recorded full-run harness exists.

- [ ] **Step 3: Implement evaluation and CI artifacts**

Add at least 12 cases, including the adversarial filing and missing-XBRL scenarios from the design. Make GitHub Actions run `ruff check`, `pytest`, and the recorded evaluator without API keys. Document local setup, free-tier environment configuration, rate limits, architecture, and disclaimer.

- [ ] **Step 4: Run full verification**

Run: `ruff check src tests evals && pytest -q && python evals/run_evals.py --recorded`

Expected: exit 0, with zero uncited numeric claims and all required scenario gates reported as passed.

- [ ] **Step 5: Commit**

```bash
git add evals tests/integration .github/workflows/ci.yml README.md
git commit -m "feat: add evaluation suite and CI"
```

## Final verification

- [ ] Confirm every design acceptance criterion maps to Tasks 1–9.
- [ ] Run the Task 9 full-verification command in a clean virtual environment.
- [ ] Verify `git status --short` is empty after committing.
- [ ] Manually inspect a recorded report and trace: citations, timestamps, disclaimer, warnings, and failed-run behavior must be visible.
