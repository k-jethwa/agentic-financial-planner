# Public Equity Research Assistant — Technical Design

## Purpose

Build a personal, research-only assistant that turns a public-equity ticker and question into a source-grounded investment memo. It is designed as a portfolio project: its agent decisions, tool calls, evidence, failures, and evaluation results must be inspectable rather than hidden behind a chat response.

The system provides decision support, not financial advice. It may express a structured bull/base/bear view and valuation range, but never executes trades, connects to a brokerage, or presents a result as personalized investment advice.

## Product scope

### In scope for v1

- US-listed common-stock ticker research.
- A chat-style prompt plus a “full research report” mode.
- Public/free sources: SEC EDGAR submissions, SEC companyfacts XBRL, company investor-relations pages, Yahoo Finance price history, and RSS/web news.
- Retrieval and citation of 10-K, 10-Q, 8-K, earnings releases, and transcripts when publicly available.
- A reproducible Markdown/JSON report with all material claims linked to evidence.
- Run history, agent trace, source timestamps, explicit uncertainty, and a critic gate.
- Local development with Python, LangChain, LlamaIndex, Pinecone free tier, Groq free developer tier, FastAPI, Streamlit, and SQLite.

### Explicitly out of scope for v1

- Trading, order routing, portfolio optimization, tax advice, alerts, user accounts, or paid market data.
- Non-US equities, real-time quotes that require a paid exchange license, intraday strategy backtesting, and autonomous background investment decisions.
- Persistent personal financial data.

## Architecture

FastAPI exposes a versioned research-run API and owns durable state. Streamlit is a thin client for creating and reading runs. LangGraph is the execution authority: it chooses branches, executes them, persists state transitions, retries safe failures, and emits trace events. LangChain provides model and tool abstractions plus Pydantic structured outputs. LlamaIndex owns document parsing, chunking, metadata preservation, and retrieval. Pinecone stores filing/document embeddings; Groq hosts the inference model through its free tier.

The design intentionally avoids unrestricted agent-to-agent chat. Every node reads and writes a typed `ResearchState`; all cross-agent facts pass through a normalized evidence ledger. This makes the workflow reproducible and gives the critic a complete audit surface.

```text
User / Streamlit
      │ POST /runs
FastAPI → Run repository (SQLite) → LangGraph supervisor
                                      ├─ market-data branch
                                      ├─ fundamentals branch ← SEC XBRL
                                      ├─ document/RAG branch ← EDGAR → LlamaIndex → Pinecone
                                      ├─ news/catalyst branch
                                      └─ valuation branch
                                                    │
                                        evidence validator / critic
                                                    │
                                     memo renderer → Markdown + JSON
```

## Core domain contracts

`ResearchRequest` contains `ticker`, `question`, `report_mode`, `as_of_date`, and `run_id`. `ResearchState` contains the validated request, execution plan, source records, evidence records, normalized financial facts, analyses, errors, and final report.

`Evidence` is the central contract:

```python
class Evidence(BaseModel):
    evidence_id: str
    claim: str
    source_type: Literal["sec_filing", "sec_xbrl", "market_data", "news", "company_ir"]
    source_url: HttpUrl
    retrieved_at: datetime
    published_at: date | None
    locator: str                 # filing section, page, table cell, or article paragraph
    excerpt: str | None
    numeric_value: Decimal | None
    unit: str | None
    confidence: Literal["high", "medium", "low"]
```

An output claim is reportable only when it references one or more evidence IDs. Prices, financial-statement data, and derived calculations retain an `as_of` date and unit. Derived values also retain their formula and input evidence IDs.

## Agent graph

1. **Intake/Planner** validates the ticker, resolves its CIK, classifies the request, and creates a deterministic task plan. A schema validator rejects an unbounded plan or unsupported tool.
2. **Market Data** retrieves daily adjusted prices, computes returns/volatility/drawdown, and returns no narrative claims that lack an observation.
3. **Fundamentals** maps SEC companyfacts tags into normalized income statement, cash-flow, and balance-sheet values; it records the reporting period and accession/filing link.
4. **Document Research** downloads the selected filings, performs LlamaIndex chunking/retrieval, and returns only passages with locators and SEC URLs.
5. **News/Catalyst** gathers a bounded, deduplicated set of recent public news/RSS items and distinguishes reported facts from its impact hypothesis.
6. **Valuation** creates a transparent DCF and multiples range only when sufficient inputs exist; otherwise it reports an unavailable valuation with missing inputs.
7. **Synthesizer** creates the structured memo strictly from ledger evidence.
8. **Critic** checks citation coverage, source freshness, contradictory facts, calculation consistency, and overconfident language. It can return the state to a specific worker once, then marks unresolved issues in the final report instead of hallucinating repair.

Agent calls are bounded by per-node timeout, retry count, maximum source count, and a global run budget. The initial MVP runs workers sequentially for debuggability; independent retrieval workers can be enabled concurrently behind a configuration flag once trace and rate-limit tests pass.

## Data acquisition and storage

SEC access uses a named `User-Agent`, request throttling, exponential backoff, local HTTP cache, and no bypassing of robots/rate limits. SEC submissions identify filings; companyfacts supplies standardized financial values; primary filing HTML is the citation source when a fact depends on wording or a table interpretation.

Market data is explicitly “end-of-day delayed/as provided by source,” never real-time. The report displays source timestamp, market session date, and the retrieval timestamp separately. News uses a source allowlist and RSS/web adapters with canonical URL deduplication.

SQLite stores users only as a local anonymous browser session, research-run metadata, status, trace events, errors, and report locations. Pinecone metadata stores ticker, CIK, filing accession, form, filing date, chunk ID, section locator, document URL, and content hash. Raw documents live in a local ignored cache, not in Git. Secrets reside exclusively in `.env`; `.env.example` documents names without values.

## Report contract

Every report produces `report.json` (machine-readable) and `report.md` (human-readable), with this fixed order:

1. Scope, disclaimer, ticker, company, and as-of timestamp.
2. One-paragraph conclusion: bull/base/bear valuation range, confidence band, and data limitations.
3. Business and financial snapshot.
4. Valuation assumptions, formula, sensitivity table, and comparison with current price.
5. Evidence-backed bull thesis, bear thesis, catalysts, and risks.
6. “What would change the view” measurable triggers.
7. Source ledger and critic findings.
8. Agent trace summary and reproducibility metadata (model, prompts version, source timestamps, software version).

The UI must prominently show “Research and educational use only — not investment advice.” It must not show a trade button, position size, expected return, or personal suitability language.

## Reliability, safety, and observability

- Validate all agent outputs with Pydantic before they alter state.
- Treat retrieved text as untrusted: separate system instructions from source content, detect prompt-injection patterns, and never grant retrieved content tool authority.
- Log tool input shape, response metadata, duration, retry, and error class; redact keys and source text where storage is unnecessary.
- Fail closed for critical sources: a missing SEC filing invalidates filing-dependent claims rather than substituting a model guess.
- Render warnings for stale data, insufficient valuation inputs, rate-limit fallbacks, unsupported ticker types, and unresolved critic issues.

## Evaluation and acceptance criteria

Maintain a curated fixture set of at least 12 ticker/query cases: large-cap, small-cap, bank, pre-profit company, ADR/foreign issuer edge case, recent earnings event, missing XBRL tag, and adversarial prompt-injection document. Each fixture includes expected source types, required citations, and prohibited claims.

Release gates:

- 100% of numeric report claims have evidence IDs and timestamps.
- 100% of report citations resolve to an allowed source URL in integration tests using recorded HTTP fixtures.
- Derived metric and DCF tests pass with deterministic fixtures.
- Critic blocks a deliberately unsupported claim and flags conflicting values.
- A failed optional news source still yields a clearly marked partial report; a failed required SEC source yields a failed run with a useful error.
- Trace viewer shows all node transitions and terminal status for a completed and a failed run.

## Delivery sequence

Deliver the product in eight testable milestones: scaffold/contracts; SEC and market adapters; filing ingestion/RAG; ledger and financial calculations; LangGraph workers; synthesis/critic/report renderer; FastAPI/Streamlit; evaluation, observability, and deployment documentation. The accompanying implementation plan assigns file-level interfaces and verification to each milestone.
