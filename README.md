# Agentic Public Equity Research Assistant

A personal, portfolio-grade project: a ticker + a question go in, and a
reproducible Markdown/JSON research memo comes out, with every material
claim traceable to a specific SEC filing, XBRL fact, price observation, or
news item — plus the full agent trace and critic findings that produced
it. See `docs/superpowers/specs/2026-08-19-public-equity-research-design.md`
for the full design spec and `docs/superpowers/plans/2026-08-19-agentic-public-equity-research.md`
for the task-by-task implementation plan this codebase followed.

## Architecture

```text
User / Streamlit UI
      │ POST /v1/runs
FastAPI ──▶ SQLite (runs, trace events) ──▶ LangGraph research workflow
                                              │
                planner (deterministic CIK resolution + task plan)
                              │
             ┌────────────┬──────────────┬─────────┬───────────┐
             ▼            ▼              ▼         ▼           ▼
        market_data  fundamentals     filings     news     valuation
       (Yahoo daily   (SEC XBRL      (SEC filing (RSS/    (transparent
        prices)       companyfacts)   RAG via     Google    DCF, 3x3
                                       LlamaIndex/ News)     sensitivity)
                                       Pinecone)
             └────────────┴──────────────┴─────────┴───────────┘
                              │
                    critic node ("critic")
                 synthesize (template writer) →
                 critique (citation coverage, freshness,
                           contradictions, overconfidence) →
                 repair once (drop unsupported claims) →
                 render (Markdown + JSON, fixed section order)
                              │
                         state.report
```

Every graph node reads and writes one typed `ResearchState`; cross-agent
facts travel only as `Evidence`/`DerivedMetric` records, never as free-form
text. Workers run sequentially (not concurrently) for debuggability. The
critic is the only path into a final report: an unsupported claim (one
citing an evidence ID that isn't in the ledger) is dropped in a single
repair pass rather than silently rendered or endlessly retried.

**Stack:** Python 3.12, FastAPI, Streamlit, LangGraph, LangChain
(`langchain-groq`, wired but not yet used for narrative generation — see
[Known limitations](#known-limitations)), LlamaIndex, Pinecone (optional),
Groq (optional), Pydantic v2, SQLAlchemy/SQLite, pytest, Ruff.

## Project layout

```text
src/equity_research/
  core/          Domain contracts: ResearchRequest, ResearchState, Evidence,
                 DerivedMetric, Settings, domain exceptions.
  data/          SEC EDGAR, Yahoo Finance, RSS news adapters; shared cached/
                 rate-limited HTTP client; on-disk HTTP cache.
  retrieval/     Filing HTML → citation-preserving chunks (LlamaIndex) →
                 vector store (in-memory fake, or Pinecone when configured)
                 → cited Evidence.
  agents/        One module per graph node (planner, market_data,
                 fundamentals, filings, news, valuation, synthesis, critic),
                 llm.py (Groq adapter), graph.py (LangGraph assembly),
                 dependencies.py (production wiring).
  reports/       InvestmentReport contract + deterministic Markdown/JSON
                 renderer.
  storage/       SQLite persistence for runs and append-only trace events.
  api/           FastAPI app: POST/GET /v1/runs, /report, /trace.
ui/streamlit_app.py   Thin client over the API.
evals/           Recorded evaluation suite (12 scenarios) + runner.
tests/           unit/, integration/, fixtures/, support/ (shared test fakes).
```

## Local setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,ui]"
cp .env.example .env                # fill in values you actually have
```

### Environment variables

All secrets are read from the environment (optionally via `.env`, which is
git-ignored). See `.env.example` for the full list. Nothing in the test
suite or CI ever reads a real one.

| Variable | Required for | Notes |
| --- | --- | --- |
| `SEC_USER_AGENT` | Any real SEC EDGAR request | SEC requires a named contact, e.g. `"my-app you@example.com"`. The app fails closed at startup without it — it will not silently skip identification. |
| `GROQ_API_KEY` | Nothing yet | `agents/llm.py` builds a `ChatGroq` model behind this key, but no graph node calls it yet (see [Known limitations](#known-limitations)). Free developer tier at [console.groq.com](https://console.groq.com). |
| `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT` | Persistent filing-chunk retrieval | Without it, `agents/dependencies.py` falls back to `InMemoryVectorStore` (works fully, just not persisted across process restarts). Free tier at [pinecone.io](https://www.pinecone.io). |

Non-secret overrides (`GROQ_MODEL`, `EQUITY_RESEARCH_DB_PATH`,
`EQUITY_RESEARCH_HTTP_CACHE_DIR`, ...) are also in `.env.example`.

### Rate limits

- **SEC EDGAR**: `data/sec.py`'s production client sends a token-bucket-limited
  ~8 requests/second with a named `User-Agent`, retries only
  timeout/429/5xx, and serves repeat requests from an on-disk cache
  (`EQUITY_RESEARCH_HTTP_CACHE_DIR`) — never re-fetches a URL it already
  has. This stays comfortably under SEC's documented fair-access policy.
- **Yahoo Finance / filing HTML**: the same policy via a shared
  `CachedHttpClient` (`data/http.py`), at ~4 req/s and ~8 req/s
  respectively.
- **News**: Google News RSS search, no key required, one request per run.
- Every node is additionally bounded by `Settings.max_sources_per_node`,
  `request_timeout_seconds`, and `max_retries`.

## Running the tests, evals, and full verification

```bash
ruff check src tests evals
pytest -q
python evals/run_evals.py --recorded
```

All three are network-free and secret-free — this is exactly what CI runs
on every push (`.github/workflows/ci.yml`).

- `pytest` — unit tests per module plus integration tests for the API
  (`tests/integration/api`) and a full graph run (`tests/integration/
  test_end_to_end.py`), all against fakes/fixtures (`tests/support`,
  `tests/fixtures`).
- `evals/run_evals.py --recorded` — 12 curated scenarios in
  `evals/cases.json` (large-cap, small-cap, bank, pre-profit, ADR/foreign-
  issuer edge case, recent earnings event, missing XBRL tag, adversarial
  prompt-injection filing, a required-source outage, an optional-source
  outage, per-share valuation, and question mode). Writes a summary —
  citation coverage, critic-status breakdown, and per-case pass/fail — to
  `evals/results.json` and exits non-zero if any gate fails.

## Running it locally

```bash
# Terminal 1: the API (needs SEC_USER_AGENT in .env for real runs)
uvicorn equity_research.api.app:create_app --factory --reload

# Terminal 2: the UI
streamlit run ui/streamlit_app.py
```

`POST /v1/runs` executes the research graph **synchronously** inside the
request — there is no background worker/queue in this MVP, so a client can
`GET` a terminal-status run immediately after `POST` returns. A future
async worker could replace that without changing the API contract.

## Known limitations

- **Synthesis is fully deterministic, not LLM-authored.**
  `agents/synthesis.TemplateSynthesisModel` builds every claim directly
  from the evidence ledger via string templates — nothing to hallucinate,
  and fully testable without a live model. `agents/llm.py` builds the Groq
  chat model and `wrap_untrusted_source` delimits retrieved text as data
  for exactly this purpose, but no node calls it yet. Swapping in an
  LLM-backed `SynthesisModel` is a drop-in change behind that protocol;
  `agents.critic` is what would keep it honest.
- **Dev-mode filing retrieval uses naive keyword overlap**
  (`InMemoryVectorStore`), not semantic search — a query must share actual
  words with a filing chunk (including its section heading) to retrieve
  it. Configuring `PINECONE_API_KEY` switches to `PineconeVectorStore`,
  which uses Pinecone's server-side embedding instead.
- **No shares-outstanding fetch yet.** The DCF exposes `value_per_share`
  whenever `DcfAssumptions.shares_outstanding` is supplied, but
  `agents/fundamentals.py` doesn't fetch a shares-outstanding XBRL tag to
  populate it automatically — it's `None` by default in production.
- Runs execute synchronously in the request handler (see above) rather
  than via a background worker.

## Disclaimer

This system is a research and educational tool. It may express a
structured bull/base/bear view and a valuation range, sourced and dated,
but it does not execute trades, connect to a brokerage, size a position,
state an expected return, or substitute for professional financial
advice. Every material claim in a generated report links to a specific,
timestamped source — read the source ledger and critic findings before
acting on anything.
