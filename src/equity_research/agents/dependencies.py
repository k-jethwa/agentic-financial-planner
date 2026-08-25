"""Production dependency wiring for the research graph.

Builds real SEC/market/filing/news adapters from `Settings`. Test suites
never call this: they build their own `GraphDependencies` directly with
fakes (see `tests/unit/agents/test_graph.py`), so nothing here needs to be
network-testable in CI. `SecClient` construction fails closed (raises
`ValueError`) if `SEC_USER_AGENT` is not configured, per design spec.
"""

from __future__ import annotations

from equity_research.agents.graph import GraphDependencies
from equity_research.core.config import Settings
from equity_research.data.cache import DiskHttpCache, TokenBucketLimiter
from equity_research.data.http import CachedHttpClient, HttpFilingHtmlFetcher
from equity_research.data.market import MarketDataClient
from equity_research.data.news import NewsClient
from equity_research.data.sec import HttpxSecHttpClient, SecClient
from equity_research.retrieval.vector_store import InMemoryVectorStore, VectorStore

# Free-form Google News RSS search: no API key required. "{ticker}" is
# filled in per-request by NewsClient.recent_items.
DEFAULT_NEWS_FEED_URLS = [
    "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
]

# Conservative request rates, well under each source's documented limits.
SEC_REQUESTS_PER_SECOND = 8
MARKET_REQUESTS_PER_SECOND = 4
FILINGS_REQUESTS_PER_SECOND = 8

PINECONE_INDEX_NAME = "equity-research-filings"


def build_production_graph_dependencies(settings: Settings) -> GraphDependencies:
    sec_client = SecClient(
        http=HttpxSecHttpClient(
            settings.sec_user_agent or "",
            cache=DiskHttpCache(settings.http_cache_dir / "sec"),
            limiter=TokenBucketLimiter(
                rate_per_second=SEC_REQUESTS_PER_SECOND, capacity=SEC_REQUESTS_PER_SECOND
            ),
            max_retries=settings.max_retries,
            timeout=settings.request_timeout_seconds,
        )
    )

    market_client = MarketDataClient(
        http=CachedHttpClient(
            cache=DiskHttpCache(settings.http_cache_dir / "market"),
            limiter=TokenBucketLimiter(
                rate_per_second=MARKET_REQUESTS_PER_SECOND, capacity=MARKET_REQUESTS_PER_SECOND
            ),
            max_retries=settings.max_retries,
            timeout=settings.request_timeout_seconds,
        )
    )

    filings_http = CachedHttpClient(
        cache=DiskHttpCache(settings.http_cache_dir / "filings"),
        limiter=TokenBucketLimiter(
            rate_per_second=FILINGS_REQUESTS_PER_SECOND, capacity=FILINGS_REQUESTS_PER_SECOND
        ),
        max_retries=settings.max_retries,
        timeout=settings.request_timeout_seconds,
        headers={"User-Agent": settings.sec_user_agent} if settings.sec_user_agent else None,
    )

    return GraphDependencies(
        sec_client=sec_client,
        market_client=market_client,
        vector_store=_build_vector_store(settings),
        filing_html_fetcher=HttpFilingHtmlFetcher(filings_http),
        news_client=NewsClient(feed_urls=DEFAULT_NEWS_FEED_URLS),
    )


def _build_vector_store(settings: Settings) -> VectorStore:
    if not settings.has_pinecone_credentials:
        return InMemoryVectorStore()

    from equity_research.retrieval.vector_store import PineconeVectorStore

    assert settings.pinecone_api_key is not None  # guaranteed by has_pinecone_credentials
    return PineconeVectorStore(api_key=settings.pinecone_api_key, index_name=PINECONE_INDEX_NAME)
