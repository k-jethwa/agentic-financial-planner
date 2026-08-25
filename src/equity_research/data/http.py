"""A shared, cached, rate-limited httpx client for production adapters.

Used by the market-data client and the filing-HTML fetcher; SEC requests
keep their own `HttpxSecHttpClient` (`data.sec`) since they must always
carry a named `User-Agent` per SEC policy. Retries only timeout/429/5xx
responses and never bypasses the token-bucket limiter — the same policy as
the SEC adapter, generalized so it is not duplicated per data source.
"""

from __future__ import annotations

import httpx

from equity_research.core.exceptions import RequiredSourceUnavailableError
from equity_research.data.cache import DiskHttpCache, TokenBucketLimiter
from equity_research.data.sec import Filing

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class CachedHttpClient:
    def __init__(
        self,
        *,
        cache: DiskHttpCache,
        limiter: TokenBucketLimiter,
        max_retries: int = 3,
        timeout: float = 15.0,
        headers: dict[str, str] | None = None,
        client: httpx.Client | None = None,
    ):
        self._cache = cache
        self._limiter = limiter
        self._max_retries = max_retries
        self._headers = headers or {}
        self._client = client or httpx.Client(timeout=timeout)

    def get(
        self,
        url: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        cache_key = _cache_key(url, params)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return httpx.Response(200, request=httpx.Request("GET", url), text=cached)

        request_headers = {**self._headers, **(headers or {})}
        last_error = "retries exhausted"
        for _ in range(self._max_retries):
            self._limiter.acquire()
            try:
                response = self._client.get(url, params=params, headers=request_headers)
            except httpx.TimeoutException as exc:
                last_error = f"timeout: {exc}"
                continue
            if response.status_code in RETRYABLE_STATUS:
                last_error = f"status {response.status_code}"
                continue
            response.raise_for_status()
            self._cache.set(cache_key, response.text)
            return response
        raise RequiredSourceUnavailableError(url, last_error)

    def get_text(self, url: str) -> str:
        return self.get(url).text


class HttpFilingHtmlFetcher:
    """A `filings.FilingHtmlFetcher` backed by a `CachedHttpClient`."""

    def __init__(self, http: CachedHttpClient):
        self._http = http

    def __call__(self, filing: Filing) -> str:
        return self._http.get_text(filing.url)


def _cache_key(url: str, params: dict[str, object] | None) -> str:
    if not params:
        return url
    query = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{url}?{query}"
