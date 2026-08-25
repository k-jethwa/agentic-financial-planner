import httpx
import pytest

from equity_research.core.exceptions import RequiredSourceUnavailableError
from equity_research.data.cache import DiskHttpCache, TokenBucketLimiter
from equity_research.data.http import CachedHttpClient, HttpFilingHtmlFetcher
from equity_research.data.sec import Filing


class _FakeTransport(httpx.BaseTransport):
    def __init__(self, responses: list[httpx.Response]):
        self._responses = list(responses)
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self._responses.pop(0)


def _client_with(
    cache_dir, *, responses: list[httpx.Response]
) -> tuple[CachedHttpClient, _FakeTransport]:
    transport = _FakeTransport(responses)
    httpx_client = httpx.Client(transport=transport)
    cache = DiskHttpCache(cache_dir)
    limiter = TokenBucketLimiter(rate_per_second=1000, capacity=1000)
    return (
        CachedHttpClient(cache=cache, limiter=limiter, max_retries=3, client=httpx_client),
        transport,
    )


def test_get_caches_successful_response(tmp_path):
    responses = [httpx.Response(200, text='{"ok": true}', request=httpx.Request("GET", "https://x/a"))]
    client, transport = _client_with(tmp_path, responses=responses)

    first = client.get("https://x/a")
    second = client.get("https://x/a")  # served from cache, no second transport call

    assert first.json() == {"ok": True}
    assert second.json() == {"ok": True}
    assert transport.calls == 1


def test_get_retries_5xx_then_succeeds(tmp_path):
    responses = [
        httpx.Response(503, text="", request=httpx.Request("GET", "https://x/b")),
        httpx.Response(200, text="ok", request=httpx.Request("GET", "https://x/b")),
    ]
    client, transport = _client_with(tmp_path, responses=responses)

    response = client.get("https://x/b")
    assert response.text == "ok"
    assert transport.calls == 2


def test_get_raises_required_source_unavailable_after_exhausting_retries(tmp_path):
    responses = [
        httpx.Response(503, text="", request=httpx.Request("GET", "https://x/c")) for _ in range(3)
    ]
    client, _ = _client_with(tmp_path, responses=responses)

    with pytest.raises(RequiredSourceUnavailableError):
        client.get("https://x/c")


def test_filing_html_fetcher_returns_text(tmp_path):
    responses = [
        httpx.Response(200, text="<html>filing</html>", request=httpx.Request("GET", "https://x/f.htm"))
    ]
    client, _ = _client_with(tmp_path, responses=responses)
    fetcher = HttpFilingHtmlFetcher(client)

    from datetime import date

    filing = Filing(accession="a", form="10-K", filing_date=date(2024, 1, 1), url="https://x/f.htm")
    assert fetcher(filing) == "<html>filing</html>"
