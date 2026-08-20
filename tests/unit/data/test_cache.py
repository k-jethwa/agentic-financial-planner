from equity_research.data.cache import DiskHttpCache, TokenBucketLimiter


def test_disk_cache_round_trips_by_url(tmp_path):
    cache = DiskHttpCache(tmp_path)
    assert cache.get("https://example.com/a") is None
    cache.set("https://example.com/a", "payload-a")
    cache.set("https://example.com/b", "payload-b")
    assert cache.get("https://example.com/a") == "payload-a"
    assert cache.get("https://example.com/b") == "payload-b"


def test_token_bucket_allows_burst_up_to_capacity():
    limiter = TokenBucketLimiter(rate_per_second=100, capacity=3)
    # Should not block: three tokens are available immediately.
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
