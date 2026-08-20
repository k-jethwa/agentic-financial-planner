"""Rate limiting and on-disk HTTP caching shared by data adapters.

Every adapter that calls a live source is expected to go through a
`TokenBucketLimiter` (never bypass a provider's rate limit) and a
`DiskHttpCache` (never re-fetch a URL already answered in this cache).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path


class TokenBucketLimiter:
    """A single-process token bucket for outbound request throttling."""

    def __init__(self, rate_per_second: float, capacity: int | None = None):
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._rate = rate_per_second
        self._capacity = capacity or max(1, int(rate_per_second))
        self._tokens = float(self._capacity)
        self._last_refill = time.monotonic()

    def acquire(self) -> None:
        """Block, if necessary, until a token is available, then consume one."""
        while True:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._last_refill = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            if self._tokens >= 1:
                self._tokens -= 1
                return
            time.sleep(max(0.0, (1 - self._tokens) / self._rate))


class DiskHttpCache:
    """A flat-file cache keyed by the SHA-256 hash of the request URL."""

    def __init__(self, cache_dir: Path | str):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, url: str) -> str | None:
        path = self._path(url)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def set(self, url: str, content: str) -> None:
        self._path(url).write_text(content, encoding="utf-8")

    def _path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.cache"
