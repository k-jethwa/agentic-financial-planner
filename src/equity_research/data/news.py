"""News/RSS adapter: bounded, deduplicated recent public news items.

News is always an optional source: a failed or empty feed degrades the
report to a clearly marked partial result (per design spec) and never
fails the run by itself. Items are deduplicated by canonical URL across
every configured feed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class NewsItem:
    title: str
    url: str
    published_at: datetime | None
    source: str
    summary: str | None = None

    @property
    def item_id(self) -> str:
        return hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:16]


class FeedParser(Protocol):
    def parse(self, url: str) -> object: ...


class NewsClient:
    """Fetches and normalizes items from a bounded set of RSS/Atom feeds."""

    def __init__(self, feed_urls: list[str], *, parser: FeedParser | None = None):
        self._feed_urls = feed_urls
        self._parser = parser or _default_parser()

    def recent_items(self, ticker: str, *, max_items: int = 8) -> list[NewsItem]:
        seen_urls: set[str] = set()
        items: list[NewsItem] = []
        for feed_url in self._feed_urls:
            parsed = self._parser.parse(feed_url.format(ticker=ticker))
            for entry in getattr(parsed, "entries", []):
                url = _canonicalize(entry.get("link", ""))
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                items.append(
                    NewsItem(
                        title=(entry.get("title") or "").strip(),
                        url=url,
                        published_at=_parse_published(entry),
                        source=feed_url,
                        summary=entry.get("summary"),
                    )
                )
        items.sort(
            key=lambda item: item.published_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )
        return items[:max_items]


def _default_parser() -> FeedParser:
    import feedparser

    return feedparser


def _canonicalize(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _parse_published(entry) -> datetime | None:
    parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed_time:
        return None
    return datetime(*parsed_time[:6], tzinfo=UTC)
