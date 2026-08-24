"""SEC EDGAR adapter: ticker/CIK resolution, filings, and companyfacts XBRL.

Production requests always carry a named `User-Agent` (SEC policy), are
throttled through a token bucket, retry only on timeout/429/5xx, and are
served from a disk cache when available. A required SEC source that
cannot be retrieved raises `RequiredSourceUnavailableError` rather than
falling back to a guessed value (fail closed, per the design spec).

`SecClient` itself depends only on the small `HttpClient` protocol below,
so tests can inject a fixture-backed fake instead of making network calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol

import httpx

from equity_research.core.exceptions import RequiredSourceUnavailableError, UnsupportedTickerError
from equity_research.data.cache import DiskHttpCache, TokenBucketLimiter

TICKER_INDEX_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

# Only annual, full-year observations count as "the" annual fact.
ANNUAL_FORMS = {"10-K", "10-K/A"}
ANNUAL_FISCAL_PERIOD = "FY"


class HttpResponse(Protocol):
    status_code: int

    def json(self) -> dict: ...


class HttpClient(Protocol):
    def get(self, url: str, headers: dict[str, str] | None = None) -> HttpResponse: ...


@dataclass(frozen=True)
class Filing:
    accession: str
    form: str
    filing_date: date
    url: str


@dataclass(frozen=True)
class NormalizedFact:
    """One standardized XBRL observation with its reporting context."""

    tag: str
    taxonomy: str
    unit: str
    value: Decimal
    fiscal_year: int
    fiscal_period: str
    period_start: date | None
    period_end: date
    accession: str
    filed: date
    form: str


class HttpxSecHttpClient:
    """Production `HttpClient`: named User-Agent, bounded retries, disk cache."""

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        user_agent: str,
        *,
        cache: DiskHttpCache,
        limiter: TokenBucketLimiter,
        max_retries: int = 3,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ):
        if not user_agent:
            raise ValueError("SEC_USER_AGENT is required to make SEC requests")
        self._user_agent = user_agent
        self._cache = cache
        self._limiter = limiter
        self._max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)

    def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        cached = self._cache.get(url)
        if cached is not None:
            return httpx.Response(200, request=httpx.Request("GET", url), text=cached)

        request_headers = {"User-Agent": self._user_agent, **(headers or {})}
        last_error = "retries exhausted"
        for _ in range(self._max_retries):
            self._limiter.acquire()
            try:
                response = self._client.get(url, headers=request_headers)
            except httpx.TimeoutException as exc:
                last_error = f"timeout: {exc}"
                continue
            if response.status_code in self.RETRYABLE_STATUS:
                last_error = f"status {response.status_code}"
                continue
            response.raise_for_status()
            self._cache.set(url, response.text)
            return response
        raise RequiredSourceUnavailableError(url, last_error)


class SecClient:
    def __init__(self, http: HttpClient):
        self._http = http

    def resolve_cik(self, ticker: str) -> str:
        index = self._http.get(TICKER_INDEX_URL).json()
        for entry in index.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
        raise UnsupportedTickerError(ticker, "CIK not found in SEC ticker index")

    def company_facts(self, cik: str) -> dict:
        url = COMPANY_FACTS_URL.format(cik10=cik.zfill(10))
        return self._http.get(url).json()

    def filings(self, cik: str) -> list[Filing]:
        url = SUBMISSIONS_URL.format(cik10=cik.zfill(10))
        payload = self._http.get(url).json()
        recent = payload.get("filings", {}).get("recent", {})
        cik_int = str(int(cik))

        filings: list[Filing] = []
        for accession, form, filed, primary_doc in zip(
            recent.get("accessionNumber", []),
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("primaryDocument", []),
            strict=True,
        ):
            accession_no_dashes = accession.replace("-", "")
            filings.append(
                Filing(
                    accession=accession,
                    form=form,
                    filing_date=date.fromisoformat(filed),
                    url=(
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik_int}/{accession_no_dashes}/{primary_doc}"
                    ),
                )
            )
        return filings

    def latest_annual_fact(self, cik: str, tag: str, taxonomy: str = "us-gaap") -> NormalizedFact:
        return self.annual_facts(cik, tag, taxonomy, limit=1)[0]

    def annual_facts(
        self, cik: str, tag: str, taxonomy: str = "us-gaap", *, limit: int = 1
    ) -> list[NormalizedFact]:
        """The `limit` most recent annual (10-K/FY) observations, newest first.

        Used, e.g., to pull both the current and prior fiscal year's revenue
        for a year-over-year growth calculation.
        """
        facts = self.company_facts(cik)
        try:
            tag_data = facts["facts"][taxonomy][tag]
        except KeyError as exc:
            raise RequiredSourceUnavailableError(
                f"companyfacts:{cik}", f"missing tag {taxonomy}:{tag}"
            ) from exc

        candidates: list[tuple[str, dict]] = [
            (unit, entry)
            for unit, entries in tag_data.get("units", {}).items()
            for entry in entries
            if entry.get("form") in ANNUAL_FORMS and entry.get("fp") == ANNUAL_FISCAL_PERIOD
        ]
        if not candidates:
            raise RequiredSourceUnavailableError(
                f"companyfacts:{cik}", f"no annual (10-K/FY) observation for {taxonomy}:{tag}"
            )

        ranked = sorted(candidates, key=lambda pair: pair[1]["fy"], reverse=True)
        return [
            NormalizedFact(
                tag=tag,
                taxonomy=taxonomy,
                unit=unit,
                value=Decimal(str(entry["val"])),
                fiscal_year=entry["fy"],
                fiscal_period=entry["fp"],
                period_start=date.fromisoformat(entry["start"]) if entry.get("start") else None,
                period_end=date.fromisoformat(entry["end"]),
                accession=entry["accn"],
                filed=date.fromisoformat(entry["filed"]),
                form=entry["form"],
            )
            for unit, entry in ranked[:limit]
        ]
