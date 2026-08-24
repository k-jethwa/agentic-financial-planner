"""Domain exceptions.

These signal conditions the system must fail closed on rather than paper
over with a guess (e.g. an unresolvable ticker, or a required SEC source
that could not be retrieved).
"""

from __future__ import annotations


class EquityResearchError(Exception):
    """Base class for all domain errors raised by the research assistant."""


class UnsupportedTickerError(EquityResearchError):
    """Raised when a ticker is not a supported US-listed common stock.

    Covers malformed symbols, non-US issuers, and tickers whose CIK cannot
    be resolved via SEC EDGAR.
    """

    def __init__(self, ticker: str, reason: str = "ticker is not supported for v1"):
        self.ticker = ticker
        self.reason = reason
        super().__init__(f"Unsupported ticker '{ticker}': {reason}")


class RequiredSourceUnavailableError(EquityResearchError):
    """Raised when a required SEC source cannot be retrieved.

    A missing required source must invalidate dependent claims rather than
    fall back to a model guess (fail closed, per design spec).
    """

    def __init__(self, source: str, detail: str):
        self.source = source
        self.detail = detail
        super().__init__(f"Required source unavailable: {source} ({detail})")


class InvalidRunTransitionError(EquityResearchError):
    """Raised when a research run's status transition is not permitted."""

    def __init__(self, current: str, requested: str):
        self.current = current
        self.requested = requested
        super().__init__(f"Cannot transition run from '{current}' to '{requested}'")
