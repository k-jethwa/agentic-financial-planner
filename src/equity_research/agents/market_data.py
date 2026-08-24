"""Market Data node: daily adjusted prices and derived return/volatility/
drawdown metrics.

Every derived metric keeps its formula and the evidence ID of the price
observation it depends on; the node never emits a narrative claim that
lacks a backing price observation. Market data failures are non-fatal: a
report can still be produced (with a recorded warning) from filings and
fundamentals alone.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from equity_research.core.evidence import DerivedMetric, Evidence
from equity_research.core.exceptions import EquityResearchError
from equity_research.core.models import ResearchError, ResearchState, TraceEvent
from equity_research.data.market import MarketDataClient, PriceSeries

LOOKBACK_DAYS = 365


def research_market_data(
    state: ResearchState,
    *,
    market_client: MarketDataClient,
    as_of: date | None = None,
) -> dict:
    end = as_of or state.request.as_of_date or datetime.now(UTC).date()
    start = end - timedelta(days=LOOKBACK_DAYS)

    try:
        series = market_client.history(state.request.ticker, start, end)
    except EquityResearchError as exc:
        return _failure(state, str(exc))

    if not series.points:
        return _failure(state, "no price observations returned")

    evidence, metrics = _price_evidence_and_metrics(series)
    return {
        "evidence": [*state.evidence, *evidence],
        "derived_metrics": [*state.derived_metrics, *metrics],
        "trace": [
            *state.trace,
            TraceEvent(node="market_data", detail=f"{len(series.points)} sessions"),
        ],
    }


def _failure(state: ResearchState, message: str) -> dict:
    return {
        "errors": [*state.errors, ResearchError(node="market_data", message=message, fatal=False)],
        "trace": [*state.trace, TraceEvent(node="market_data", status="failed", detail=message)],
    }


def _price_evidence_and_metrics(series: PriceSeries) -> tuple[list[Evidence], list[DerivedMetric]]:
    latest, first = series.points[-1], series.points[0]
    source_url = f"https://finance.yahoo.com/quote/{series.ticker}/history"

    price_evidence = Evidence(
        evidence_id=f"market_data:{series.ticker}:{latest.session_date.isoformat()}",
        claim=(
            f"{series.ticker} closed at {latest.adjusted_close} (adjusted) "
            f"on {latest.session_date}"
        ),
        source_type="market_data",
        source_url=source_url,
        retrieved_at=series.retrieved_at,
        published_at=latest.session_date,
        locator=f"session {latest.session_date.isoformat()}",
        numeric_value=latest.adjusted_close,
        unit="USD",
        confidence="high",
    )

    closes = [point.adjusted_close for point in series.points]
    metrics: list[DerivedMetric] = []

    if first.adjusted_close != 0:
        metrics.append(
            DerivedMetric(
                name="period_return",
                value=(latest.adjusted_close - first.adjusted_close) / first.adjusted_close,
                formula="(latest_close - first_close) / first_close",
                input_evidence_ids=[price_evidence.evidence_id],
                unit="ratio",
                as_of=latest.session_date,
            )
        )

    metrics.append(
        DerivedMetric(
            name="max_drawdown",
            value=_max_drawdown(closes),
            formula="min((close - running_peak) / running_peak)",
            input_evidence_ids=[price_evidence.evidence_id],
            unit="ratio",
            as_of=latest.session_date,
        )
    )

    volatility = _volatility(closes)
    if volatility is not None:
        metrics.append(
            DerivedMetric(
                name="volatility_daily_stdev",
                value=volatility,
                formula="stdev(daily_return_t for t in sessions)",
                input_evidence_ids=[price_evidence.evidence_id],
                unit="ratio",
                as_of=latest.session_date,
            )
        )

    return [price_evidence], metrics


def _max_drawdown(closes: list[Decimal]) -> Decimal:
    peak = closes[0]
    worst = Decimal("0")
    for price in closes:
        peak = max(peak, price)
        if peak != 0:
            worst = min(worst, (price - peak) / peak)
    return worst


def _volatility(closes: list[Decimal]) -> Decimal | None:
    daily_returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1] != 0
    ]
    if not daily_returns:
        return None
    mean = sum(daily_returns) / Decimal(len(daily_returns))
    variance = sum((r - mean) ** 2 for r in daily_returns) / Decimal(len(daily_returns))
    return variance.sqrt()
