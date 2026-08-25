"""Synthesizer: builds the structured memo strictly from ledger evidence.

`Claim.evidence_ids` cannot be empty (see `reports.models`), so nothing
this module produces can be unlinked from the evidence ledger. The
`SynthesisModel` protocol keeps prose generation swappable: v1 ships only
`TemplateSynthesisModel`, a fully deterministic writer built directly from
evidence and derived metrics — nothing to hallucinate, and fully testable
without a live API (per the design spec's CI constraint: no test may
depend on a live model or secret). An LLM-backed model (built on
`agents.llm.create_research_llm` and `wrap_untrusted_source`, so retrieved
filing/news text is always passed as delimited data, never instructions)
can be added later behind this same protocol without touching the graph;
`agents.critic` remains the sole gate that decides whether any drafted
claim is fit to report.
"""

from __future__ import annotations

from typing import Protocol

from equity_research.agents.valuation import ValuationResult, ValuationUnavailable
from equity_research.core.evidence import DerivedMetric, Evidence
from equity_research.core.models import ResearchRequest
from equity_research.reports.models import Claim, SynthesisDraft

PROMPTS_VERSION = "template-v1"

_MAX_CATALYSTS = 3
_MAX_RISKS = 3
_EXCERPT_CHARS = 280


class SynthesisModel(Protocol):
    @property
    def model_name(self) -> str | None: ...

    def draft(
        self,
        *,
        request: ResearchRequest,
        evidence: list[Evidence],
        derived_metrics: list[DerivedMetric],
        valuation: ValuationResult | ValuationUnavailable | None,
    ) -> SynthesisDraft: ...


class TemplateSynthesisModel:
    """A deterministic, no-model synthesizer built entirely from the ledger."""

    @property
    def model_name(self) -> str | None:
        return None

    def draft(
        self,
        *,
        request: ResearchRequest,
        evidence: list[Evidence],
        derived_metrics: list[DerivedMetric],
        valuation: ValuationResult | ValuationUnavailable | None,
    ) -> SynthesisDraft:
        by_type = _group_by_source_type(evidence)
        metrics_by_name = {metric.name: metric for metric in derived_metrics}

        bull, bear = _thesis_claims(metrics_by_name, by_type, valuation)
        return SynthesisDraft(
            conclusion=_conclusion(request, metrics_by_name, valuation, len(evidence)),
            bull_thesis=bull,
            bear_thesis=bear,
            catalysts=_catalyst_claims(by_type.get("news", [])),
            risks=_risk_claims(by_type.get("sec_filing", []), metrics_by_name),
            what_would_change_the_view=_triggers(valuation),
        )


def _group_by_source_type(evidence: list[Evidence]) -> dict[str, list[Evidence]]:
    groups: dict[str, list[Evidence]] = {}
    for item in evidence:
        groups.setdefault(item.source_type, []).append(item)
    return groups


def _thesis_claims(
    metrics: dict[str, DerivedMetric],
    by_type: dict[str, list[Evidence]],
    valuation: ValuationResult | ValuationUnavailable | None,
) -> tuple[list[Claim], list[Claim]]:
    bull: list[Claim] = []
    bear: list[Claim] = []

    growth = metrics.get("revenue_growth_yoy")
    if growth is not None:
        claim = Claim(
            text=f"Revenue grew {growth.value:.1%} year over year.",
            evidence_ids=list(growth.input_evidence_ids),
        )
        (bull if growth.value >= 0 else bear).append(claim)

    fcf_margin = metrics.get("fcf_margin")
    if fcf_margin is not None:
        claim = Claim(
            text=f"Free cash flow margin was {fcf_margin.value:.1%} of revenue.",
            evidence_ids=list(fcf_margin.input_evidence_ids),
        )
        (bull if fcf_margin.value >= 0 else bear).append(claim)

    market_evidence = by_type.get("market_data", [])
    period_return = metrics.get("period_return")
    if period_return is not None and market_evidence:
        claim = Claim(
            text=(
                "The stock's price return over the observed window was "
                f"{period_return.value:.1%}."
            ),
            evidence_ids=[market_evidence[0].evidence_id],
        )
        (bull if period_return.value >= 0 else bear).append(claim)

    drawdown = metrics.get("max_drawdown")
    if drawdown is not None and market_evidence and drawdown.value < 0:
        bear.append(
            Claim(
                text=(
                    f"The stock experienced a maximum drawdown of {drawdown.value:.1%} "
                    "over the observed window."
                ),
                evidence_ids=[market_evidence[0].evidence_id],
            )
        )

    if isinstance(valuation, ValuationResult) and market_evidence and valuation.value_per_share:
        latest_price = market_evidence[0].numeric_value
        if latest_price:
            gap = (valuation.value_per_share - latest_price) / latest_price
            claim = Claim(
                text=(
                    f"The DCF value per share of {valuation.value_per_share:.2f} implies "
                    f"{'upside' if gap >= 0 else 'downside'} of {abs(gap):.1%} versus the "
                    f"last observed price of {latest_price}."
                ),
                evidence_ids=[
                    market_evidence[0].evidence_id,
                    *valuation.formula_input_evidence_ids,
                ],
            )
            (bull if gap >= 0 else bear).append(claim)

    return bull, bear


def _catalyst_claims(news_evidence: list[Evidence]) -> list[Claim]:
    return [
        Claim(text=f"Recent news: {item.claim}", evidence_ids=[item.evidence_id])
        for item in news_evidence[:_MAX_CATALYSTS]
    ]


def _risk_claims(
    filing_evidence: list[Evidence], metrics: dict[str, DerivedMetric]
) -> list[Claim]:
    risks = [
        Claim(
            text=(
                f"Filing risk disclosure ({item.locator}): "
                f"{(item.excerpt or item.claim)[:_EXCERPT_CHARS]}"
            ),
            evidence_ids=[item.evidence_id],
        )
        for item in filing_evidence
        if "risk" in item.locator.lower()
    ][:_MAX_RISKS]

    volatility = metrics.get("volatility_daily_stdev")
    if volatility is not None:
        risks.append(
            Claim(
                text=f"Observed daily price volatility (stdev) was {volatility.value:.2%}.",
                evidence_ids=list(volatility.input_evidence_ids),
            )
        )
    return risks


def _conclusion(
    request: ResearchRequest,
    metrics: dict[str, DerivedMetric],
    valuation: ValuationResult | ValuationUnavailable | None,
    evidence_count: int,
) -> str:
    if isinstance(valuation, ValuationResult):
        values = list(valuation.sensitivity.values()) or [valuation.equity_value]
        low, high = min(values), max(values)
        valuation_text = (
            f"a DCF-implied equity value range of roughly {low:,.0f} to {high:,.0f} "
            f"(base case {valuation.equity_value:,.0f})"
        )
    elif isinstance(valuation, ValuationUnavailable):
        valuation_text = f"no DCF valuation (missing inputs: {', '.join(valuation.missing_inputs)})"
    else:
        valuation_text = "no valuation was attempted"

    growth = metrics.get("revenue_growth_yoy")
    growth_text = (
        f"revenue growth of {growth.value:.1%} year over year"
        if growth is not None
        else "no revenue growth figure available"
    )

    return (
        f"{request.ticker} research based on {evidence_count} cited source(s) shows "
        f"{growth_text} and {valuation_text}. This is a structured research output with "
        "an explicit confidence band determined by data completeness, not a personalized "
        "recommendation — see the source ledger and critic findings below for every "
        "citation and unresolved limitation."
    )


def _triggers(valuation: ValuationResult | ValuationUnavailable | None) -> list[str]:
    if isinstance(valuation, ValuationResult):
        assumptions = valuation.assumptions
        return [
            f"A sustained change in the discount rate away from the assumed WACC of "
            f"{assumptions.wacc:.1%} would move the DCF range shown above.",
            f"A free-cash-flow growth trajectory materially above or below the assumed "
            f"{assumptions.fcf_growth_rate:.1%} would change the base-case value.",
        ]
    return [
        "Publication of the missing financial data required for a DCF would allow a "
        "valuation range to be produced."
    ]
