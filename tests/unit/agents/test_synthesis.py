"""TemplateSynthesisModel: every claim it drafts must cite real ledger evidence."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from equity_research.agents.synthesis import TemplateSynthesisModel
from equity_research.agents.valuation import ValuationUnavailable
from equity_research.core.evidence import DerivedMetric, Evidence
from equity_research.core.models import ReportMode, ResearchRequest


def _request() -> ResearchRequest:
    return ResearchRequest(ticker="MSFT", question="Assess risk", report_mode=ReportMode.FULL)


def _price_evidence() -> Evidence:
    return Evidence(
        evidence_id="market_data:MSFT:2024-07-30",
        claim="MSFT closed at 400",
        source_type="market_data",
        source_url="https://finance.yahoo.com/quote/MSFT/history",
        retrieved_at=datetime(2024, 7, 30, tzinfo=UTC),
        published_at=date(2024, 7, 30),
        locator="session 2024-07-30",
        numeric_value=Decimal("400"),
        unit="USD",
        confidence="high",
    )


def test_all_drafted_claims_cite_evidence_present_in_the_ledger():
    evidence = [_price_evidence()]
    metrics = [
        DerivedMetric(
            name="period_return",
            value=Decimal("0.05"),
            formula="(latest_close - first_close) / first_close",
            input_evidence_ids=[evidence[0].evidence_id],
        )
    ]
    draft = TemplateSynthesisModel().draft(
        request=_request(), evidence=evidence, derived_metrics=metrics, valuation=None
    )

    evidence_ids = {e.evidence_id for e in evidence}
    for claim in draft.all_claims():
        assert set(claim.evidence_ids) <= evidence_ids


def test_negative_growth_lands_in_bear_thesis():
    evidence = [_price_evidence()]
    metrics = [
        DerivedMetric(
            name="revenue_growth_yoy",
            value=Decimal("-0.10"),
            formula="(current - prior) / abs(prior)",
            input_evidence_ids=[evidence[0].evidence_id],
        )
    ]
    draft = TemplateSynthesisModel().draft(
        request=_request(), evidence=evidence, derived_metrics=metrics, valuation=None
    )

    assert any("Revenue grew -10.0%" in c.text for c in draft.bear_thesis)
    assert not any("Revenue grew" in c.text for c in draft.bull_thesis)


def test_draft_with_no_evidence_still_produces_a_conclusion():
    draft = TemplateSynthesisModel().draft(
        request=_request(), evidence=[], derived_metrics=[], valuation=None
    )
    assert draft.conclusion
    assert draft.bull_thesis == []
    assert draft.bear_thesis == []


def test_valuation_unavailable_is_reflected_in_conclusion_and_triggers():
    draft = TemplateSynthesisModel().draft(
        request=_request(),
        evidence=[],
        derived_metrics=[],
        valuation=ValuationUnavailable(ticker="MSFT", missing_inputs=["operating_cash_flow"]),
    )
    assert "operating_cash_flow" in draft.conclusion
    assert draft.what_would_change_the_view


def test_template_synthesis_model_reports_no_llm():
    assert TemplateSynthesisModel().model_name is None
