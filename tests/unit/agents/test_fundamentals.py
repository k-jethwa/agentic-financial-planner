from datetime import date
from decimal import Decimal

from equity_research.agents.fundamentals import build_snapshot, calculate_growth
from equity_research.data.sec import NormalizedFact

CIK = "0000789019"


def _fact(tag: str, value: str, fiscal_year: int, form: str = "10-K") -> NormalizedFact:
    return NormalizedFact(
        tag=tag,
        taxonomy="us-gaap",
        unit="USD",
        value=Decimal(value),
        fiscal_year=fiscal_year,
        fiscal_period="FY",
        period_start=date(fiscal_year - 1, 7, 1),
        period_end=date(fiscal_year, 6, 30),
        accession=f"0000789019-{fiscal_year}-000012",
        filed=date(fiscal_year, 7, 30),
        form=form,
    )


def _full_snapshot():
    return build_snapshot(
        ticker="MSFT",
        cik=CIK,
        revenue=_fact("Revenues", "245122000000", 2024),
        prior_revenue=_fact("Revenues", "211915000000", 2023),
        operating_cash_flow=_fact(
            "NetCashProvidedByUsedInOperatingActivities", "118548000000", 2024
        ),
        capital_expenditures=_fact(
            "PaymentsToAcquirePropertyPlantAndEquipment", "44477000000", 2024
        ),
    )


def test_build_snapshot_attaches_evidence_to_each_fact():
    snapshot = _full_snapshot()
    assert snapshot.revenue.evidence.numeric_value == Decimal("245122000000")
    assert snapshot.revenue.evidence.source_type == "sec_xbrl"
    assert "Revenues" in snapshot.revenue.evidence.locator
    assert snapshot.missing_inputs == []


def test_build_snapshot_reports_missing_inputs():
    snapshot = build_snapshot(ticker="MSFT", cik=CIK, revenue=_fact("Revenues", "100", 2024))
    assert snapshot.missing_inputs == ["operating_cash_flow", "capital_expenditures"]
    assert snapshot.free_cash_flow is None


def test_free_cash_flow_is_operating_cash_flow_minus_capex():
    snapshot = _full_snapshot()
    assert snapshot.free_cash_flow == Decimal("118548000000") - Decimal("44477000000")


def test_calculate_growth_keeps_input_evidence_ids():
    snapshot = _full_snapshot()
    metrics = calculate_growth(snapshot)
    names = {m.name: m for m in metrics}

    growth = names["revenue_growth_yoy"]
    assert growth.value == (Decimal("245122000000") - Decimal("211915000000")) / Decimal(
        "211915000000"
    )
    assert growth.input_evidence_ids == [
        snapshot.revenue.evidence.evidence_id,
        snapshot.prior_revenue.evidence.evidence_id,
    ]

    margin = names["fcf_margin"]
    assert margin.value == snapshot.free_cash_flow / snapshot.revenue.fact.value
    assert set(margin.input_evidence_ids) == {
        snapshot.operating_cash_flow.evidence.evidence_id,
        snapshot.capital_expenditures.evidence.evidence_id,
        snapshot.revenue.evidence.evidence_id,
    }


def test_calculate_growth_omits_metrics_with_missing_inputs():
    snapshot = build_snapshot(ticker="MSFT", cik=CIK, revenue=_fact("Revenues", "100", 2024))
    assert calculate_growth(snapshot) == []
