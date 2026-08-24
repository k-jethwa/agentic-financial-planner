from datetime import date
from decimal import Decimal

from equity_research.agents.fundamentals import build_snapshot
from equity_research.agents.valuation import DcfAssumptions, ValuationUnavailable, build_dcf
from equity_research.data.sec import NormalizedFact

CIK = "0000789019"


def _fact(tag: str, value: str) -> NormalizedFact:
    return NormalizedFact(
        tag=tag,
        taxonomy="us-gaap",
        unit="USD",
        value=Decimal(value),
        fiscal_year=2024,
        fiscal_period="FY",
        period_start=date(2023, 7, 1),
        period_end=date(2024, 6, 30),
        accession="0000789019-24-000012",
        filed=date(2024, 7, 30),
        form="10-K",
    )


def snapshot_with_cash_flows():
    return build_snapshot(
        ticker="MSFT",
        cik=CIK,
        revenue=_fact("Revenues", "245122000000"),
        operating_cash_flow=_fact("NetCashProvidedByUsedInOperatingActivities", "118548000000"),
        capital_expenditures=_fact("PaymentsToAcquirePropertyPlantAndEquipment", "44477000000"),
    )


def test_dcf_exposes_all_inputs_and_sensitivity():
    result = build_dcf(
        snapshot_with_cash_flows(),
        DcfAssumptions(wacc=Decimal("0.10"), terminal_growth=Decimal("0.025")),
    )
    assert result.formula_input_evidence_ids
    assert result.sensitivity["wacc=0.09,g=0.025"] > 0
    assert result.enterprise_value > 0
    assert len(result.projected_fcfs) == 5
    # 3x3 grid, minus any combos where wacc <= terminal_growth.
    assert len(result.sensitivity) <= 9


def test_dcf_computes_value_per_share_when_shares_given():
    result = build_dcf(
        snapshot_with_cash_flows(),
        DcfAssumptions(
            wacc=Decimal("0.10"),
            terminal_growth=Decimal("0.025"),
            shares_outstanding=Decimal("7400000000"),
        ),
    )
    assert result.value_per_share == result.equity_value / Decimal("7400000000")


def test_dcf_omits_value_per_share_without_shares_outstanding():
    result = build_dcf(
        snapshot_with_cash_flows(),
        DcfAssumptions(wacc=Decimal("0.10"), terminal_growth=Decimal("0.025")),
    )
    assert result.value_per_share is None


def test_dcf_unavailable_when_cash_flow_inputs_missing():
    snapshot = build_snapshot(ticker="MSFT", cik=CIK, revenue=_fact("Revenues", "245122000000"))
    result = build_dcf(
        snapshot, DcfAssumptions(wacc=Decimal("0.10"), terminal_growth=Decimal("0.025"))
    )
    assert isinstance(result, ValuationUnavailable)
    assert set(result.missing_inputs) == {"operating_cash_flow", "capital_expenditures"}


def test_dcf_unavailable_when_wacc_does_not_exceed_terminal_growth():
    result = build_dcf(
        snapshot_with_cash_flows(),
        DcfAssumptions(wacc=Decimal("0.02"), terminal_growth=Decimal("0.025")),
    )
    assert isinstance(result, ValuationUnavailable)
    assert "wacc_must_exceed_terminal_growth" in result.missing_inputs


def test_dcf_never_invents_missing_inputs_as_zero():
    # A snapshot missing capex must not silently treat capex as zero.
    from equity_research.agents.fundamentals import build_snapshot as _build

    snapshot = _build(
        ticker="MSFT",
        cik=CIK,
        revenue=_fact("Revenues", "245122000000"),
        operating_cash_flow=_fact("NetCashProvidedByUsedInOperatingActivities", "118548000000"),
    )
    result = build_dcf(
        snapshot, DcfAssumptions(wacc=Decimal("0.10"), terminal_growth=Decimal("0.025"))
    )
    assert isinstance(result, ValuationUnavailable)
    assert result.missing_inputs == ["capital_expenditures"]
