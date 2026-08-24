"""Transparent DCF valuation.

`build_dcf` never invents a missing input: when the snapshot lacks the free
cash flow it depends on, or the assumptions are internally inconsistent
(e.g. `wacc <= terminal_growth`), it returns a `ValuationUnavailable` naming
every missing input instead of a `ValuationResult`. Every number in a
`ValuationResult` traces back to the evidence IDs of the SEC facts that fed
it (`formula_input_evidence_ids`), and the model exposes a 3x3 WACC /
terminal-growth sensitivity table so a reader can see how fragile the
headline number is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from equity_research.agents.fundamentals import FundamentalSnapshot

DCF_FORMULA = (
    "equity_value = sum(fcf_0*(1+growth)^t / (1+wacc)^t for t in 1..n) "
    "+ [fcf_n*(1+terminal_growth) / (wacc-terminal_growth)] / (1+wacc)^n "
    "- net_debt"
)

_SENSITIVITY_WACC_STEP = Decimal("0.01")
_SENSITIVITY_GROWTH_STEP = Decimal("0.005")
_SENSITIVITY_OFFSETS = (-1, 0, 1)


@dataclass(frozen=True)
class DcfAssumptions:
    """Explicit, reportable DCF inputs. Nothing here is inferred silently."""

    wacc: Decimal
    terminal_growth: Decimal
    fcf_growth_rate: Decimal = Decimal("0.05")
    projection_years: int = 5
    shares_outstanding: Decimal | None = None
    net_debt: Decimal = Decimal("0")


@dataclass(frozen=True)
class ValuationResult:
    ticker: str
    enterprise_value: Decimal
    equity_value: Decimal
    value_per_share: Decimal | None
    projected_fcfs: list[Decimal]
    terminal_value: Decimal
    assumptions: DcfAssumptions
    formula: str
    formula_input_evidence_ids: list[str]
    sensitivity: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True)
class ValuationUnavailable:
    """Returned instead of a `ValuationResult` when inputs are insufficient.

    Not raised as an exception: an unavailable valuation is an expected,
    reportable outcome, not a failure that should abort the run.
    """

    ticker: str
    missing_inputs: list[str]


def build_dcf(
    snapshot: FundamentalSnapshot, assumptions: DcfAssumptions
) -> ValuationResult | ValuationUnavailable:
    missing = _missing_inputs(snapshot, assumptions)
    if missing:
        return ValuationUnavailable(ticker=snapshot.ticker, missing_inputs=missing)

    base_fcf = snapshot.free_cash_flow
    assert base_fcf is not None  # guaranteed by _missing_inputs above

    projected_fcfs = _project_fcfs(
        base_fcf, assumptions.fcf_growth_rate, assumptions.projection_years
    )
    terminal_value = _terminal_value(
        projected_fcfs[-1], assumptions.wacc, assumptions.terminal_growth
    )
    enterprise_value = _present_value(projected_fcfs, assumptions.wacc) + terminal_value / (
        (1 + assumptions.wacc) ** assumptions.projection_years
    )
    equity_value = enterprise_value - assumptions.net_debt
    value_per_share = (
        equity_value / assumptions.shares_outstanding
        if assumptions.shares_outstanding
        else None
    )

    return ValuationResult(
        ticker=snapshot.ticker,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        projected_fcfs=projected_fcfs,
        terminal_value=terminal_value,
        assumptions=assumptions,
        formula=DCF_FORMULA,
        formula_input_evidence_ids=[
            snapshot.operating_cash_flow.evidence.evidence_id,
            snapshot.capital_expenditures.evidence.evidence_id,
        ],
        sensitivity=_sensitivity_table(base_fcf, assumptions),
    )


def _missing_inputs(snapshot: FundamentalSnapshot, assumptions: DcfAssumptions) -> list[str]:
    missing: list[str] = []
    if snapshot.operating_cash_flow is None:
        missing.append("operating_cash_flow")
    if snapshot.capital_expenditures is None:
        missing.append("capital_expenditures")
    if assumptions.projection_years <= 0:
        missing.append("projection_years_must_be_positive")
    if assumptions.wacc <= assumptions.terminal_growth:
        missing.append("wacc_must_exceed_terminal_growth")
    return missing


def _project_fcfs(base_fcf: Decimal, growth: Decimal, years: int) -> list[Decimal]:
    return [base_fcf * (1 + growth) ** year for year in range(1, years + 1)]


def _terminal_value(final_fcf: Decimal, wacc: Decimal, terminal_growth: Decimal) -> Decimal:
    return final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)


def _present_value(cash_flows: list[Decimal], wacc: Decimal) -> Decimal:
    return sum(
        (cash_flow / (1 + wacc) ** year for year, cash_flow in enumerate(cash_flows, start=1)),
        start=Decimal("0"),
    )


def _equity_value_for(
    base_fcf: Decimal, wacc: Decimal, terminal_growth: Decimal, assumptions: DcfAssumptions
) -> Decimal | None:
    if wacc <= terminal_growth:
        return None
    projected = _project_fcfs(base_fcf, assumptions.fcf_growth_rate, assumptions.projection_years)
    terminal = _terminal_value(projected[-1], wacc, terminal_growth)
    enterprise_value = _present_value(projected, wacc) + terminal / (
        (1 + wacc) ** assumptions.projection_years
    )
    return enterprise_value - assumptions.net_debt


def _sensitivity_table(base_fcf: Decimal, assumptions: DcfAssumptions) -> dict[str, Decimal]:
    table: dict[str, Decimal] = {}
    for wacc_offset in _SENSITIVITY_OFFSETS:
        wacc = assumptions.wacc + wacc_offset * _SENSITIVITY_WACC_STEP
        for growth_offset in _SENSITIVITY_OFFSETS:
            terminal_growth = assumptions.terminal_growth + growth_offset * _SENSITIVITY_GROWTH_STEP
            value = _equity_value_for(base_fcf, wacc, terminal_growth, assumptions)
            if value is not None:
                table[f"wacc={wacc},g={terminal_growth}"] = value
    return table
