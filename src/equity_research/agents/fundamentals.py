"""Financial normalization: SEC XBRL facts -> a typed fundamental snapshot.

Every field on `FundamentalSnapshot` carries the `Evidence` record it was
built from, so a snapshot can never drift from its citation trail. Derived
values (growth, margins, ...) keep their formula and input evidence IDs per
`DerivedMetric`. A tag missing from the source facts simply leaves the
corresponding field `None` and drops any metric that depends on it, rather
than inventing a value (fail closed, per design spec).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from equity_research.core.evidence import DerivedMetric, Evidence
from equity_research.data.sec import COMPANY_FACTS_URL, NormalizedFact

REVENUE_GROWTH_FORMULA = "(current_revenue - prior_revenue) / abs(prior_revenue)"
FCF_FORMULA = "operating_cash_flow - capital_expenditures"
FCF_MARGIN_FORMULA = "free_cash_flow / revenue"


@dataclass(frozen=True)
class FinancialLineItem:
    """One normalized SEC fact plus the evidence record backing it."""

    fact: NormalizedFact
    evidence: Evidence


@dataclass(frozen=True)
class FundamentalSnapshot:
    """A ticker's normalized annual financial facts, each evidence-linked."""

    ticker: str
    cik: str
    revenue: FinancialLineItem | None = None
    prior_revenue: FinancialLineItem | None = None
    net_income: FinancialLineItem | None = None
    operating_cash_flow: FinancialLineItem | None = None
    capital_expenditures: FinancialLineItem | None = None

    @property
    def free_cash_flow(self):
        if self.operating_cash_flow is None or self.capital_expenditures is None:
            return None
        return self.operating_cash_flow.fact.value - self.capital_expenditures.fact.value

    @property
    def missing_inputs(self) -> list[str]:
        missing = []
        if self.revenue is None:
            missing.append("revenue")
        if self.operating_cash_flow is None:
            missing.append("operating_cash_flow")
        if self.capital_expenditures is None:
            missing.append("capital_expenditures")
        return missing


def build_snapshot(
    ticker: str,
    cik: str,
    *,
    revenue: NormalizedFact | None = None,
    prior_revenue: NormalizedFact | None = None,
    net_income: NormalizedFact | None = None,
    operating_cash_flow: NormalizedFact | None = None,
    capital_expenditures: NormalizedFact | None = None,
    retrieved_at: datetime | None = None,
) -> FundamentalSnapshot:
    """Wrap already-fetched `NormalizedFact`s into an evidence-linked snapshot.

    Fetching (via `SecClient.latest_annual_fact`) is the caller's
    responsibility so this stays a pure, easily testable normalization step.
    """
    retrieved_at = retrieved_at or datetime.now(UTC)

    def wrap(tag: str, fact: NormalizedFact | None) -> FinancialLineItem | None:
        if fact is None:
            return None
        return FinancialLineItem(fact=fact, evidence=_fact_evidence(cik, tag, fact, retrieved_at))

    return FundamentalSnapshot(
        ticker=ticker,
        cik=cik,
        revenue=wrap("Revenues", revenue),
        prior_revenue=wrap("Revenues", prior_revenue),
        net_income=wrap("NetIncomeLoss", net_income),
        operating_cash_flow=wrap(
            "NetCashProvidedByUsedInOperatingActivities", operating_cash_flow
        ),
        capital_expenditures=wrap(
            "PaymentsToAcquirePropertyPlantAndEquipment", capital_expenditures
        ),
    )


def calculate_growth(snapshot: FundamentalSnapshot) -> list[DerivedMetric]:
    """Compute deterministic, evidence-linked derived metrics for a snapshot.

    A metric is only returned when every fact it depends on is present;
    partial data yields a shorter list, never a guessed value.
    """
    metrics: list[DerivedMetric] = []

    if snapshot.revenue is not None and snapshot.prior_revenue is not None:
        current = snapshot.revenue.fact.value
        prior = snapshot.prior_revenue.fact.value
        if prior != 0:
            metrics.append(
                DerivedMetric(
                    name="revenue_growth_yoy",
                    value=(current - prior) / abs(prior),
                    formula=REVENUE_GROWTH_FORMULA,
                    input_evidence_ids=[
                        snapshot.revenue.evidence.evidence_id,
                        snapshot.prior_revenue.evidence.evidence_id,
                    ],
                    unit="ratio",
                    as_of=snapshot.revenue.fact.period_end,
                )
            )

    fcf = snapshot.free_cash_flow
    if fcf is not None and snapshot.revenue is not None and snapshot.revenue.fact.value != 0:
        metrics.append(
            DerivedMetric(
                name="fcf_margin",
                value=fcf / snapshot.revenue.fact.value,
                formula=FCF_MARGIN_FORMULA,
                input_evidence_ids=[
                    snapshot.operating_cash_flow.evidence.evidence_id,
                    snapshot.capital_expenditures.evidence.evidence_id,
                    snapshot.revenue.evidence.evidence_id,
                ],
                unit="ratio",
                as_of=snapshot.revenue.fact.period_end,
            )
        )

    return metrics


def _fact_evidence(cik: str, tag: str, fact: NormalizedFact, retrieved_at: datetime) -> Evidence:
    return Evidence(
        evidence_id=f"sec_xbrl:{cik}:{tag}:{fact.fiscal_year}",
        claim=f"{tag} for FY{fact.fiscal_year} was {fact.value} {fact.unit}",
        source_type="sec_xbrl",
        source_url=COMPANY_FACTS_URL.format(cik10=cik.zfill(10)),
        retrieved_at=retrieved_at,
        published_at=fact.filed,
        locator=f"{tag} FY{fact.fiscal_year} ({fact.form}, accession {fact.accession})",
        numeric_value=fact.value,
        unit=fact.unit,
        confidence="high",
    )
