"""Typed report contract: the structured investment memo and its findings.

A `Claim` cannot be constructed without at least one evidence ID: "an
output claim is reportable only when it references one or more evidence
IDs" (design spec) is enforced structurally here, not just by convention.
`InvestmentReport` is the terminal artifact the graph produces (assembled
by `build_report`); `reports.renderer.render_report` turns it into the
fixed Markdown/JSON contract from the design spec.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from equity_research.core.evidence import DerivedMetric

DISCLAIMER = (
    "Research and educational use only — not investment advice. This report "
    "is not a recommendation to buy or sell any security, is not personalized "
    "to any individual's financial situation, and must not be the sole basis "
    "for an investment decision."
)


class Claim(BaseModel):
    """One reportable sentence. Never constructable without evidence."""

    model_config = ConfigDict(frozen=True)

    text: str
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim text must not be blank")
        return value

    @property
    def is_numeric(self) -> bool:
        return any(character.isdigit() for character in self.text)


class SynthesisDraft(BaseModel):
    """The synthesizer's structured output, before critic review."""

    model_config = ConfigDict(frozen=True)

    conclusion: str
    bull_thesis: list[Claim] = Field(default_factory=list)
    bear_thesis: list[Claim] = Field(default_factory=list)
    catalysts: list[Claim] = Field(default_factory=list)
    risks: list[Claim] = Field(default_factory=list)
    what_would_change_the_view: list[str] = Field(default_factory=list)

    def all_claims(self) -> list[Claim]:
        return [*self.bull_thesis, *self.bear_thesis, *self.catalysts, *self.risks]

    def without_claims(self, dropped_texts: set[str]) -> SynthesisDraft:
        """Return a copy with every claim in `dropped_texts` removed.

        Used by the critic's one-shot repair: it never invents a fix, it
        only removes the offending claim(s) it already flagged.
        """

        def keep(claims: list[Claim]) -> list[Claim]:
            return [claim for claim in claims if claim.text not in dropped_texts]

        return self.model_copy(
            update={
                "bull_thesis": keep(self.bull_thesis),
                "bear_thesis": keep(self.bear_thesis),
                "catalysts": keep(self.catalysts),
                "risks": keep(self.risks),
            }
        )


class CriticFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: Literal["blocking", "warning"]
    message: str
    claim_text: str | None = None


class CriticResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["approved", "repair_required", "repaired", "unresolved"]
    findings: list[CriticFinding] = Field(default_factory=list)

    @property
    def blocking(self) -> list[CriticFinding]:
        return [finding for finding in self.findings if finding.severity == "blocking"]

    @property
    def warnings(self) -> list[CriticFinding]:
        return [finding for finding in self.findings if finding.severity == "warning"]


class ValuationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    available: bool
    formula: str | None = None
    wacc: str | None = None
    terminal_growth: str | None = None
    equity_value: str | None = None
    value_per_share: str | None = None
    sensitivity: dict[str, str] = Field(default_factory=dict)
    missing_inputs: list[str] = Field(default_factory=list)


class SourceLedgerEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    source_type: str
    source_url: str
    locator: str
    retrieved_at: datetime
    published_at: date | None
    confidence: str


class InvestmentReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    question: str
    report_mode: str
    as_of: date
    generated_at: datetime
    disclaimer: str = DISCLAIMER
    conclusion: str
    bull_thesis: list[Claim]
    bear_thesis: list[Claim]
    catalysts: list[Claim]
    risks: list[Claim]
    what_would_change_the_view: list[str]
    valuation: ValuationSummary
    derived_metrics: list[DerivedMetric]
    sources: list[SourceLedgerEntry]
    critic: CriticResult
    warnings: list[str]
    model_name: str | None
    prompts_version: str
    software_version: str

    @property
    def numeric_claims(self) -> list[Claim]:
        return [claim for claim in self.all_claims() if claim.is_numeric]

    def all_claims(self) -> list[Claim]:
        return [*self.bull_thesis, *self.bear_thesis, *self.catalysts, *self.risks]


def build_report(
    *,
    ticker: str,
    question: str,
    report_mode: str,
    as_of: date,
    generated_at: datetime,
    draft: SynthesisDraft,
    valuation: ValuationSummary,
    derived_metrics: list[DerivedMetric],
    sources: list[SourceLedgerEntry],
    critic: CriticResult,
    warnings: list[str],
    model_name: str | None,
    prompts_version: str,
    software_version: str,
) -> InvestmentReport:
    return InvestmentReport(
        ticker=ticker,
        question=question,
        report_mode=report_mode,
        as_of=as_of,
        generated_at=generated_at,
        conclusion=draft.conclusion,
        bull_thesis=draft.bull_thesis,
        bear_thesis=draft.bear_thesis,
        catalysts=draft.catalysts,
        risks=draft.risks,
        what_would_change_the_view=draft.what_would_change_the_view,
        valuation=valuation,
        derived_metrics=derived_metrics,
        sources=sources,
        critic=critic,
        warnings=warnings,
        model_name=model_name,
        prompts_version=prompts_version,
        software_version=software_version,
    )
