"""Deterministic Markdown/JSON rendering of the fixed report contract.

Section order matches the design spec exactly: scope/disclaimer, one-
paragraph conclusion, business/financial snapshot placeholders, valuation,
evidence-backed theses, "what would change the view", source ledger plus
critic findings, and trace/reproducibility metadata. Rendering never
inspects untrusted source text directly — every claim and source entry it
prints came from the typed `InvestmentReport`, already validated.
"""

from __future__ import annotations

from equity_research.reports.models import (
    Claim,
    CriticResult,
    InvestmentReport,
    SourceLedgerEntry,
    ValuationSummary,
)


def render_report(report: InvestmentReport) -> tuple[str, dict]:
    """Render both the machine-readable payload and the human-readable memo."""
    payload = report.model_dump(mode="json")
    return _render_markdown(report), payload


def _render_markdown(report: InvestmentReport) -> str:
    warning_lines = [f"- {warning}" for warning in report.warnings] or ["- none"]
    lines: list[str] = [
        f"# {report.ticker} Research Memo",
        "",
        f"**{report.disclaimer}**",
        "",
        f"- Ticker: {report.ticker}",
        f"- Question: {report.question}",
        f"- Report mode: {report.report_mode}",
        f"- As of: {report.as_of.isoformat()}",
        f"- Generated at: {report.generated_at.isoformat()}",
        "",
        "## Conclusion",
        report.conclusion,
        "",
        "## Valuation",
        _render_valuation(report.valuation),
        "",
        "## Bull Thesis",
        *_render_claims(report.bull_thesis),
        "",
        "## Bear Thesis",
        *_render_claims(report.bear_thesis),
        "",
        "## Catalysts",
        *_render_claims(report.catalysts),
        "",
        "## Risks",
        *_render_claims(report.risks),
        "",
        "## What Would Change the View",
        *(f"- {trigger}" for trigger in report.what_would_change_the_view),
        "",
        "## Source Ledger",
        *_render_sources(report.sources),
        "",
        "## Critic Findings",
        *_render_critic(report.critic),
        "",
        "## Warnings",
        *warning_lines,
        "",
        "## Reproducibility",
        f"- Model: {report.model_name or 'template (no LLM)'}",
        f"- Prompts version: {report.prompts_version}",
        f"- Software version: {report.software_version}",
    ]
    return "\n".join(lines)


def _render_claims(claims: list[Claim]) -> list[str]:
    if not claims:
        return ["- none"]
    return [f"- {claim.text} [{', '.join(claim.evidence_ids)}]" for claim in claims]


def _render_sources(sources: list[SourceLedgerEntry]) -> list[str]:
    if not sources:
        return ["- none"]
    lines = []
    for source in sources:
        published = f", published {source.published_at.isoformat()}" if source.published_at else ""
        lines.append(
            f"- `{source.evidence_id}` ({source.source_type}, confidence={source.confidence}): "
            f"{source.source_url} — {source.locator} "
            f"(retrieved {source.retrieved_at.isoformat()}{published})"
        )
    return lines


def _render_critic(critic: CriticResult) -> list[str]:
    lines = [f"- Status: {critic.status}"]
    if not critic.findings:
        return [*lines, "- No findings."]
    return [*lines, *(f"- [{f.severity}] {f.code}: {f.message}" for f in critic.findings)]


def _render_valuation(valuation: ValuationSummary) -> str:
    if not valuation.available:
        missing = ", ".join(valuation.missing_inputs) or "unspecified"
        return f"Valuation unavailable. Missing inputs: {missing}."

    lines = [
        f"Formula: `{valuation.formula}`",
        f"Assumptions: WACC={valuation.wacc}, terminal growth={valuation.terminal_growth}",
        f"Equity value: {valuation.equity_value}",
    ]
    if valuation.value_per_share:
        lines.append(f"Value per share: {valuation.value_per_share}")
    if valuation.sensitivity:
        lines.append("Sensitivity:")
        lines.extend(f"  - {key}: {value}" for key, value in valuation.sensitivity.items())
    return "\n".join(lines)
