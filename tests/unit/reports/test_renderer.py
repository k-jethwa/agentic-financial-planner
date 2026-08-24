from datetime import UTC, date, datetime

from equity_research.reports.models import (
    Claim,
    CriticFinding,
    CriticResult,
    InvestmentReport,
    SourceLedgerEntry,
    SynthesisDraft,
    ValuationSummary,
    build_report,
)
from equity_research.reports.renderer import render_report


def _report(**overrides) -> InvestmentReport:
    draft = SynthesisDraft(
        conclusion="MSFT looks fine.",
        bull_thesis=[Claim(text="Revenue grew 10%.", evidence_ids=["e1"])],
        bear_thesis=[],
        catalysts=[],
        risks=[Claim(text="Supply chain risk noted.", evidence_ids=["e2"])],
        what_would_change_the_view=["A change in WACC would move the range."],
    )
    defaults = dict(
        ticker="MSFT",
        question="Assess valuation",
        report_mode="full",
        as_of=date(2024, 8, 1),
        generated_at=datetime(2024, 8, 1, 12, 0, tzinfo=UTC),
        draft=draft,
        valuation=ValuationSummary(available=True, formula="dcf", equity_value="1000"),
        derived_metrics=[],
        sources=[
            SourceLedgerEntry(
                evidence_id="e1",
                source_type="sec_xbrl",
                source_url="https://data.sec.gov/x",
                locator="Revenues FY2024",
                retrieved_at=datetime(2024, 8, 1, tzinfo=UTC),
                published_at=date(2024, 7, 30),
                confidence="high",
            )
        ],
        critic=CriticResult(status="approved", findings=[]),
        warnings=[],
        model_name=None,
        prompts_version="template-v1",
        software_version="0.1.0",
    )
    defaults.update(overrides)
    return build_report(**defaults)


def test_report_contains_disclaimer_and_source_ledger():
    report = _report()
    markdown, payload = render_report(report)

    assert "not investment advice" in markdown.lower()
    assert payload["sources"][0]["evidence_id"] == "e1"


def test_markdown_sections_appear_in_design_spec_order():
    markdown, _ = render_report(_report())
    expected_order = [
        "## Conclusion",
        "## Valuation",
        "## Bull Thesis",
        "## Bear Thesis",
        "## Catalysts",
        "## Risks",
        "## What Would Change the View",
        "## Source Ledger",
        "## Critic Findings",
        "## Warnings",
        "## Reproducibility",
    ]
    positions = [markdown.index(section) for section in expected_order]
    assert positions == sorted(positions)


def test_claims_render_with_their_evidence_ids():
    markdown, _ = render_report(_report())
    assert "Revenue grew 10%. [e1]" in markdown
    assert "Supply chain risk noted. [e2]" in markdown


def test_empty_sections_render_none_not_a_crash():
    markdown, _ = render_report(_report())
    assert "## Bear Thesis\n- none" in markdown
    assert "## Catalysts\n- none" in markdown


def test_unavailable_valuation_renders_missing_inputs():
    report = _report(
        valuation=ValuationSummary(available=False, missing_inputs=["operating_cash_flow"])
    )
    markdown, payload = render_report(report)

    assert "Valuation unavailable" in markdown
    assert "operating_cash_flow" in markdown
    assert payload["valuation"]["available"] is False


def test_critic_findings_and_warnings_are_rendered():
    report = _report(
        critic=CriticResult(
            status="repaired",
            findings=[
                CriticFinding(
                    code="stale_source", severity="warning", message="e1 is stale"
                )
            ],
        ),
        warnings=["operating_cash_flow tag was unavailable"],
    )
    markdown, payload = render_report(report)

    assert "[warning] stale_source: e1 is stale" in markdown
    assert "operating_cash_flow tag was unavailable" in markdown
    assert payload["critic"]["status"] == "repaired"


def test_payload_is_json_serializable_round_trip():
    import json

    _, payload = render_report(_report())
    # model_dump(mode="json") must already be plain JSON-safe types.
    json.dumps(payload)
