"""Critic tests: citation coverage, freshness, contradiction, and
overconfidence checks, plus the one-shot claim-dropping repair."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from equity_research.agents.critic import critique, repair
from equity_research.core.evidence import Evidence
from equity_research.reports.models import Claim, SynthesisDraft

AS_OF = date(2024, 8, 1)


def _evidence(
    evidence_id: str,
    *,
    source_type: str = "market_data",
    locator: str = "session 2024-07-30",
    numeric_value: Decimal | None = Decimal("100"),
    published_at: date | None = date(2024, 7, 30),
    retrieved_at: datetime | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        claim="a claim",
        source_type=source_type,
        source_url="https://example.com/x",
        retrieved_at=retrieved_at or datetime(2024, 7, 30, tzinfo=UTC),
        published_at=published_at,
        locator=locator,
        numeric_value=numeric_value,
        confidence="high",
    )


def test_state_with_unsupported_claim_is_blocked():
    draft = SynthesisDraft(
        conclusion="conclusion",
        bull_thesis=[Claim(text="Revenue rose 10%.", evidence_ids=["e-missing"])],
    )
    result = critique(draft, evidence=[], as_of=AS_OF)

    assert result.status == "repair_required"
    assert result.blocking
    assert result.blocking[0].code == "unsupported_claim"


def test_claim_backed_by_real_evidence_is_approved():
    evidence = [_evidence("e1")]
    draft = SynthesisDraft(
        conclusion="conclusion",
        bull_thesis=[Claim(text="Revenue rose 10%.", evidence_ids=["e1"])],
    )
    result = critique(draft, evidence=evidence, as_of=AS_OF)

    assert result.status == "approved"
    assert result.blocking == []


def test_repair_drops_only_the_unsupported_claim():
    evidence = [_evidence("e1")]
    good_claim = Claim(text="Revenue rose 10%.", evidence_ids=["e1"])
    bad_claim = Claim(text="Margin expanded to 50%.", evidence_ids=["e-missing"])
    draft = SynthesisDraft(conclusion="conclusion", bull_thesis=[good_claim, bad_claim])

    first_pass = critique(draft, evidence=evidence, as_of=AS_OF)
    repaired_draft = repair(draft, first_pass)
    second_pass = critique(repaired_draft, evidence=evidence, as_of=AS_OF)

    assert repaired_draft.bull_thesis == [good_claim]
    assert second_pass.status == "approved"


def test_overconfident_language_is_flagged_as_a_warning_not_blocking():
    evidence = [_evidence("e1")]
    draft = SynthesisDraft(
        conclusion="conclusion",
        bull_thesis=[Claim(text="This stock is guaranteed to rise 10%.", evidence_ids=["e1"])],
    )
    result = critique(draft, evidence=evidence, as_of=AS_OF)

    assert result.status == "approved"  # a warning alone does not block
    assert any(f.code == "overconfident_language" for f in result.warnings)


def test_stale_required_source_is_flagged():
    stale_evidence = _evidence("e1", source_type="sec_xbrl", published_at=date(2022, 1, 1))
    draft = SynthesisDraft(conclusion="conclusion")
    result = critique(draft, evidence=[stale_evidence], as_of=date(2024, 8, 1))

    assert any(f.code == "stale_source" for f in result.warnings)


def test_fresh_required_source_is_not_flagged():
    fresh_evidence = _evidence("e1", source_type="sec_xbrl", published_at=date(2024, 7, 1))
    draft = SynthesisDraft(conclusion="conclusion")
    result = critique(draft, evidence=[fresh_evidence], as_of=date(2024, 8, 1))

    assert not any(f.code == "stale_source" for f in result.findings)


def test_market_data_freshness_is_not_checked():
    # Only "required" SEC sources are checked for staleness; market data and
    # news are expected to roll forward and are not treated as stale.
    old_price = _evidence("e1", source_type="market_data", published_at=date(2020, 1, 1))
    draft = SynthesisDraft(conclusion="conclusion")
    result = critique(draft, evidence=[old_price], as_of=AS_OF)

    assert result.findings == []


def test_contradictory_values_for_the_same_fact_are_blocking():
    conflicting = [
        _evidence(
            "e1", source_type="sec_xbrl", locator="Revenues FY2024", numeric_value=Decimal("100")
        ),
        _evidence(
            "e2", source_type="sec_xbrl", locator="Revenues FY2024", numeric_value=Decimal("999")
        ),
    ]
    draft = SynthesisDraft(conclusion="conclusion")
    result = critique(draft, evidence=conflicting, as_of=AS_OF)

    assert result.status == "repair_required"
    assert any(f.code == "contradictory_values" for f in result.blocking)


def test_contradiction_with_no_claim_to_drop_survives_repair_as_unresolved():
    conflicting = [
        _evidence(
            "e1", source_type="sec_xbrl", locator="Revenues FY2024", numeric_value=Decimal("100")
        ),
        _evidence(
            "e2", source_type="sec_xbrl", locator="Revenues FY2024", numeric_value=Decimal("999")
        ),
    ]
    draft = SynthesisDraft(conclusion="conclusion")
    first_pass = critique(draft, evidence=conflicting, as_of=AS_OF)
    repaired_draft = repair(draft, first_pass)  # nothing to drop: no claim caused it
    second_pass = critique(repaired_draft, evidence=conflicting, as_of=AS_OF)

    assert second_pass.blocking  # still unresolved after the one repair pass
