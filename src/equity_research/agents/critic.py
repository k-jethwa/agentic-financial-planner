"""Critic: the sole gate a drafted memo passes through before it can become
the final report.

Checks citation coverage (every evidence ID a claim cites must resolve to
a real ledger entry), source freshness, contradictory numeric facts, and
overconfident/false-certainty language. On a blocking citation finding it
can repair a draft exactly once, by dropping the offending claim(s) —
never by inventing a fix — then re-runs the checks. Anything still
blocking after that (e.g. a data-integrity contradiction with no single
claim to drop) is left for the caller to record as an unresolved report
warning instead of looping indefinitely.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from equity_research.core.evidence import Evidence
from equity_research.reports.models import Claim, CriticFinding, CriticResult, SynthesisDraft

# Roughly one annual reporting cycle plus a filing lag; a required SEC
# source older than this is worth flagging even if it is the best we have.
STALE_AFTER_DAYS = 400

_OVERCONFIDENT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bguarantee[sd]?\b",
        r"\bcertain(ly)? to\b",
        r"\brisk[- ]free\b",
        r"\bcannot lose\b",
        r"\bwill definitely\b",
        r"\bnever fails?\b",
        r"\b100% (safe|certain|guaranteed)\b",
    )
)


def critique(draft: SynthesisDraft, evidence: list[Evidence], as_of: date) -> CriticResult:
    """Run every check against a draft and return an approved/repair-required result."""
    evidence_by_id = {item.evidence_id: item for item in evidence}
    findings: list[CriticFinding] = []

    for claim in draft.all_claims():
        findings.extend(_citation_findings(claim, evidence_by_id))
        findings.extend(_overconfidence_findings(claim))

    findings.extend(_freshness_findings(evidence, as_of))
    findings.extend(_contradiction_findings(evidence))

    status = "repair_required" if any(f.severity == "blocking" for f in findings) else "approved"
    return CriticResult(status=status, findings=findings)


def repair(draft: SynthesisDraft, result: CriticResult) -> SynthesisDraft:
    """Drop every claim named in a blocking finding. Never invents a fix."""
    dropped_texts = {finding.claim_text for finding in result.blocking if finding.claim_text}
    return draft.without_claims(dropped_texts)


def _citation_findings(claim: Claim, evidence_by_id: dict[str, Evidence]) -> list[CriticFinding]:
    return [
        CriticFinding(
            code="unsupported_claim",
            severity="blocking",
            message=f"claim cites unknown evidence ID {evidence_id!r}: {claim.text!r}",
            claim_text=claim.text,
        )
        for evidence_id in claim.evidence_ids
        if evidence_id not in evidence_by_id
    ]


def _overconfidence_findings(claim: Claim) -> list[CriticFinding]:
    for pattern in _OVERCONFIDENT_PATTERNS:
        if pattern.search(claim.text):
            return [
                CriticFinding(
                    code="overconfident_language",
                    severity="warning",
                    message=f"claim uses absolute/certainty language: {claim.text!r}",
                    claim_text=claim.text,
                )
            ]
    return []


def _freshness_findings(evidence: list[Evidence], as_of: date) -> list[CriticFinding]:
    findings: list[CriticFinding] = []
    for item in evidence:
        if item.source_type not in ("sec_filing", "sec_xbrl"):
            continue
        reference_date = item.published_at or item.retrieved_at.date()
        if (as_of - reference_date) > timedelta(days=STALE_AFTER_DAYS):
            findings.append(
                CriticFinding(
                    code="stale_source",
                    severity="warning",
                    message=(
                        f"{item.evidence_id} is dated {reference_date}, more than "
                        f"{STALE_AFTER_DAYS} days before as-of {as_of}"
                    ),
                )
            )
    return findings


def _contradiction_findings(evidence: list[Evidence]) -> list[CriticFinding]:
    """Flag two evidence items that describe the same fact with different values.

    "Same fact" is approximated as (source_type, locator): two SEC facts or
    filing passages filed under the same tag/section that disagree on a
    numeric value are a data-integrity problem, not something a claim-level
    repair can fix by simply dropping a claim.
    """
    by_key: dict[tuple[str, str], list[Evidence]] = {}
    for item in evidence:
        if item.numeric_value is None:
            continue
        by_key.setdefault((item.source_type, item.locator), []).append(item)

    findings: list[CriticFinding] = []
    for (source_type, locator), items in by_key.items():
        values = {item.numeric_value for item in items}
        if len(values) > 1:
            findings.append(
                CriticFinding(
                    code="contradictory_values",
                    severity="blocking",
                    message=(
                        f"conflicting numeric values for {source_type}/{locator}: "
                        f"{sorted(str(v) for v in values)}"
                    ),
                )
            )
    return findings
