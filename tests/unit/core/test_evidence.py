from decimal import Decimal

import pytest
from pydantic import ValidationError

from equity_research.core.evidence import DerivedMetric, Evidence


def test_evidence_requires_url_and_retrieval_time():
    with pytest.raises(ValidationError):
        Evidence(evidence_id="e1", claim="Revenue rose", source_type="sec_xbrl")


def test_evidence_accepts_a_complete_record():
    evidence = Evidence(
        evidence_id="e1",
        claim="Revenue rose 12% YoY",
        source_type="sec_xbrl",
        source_url="https://www.sec.gov/cgi-bin/browse-edgar",
        retrieved_at="2026-08-20T12:00:00Z",
        locator="Revenues[FY2025]",
        numeric_value=Decimal("1.12"),
        unit="ratio",
        confidence="high",
    )
    assert evidence.evidence_id == "e1"
    assert evidence.confidence == "high"


def test_evidence_rejects_unsupported_source_type():
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="e1",
            claim="Revenue rose",
            source_type="tweet",
            source_url="https://example.com",
            retrieved_at="2026-08-20T12:00:00Z",
        )


def test_derived_metric_keeps_input_evidence_ids():
    metric = DerivedMetric(
        name="fcf_margin",
        value=Decimal("0.12"),
        formula="fcf/revenue",
        input_evidence_ids=["e1", "e2"],
    )
    assert metric.input_evidence_ids == ["e1", "e2"]


def test_derived_metric_requires_formula():
    with pytest.raises(ValidationError):
        DerivedMetric(name="fcf_margin", value=Decimal("0.12"), input_evidence_ids=["e1"])
