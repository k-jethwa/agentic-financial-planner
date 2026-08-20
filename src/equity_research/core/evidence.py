"""The evidence ledger: the sole contract for cross-agent facts.

Every worker in the research graph reads and writes typed state; any fact
that crosses from one worker to another, or into the final report, must be
represented as an `Evidence` record or a `DerivedMetric` that traces back to
one or more evidence IDs. A claim with no evidence ID is not reportable.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl

SourceType = Literal["sec_filing", "sec_xbrl", "market_data", "news", "company_ir"]
Confidence = Literal["high", "medium", "low"]


class Evidence(BaseModel):
    """A single, citable fact backing a report claim.

    `source_url` and `retrieved_at` are required for every piece of
    evidence: a claim without a resolvable source and a retrieval
    timestamp cannot enter the ledger.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    claim: str
    source_type: SourceType
    source_url: HttpUrl
    retrieved_at: datetime
    published_at: date | None = None
    locator: str = ""  # filing section, page, table cell, or article paragraph
    excerpt: str | None = None
    numeric_value: Decimal | None = None
    unit: str | None = None
    confidence: Confidence = "medium"


class DerivedMetric(BaseModel):
    """A calculated value that retains its formula and input evidence.

    Derived values (growth rates, margins, DCF outputs, ...) must never be
    reportable without a formula string and the evidence IDs of every input
    that fed the calculation.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    value: Decimal
    formula: str
    input_evidence_ids: list[str]
    unit: str | None = None
    as_of: date | None = None
