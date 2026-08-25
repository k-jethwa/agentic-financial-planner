#!/usr/bin/env python
"""Recorded evaluation harness.

Runs every scenario in `evals/cases.json` against the research graph built
entirely from fakes (`tests/support/graph_fakes`) and checks each case's
declared expectations. No network access and no API key is ever required
-- every case's inputs are fully deterministic, canned data, matching the
design spec's CI constraint. Writes a JSON summary (citation coverage,
critic behavior, and per-case pass/fail) to `evals/results.json` and exits
non-zero if any case's gates fail, so CI can treat this as a release gate.

Usage: python evals/run_evals.py --recorded
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from support.graph_fakes import build_fake_deps  # noqa: E402

from equity_research.agents.graph import build_research_graph  # noqa: E402
from equity_research.agents.valuation import DcfAssumptions  # noqa: E402
from equity_research.core.models import ReportMode, ResearchRequest, ResearchState  # noqa: E402

CLAIM_GROUPS = ("bull_thesis", "bear_thesis", "catalysts", "risks")
DEFAULT_DCF_ASSUMPTIONS = {"wacc": Decimal("0.09"), "terminal_growth": Decimal("0.025")}


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": None if passed else detail}


def _build_deps(scenario: dict[str, Any]):
    kwargs: dict[str, Any] = {}
    if "resolves" in scenario:
        kwargs["resolves"] = scenario["resolves"]
    if scenario.get("annual_facts") is not None:
        kwargs["annual_facts"] = {
            tag: [tuple(entry) for entry in entries]
            for tag, entries in scenario["annual_facts"].items()
        }
    if scenario.get("raise_on_filings"):
        kwargs["raise_on_filings"] = True
    if scenario.get("news_raises"):
        kwargs["news_raises"] = True
    if scenario.get("filing_html"):
        kwargs["filing_html"] = scenario["filing_html"]
    if scenario.get("shares_outstanding") is not None:
        kwargs["dcf_assumptions"] = DcfAssumptions(
            shares_outstanding=Decimal(str(scenario["shares_outstanding"])),
            **DEFAULT_DCF_ASSUMPTIONS,
        )
    return build_fake_deps(**kwargs)


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    scenario = case.get("scenario", {})
    deps = _build_deps(scenario)
    graph = build_research_graph(deps)

    request = ResearchRequest(
        ticker=case["ticker"],
        question=case["question"],
        report_mode=ReportMode(case.get("report_mode", "full")),
        as_of_date=(
            date.fromisoformat(scenario["as_of_date"]) if scenario.get("as_of_date") else None
        ),
    )
    result = graph.invoke(ResearchState(request=request))

    exp = case["expectations"]
    checks: list[dict] = []

    has_fatal = any(error.fatal for error in result["errors"])
    report = result.get("report")
    actual_status = "failed" if (has_fatal or report is None) else "completed"
    checks.append(
        _check(
            "run_status",
            actual_status == exp["run_status"],
            f"expected {exp['run_status']}, got {actual_status}",
        )
    )

    if "has_report" in exp:
        checks.append(
            _check(
                "has_report",
                (report is not None) == exp["has_report"],
                f"report is None: {report is None}",
            )
        )

    report_json = report["json"] if report else None
    markdown = report["markdown"] if report else ""
    numeric_claims_total = 0
    numeric_claims_cited = 0

    if report_json is not None:
        numeric_claims = [
            claim
            for group in CLAIM_GROUPS
            for claim in report_json[group]
            if any(character.isdigit() for character in claim["text"])
        ]
        uncited = [claim for claim in numeric_claims if not claim["evidence_ids"]]
        numeric_claims_total = len(numeric_claims)
        numeric_claims_cited = numeric_claims_total - len(uncited)
        checks.append(
            _check(
                "numeric_claims_cited",
                not uncited,
                f"{len(uncited)} of {numeric_claims_total} numeric claims lack evidence IDs",
            )
        )

        non_https_sources = [
            s for s in report_json["sources"] if not s["source_url"].startswith("https://")
        ]
        checks.append(
            _check(
                "source_urls_resolve",
                not non_https_sources,
                f"{len(non_https_sources)} sources without a resolvable https URL",
            )
        )

    evidence_source_types = {e.source_type for e in result["evidence"]}
    for source_type in exp.get("min_evidence_source_types", []):
        checks.append(
            _check(
                f"evidence_present:{source_type}",
                source_type in evidence_source_types,
                f"expected {source_type} evidence, found none",
            )
        )
    for source_type in exp.get("absent_evidence_source_types", []):
        checks.append(
            _check(
                f"evidence_absent:{source_type}",
                source_type not in evidence_source_types,
                f"expected no {source_type} evidence, but some was present",
            )
        )

    if "critic_status_in" in exp:
        actual = report_json["critic"]["status"] if report_json else None
        checks.append(
            _check(
                "critic_status",
                actual in exp["critic_status_in"],
                f"critic status {actual!r} not in {exp['critic_status_in']}",
            )
        )

    critic_codes = {f["code"] for f in report_json["critic"]["findings"]} if report_json else set()
    for code in exp.get("critic_finding_codes_present", []):
        checks.append(
            _check(
                f"critic_finding_present:{code}",
                code in critic_codes,
                f"missing finding {code}",
            )
        )
    for code in exp.get("critic_finding_codes_absent", []):
        checks.append(
            _check(
                f"critic_finding_absent:{code}",
                code not in critic_codes,
                f"unexpected finding {code}",
            )
        )

    for substring in exp.get("prohibited_substrings", []):
        checks.append(
            _check(
                f"prohibited_absent:{substring}",
                substring.lower() not in markdown.lower(),
                f"prohibited text {substring!r} found in the rendered report",
            )
        )
    for substring in exp.get("conclusion_prohibited_substrings", []):
        conclusion = (report_json or {}).get("conclusion", "")
        checks.append(
            _check(
                f"conclusion_prohibited_absent:{substring}",
                substring.lower() not in conclusion.lower(),
                f"prohibited text {substring!r} leaked into the conclusion",
            )
        )
    for substring in exp.get("required_substrings", []):
        checks.append(
            _check(
                f"required_present:{substring}",
                substring.lower() in markdown.lower(),
                f"expected text {substring!r} not found in the rendered report",
            )
        )

    if "expect_warnings_nonempty" in exp:
        expected_nonempty = exp["expect_warnings_nonempty"]
        actual_nonempty = bool(report_json["warnings"]) if report_json else False
        checks.append(
            _check(
                "warnings_nonempty",
                actual_nonempty == expected_nonempty,
                f"expected warnings present={expected_nonempty}, got {actual_nonempty}",
            )
        )

    if "valuation_available" in exp:
        expected_available = exp["valuation_available"]
        actual_available = report_json["valuation"]["available"] if report_json else False
        checks.append(
            _check(
                "valuation_available",
                actual_available == expected_available,
                f"expected valuation available={expected_available}, got {actual_available}",
            )
        )

    if "value_per_share_present" in exp:
        value_per_share = (report_json or {}).get("valuation", {}).get("value_per_share")
        checks.append(
            _check(
                "value_per_share_present",
                (value_per_share is not None) == exp["value_per_share_present"],
                f"value_per_share present={value_per_share is not None}",
            )
        )

    if "plan_nodes" in exp:
        actual_nodes = [task.node for task in result["plan"].tasks]
        checks.append(
            _check(
                "plan_nodes",
                actual_nodes == exp["plan_nodes"],
                f"expected plan nodes {exp['plan_nodes']}, got {actual_nodes}",
            )
        )

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "critic_status": report_json["critic"]["status"] if report_json else None,
        "numeric_claims_total": numeric_claims_total,
        "numeric_claims_cited": numeric_claims_cited,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorded",
        action="store_true",
        help="Run entirely against recorded/fake fixtures. This is currently the only "
        "supported mode; the flag is accepted for compatibility with CI invocations.",
    )
    parser.add_argument("--cases", type=Path, default=REPO_ROOT / "evals" / "cases.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evals" / "results.json")
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text())
    results = [run_case(case) for case in cases]

    total_numeric = sum(r["numeric_claims_total"] for r in results)
    cited_numeric = sum(r["numeric_claims_cited"] for r in results)
    critic_statuses: dict[str, int] = {}
    for r in results:
        if r["critic_status"]:
            critic_statuses[r["critic_status"]] = critic_statuses.get(r["critic_status"], 0) + 1

    summary = {
        "total_cases": len(results),
        "passed_cases": sum(1 for r in results if r["passed"]),
        "failed_cases": sum(1 for r in results if not r["passed"]),
        "categories": sorted({r["category"] for r in results}),
        "citation_coverage": {
            "total_numeric_claims": total_numeric,
            "cited_numeric_claims": cited_numeric,
            "coverage_pct": (
                100.0 if total_numeric == 0 else round(100 * cited_numeric / total_numeric, 2)
            ),
        },
        "critic_behavior": critic_statuses,
        "results": results,
    }
    args.out.write_text(json.dumps(summary, indent=2, default=str))

    print(f"{summary['passed_cases']}/{summary['total_cases']} eval cases passed.")
    print(f"Citation coverage: {summary['citation_coverage']['coverage_pct']}%")
    print(f"Full results written to {args.out}")
    for r in results:
        if not r["passed"]:
            print(f"FAILED: {r['id']}")
            for c in r["checks"]:
                if not c["passed"]:
                    print(f"  - {c['name']}: {c['detail']}")

    return 0 if summary["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
