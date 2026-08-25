"""Streamlit run viewer: a thin client over the research-run API.

Creates and reads runs against the FastAPI backend (`equity_research.api`).
Prominently shows the "not investment advice" disclaimer, run status,
report sections, sources, warnings, and the chronological graph trace.
Never renders untrusted source HTML: filing/news text only ever reaches
this page as plain strings already extracted into typed `Evidence`/
`Claim` records by the backend, and every value is shown via
`st.write`/`st.markdown` with unsafe_allow_html left at its default
(off) — nothing here echoes raw HTML from a filing or news source.
"""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("EQUITY_RESEARCH_API_URL", "http://localhost:8000")
DISCLAIMER = "Research and educational use only — not investment advice."
POLL_INTERVAL_SECONDS = 1.5
POLL_TIMEOUT_SECONDS = 120

st.set_page_config(page_title="Equity Research Assistant", layout="wide")


def _api(path: str, *, method: str = "GET", json: dict | None = None) -> dict:
    url = f"{API_BASE_URL}{path}"
    response = httpx.request(method, url, json=json, timeout=30.0)
    response.raise_for_status()
    return response.json()


def _create_run(ticker: str, question: str, report_mode: str) -> str:
    payload = {"ticker": ticker.upper(), "question": question, "report_mode": report_mode}
    created = _api("/v1/runs", method="POST", json=payload)
    return created["run_id"]


def _wait_for_terminal_status(run_id: str) -> dict:
    """Poll GET /v1/runs/{id} until the run reaches a terminal status.

    The MVP backend executes a run synchronously inside POST /v1/runs, so
    this loop typically resolves on its first iteration; polling is kept so
    the UI keeps working unchanged if a background worker replaces that.
    """
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    placeholder = st.empty()
    while time.monotonic() < deadline:
        run = _api(f"/v1/runs/{run_id}")
        placeholder.info(f"Run status: **{run['status']}**")
        if run["status"] in ("completed", "failed"):
            placeholder.empty()
            return run
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Run {run_id} did not reach a terminal status within {POLL_TIMEOUT_SECONDS}s"
    )


def _render_claims(title: str, claims: list[dict]) -> None:
    st.subheader(title)
    if not claims:
        st.write("_none_")
        return
    for claim in claims:
        evidence_ids = ", ".join(claim["evidence_ids"])
        st.markdown(f"- {claim['text']} `[{evidence_ids}]`")


def _render_valuation(valuation: dict) -> None:
    st.subheader("Valuation")
    if not valuation.get("available"):
        missing = ", ".join(valuation.get("missing_inputs", [])) or "unspecified"
        st.warning(f"Valuation unavailable. Missing inputs: {missing}.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Equity value", valuation.get("equity_value") or "—")
    col2.metric("Value per share", valuation.get("value_per_share") or "—")
    wacc, terminal_growth = valuation.get("wacc"), valuation.get("terminal_growth")
    col3.metric("WACC / terminal growth", f"{wacc} / {terminal_growth}")
    st.caption(f"Formula: `{valuation.get('formula')}`")
    sensitivity = valuation.get("sensitivity") or {}
    if sensitivity:
        st.write("Sensitivity:")
        st.table({"scenario": list(sensitivity.keys()), "equity value": list(sensitivity.values())})


def _render_report(report: dict) -> None:
    st.success(DISCLAIMER)
    st.header(f"{report['ticker']} Research Memo")
    st.caption(
        f"Question: {report['question']} · As of {report['as_of']} · Mode: {report['report_mode']}"
    )

    st.subheader("Conclusion")
    st.write(report["conclusion"])

    _render_valuation(report["valuation"])
    _render_claims("Bull Thesis", report["bull_thesis"])
    _render_claims("Bear Thesis", report["bear_thesis"])
    _render_claims("Catalysts", report["catalysts"])
    _render_claims("Risks", report["risks"])

    st.subheader("What Would Change the View")
    for trigger in report["what_would_change_the_view"]:
        st.markdown(f"- {trigger}")

    st.subheader("Source Ledger")
    sources = report["sources"]
    if sources:
        st.dataframe(
            [
                {
                    "evidence_id": s["evidence_id"],
                    "type": s["source_type"],
                    "confidence": s["confidence"],
                    "locator": s["locator"],
                    "url": s["source_url"],
                    "published": s["published_at"],
                }
                for s in sources
            ],
            width="stretch",
        )

    st.subheader("Critic Findings")
    critic = report["critic"]
    st.write(f"Status: **{critic['status']}**")
    for finding in critic["findings"]:
        (st.warning if finding["severity"] == "warning" else st.error)(
            f"[{finding['code']}] {finding['message']}"
        )

    if report["warnings"]:
        st.subheader("Warnings")
        for warning in report["warnings"]:
            st.warning(warning)

    with st.expander("Reproducibility"):
        st.write(f"Model: {report.get('model_name') or 'template (no LLM)'}")
        st.write(f"Prompts version: {report['prompts_version']}")
        st.write(f"Software version: {report['software_version']}")
        st.write(f"Generated at: {report['generated_at']}")


def _render_trace(run_id: str) -> None:
    trace = _api(f"/v1/runs/{run_id}/trace")
    st.subheader("Agent Trace")
    for event in trace["trace"]:
        icon = {"completed": "✅", "failed": "❌", "started": "▶️"}.get(event["status"], "•")
        st.write(f"{icon} `{event['occurred_at']}` **{event['node']}** — {event['status']}")
        if event["detail"]:
            st.caption(event["detail"])


def main() -> None:
    st.title("📊 Equity Research Assistant")
    st.info(DISCLAIMER)

    with st.form("new_run"):
        col1, col2 = st.columns([1, 3])
        ticker = col1.text_input("Ticker", value="MSFT", max_chars=10)
        question = col2.text_input("Question", value="Assess the current investment thesis")
        report_mode = st.radio("Report mode", ["full", "question"], horizontal=True)
        submitted = st.form_submit_button("Run research")

    if submitted:
        if not ticker.strip() or not question.strip():
            st.error("Ticker and question are both required.")
            return
        try:
            run_id = _create_run(ticker.strip(), question.strip(), report_mode)
            st.session_state["run_id"] = run_id
        except httpx.HTTPStatusError as exc:
            st.error(f"Could not start run: {exc.response.text}")
            return

    run_id = st.session_state.get("run_id")
    if not run_id:
        return

    st.caption(f"Run ID: `{run_id}`")
    try:
        run = _wait_for_terminal_status(run_id)
    except TimeoutError as exc:
        st.error(str(exc))
        return

    if run["status"] == "failed":
        st.error("This run failed. See the trace below for details.")
        _render_trace(run_id)
        return

    report_response = _api(f"/v1/runs/{run_id}/report")
    if report_response["report"] is None:
        st.warning("Run completed but no report was produced.")
    else:
        _render_report(report_response["report"])

    with st.expander("Chronological graph trace", expanded=False):
        _render_trace(run_id)


if __name__ == "__main__":
    main()
