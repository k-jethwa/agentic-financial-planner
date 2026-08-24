import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def load_fixture():
    def _load(relative_path: str) -> dict:
        # relative_path is rooted at tests/fixtures/, e.g. "sec/companyfacts_msft.json"
        path = FIXTURES_DIR / relative_path
        return json.loads(path.read_text(encoding="utf-8"))

    return _load


class FakeResponse:
    """A minimal stand-in for an httpx.Response, JSON-only."""

    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class FixtureHttp:
    """A fake HttpClient that returns one canned fixture payload for every GET.

    Sufficient for adapter unit tests, which each exercise a single
    endpoint call against a recorded fixture rather than live SEC/Yahoo
    traffic.
    """

    def __init__(self, payload: dict):
        self._payload = payload
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> FakeResponse:
        self.calls.append(url)
        return FakeResponse(self._payload)


@pytest.fixture
def fixture_http():
    return FixtureHttp
