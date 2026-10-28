"""fetch_sections retries transient failures (T441983).

Enumeration had no retry while generation did, so one 503 or timeout
ended an article's run with a dead letter. A 200-article trial lost 9
articles that way. The taxonomy must match generate_one: 5xx and
transport errors are transient, 4xx is deterministic and must not be
retried.

No service needed: a fake session returns the status sequence under test.
"""

import pytest
import requests

from src.models.tts_section_generator.scripts import batch_generate


class _Response:
    def __init__(self, status_code: int):
        self.status_code = status_code
        self.reason = "Test"
        self.url = "http://test/sections"

    def json(self) -> dict:
        return {"sections": []}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)


class _Session:
    """Returns the given sequence; "transport" raises instead."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if outcome == "transport":
            raise requests.ConnectionError("simulated transport failure")
        return _Response(outcome)


ARTICLE = {"page_id": 1, "rev_id": 2, "title": "Test"}


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """The retry sleeps are real; the test should not."""
    monkeypatch.setattr(batch_generate, "BACKOFF_S", 0.0)


@pytest.mark.parametrize(
    "outcomes,expected_calls",
    [
        ([200], 1),
        ([503, 200], 2),
        ([500, 500, 200], 3),
        (["transport", 200], 2),
    ],
)
def test_transient_failures_are_retried(outcomes, expected_calls):
    session = _Session(outcomes)
    assert batch_generate.fetch_sections("http://test", ARTICLE, session) == {
        "sections": []
    }
    assert session.calls == expected_calls


def test_persistent_failure_raises_after_the_retry_budget():
    session = _Session([500, 500, 500])
    with pytest.raises(requests.RequestException):
        batch_generate.fetch_sections("http://test", ARTICLE, session)
    assert session.calls == 1 + batch_generate.TRANSIENT_RETRIES


def test_client_errors_are_not_retried():
    """A 4xx is deterministic: retrying cannot change the answer, and
    doing so would turn a fast skip into a slow one."""
    session = _Session([404])
    with pytest.raises(requests.HTTPError):
        batch_generate.fetch_sections("http://test", ARTICLE, session)
    assert session.calls == 1
