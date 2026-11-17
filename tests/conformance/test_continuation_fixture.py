"""Continuation controls fail as fixture errors, before product execution."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._continuation import ContinuationFailedError, scripted_continuation
from tests.conformance._shape import unknown_members


@pytest.mark.parametrize(
    "policy",
    [
        None,
        [],
        {},
        {"prefix": "I"},
        {"prefix": 1, "nudge": "go"},
        {"prefix": "I", "nudge": False},
        {"prefix": "I", "error": None},
        {"prefix": "I", "nudge": None, "error": "failed"},
        {"prefix": "I", "nudge": "go", "misspelled": True},
    ],
)
def test_malformed_policy_is_refused_before_execution(policy: Any) -> None:
    scenario = {"setup": {"channels": [{"id": "agent", "type": "ai", "continuation": policy}]}}
    assert any("continuation" in problem for problem in unknown_members(scenario))
    with pytest.raises(ValueError, match="continuation"):
        scripted_continuation(policy, [])


@pytest.mark.parametrize("nudge", [None, "", "  go\n"])
def test_the_fixture_records_exact_inputs_and_preserves_the_nudge(nudge: str | None) -> None:
    calls: list[str] = []
    policy = scripted_continuation({"prefix": "I will", "nudge": nudge}, calls)
    assert policy("I will check.\n") == nudge
    assert policy("Done.") is None
    assert calls == ["I will check.\n", "Done."]


def test_a_configured_failure_is_not_a_fixture_failure() -> None:
    calls: list[str] = []
    policy = scripted_continuation({"prefix": "I will", "error": "policy failed"}, calls)
    with pytest.raises(ContinuationFailedError, match="policy failed"):
        policy("I will check.")
    assert calls == ["I will check."]
