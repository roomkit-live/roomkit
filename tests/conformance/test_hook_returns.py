"""Invalid hook returns are deliberate fixtures, never malformed scenarios."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from tests.conformance._script import hook_handler
from tests.conformance._shape import unknown_members


@pytest.mark.parametrize(
    "reaction",
    [
        {"invalid_result": {"kind": "typo"}},
        {"invalid_result": {"kind": "no_result", "tasks": []}},
        {"invalid_result": {"kind": "missing_payload", "task": []}},
        {"invalid_result": {"kind": "no_result"}, "result": {"action": "allow"}},
        {"invalid_result": {"kind": "no_result"}, "fail": "error"},
    ],
)
def test_a_malformed_invalid_return_is_refused_before_execution(
    reaction: dict[str, Any],
) -> None:
    hook = {"name": "check", "trigger": "before_broadcast", "reactions": [reaction]}
    assert unknown_members({"setup": {"hooks": [hook]}})


@pytest.mark.parametrize("field", ["tasks", "observations", "injected_events"])
def test_a_malformed_side_effect_is_not_a_simulated_hook_error(field: str) -> None:
    invalid = {"kind": "missing_payload", field: [{}]}
    hook = {
        "name": "check",
        "trigger": "before_broadcast",
        "reactions": [{"invalid_result": invalid}],
    }
    with pytest.raises((ValidationError, KeyError)):
        hook_handler(hook, Mock())


@pytest.mark.parametrize("field", ["tasks", "observations", "injected_events"])
@pytest.mark.parametrize("value", [{}, "", 5])
def test_side_effects_of_the_wrong_shape_are_never_silently_emptied(
    field: str, value: Any
) -> None:
    hook = {
        "name": "check",
        "trigger": "before_broadcast",
        "reactions": [{"invalid_result": {"kind": "missing_payload", field: value}}],
    }
    assert unknown_members({"setup": {"hooks": [hook]}})
    with pytest.raises(ValueError, match="must be an array"):
        hook_handler(hook, Mock())


@pytest.mark.parametrize(
    "setup",
    [
        {"hooks": [{"name": "check", "trigger": "after_broadcast"}]},
        {"hooks": [{"name": "check", "trigger": "before_broadcast", "execution": "async"}]},
        {"channels": [{"id": "sms", "type": "sms"}]},
    ],
)
def test_an_invalid_return_on_an_unhandled_callback_is_refused(setup: dict[str, Any]) -> None:
    kind = next(iter(setup))
    field = "reactions" if kind == "hooks" else "deliver"
    setup[kind][0][field] = [{"invalid_result": {"kind": "no_result"}}]
    assert unknown_members({"setup": setup})
