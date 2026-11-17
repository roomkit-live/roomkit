"""Planning fixture settings cannot silently coerce malformed booleans."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._shape import unknown_members


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_enable_planning_refuses_non_booleans(value: Any) -> None:
    scenario = {"setup": {"channels": [{"id": "agent", "type": "ai", "enable_planning": value}]}}
    assert any("enable_planning: expected a boolean" in p for p in unknown_members(scenario))


@pytest.mark.parametrize("value", [False, True])
def test_enable_planning_accepts_both_public_boolean_values(value: bool) -> None:
    scenario = {"setup": {"channels": [{"id": "agent", "type": "ai", "enable_planning": value}]}}
    assert unknown_members(scenario) == []
