"""Per-turn bounds keep integer milliseconds explicit before execution."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._shape import unknown_members


@pytest.mark.parametrize("bounds", [[], "", {"slow": True}, {"slow": -1}, {"slow": 1.5}])
def test_malformed_turn_bounds_are_rejected(bounds: Any) -> None:
    scenario = {
        "setup": {
            "channels": [
                {
                    "id": "ai",
                    "type": "ai",
                    "config_provider": {
                        "room": {"tool_timeouts_ms": bounds},
                    },
                }
            ]
        }
    }
    assert unknown_members(scenario)


@pytest.mark.parametrize("bounds", [None, {}, {"slow": None}, {"slow": 0}, {"slow": 20}])
def test_zero_reaches_the_public_turn_validation(bounds: Any) -> None:
    scenario = {
        "setup": {
            "channels": [
                {
                    "id": "ai",
                    "type": "ai",
                    "config_provider": {
                        "room": {"tool_timeouts_ms": bounds},
                    },
                }
            ]
        }
    }
    assert unknown_members(scenario) == []
