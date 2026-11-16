"""A scripted in-memory skill is registered through the public add API."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._shape import unknown_members
from tests.conformance._skills import scripted_skill_registry


def test_the_fixture_adds_an_in_memory_skill_without_a_skill_md() -> None:
    registry = scripted_skill_registry(
        [
            {
                "name": "Store Handbook",
                "description": "Stored outside the filesystem",
                "instructions": "Follow the store's instructions.",
                "references": {"notes.md": "Stored reference"},
                "in_memory": True,
            }
        ]
    )

    skill = registry.get_skill("Store Handbook")
    assert skill is not None
    assert skill.instructions == "Follow the store's instructions."
    assert skill.read_reference("notes.md") == "Stored reference"
    assert not (skill.path / "SKILL.md").exists()


@pytest.mark.parametrize("value", [None, 1, "true", [], {}])
def test_the_in_memory_control_requires_a_boolean(value: Any) -> None:
    scenario = {
        "setup": {
            "channels": [
                {
                    "id": "agent",
                    "type": "ai",
                    "skills": [{"name": "local", "description": "d", "in_memory": value}],
                }
            ]
        }
    }
    assert any("in_memory: expected a boolean" in error for error in unknown_members(scenario))
