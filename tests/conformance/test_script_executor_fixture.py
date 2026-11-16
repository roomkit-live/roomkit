"""The fixture must expose host entry, including a missing framework guard."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.skills import SkillPathError
from tests.conformance._script_executor import ScriptedScriptExecutor
from tests.conformance._shape import unknown_members
from tests.conformance._skills import scripted_skill_registry


async def test_entry_is_recorded_before_the_fixture_checks_the_path() -> None:
    registry = scripted_skill_registry(
        [{"name": "local", "description": "Scripts", "scripts": {"build.sh": "source text"}}]
    )
    skill = registry.get_skill("local")
    assert skill is not None
    record: dict[str, Any] = {"calls": [], "closes": 0}
    executor = ScriptedScriptExecutor({}, record)

    with pytest.raises(SkillPathError):
        await executor.execute(skill, "../build.sh")
    assert record["calls"] == [
        {"skill_name": "local", "script_name": "../build.sh", "arguments": None}
    ]

    result = await executor.execute(skill, "build.sh", {})
    assert result.exit_code == 0 and result.success
    assert record["calls"][1]["contents"] == "source text"
    assert record["calls"][1]["arguments"] == {}
    await executor.close()
    assert record["closes"] == 1


@pytest.mark.parametrize(
    ("executor", "member"),
    [
        ({"reslut": {}}, "reslut"),
        ({"result": {"exit_code": 0, "sucess": False}}, "sucess"),
        ({"raises": {"kind": "Error", "mesage": "failure"}}, "mesage"),
    ],
)
def test_executor_controls_do_not_silently_ignore_unknown_members(
    executor: dict[str, Any], member: str
) -> None:
    scenario = {
        "setup": {"channels": [{"id": "agent", "type": "ai", "script_executor": executor}]}
    }
    assert any(member in failure for failure in unknown_members(scenario))
