"""``run_skill_script`` served by a realtime channel through the public tool (RFC §24).

A realtime voice channel whose skills belong to another agent runs their
scripts behind its own gate: ``RunSkillScriptTool`` in its ``tools=`` runs
them through the one handler every channel uses. A script runs; an unknown
skill and a script outside its skill answer an error, and nothing runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roomkit import RoomKit, RunSkillScriptTool, ToolRefusedError
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.skills import SkillRegistry
from roomkit.tools import extract_tools
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import until
from tests.test_skills import _make_skill_dir_full
from tests.test_skills_integration import MockScriptExecutor


async def _call(tmp_path: Path, arguments: dict[str, object]) -> tuple[str, MockScriptExecutor]:
    _make_skill_dir_full(tmp_path, "reports", scripts=["build.py"])
    skills = SkillRegistry()
    skills.discover(tmp_path)
    executor = MockScriptExecutor()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[RunSkillScriptTool(skills, executor)],
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")

    await provider.simulate_tool_call(session, "c1", "run_skill_script", arguments)
    await until(lambda: bool(provider.tool_results))
    await kit.close()
    return provider.tool_results[0][2], executor


async def test_a_script_of_the_skill_runs(tmp_path: Path) -> None:
    result, executor = await _call(
        tmp_path, {"skill_name": "reports", "script_name": "build.py", "arguments": {"q": "1"}}
    )

    assert executor.calls == [("reports", "build.py", {"q": "1"})]
    assert json.loads(result)["stdout"] == "OK"


@pytest.mark.parametrize(
    "arguments",
    [
        {"skill_name": "nowhere", "script_name": "build.py"},
        {"skill_name": "reports", "script_name": "../../etc/passwd"},
    ],
    ids=["unknown-skill", "outside-the-skill"],
)
async def test_nothing_runs_for_an_unknown_skill_or_a_script_outside_it(
    tmp_path: Path, arguments: dict[str, object]
) -> None:
    result, executor = await _call(tmp_path, arguments)

    assert executor.calls == []
    assert "error" in result


def test_the_tool_declares_the_one_schema_every_channel_declares(tmp_path: Path) -> None:
    tool = RunSkillScriptTool(SkillRegistry(), MockScriptExecutor())

    assert tool.definition["name"] == RunSkillScriptTool.name == "run_skill_script"
    assert set(tool.definition["parameters"]["required"]) == {"skill_name", "script_name"}


def test_standalone_definitions_own_their_nested_schema() -> None:
    first = RunSkillScriptTool(SkillRegistry(), MockScriptExecutor())
    second = RunSkillScriptTool(SkillRegistry(), MockScriptExecutor())
    parameters = first.definition["parameters"]
    parameters["properties"]["skill_name"]["description"] = "Custom wording"
    parameters["required"].clear()
    for definition in (first.definition, second.definition):
        assert (
            definition["parameters"]["properties"]["skill_name"]["description"]
            == "Name of the skill."
        )
        assert definition["parameters"]["required"] == ["skill_name", "script_name"]


async def test_standalone_tool_serves_the_registry_supplied_by_the_host(tmp_path: Path) -> None:
    skills = SkillRegistry()
    executor = MockScriptExecutor()
    tool = RunSkillScriptTool(skills, executor)
    definitions, handler = extract_tools([tool])
    assert [definition.name for definition in definitions] == [tool.name]
    assert handler is not None

    _make_skill_dir_full(tmp_path, "reports", scripts=["build.py"])
    skills.discover(tmp_path)
    arguments = {"skill_name": "reports", "script_name": "build.py", "arguments": {"q": "1"}}
    answer = await handler(tool.name, arguments)
    assert isinstance(answer, str)
    assert json.loads(answer)["stdout"] == "OK"

    with pytest.raises(ToolRefusedError, match="nowhere"):
        await tool.handler(tool.name, {"skill_name": "nowhere", "script_name": "build.py"})
    missing_file = await tool.handler(
        tool.name, {"skill_name": "reports", "script_name": "absent"}
    )
    assert "error" in json.loads(missing_file)
    assert executor.calls == [("reports", "build.py", {"q": "1"})]
