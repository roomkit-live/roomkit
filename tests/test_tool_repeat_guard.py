"""Anti-loop guard: identical (tool, arguments) calls short-circuit in a turn.

The observed failure mode: a small model re-issues the SAME find_tools query
(or the same tool call) round after round and burns the whole turn without
answering. The guard counts identical calls per tool loop: find_tools /
list_tools are pure within a turn so the second identical call is refused;
regular tools get one retry (transient failures) and the third identical
call is refused, with an explicit instruction to stop.
"""

from __future__ import annotations

import json

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_ECHO_TOOL = {
    "name": "echo",
    "description": "Echo a value back.",
    "parameters": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
}


def _binding(tools: list[dict]) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": tools},
    )


def _tool_results(context) -> list[dict]:
    return [json.loads(m.content[0].result) for m in context.messages if m.role == "tool"]


def _same_call_response(call_id: str, arguments: dict) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name="echo", arguments=arguments)],
    )


async def test_third_identical_call_is_refused(streaming: bool) -> None:
    executions: list[dict] = []

    async def handler(name: str, arguments: dict) -> str:
        executions.append(arguments)
        return json.dumps({"ok": True})

    args = {"value": "same"}
    provider = MockAIProvider(
        ai_responses=[
            _same_call_response("t1", args),
            _same_call_response("t2", args),
            _same_call_response("t3", args),
            AIResponse(content="done", finish_reason="stop"),
        ],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=handler)
    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding([_ECHO_TOOL]),
        RoomContext(room=Room(id="r1")),
    )

    # Two executions allowed (retry latitude); the third is short-circuited.
    assert len(executions) == 2
    third = _tool_results(provider.calls[3])[-1]
    assert "EXACT arguments" in third["error"]
    assert "STOP" in third["hint"]


@pytest.mark.parametrize(("changes_state", "runs"), [(False, 2), (True, 3)])
async def test_a_rerun_after_a_change_runs_only_when_the_tool_changes_state(
    streaming: bool, changes_state: bool, runs: int
) -> None:
    """A test rerun after each edit, as an unattended coding turn does: an
    edit declared as changing state starts the turn's counts again, so every
    rerun runs; an edit that declares nothing leaves the third rerun refused
    (RFC §6.4)."""
    ran: list[dict] = []
    edits: list[dict] = []

    async def handler(name: str, arguments: dict) -> str:
        (ran if name == "echo" else edits).append(arguments)
        return json.dumps({"ok": True})

    def edit(call_id: str, value: str) -> AIResponse:
        return AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=call_id, name="edit", arguments={"value": value})],
        )

    args = {"value": "same"}
    provider = MockAIProvider(
        ai_responses=[
            _same_call_response("t1", args),
            edit("e1", "fix-1"),
            _same_call_response("t2", args),
            edit("e2", "fix-2"),
            _same_call_response("t3", args),
            AIResponse(content="done", finish_reason="stop"),
        ],
        streaming=streaming,
    )
    edit_tool = AITool(
        name="edit",
        description="Edit a file.",
        parameters=_ECHO_TOOL["parameters"],
        changes_state=changes_state,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=handler, tools=[edit_tool])
    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding([_ECHO_TOOL]),
        RoomContext(room=Room(id="r1")),
    )

    assert len(edits) == 2
    assert len(ran) == runs
    third = _tool_results(provider.calls[5])[-1]
    assert ("error" in third) is not changes_state


@pytest.mark.parametrize("mutates", [False, True])
async def test_a_change_made_again_and_again_is_still_refused(
    streaming: bool, mutates: bool
) -> None:
    """The change keeps its own count, as the guard counted it before its
    handler ran, which may change the arguments it is handed: the same write
    three times is a loop."""
    writes: list[dict] = []

    async def handler(name: str, arguments: dict) -> str:
        writes.append(dict(arguments))
        if mutates:
            arguments.pop("value")
        return json.dumps({"ok": True})

    def write(call_id: str) -> AIResponse:
        return AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=call_id, name="write", arguments={"value": "same"})],
        )

    provider = MockAIProvider(
        ai_responses=[write("w1"), write("w2"), write("w3"), AIResponse(content="done")],
        streaming=streaming,
    )
    write_tool = AITool(
        name="write",
        description="Write a file.",
        parameters=_ECHO_TOOL["parameters"],
        changes_state=True,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=handler, tools=[write_tool])
    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding([]),
        RoomContext(room=Room(id="r1")),
    )

    assert len(writes) == 2
    assert "EXACT arguments" in _tool_results(provider.calls[3])[-1]["error"]


async def test_two_changes_made_in_turn_start_each_other_s_counts_again(
    streaming: bool,
) -> None:
    """Each change starts the other's count again: two edits made in turn,
    each three times with the same arguments, all run (RMK-1541's review)."""
    edits: list[str] = []

    async def handler(name: str, arguments: dict) -> str:
        edits.append(arguments["value"])
        return json.dumps({"ok": True})

    def edit(call_id: str, value: str) -> AIResponse:
        return AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=call_id, name="edit", arguments={"value": value})],
        )

    provider = MockAIProvider(
        ai_responses=[
            *(edit(f"e{i}", "a" if i % 2 == 0 else "b") for i in range(6)),
            AIResponse(content="done"),
        ],
        streaming=streaming,
    )
    edit_tool = AITool(
        name="edit",
        description="Edit a file.",
        parameters=_ECHO_TOOL["parameters"],
        changes_state=True,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=handler, tools=[edit_tool])
    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding([]),
        RoomContext(room=Room(id="r1")),
    )

    assert edits == ["a", "b", "a", "b", "a", "b"]


async def test_different_arguments_are_not_guarded(streaming: bool) -> None:
    executions: list[dict] = []

    async def handler(name: str, arguments: dict) -> str:
        executions.append(arguments)
        return json.dumps({"ok": True})

    provider = MockAIProvider(
        ai_responses=[
            _same_call_response("t1", {"value": "a"}),
            _same_call_response("t2", {"value": "b"}),
            _same_call_response("t3", {"value": "c"}),
            AIResponse(content="done", finish_reason="stop"),
        ],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=handler)
    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding([_ECHO_TOOL]),
        RoomContext(room=Room(id="r1")),
    )
    assert len(executions) == 3


async def test_second_identical_find_tools_is_refused(streaming: bool) -> None:
    async def handler(name: str, arguments: dict) -> str:
        return json.dumps({"ok": True})

    find_call = AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="t", name="find_tools", arguments={"query": "send sms"})],
    )
    provider = MockAIProvider(
        ai_responses=[find_call, find_call, AIResponse(content="done")], streaming=streaming
    )
    ch = AIChannel("ai1", provider=provider, tool_search=True, tool_handler=handler)
    noise = [{"name": f"widget_{i}", "description": f"Operate widget {i}."} for i in range(30)]
    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding(noise),
        RoomContext(room=Room(id="r1")),
    )

    first = _tool_results(provider.calls[1])[0]
    assert "matches" in first
    # Identical query: pure within the turn, refused on the second call.
    second = _tool_results(provider.calls[2])[-1]
    assert "EXACT arguments" in second["error"]


async def test_unknown_skill_matching_tools_redirects_and_reveals(
    tmp_path, streaming: bool
) -> None:
    """Small models confuse skills with tools ("activate the Spotify skill"
    when SpotifySearch/... are tools). The dead-end error must reveal the
    matching tools and say to call them directly."""
    from pathlib import Path

    from roomkit.skills.registry import SkillRegistry

    skill_dir = Path(tmp_path) / "about-me"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: about-me\ndescription: Bio.\n---\nBody.")
    registry = SkillRegistry()
    registry.discover(Path(tmp_path))

    spotify_tools = [
        {"name": "SpotifySearch", "description": "Search Spotify."},
        {"name": "SpotifyPlay", "description": "Play a track."},
    ]
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(id="t1", name="activate_skill", arguments={"name": "Spotify"})
                ],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ],
        streaming=streaming,
    )

    async def handler(name: str, arguments: dict) -> str:
        return json.dumps({"ok": True})

    ch = AIChannel("ai1", provider=provider, skills=registry, tool_handler=handler)
    await respond(
        ch,
        make_event(body="play music", channel_id="sms1"),
        _binding(spotify_tools),
        RoomContext(room=Room(id="r1")),
    )

    result = _tool_results(provider.calls[1])[-1]
    assert "not found" in result["error"]
    assert "SpotifySearch" in result["tools_hint"]
    assert "call one directly" in result["tools_hint"]


async def test_force_stop_ends_loop_when_model_ignores_guard(streaming: bool) -> None:
    """When a model keeps re-issuing a blocked identical call, the guard pulls
    the ripcord: one last generation is asked for a plain-text answer, instead
    of hammering the same call to the round limit (observed: 37×)."""
    executions = 0

    async def handler(name: str, arguments: dict) -> str:
        nonlocal executions
        executions += 1
        return json.dumps({"ok": True})

    args = {"value": ""}
    # The model insists on the same call far more than the guard tolerates;
    # after force-stop it must produce the final answer.
    repeats = [_same_call_response(f"t{i}", args) for i in range(10)]
    provider = MockAIProvider(
        ai_responses=[*repeats, AIResponse(content="done")], streaming=streaming
    )
    # Rename the echo tool call to a plain tool so it's not a pure discovery tool.
    ch = AIChannel("ai1", provider=provider, tool_handler=handler)
    await respond(
        ch,
        make_event(body="go", channel_id="sms1"),
        _binding([_ECHO_TOOL]),
        RoomContext(room=Room(id="r1")),
    )
    # Executed at most twice (retry latitude), then blocked + force-stopped —
    # nowhere near 10 calls.
    assert executions <= 2
    # The final generation kept the declaration (RFC §6.4), told to answer.
    assert provider.calls[-1].tools == provider.calls[-2].tools
    assert "further tool calls are disabled" in str(provider.calls[-1].messages[-1].content)


async def test_force_stop_also_ends_the_streaming_loop() -> None:
    """The streaming loop (ollama and other streaming providers) must honor
    force_stop too — it re-filters tools every round, so without an explicit
    check the guard flag is set but never acted on and the model keeps
    hammering the blocked call (observed live: a find repeated ~10x)."""
    executions = 0

    async def handler(name: str, arguments: dict) -> str:
        nonlocal executions
        executions += 1
        return json.dumps({"ok": True})

    args = {"command": "find . -type f"}
    repeats = [_same_call_response(f"t{i}", args) for i in range(10)]
    provider = MockAIProvider(
        ai_responses=[*repeats, AIResponse(content="done")],
        streaming=True,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=handler)
    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding([_ECHO_TOOL]),
        RoomContext(room=Room(id="r1")),
    )
    if output.response_stream is not None:
        async for _ in output.response_stream:
            pass
    # Blocked + force-stopped well before the 10 scripted repeats.
    assert executions <= 2
