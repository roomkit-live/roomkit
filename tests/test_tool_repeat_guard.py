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

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

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


async def test_third_identical_call_is_refused() -> None:
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
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=handler)
    await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding([_ECHO_TOOL]),
        RoomContext(room=Room(id="r1")),
    )

    # Two executions allowed (retry latitude); the third is short-circuited.
    assert len(executions) == 2
    third = _tool_results(provider.calls[3])[-1]
    assert "EXACT arguments" in third["error"]
    assert "STOP" in third["hint"]


async def test_different_arguments_are_not_guarded() -> None:
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
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=handler)
    await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding([_ECHO_TOOL]),
        RoomContext(room=Room(id="r1")),
    )
    assert len(executions) == 3


async def test_second_identical_find_tools_is_refused() -> None:
    async def handler(name: str, arguments: dict) -> str:
        return json.dumps({"ok": True})

    find_call = AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="t", name="find_tools", arguments={"query": "send sms"})],
    )
    provider = MockAIProvider(ai_responses=[find_call, find_call, AIResponse(content="done")])
    ch = AIChannel("ai1", provider=provider, tool_search=True, tool_handler=handler)
    noise = [{"name": f"widget_{i}", "description": f"Operate widget {i}."} for i in range(30)]
    await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding(noise),
        RoomContext(room=Room(id="r1")),
    )

    first = _tool_results(provider.calls[1])[0]
    assert "matches" in first
    # Identical query: pure within the turn, refused on the second call.
    second = _tool_results(provider.calls[2])[-1]
    assert "EXACT arguments" in second["error"]


async def test_unknown_skill_matching_tools_redirects_and_reveals(tmp_path) -> None:
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
    )

    async def handler(name: str, arguments: dict) -> str:
        return json.dumps({"ok": True})

    ch = AIChannel("ai1", provider=provider, skills=registry, tool_handler=handler)
    await ch.on_event(
        make_event(body="play music", channel_id="sms1"),
        _binding(spotify_tools),
        RoomContext(room=Room(id="r1")),
    )

    result = _tool_results(provider.calls[1])[-1]
    assert "not found" in result["error"]
    assert "SpotifySearch" in result["tools_hint"]
    assert "call one directly" in result["tools_hint"]


async def test_force_stop_ends_loop_when_model_ignores_guard() -> None:
    """When a model keeps re-issuing a blocked identical call, the guard pulls
    the ripcord: tools are stripped and a final plain-text answer is forced,
    instead of hammering the same call to the round limit (observed: 37×)."""
    executions = 0

    async def handler(name: str, arguments: dict) -> str:
        nonlocal executions
        executions += 1
        return json.dumps({"ok": True})

    args = {"command": ""}
    # The model insists on the same call far more than the guard tolerates;
    # after force-stop it must produce the final answer.
    repeats = [_same_call_response(f"t{i}", args) for i in range(10)]
    provider = MockAIProvider(ai_responses=[*repeats, AIResponse(content="done")])
    # Rename the echo tool call to a plain tool so it's not a pure discovery tool.
    ch = AIChannel("ai1", provider=provider, tool_handler=handler)
    await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding([_ECHO_TOOL]),
        RoomContext(room=Room(id="r1")),
    )
    # Executed at most twice (retry latitude), then blocked + force-stopped —
    # nowhere near 10 calls.
    assert executions <= 2
    # The final generation ran WITHOUT tools (force-stop strips them).
    last_call = provider.calls[-1]
    assert not last_call.tools


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
