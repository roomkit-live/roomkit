"""The edges of a turn's toolset read alike on every door (RMK-430, RFC §6.4, §21.1).

A provider's native tool without a name stays declared under Tool Search and
any policy; ``call_tool`` is exempt like the other search tools; the re-read
of a stored result is in the allowed names and ``list_tools`` once a round
declares it; ``AFTER_TOOL_ROUND`` sees the toolset ``BEFORE_AI_GENERATION``
sees; an ``activate_skill`` named after tools hints and reveals them alike on
text and realtime, the reveal kept for the next turn.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from roomkit import (
    ConferenceRealtimeConfig,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.hooks import SyncPipelineResult
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.tools import current_tool_allowed_names
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.tools.policy import ToolPolicy
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit
from tests.conftest import make_event
from tests.tool_loop_modes import respond

PARAMS: dict[str, Any] = {"type": "object", "properties": {}}
NATIVE: dict[str, Any] = {"google_search": {}}


def _schema(name: str) -> dict[str, Any]:
    return {"name": name, "description": f"{name} tool", "parameters": PARAMS}


def _tool(name: str) -> AITool:
    return AITool(name=name, description=f"{name} tool", parameters=PARAMS)


def _calling(tool: str, **arguments: Any) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"call-{tool}", name=tool, arguments=arguments)],
    )


def _skills(tmp_path: Path, name: str, gates: str | None = None) -> SkillRegistry:
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    front = [f"name: {name}", f"description: {name} skill"]
    if gates:
        front.append(f"allowed_tools: {gates}")
    (folder / "SKILL.md").write_text("---\n" + "\n".join(front) + "\n---\nBody.", encoding="utf-8")
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


class _Recorder:
    """A host tool handler recording the allowed names each call reads."""

    def __init__(self) -> None:
        self.allowed: list[set[str] | None] = []

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.allowed.append(current_tool_allowed_names())
        return f"ok:{name}"


async def _text_turn(channel: AIChannel, room_id: str = "r1") -> None:
    binding = ChannelBinding(
        channel_id=channel.channel_id,
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    event = make_event(room_id=room_id, body="go", channel_id="sms1")
    await respond(channel, event, binding, RoomContext(room=Room(id=room_id)))


async def _session(channel: RealtimeVoiceChannel) -> tuple[RoomKit, Any]:
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", channel.channel_id)
    return kit, await channel.start_session("r1", "u", "ws")


async def _call(
    channel: RealtimeVoiceChannel,
    provider: MockRealtimeProvider,
    session: Any,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    before = len(provider.tool_results)
    await provider.simulate_tool_call(session, f"c-{name}-{before}", name, arguments or {})
    for _ in range(100):
        await asyncio.gather(*list(channel._scheduled_tasks), return_exceptions=True)
        if len(provider.tool_results) > before:
            break
        await asyncio.sleep(0.01)
    return provider.tool_results[-1][2]


def _declared(provider: MockRealtimeProvider) -> list[Any]:
    calls = [c for c in provider.calls if c.method in ("connect", "reconfigure")]
    return list(calls[-1].args.get("tools") or [])


@pytest.mark.parametrize(
    "settings",
    [{"tool_search": True}, {"tool_policy": ToolPolicy(allow=["crm_*"])}],
)
async def test_a_native_tool_without_a_name_stays_declared(settings: dict[str, Any]) -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[NATIVE, _schema("crm_lookup"), _schema("crm_book")],
        tool_handler=_Recorder(),
        **settings,
    )
    kit, _ = await _session(channel)

    assert NATIVE in _declared(provider)
    await kit.close()


async def test_a_conference_keeps_a_native_tool_under_an_allow_list() -> None:
    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(
        provider=provider,
        tools=[NATIVE, _schema("crm_lookup")],
        tool_handler=lambda *a: "ok",
        tool_policy=ToolPolicy(allow=["crm_*"]),
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    await channel._realtime.ensure_session(ROOM)

    assert NATIVE in _declared(provider)
    await kit.close()


class _FixedProvider(MockRealtimeProvider):
    @property
    def supports_mid_session_reconfigure(self) -> bool:
        return False


async def test_call_tool_is_an_allowed_name_under_an_allow_list() -> None:
    recorder = _Recorder()
    provider = _FixedProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[_schema("crm_a"), _schema("crm_b")],
        tool_handler=recorder,
        tool_policy=ToolPolicy(allow=["crm_*"]),
        tool_search=True,
    )
    kit, session = await _session(channel)

    await _call(channel, provider, session, "call_tool", {"name": "crm_a", "arguments_json": "{}"})

    [allowed] = recorder.allowed
    assert allowed is not None and "call_tool" in allowed
    assert "call_tool" in [t.get("name") for t in _declared(provider)]
    await kit.close()


async def test_the_reread_tool_is_offered_before_any_result_is_stored() -> None:
    recorder = _Recorder()
    provider = MockAIProvider(
        ai_responses=[_calling("lookup"), _calling("list_tools"), AIResponse(content="done")]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[_tool("lookup")],
        tool_handler=recorder,
        tool_search=True,
    )

    await _text_turn(channel)

    [allowed] = recorder.allowed
    assert allowed is not None and "read_stored_result" in allowed
    listed = [
        json.loads(str(part.result))
        for message in provider.calls[2].messages
        if message.role == "tool"
        for part in message.content
        if part.name == "list_tools"
    ]
    assert "read_stored_result" in {t["name"] for t in listed[0]["tools"]}


@pytest.mark.parametrize("tool_search", [False, True])
async def test_after_tool_round_sees_the_toolset_before_ai_generation_sees(
    tmp_path: Path, tool_search: bool
) -> None:
    provider = MockAIProvider(ai_responses=[_calling("lookup"), AIResponse(content="done")])
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[_tool(n) for n in ("lookup", "secret", "gated_cal")],
        tool_handler=_Recorder(),
        tool_policy=ToolPolicy(deny=["secret"]),
        skills=_skills(tmp_path, "cal", gates="gated_cal"),
        tool_search=tool_search,
    )
    seen: dict[str, set[str]] = {}

    async def before(event: Any) -> SyncPipelineResult:
        seen["before"] = {t.name for t in event.ai_context.tools}
        return SyncPipelineResult(allowed=True, event=event)

    async def after(event: Any) -> None:
        seen["after"] = set(event.tools)

    channel._before_generation_hook = before
    channel._after_tool_round_hook = after
    await _text_turn(channel)

    # The re-read is declared by the rounds, not shown to the generation hook.
    assert seen["after"] - {"read_stored_result"} == seen["before"]
    assert not {"secret", "gated_cal"} & seen["after"]


async def test_after_tool_round_leaves_out_a_withdrawn_tool() -> None:
    provider = MockAIProvider(
        ai_responses=[_calling("lookup"), _calling("lookup"), AIResponse(content="done")]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[_tool("lookup"), _tool("book")],
        tool_handler=_Recorder(),
        tool_search=False,
    )
    rounds: list[set[str]] = []

    async def after(event: Any) -> None:
        rounds.append(set(event.tools))
        event.withdraw("book")

    channel._after_tool_round_hook = after
    await _text_turn(channel)

    assert "book" in rounds[0] and "book" not in rounds[1]


async def test_a_text_activation_named_after_tools_reveals_them_for_the_next_turn(
    tmp_path: Path,
) -> None:
    provider = MockAIProvider(
        ai_responses=[
            _calling("activate_skill", name="spotify"),
            AIResponse(content="done"),
            AIResponse(content="next turn"),
        ]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[
            _tool(n) for n in ("spotify_play", "spotify_search", *(f"x{i}" for i in range(30)))
        ],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path, "guide"),
        tool_search=True,
    )

    await _text_turn(channel)
    await _text_turn(channel)

    declared_next_turn = {t.name for t in provider.calls[2].tools or [] if not t.defer_loading}
    assert {"spotify_play", "spotify_search"} <= declared_next_turn


async def test_a_realtime_activation_named_after_tools_hints_and_reveals_them(
    tmp_path: Path,
) -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[
            _schema(n) for n in ("spotify_play", "spotify_search", *(f"x{i}" for i in range(30)))
        ],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path, "guide"),
        tool_search=True,
    )
    kit, session = await _session(channel)
    assert "spotify_play" not in [t.get("name") for t in _declared(provider)]

    result = json.loads(
        await _call(channel, provider, session, "activate_skill", {"name": "spotify"})
    )

    assert "spotify_play, spotify_search" in result["tools_hint"]
    assert {"spotify_play", "spotify_search"} <= {t.get("name") for t in _declared(provider)}
    await kit.close()


async def test_a_withdrawn_tool_leaves_the_allowed_names() -> None:
    recorder = _Recorder()
    provider = MockAIProvider(
        ai_responses=[_calling("lookup"), _calling("lookup"), AIResponse(content="done")]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[_tool("lookup"), _tool("book")],
        tool_handler=recorder,
        tool_search=False,
    )

    async def after(event: Any) -> None:
        event.withdraw("book")

    channel._after_tool_round_hook = after
    await _text_turn(channel)

    first, second = recorder.allowed
    assert first is not None and "book" in first
    assert second is not None and "book" not in second


async def test_an_unavailable_skill_named_like_tools_gets_its_reason_not_a_hint(
    tmp_path: Path,
) -> None:
    skills = _skills(tmp_path, "spotify")
    skills.mark_unavailable("spotify", "its tools are not granted here")
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[_schema("spotify_play")],
        tool_handler=_Recorder(),
        skills=skills,
    )
    kit, session = await _session(channel)

    result = json.loads(
        await _call(channel, provider, session, "activate_skill", {"name": "spotify"})
    )

    assert "unavailable" in result["error"] and "tools_hint" not in result
    await kit.close()


SPOTIFY = ("spotify_play", "spotify_search", *(f"x{i}" for i in range(30)))


async def test_a_blocked_text_activation_reveals_nothing(tmp_path: Path) -> None:
    provider = MockAIProvider(
        ai_responses=[_calling("activate_skill", name="spotify"), AIResponse(content="done")]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[_tool(n) for n in SPOTIFY],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path, "guide"),
        tool_search=True,
    )
    kit = RoomKit()
    kit.register_channel(channel)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="no_skills")
    async def block(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("no skills here")

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1")
    binding = ChannelBinding(channel_id="ai1", room_id="r1", channel_type=ChannelType.AI)
    event = make_event(room_id="r1", body="go", channel_id="sms1")
    await respond(channel, event, binding, await kit._build_context("r1"))

    declared = {t.name for t in provider.calls[1].tools or [] if not t.defer_loading}
    assert not {"spotify_play", "spotify_search"} & declared
    await kit.close()


async def test_a_text_activation_hints_no_tool_the_channel_serves_itself(tmp_path: Path) -> None:
    provider = MockAIProvider(
        ai_responses=[_calling("activate_skill", name="skill"), AIResponse(content="done")]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[_tool("lookup")],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path, "guide"),
    )

    await _text_turn(channel)

    [answer] = [
        json.loads(str(part.result))
        for message in provider.calls[1].messages
        if message.role == "tool"
        for part in message.content
    ]
    assert "tools_hint" not in answer


@pytest.mark.parametrize("wanted", ["skill", "tools"])
async def test_a_realtime_activation_hints_no_tool_the_channel_serves_itself(
    tmp_path: Path, wanted: str
) -> None:
    """As on the text path (RFC §24.4): only the session's own catalogue is
    matched, never activate_skill, read_skill_reference or Tool Search's."""
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[_schema(n) for n in ("weather_now", "reader_feed", *(f"x{i}" for i in range(30)))],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path, "guide"),
        tool_search=True,
    )
    kit, session = await _session(channel)

    result = json.loads(
        await _call(channel, provider, session, "activate_skill", {"name": wanted})
    )
    await kit.close()

    assert "tools_hint" not in result


async def test_a_fixed_provider_hint_points_to_call_tool(tmp_path: Path) -> None:
    provider = _FixedProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[_schema(n) for n in SPOTIFY],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path, "guide"),
        tool_search=True,
    )
    kit, session = await _session(channel)

    result = json.loads(
        await _call(channel, provider, session, "activate_skill", {"name": "spotify"})
    )

    assert (
        "call_tool" in result["tools_hint"] and "now in your tool list" not in result["tools_hint"]
    )
    await kit.close()


async def test_a_hint_naming_only_a_pinned_tool_reconfigures_nothing(tmp_path: Path) -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[_schema(n) for n in ("spotify_play", *(f"x{i}" for i in range(30)))],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path, "guide"),
        tool_search=True,
        tool_search_pinned=["spotify_play"],
    )
    kit, session = await _session(channel)
    before = len(provider.calls)

    result = json.loads(
        await _call(channel, provider, session, "activate_skill", {"name": "spotify"})
    )

    assert "spotify_play" in result["tools_hint"]
    assert [c.method for c in provider.calls[before:]] == ["submit_tool_result"]
    await kit.close()


async def test_a_blocked_realtime_activation_reveals_nothing(tmp_path: Path) -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[_schema(n) for n in SPOTIFY],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path, "guide"),
        tool_search=True,
    )
    kit, session = await _session(channel)

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="no_skills")
    async def block(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("no skills here")

    await _call(channel, provider, session, "activate_skill", {"name": "spotify"})

    assert "spotify_play" not in [t.get("name") for t in _declared(provider)]
    await kit.close()


async def test_a_native_tool_survives_skill_gating(tmp_path: Path) -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[NATIVE, _schema("crm_lookup")],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path, "cal", gates="*"),
    )
    kit, session = await _session(channel)

    assert NATIVE in _declared(provider)
    # Activating a skill reads the catalogue by name: a native tool is skipped.
    result = await _call(channel, provider, session, "activate_skill", {"name": "cal"})
    assert "KeyError" not in result
    await kit.close()


async def test_a_native_tool_does_not_count_toward_tool_search(tmp_path: Path) -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[NATIVE, *(_schema(f"t{i}") for i in range(3))],
        tool_handler=_Recorder(),
        tool_search_threshold=3,
    )
    kit, _ = await _session(channel)

    assert "find_tools" not in [t.get("name") for t in _declared(provider)]
    await kit.close()


def _person() -> HumanInputToolHandler:
    ask = AITool(
        name="ask_person", description="Ask the person", parameters=_schema("x")["parameters"]
    )
    return HumanInputToolHandler({"ask_person"}, timeout=2.0, tool_definitions=[ask])


async def test_an_activation_hints_no_human_input_tool_on_either_path(tmp_path: Path) -> None:
    """The channel declares and serves its human-input tools itself (RFC
    §9.3), so a hint names them on neither path (RFC §24.4)."""
    text_provider = MockAIProvider(
        ai_responses=[_calling("activate_skill", name="ask"), AIResponse(content="done")]
    )
    text = AIChannel(
        "ai1",
        provider=text_provider,
        tools=[_tool("ask_db")],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path / "text", "guide"),
        human_input_handler=_person(),
        tool_search=False,
    )
    await _text_turn(text)
    [text_answer] = [
        json.loads(str(part.result))
        for message in text_provider.calls[1].messages
        if message.role == "tool"
        for part in message.content
    ]

    provider = MockRealtimeProvider()
    realtime = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[_schema("ask_db")],
        tool_handler=_Recorder(),
        skills=_skills(tmp_path / "realtime", "guide"),
        human_input_handler=_person(),
        tool_search=False,
    )
    kit, session = await _session(realtime)
    realtime_answer = json.loads(
        await _call(realtime, provider, session, "activate_skill", {"name": "ask"})
    )
    await kit.close()

    for answer in (text_answer, realtime_answer):
        assert "ask_db" in answer["tools_hint"]
        assert "ask_person" not in answer["tools_hint"]
