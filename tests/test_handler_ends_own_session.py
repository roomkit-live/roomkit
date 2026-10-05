"""A tool call whose own handler ends its session runs on, on every host and
door (RMK-515, RMK-520, RFC §12.4).

A hang-up tool ends its session, detaches its room, unplugs the provider or
closes its channel, itself or through a task it starts: its call is not
interrupted by the ending it caused, and reports its own outcome; another
call of the session the ending reaches is interrupted and reported
cancelled; the ending completes and the kit closes cleanly. A spared call
still running when the kit closes is waited for, then cut, and reported
once; nothing of it outlives the close.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, HookExecution, HookTrigger, RoomKit
from roomkit.channels import _realtime_endings
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel, get_current_voice_session
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from tests.conference.test_conference_realtime import ROOM, realtime_kit

TOOLS = [
    {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}
    for name in ("hangup", "lookup")
]


class _Calls:
    """The handlers' side: the hang-up call ends the session once the other
    call is running; the other runs until it is cancelled."""

    def __init__(self, ending: Any) -> None:
        self.ending = ending
        self.lookup_running = asyncio.Event()
        self.trace: list[str] = []

    async def serve(self, name: str) -> str:
        if name == "lookup":
            self.lookup_running.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.trace.append("lookup cancelled")
                raise
            return "late"
        await self.lookup_running.wait()
        await self.ending()
        self.trace.append("hangup returned")
        return "bye"


def _audit(kit: RoomKit) -> list[tuple[str, bool]]:
    seen: list[tuple[str, bool]] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        seen.append((event.tool_call_id, event.cancelled))

    return seen


async def _realtime(how: str) -> tuple[_Calls, list[tuple[str, bool]], RoomKit]:
    provider = MockRealtimeProvider()
    kit = RoomKit()
    holder: dict[str, Any] = {}

    async def ending() -> None:
        channel = holder["channel"]
        session = get_current_voice_session()
        assert session is not None
        if how == "end_session":
            await channel.end_session(session)
        elif how == "spawned_end_session":
            await asyncio.create_task(channel.end_session(session))
        elif how == "shielded_end_session":
            await asyncio.shield(channel.end_session(session))
        elif how == "close_channel":
            await channel.close()
        elif how == "kit_close":
            await kit.close()
        else:
            await kit.detach_channel("r1", "rt")

    calls = _Calls(ending)

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return await calls.serve(name)

    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=TOOLS,
        tool_handler=handler,
    )
    holder["channel"] = channel
    kit.register_channel(channel)
    reports = _audit(kit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    await provider.simulate_tool_call(session, "c2", "lookup", {})
    await provider.simulate_tool_call(session, "c1", "hangup", {})
    return calls, reports, kit


async def _conference(how: str) -> tuple[_Calls, list[tuple[str, bool]], RoomKit]:
    provider = MockRealtimeProvider()
    holder: dict[str, Any] = {}

    async def ending() -> None:
        kit, channel = holder["kit"], holder["channel"]
        if how == "detach":
            await kit.detach_channel(ROOM, "conf")
        elif how == "close_channel":
            await channel.close()
        elif how == "kit_close":
            await kit.close()
        else:
            await channel.unplug_realtime()

    calls = _Calls(ending)

    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        return await calls.serve(name)

    config = ConferenceRealtimeConfig(provider=provider, tools=TOOLS, tool_handler=handler)
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    holder.update(kit=kit, channel=channel)
    reports = _audit(kit)
    session = await channel._realtime.ensure_session(ROOM)
    await provider.simulate_tool_call(session, "c2", "lookup", {})
    await provider.simulate_tool_call(session, "c1", "hangup", {})
    return calls, reports, kit


_CASES = [
    ("realtime", "end_session"),
    ("realtime", "spawned_end_session"),
    ("realtime", "shielded_end_session"),
    ("realtime", "detach"),
    ("realtime", "close_channel"),
    ("conference", "detach"),
    ("conference", "unplug"),
    ("conference", "close_channel"),
]


@pytest.mark.parametrize(("host", "how"), _CASES)
async def test_the_call_that_ends_its_session_runs_on_and_the_other_is_interrupted(
    host: str, how: str
) -> None:
    run = _realtime if host == "realtime" else _conference
    calls, reports, kit = await run(how)
    for _ in range(300):
        if len(reports) == 2:
            break
        await asyncio.sleep(0.01)

    await asyncio.wait_for(kit.close(), 5)

    assert sorted(calls.trace) == ["hangup returned", "lookup cancelled"]
    assert sorted(reports) == [("c1", False), ("c2", True)]


@pytest.mark.parametrize("host", ["realtime", "conference"])
async def test_a_call_that_closes_the_kit_runs_on_and_the_kit_closes(host: str) -> None:
    """The close it caused completes and interrupts the other call; the
    call runs on, but finishes after the framework is gone: its outcome
    reaches no hook (RFC §12.4)."""
    run = _realtime if host == "realtime" else _conference
    calls, reports, kit = await run("kit_close")
    for _ in range(300):
        if len(calls.trace) == 2:
            break
        await asyncio.sleep(0.01)

    await asyncio.wait_for(kit.close(), 5)

    assert sorted(calls.trace) == ["hangup returned", "lookup cancelled"]
    assert reports == [("c2", True)]


async def test_a_backend_call_that_ends_its_session_runs_on() -> None:
    """An agent reasoning backend's hang-up call: its delegation's task holds
    it, and is not cut by the ending the call caused."""
    holder: dict[str, Any] = {}

    async def ending() -> None:
        session = get_current_voice_session()
        assert session is not None
        await holder["channel"].end_session(session)

    calls = _Calls(ending)

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return await calls.serve(name)

    model = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(id="c2", name="lookup", arguments={}),
                    AIToolCall(id="c1", name="hangup", arguments={}),
                ],
            ),
            AIResponse(content="done"),
        ]
    )
    provider = MockRealtimeProvider(full_duplex=True)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=TOOLS,
        reasoning_backend=AgentReasoningBackend(Agent("reasoner", provider=model)),
    )
    holder["channel"] = channel
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    reports = _audit(kit)
    session = await channel.start_session("r", "u", "ws")
    await provider.simulate_delegation(session, "d1", "integrator")
    for _ in range(300):
        if len(calls.trace) == 2 and len(reports) == 2:
            break
        await asyncio.sleep(0.01)
    disconnected = session.id not in provider._sessions

    await asyncio.wait_for(kit.close(), 5)

    assert sorted(calls.trace) == ["hangup returned", "lookup cancelled"]
    assert disconnected
    assert sorted(reports) == [("d1:c1", False), ("d1:c2", True)]


async def _ends_then_works(host: str, work: float) -> tuple[RoomKit, list[Any], list[str]]:
    """A hang-up call that ends its session, then works *work* seconds more."""
    trace: list[str] = []
    ended = asyncio.Event()
    holder: dict[str, Any] = {}

    async def serve() -> str:
        await holder["ending"]()
        ended.set()
        await asyncio.sleep(work)
        trace.append("hangup returned")
        return "bye"

    provider = MockRealtimeProvider()
    if host == "realtime":
        kit = RoomKit()

        async def rt_handler(name: str, arguments: dict[str, Any]) -> str:
            return await serve()

        channel: Any = RealtimeVoiceChannel(
            "rt",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=TOOLS,
            tool_handler=rt_handler,
        )
        kit.register_channel(channel)
        reports = _audit(kit)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "rt")
        session = await channel.start_session("r1", "u", "ws")
        holder["ending"] = lambda: channel.end_session(session)
    else:

        async def conf_handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
            return await serve()

        config = ConferenceRealtimeConfig(
            provider=provider, tools=TOOLS, tool_handler=conf_handler
        )
        kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
        reports = _audit(kit)
        session = await channel._realtime.ensure_session(ROOM)
        holder["ending"] = lambda: kit.detach_channel(ROOM, "conf")
    await provider.simulate_tool_call(session, "c1", "hangup", {})
    await asyncio.wait_for(ended.wait(), 3)
    return kit, reports, trace


@pytest.mark.parametrize("host", ["realtime", "conference"])
async def test_a_spared_call_still_working_at_close_is_waited_for(host: str) -> None:
    kit, reports, trace = await _ends_then_works(host, 0.3)

    await asyncio.wait_for(kit.close(), 5)

    assert trace == ["hangup returned"]
    assert reports == [("c1", False)]


@pytest.mark.parametrize("host", ["realtime", "conference"])
async def test_a_spared_call_past_the_close_bound_is_cut_and_reported(
    host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_realtime_endings, "CLOSE_WAIT_S", 0.1)
    kit, reports, trace = await _ends_then_works(host, 30)

    await asyncio.wait_for(kit.close(), 5)
    await asyncio.sleep(0.05)

    assert trace == []
    assert reports == [("c1", True)]
