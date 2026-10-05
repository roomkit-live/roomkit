"""A tool call whose own handler ends its session runs on, on every host and
door (RMK-515, RMK-520, RFC §12.4).

A hang-up tool ends its session, detaches its room, unplugs the provider or
closes its channel, itself or through a task it starts, awaited or not: its
call is not interrupted by the ending it caused, and reports its own outcome;
another call of the session the ending reaches is interrupted there and then,
and reported cancelled; the ending completes and the kit closes cleanly. On a
reasoning backend's door the delegation ends once the hang-up call returns.
A spared call still running when the kit closes is waited for, then cut, and
reported once; nothing of it outlives the close.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, HookExecution, HookTrigger, RoomKit
from roomkit.channels import _realtime_endings
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until

TOOLS = [
    {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}
    for name in ("hangup", "lookup")
]

Ending = Callable[[], Awaitable[Any]]


class _Calls:
    """The handlers' side: the hang-up call ends the session once the other
    call is running, then works *after* seconds more; the other runs until
    it is cancelled."""

    def __init__(self, after: float = 0.0) -> None:
        self.ending: Ending | None = None
        self.after = after
        self.lookup_running = asyncio.Event()
        self.ended = asyncio.Event()
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
        assert self.ending is not None
        await self.ending()
        self.ended.set()
        if self.after:
            # Without a wait the handler returns at once: an ending it did
            # not await then runs while the call delivers and reports.
            await asyncio.sleep(self.after)
        self.trace.append("hangup returned")
        return "bye"


def _audit(kit: RoomKit) -> list[tuple[str, bool]]:
    seen: list[tuple[str, bool]] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        seen.append((event.tool_call_id, event.cancelled))

    return seen


@dataclass
class _Run:
    calls: _Calls
    reports: list[tuple[str, bool]]
    kit: RoomKit
    channel: Any
    session: Any
    model: MockAIProvider | None = None


def _aside(make: Callable[[], Awaitable[Any]]) -> Ending:
    """An ending run in a task the handler starts and does not wait for."""

    async def start() -> None:
        asyncio.ensure_future(make())

    return start


def _spawned(make: Callable[[], Awaitable[Any]]) -> Ending:
    """An ending run in a task the handler starts and waits for."""

    async def run() -> None:
        await asyncio.ensure_future(make())

    return run


# The endings, per host: each makes the hang-up handler's ending from the
# run's kit, channel and session.
_REALTIME_ENDINGS: dict[str, Callable[[RoomKit, Any, Any], Ending]] = {
    "end_session": lambda kit, ch, s: lambda: ch.end_session(s),
    "spawned_end_session": lambda kit, ch, s: _spawned(lambda: ch.end_session(s)),
    "shielded_end_session": lambda kit, ch, s: lambda: asyncio.shield(ch.end_session(s)),
    "unwaited_end_session": lambda kit, ch, s: _aside(lambda: ch.end_session(s)),
    "close_channel": lambda kit, ch, s: ch.close,
    "spawned_close_channel": lambda kit, ch, s: _spawned(ch.close),
    "kit_close": lambda kit, ch, s: kit.close,
}

_CONFERENCE_ENDINGS: dict[str, Callable[[RoomKit, Any, Any], Ending]] = {
    "detach": lambda kit, ch, s: lambda: kit.detach_channel(ROOM, "conf"),
    "unwaited_detach": lambda kit, ch, s: _aside(lambda: kit.detach_channel(ROOM, "conf")),
    "unplug": lambda kit, ch, s: ch.unplug_realtime,
    "close_channel": lambda kit, ch, s: ch.close,
    "spawned_close_channel": lambda kit, ch, s: _spawned(ch.close),
    "kit_close": lambda kit, ch, s: kit.close,
}


async def _realtime(ending: str, after: float = 0.0) -> _Run:
    provider = MockRealtimeProvider()
    calls = _Calls(after)

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return await calls.serve(name)

    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=TOOLS,
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    reports = _audit(kit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    calls.ending = _REALTIME_ENDINGS[ending](kit, channel, session)
    await provider.simulate_tool_call(session, "c2", "lookup", {})
    await provider.simulate_tool_call(session, "c1", "hangup", {})
    return _Run(calls, reports, kit, channel, session)


async def _conference(ending: str, after: float = 0.0) -> _Run:
    provider = MockRealtimeProvider()
    calls = _Calls(after)

    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        return await calls.serve(name)

    config = ConferenceRealtimeConfig(provider=provider, tools=TOOLS, tool_handler=handler)
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    reports = _audit(kit)
    session = await channel._realtime.ensure_session(ROOM)
    calls.ending = _CONFERENCE_ENDINGS[ending](kit, channel, session)
    await provider.simulate_tool_call(session, "c2", "lookup", {})
    await provider.simulate_tool_call(session, "c1", "hangup", {})
    return _Run(calls, reports, kit, channel, session)


def _agent_backend() -> tuple[AgentReasoningBackend, MockAIProvider]:
    """A reasoning backend whose agent calls lookup and hangup in one round."""
    round_ = [
        AIToolCall(id="c2", name="lookup", arguments={}),
        AIToolCall(id="c1", name="hangup", arguments={}),
    ]
    model = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=round_),
            AIResponse(content="done"),
        ]
    )
    return AgentReasoningBackend(Agent("reasoner", provider=model)), model


async def _backend(ending: str) -> _Run:
    calls = _Calls()

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return await calls.serve(name)

    backend, model = _agent_backend()
    provider = MockRealtimeProvider(full_duplex=True)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=TOOLS,
        reasoning_backend=backend,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    reports = _audit(kit)
    session = await channel.start_session("r", "u", "ws")
    calls.ending = _REALTIME_ENDINGS[ending](kit, channel, session)
    await provider.simulate_delegation(session, "d1", "integrator")
    return _Run(calls, reports, kit, channel, session, model)


async def _host(host: str, ending: str, after: float = 0.0) -> _Run:
    return await (_realtime if host == "realtime" else _conference)(ending, after)


_CASES = [
    *(("realtime", ending) for ending in _REALTIME_ENDINGS if ending != "kit_close"),
    *(("conference", ending) for ending in _CONFERENCE_ENDINGS if ending != "kit_close"),
]


@pytest.mark.parametrize(("host", "ending"), _CASES)
async def test_the_call_that_ends_its_session_runs_on_and_the_other_is_interrupted(
    host: str, ending: str
) -> None:
    run = await _host(host, ending)
    await until(lambda: len(run.reports) == 2 and len(run.calls.trace) == 2)

    await asyncio.wait_for(run.kit.close(), 5)

    assert sorted(run.calls.trace) == ["hangup returned", "lookup cancelled"]
    assert sorted(run.reports) == [("c1", False), ("c2", True)]


@pytest.mark.parametrize(
    "ending", ["end_session", "unwaited_end_session", "spawned_close_channel"]
)
async def test_a_backend_call_that_ends_its_session_runs_on_and_ends_its_delegation(
    ending: str,
) -> None:
    """The other call is cut at the ending itself; the hang-up call runs on;
    its delegation then ends: the agent asks its model no second round and
    keeps nothing for the ended session."""
    run = await _backend(ending)
    await asyncio.wait_for(run.calls.ended.wait(), 3)
    await until(lambda: len(run.reports) == 2 and len(run.calls.trace) == 2, timeout=1)
    await asyncio.sleep(0.05)

    assert sorted(run.calls.trace) == ["hangup returned", "lookup cancelled"]
    assert sorted(run.reports) == [("d1:c1", False), ("d1:c2", True)]
    assert run.model is not None and len(run.model.calls) == 1
    assert run.session.id not in run.channel._reasoning_backend._histories
    await asyncio.wait_for(run.kit.close(), 5)


@pytest.mark.parametrize("host", ["realtime", "conference"])
async def test_a_call_that_closes_the_kit_runs_on_and_the_kit_closes(
    host: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The close it caused completes and interrupts the other call; the
    call runs on, but finishes after the framework is gone: its outcome
    reaches no hook, and is logged (RFC §12.4)."""
    caplog.set_level(logging.INFO, logger="roomkit")
    run = await _host(host, "kit_close")
    await until(lambda: "Tool call hangup(c1) served" in caplog.text)

    await asyncio.wait_for(run.kit.close(), 5)

    assert sorted(run.calls.trace) == ["hangup returned", "lookup cancelled"]
    assert run.reports == [("c2", True)]


_ENDS_ITS_SESSION = [("realtime", "end_session"), ("conference", "detach")]


@pytest.mark.parametrize(("host", "ending"), _ENDS_ITS_SESSION)
async def test_a_spared_call_still_working_at_close_is_waited_for(host: str, ending: str) -> None:
    run = await _host(host, ending, after=0.3)
    await asyncio.wait_for(run.calls.ended.wait(), 3)

    await asyncio.wait_for(run.kit.close(), 5)

    assert sorted(run.calls.trace) == ["hangup returned", "lookup cancelled"]
    assert sorted(run.reports) == [("c1", False), ("c2", True)]


@pytest.mark.parametrize(("host", "ending"), _ENDS_ITS_SESSION)
async def test_a_spared_call_past_the_close_bound_is_cut_and_reported(
    host: str, ending: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_realtime_endings, "CLOSE_WAIT_S", 0.1)
    run = await _host(host, ending, after=30)
    await asyncio.wait_for(run.calls.ended.wait(), 3)

    await asyncio.wait_for(run.kit.close(), 5)
    await asyncio.sleep(0.05)

    assert run.calls.trace == ["lookup cancelled"]
    assert sorted(run.reports) == [("c1", True), ("c2", True)]


@pytest.mark.parametrize(("host", "ending"), _ENDS_ITS_SESSION)
async def test_a_spared_call_is_held_only_while_it_runs(host: str, ending: str) -> None:
    run = await _host(host, ending)
    await until(lambda: len(run.calls.trace) == 2)
    await asyncio.sleep(0.05)
    realtime = run.channel if host == "realtime" else run.channel._realtime

    assert len(realtime._spared_calls) == 0
    await asyncio.wait_for(run.kit.close(), 5)
