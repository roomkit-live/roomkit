"""A realtime call cut while it was reported, or issued once its session
ended, is reported once (RMK-431, RFC §9.3, §12.4).

A call whose result went out before its report (a refused or failed call)
owes the observers that outcome when an ending cuts in between; a Tool Search
call, judged before its result goes out (RMK-447), is cancelled by an ending
that cuts its judgement, and nothing is sent. A call issued once its session
ended, by the provider or a reasoning backend, runs no gate and is reported
once, cancelled.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from roomkit import (
    ConferenceRealtimeConfig,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    ToolCallEvent,
    ToolCallResult,
)
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.base import VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningOutput, ReasoningRequest
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until

MANY = [
    {
        "name": f"tool_{i}",
        "description": f"weather forecast {i}",
        "parameters": {"type": "object", "properties": {}},
    }
    for i in range(30)
]
HANGUP_THEN_LOOKUP = [
    {"name": "hangup", "parameters": {"type": "object"}},
    {"name": "lookup", "parameters": {"type": "object"}},
]


async def _until(condition: Any) -> None:
    for _ in range(300):
        if condition():
            return
        await asyncio.sleep(0.01)


async def _session(channel: RealtimeVoiceChannel) -> tuple[RoomKit, Any]:
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", channel.channel_id)
    return kit, await channel.start_session("r1", "u1", "ws")


def _observe(kit: RoomKit) -> list[ToolCallEvent]:
    seen: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        seen.append(event)

    return seen


async def test_a_search_call_whose_judgement_the_ending_cut_is_cancelled() -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=MANY,
        tool_search=True,
    )
    kit, session = await _session(channel)
    held = asyncio.Event()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="slow")
    async def slow(event: ToolCallEvent, ctx: Any) -> HookResult:
        held.set()
        await asyncio.sleep(30)
        return HookResult.allow()

    seen = _observe(kit)
    await provider.simulate_tool_call(session, "c1", "find_tools", {"query": "weather forecast"})
    await asyncio.wait_for(held.wait(), 5)
    await channel.end_session(session)
    await _until(lambda: bool(seen))
    await asyncio.sleep(0.05)

    [report] = seen
    assert (report.name, report.is_error, report.cancelled) == ("find_tools", True, True)
    assert provider.tool_results == []
    assert not channel._tool_search_support._exposed.get(session.id)
    await kit.close()


class _HangsUpThenLooksUp(ReasoningBackend):
    def __init__(self) -> None:
        self.results: list[tuple[str, ToolCallResult]] = []

    async def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        assert request.execute_tool_call is not None
        for name in ("hangup", "lookup"):
            self.results.append((name, await request.execute_tool_call(name, {})))
        yield ReasoningOutput("done", is_final=True)


async def test_a_backend_that_hangs_up_ends_its_delegation_and_its_next_call() -> None:
    """The hang-up call ended its own session: once it returns, its
    delegation ends, and the call the backend would make next never runs,
    nor is anything reported for it (RFC §12.4)."""
    provider = MockRealtimeProvider(full_duplex=True)
    holder: dict[str, Any] = {}
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        if name == "hangup":
            await holder["channel"].end_session(holder["session"])
            return "bye"
        return "found"

    backend = _HangsUpThenLooksUp()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=HANGUP_THEN_LOOKUP,
        reasoning_backend=backend,
    )
    holder["channel"] = channel
    kit, session = await _session(channel)
    holder["session"] = session
    gated: list[str] = []

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, name="gate")
    async def gate(event: ToolCallEvent, ctx: Any) -> HookResult:
        gated.append(event.name)
        return HookResult.allow()

    seen = _observe(kit)
    await provider.simulate_delegation(session, "d1", "integrator")
    await _until(lambda: bool(seen))
    await asyncio.sleep(0.1)

    assert ran == ["hangup"]
    # Whoever issued it is gone: no gate runs for it.
    assert gated == ["hangup"]
    # The delegation ended before the backend read the next call's result.
    assert [name for name, _ in backend.results] == ["hangup"]
    assert [(e.name, e.is_error, e.cancelled) for e in seen] == [("hangup", False, False)]
    await kit.close()


class _SlowRevealProvider(MockRealtimeProvider):
    """Reveals tools slowly: an ending lands after the result went out."""

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        return True

    async def reconfigure(self, session: Any, **kwargs: Any) -> None:
        self.revealing.set()
        await asyncio.sleep(30)

    def __init__(self) -> None:
        super().__init__()
        self.revealing = asyncio.Event()


async def test_a_search_call_cut_before_its_report_keeps_what_the_model_read() -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    provider = _SlowRevealProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=MANY,
        tool_search=True,
    )
    kit, session = await _session(channel)
    seen = _observe(kit)

    await provider.simulate_tool_call(session, "c1", "find_tools", {"query": "weather forecast"})
    await asyncio.wait_for(provider.revealing.wait(), 5)
    await channel.end_session(session)
    await _until(lambda: bool(seen))

    assert [(e.name, e.is_error, e.cancelled) for e in seen] == [("find_tools", False, False)]
    await kit.close()


class _SlowSubmitProvider(MockRealtimeProvider):
    """Writes a result slowly: an ending lands while it goes out."""

    def __init__(self) -> None:
        super().__init__()
        self.submitting = asyncio.Event()

    async def submit_tool_result(self, session: Any, call_id: str, result: str) -> None:
        await super().submit_tool_result(session, call_id, result)
        self.submitting.set()
        await asyncio.sleep(30)

    async def submit_tool_error(self, session: Any, call_id: str, result: str) -> None:
        await self.submit_tool_result(session, call_id, result)


async def test_a_failed_call_cut_while_it_goes_out_is_reported_once() -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        raise RuntimeError("backend down")

    provider = _SlowSubmitProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=[{"name": "lookup", "parameters": {"type": "object"}}],
    )
    kit, session = await _session(channel)
    seen = _observe(kit)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await asyncio.wait_for(provider.submitting.wait(), 5)
    await channel.end_session(session)
    await _until(lambda: bool(seen))
    await asyncio.sleep(0.05)

    assert [(e.tool_call_id, e.is_error, e.cancelled) for e in seen] == [("c1", True, False)]
    await kit.close()


async def test_a_provider_call_on_an_ended_session_is_reported_once_cancelled() -> None:
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=[{"name": "lookup", "parameters": {"type": "object"}}],
    )
    kit, session = await _session(channel)
    seen = _observe(kit)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    # The provider's connection dropped before the call was served.
    session.state = VoiceSessionState.ENDED
    await _until(lambda: bool(seen))

    assert ran == []
    assert provider.tool_results == []
    assert [(e.tool_call_id, e.is_error, e.cancelled) for e in seen] == [("c1", True, True)]
    await kit.close()


async def test_a_call_from_a_conference_session_left_behind_is_reported_once() -> None:
    ran: list[str] = []

    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(
        provider=provider,
        tools=[{"name": "lookup", "parameters": {"type": "object"}}],
        tool_handler=handler,
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None
    seen = _observe(kit)
    await kit.detach_channel(ROOM, "conf")

    # A late frame of the session the conference no longer speaks for.
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(seen))

    assert ran == []
    assert [(e.tool_call_id, e.is_error, e.cancelled) for e in seen] == [("c1", True, True)]
    await kit.close()


async def test_a_provider_call_arriving_after_the_session_ended_is_reported_once() -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=[{"name": "lookup", "parameters": {"type": "object"}}],
    )
    kit, session = await _session(channel)
    seen = _observe(kit)
    await channel.end_session(session)

    # A late frame of the connection that just closed.
    await provider.simulate_tool_call(session, "late", "lookup", {})
    await _until(lambda: bool(seen))

    assert [(e.tool_call_id, e.cancelled) for e in seen] == [("late", True)]
    await kit.close()
