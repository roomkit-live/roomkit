"""A conference's tool calls go through the realtime tool executor (RFC §12.4, RMK-306).

The handler runs inside the call's tool call context, at the depth of the
answer that issued it; a second call under an id in flight sends nothing; a
call the detach interrupts is reported once, as cancelled; a call cancelled
while ON_TOOL_CALL judges it is reported once, as cancelled.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from roomkit import (
    ConferenceRealtimeConfig,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomKit,
    ToolCallEvent,
)
from roomkit.models.event import TextContent
from roomkit.tools import current_tool_call, current_tool_room_id
from roomkit.tools.context import _current_turn_chain_depth
from roomkit.voice.realtime.mock import MockRealtimeProvider
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until

LOOKUP = {"name": "lookup", "description": "Look up", "parameters": {"type": "object"}}


class _Handler:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.seen: list[tuple[Any, ...]] = []
        self.started = 0

    async def __call__(self, room_id: str, name: str, arguments: dict[str, Any]) -> str:
        self.started += 1
        call = current_tool_call()
        self.seen.append(
            (current_tool_room_id(), call and call.tool_call_id, _current_turn_chain_depth())
        )
        await self.release.wait()
        return f"r{self.started}"


async def _conference(handler: _Handler) -> tuple[RoomKit, Any, MockRealtimeProvider, Any, list]:
    provider = MockRealtimeProvider()
    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(provider=provider, tools=[LOOKUP], tool_handler=handler),
    )
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="observe")
    async def observe(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None
    return kit, channel, provider, session, observed


async def test_the_handler_runs_in_the_calls_context_at_its_answers_depth() -> None:
    handler = _Handler()
    handler.release.set()
    kit, _, provider, session, _ = await _conference(handler)
    await kit.send_event(ROOM, "src", TextContent(body="agent says"), chain_depth=2)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results))

    assert handler.seen == [(ROOM, "c1", 3)]
    await kit.close()


async def test_a_second_call_under_an_id_in_flight_sends_nothing() -> None:
    handler = _Handler()
    kit, _, provider, session, observed = await _conference(handler)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(observed))
    handler.release.set()
    await until(lambda: bool(provider.tool_results))

    assert handler.started == 1
    assert [r[2] for r in provider.tool_results] == ["r1"]
    assert "has not had its result yet" in json.loads(observed[0].result)["error"]
    await kit.close()


async def test_a_call_the_detach_interrupts_is_reported_cancelled() -> None:
    handler = _Handler()
    kit, channel, provider, session, observed = await _conference(handler)

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: handler.started == 1)
    await channel._realtime.disconnect_detached(channel._realtime.detach_room(ROOM))
    await until(lambda: bool(observed))

    assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c1", True)]
    assert provider.tool_results == []
    await kit.close()


async def test_a_call_cancelled_while_judged_is_reported_cancelled_once() -> None:
    handler, judging, release = _Handler(), asyncio.Event(), asyncio.Event()
    handler.release.set()
    kit, _, provider, session, observed = await _conference(handler)

    @kit.hook(HookTrigger.ON_TOOL_CALL, name="slow-judge")
    async def slow_judge(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        judging.set()
        await release.wait()
        return HookResult.allow()

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(judging.is_set)
    await provider.simulate_tool_call_cancellation(session, ["c1"])
    await until(lambda: bool(observed))
    await asyncio.sleep(0.05)

    assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c1", True)]
    assert provider.tool_results == []
    await kit.close()


async def test_a_cancellation_while_the_observers_run_adds_no_second_report() -> None:
    handler = _Handler()
    handler.release.set()
    kit, _, provider, session, observed = await _conference(handler)
    in_observer, release = asyncio.Event(), asyncio.Event()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="slow-audit")
    async def slow_audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        in_observer.set()
        await release.wait()

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(in_observer.is_set)
    await provider.simulate_tool_call_cancellation(session, ["c1"])
    release.set()
    await asyncio.sleep(0.1)

    # One report, and nothing sent: the provider freed the id (RFC §12.4).
    assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c1", False)]
    assert provider.tool_results == []
    await kit.close()


class _Reconnecting:
    """A handler whose work makes the provider reconnect, orphaning the ids
    the old socket issued, its own call's included."""

    def __init__(self) -> None:
        self.provider: MockRealtimeProvider | None = None
        self.session: Any = None
        self.ran_on = False

    async def __call__(self, room_id: str, name: str, arguments: dict[str, Any]) -> str:
        assert self.provider is not None
        await self.provider.simulate_tool_call_cancellation(self.session, ["c1"])
        await asyncio.sleep(0)
        self.ran_on = True
        return "reconfigured"


async def test_a_call_whose_handler_caused_the_reconnect_runs_on() -> None:
    """Not abandoned, as on a realtime voice channel: it runs to its end, its
    result stays off the wire and it is reported served (RFC §9.3)."""
    handler = _Reconnecting()
    kit, _, provider, session, observed = await _conference(handler)  # type: ignore[arg-type]
    handler.provider, handler.session = provider, session

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(observed))

    assert handler.ran_on
    assert [(e.tool_call_id, e.cancelled, e.result) for e in observed] == [
        ("c1", False, "reconfigured")
    ]
    assert provider.tool_results == []
    await kit.close()
