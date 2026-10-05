"""A tool call whose own handler ends its session runs on, on every host (RMK-515, RFC §12.4).

A hang-up tool ends its session, detaches its room or unplugs the provider:
its call is not interrupted by the ending it caused, and reports its own
outcome; another call of the session the ending reaches is interrupted and
reported cancelled; the ending completes and the kit closes cleanly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel, get_current_voice_session
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
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
        if how == "end_session":
            session = get_current_voice_session()
            assert session is not None
            await channel.end_session(session)
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
    ("realtime", "detach"),
    ("conference", "detach"),
    ("conference", "unplug"),
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
