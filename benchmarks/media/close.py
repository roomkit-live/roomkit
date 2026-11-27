"""How long the kit's close takes against its bound (RFC §12.10.4, RK-12.10.4-056).

The close takes the grace, plus the turn backend's budget, plus the channels'
and the framework's other backends' closes, each on its own budget, plus at
most two unwinds of 1 second, which only work that ignores its cancellation
spends. Each sample builds a fresh kit with a short ``close_grace``, starts one
piece of work the close has to cut, and times ``kit.close()`` on the real clock:

- ``hook``: a synchronous ``BEFORE_BROADCAST`` hook that waits;
- ``tool``: a model turn's tool call that waits.

The work either honours its cancellation (``-honours``) or ignores it until the
sample releases it, after the close returned (``-ignores``). The same cases run
in roomkit-rs (``benchmarks/``, ``close``).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Any

from roomkit import (
    Agent,
    Channel,
    ChannelCategory,
    ChannelOutput,
    ChannelType,
    EventSource,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
)
from roomkit.core.task_utils import CLOSE_WAIT_S, CUT_UNWIND_S
from roomkit.providers.ai import AIResponse, AIToolCall, MockAIProvider

GRACE_S = 0.05
CASES = ("hook-honours", "hook-ignores", "tool-honours", "tool-ignores")
# Samples per pass: an ignoring case spends its unwinds, seconds each.
SAMPLES = {"hook-honours": 20, "hook-ignores": 4, "tool-honours": 20, "tool-ignores": 2}
# The budgets the bound adds up, in seconds (Python's): the turn backend in
# the owner's process is given one unwind; the agent's close waits for its
# turns on a budget of its own; the transport's close waits for nothing.
TURN_BACKEND_S = CUT_UNWIND_S
CHANNEL_CLOSES_S = {"hook": 0.0, "tool": CLOSE_WAIT_S}
UNWINDS_S = 2 * CUT_UNWIND_S
# The sum bounds the close's waits: its own work between them (releasing the
# store, closing a channel that waits for nothing) is allowed this much on top
# (decided 2026-11-27).
OWN_WORK_S = 0.05
_START_BOUND_S = 5.0


class _Transport(Channel):
    """A transport whose sends end at once and whose close waits for nothing."""

    channel_type = ChannelType.SMS

    async def handle_inbound(self, message: InboundMessage, context: Any) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(self, event: RoomEvent, binding: Any, context: Any) -> ChannelOutput:
        return ChannelOutput.empty()


class _Work:
    """A wait the close cuts: it honours its cancellation, or ignores it
    until :meth:`release` (the sample's, once the close returned)."""

    def __init__(self, *, ignores: bool) -> None:
        self._ignores = ignores
        self.started = asyncio.Event()
        self._released = asyncio.Event()

    async def run(self) -> None:
        self.started.set()
        if not self._ignores:
            await self._released.wait()
            return
        while not self._released.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await self._released.wait()

    def release(self) -> None:
        self._released.set()


def bound_s(case: str, grace: float = GRACE_S) -> float:
    """The RFC's sum for *case*: what its close may take at most."""
    return grace + TURN_BACKEND_S + CHANNEL_CLOSES_S[case.split("-")[0]] + UNWINDS_S


def _hook(kit: RoomKit, work: _Work) -> None:
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="held")
    async def held(event: RoomEvent, ctx: Any) -> HookResult:
        await work.run()
        return HookResult.allow()


async def _tool(kit: RoomKit, work: _Work) -> None:
    tool = {"name": "lookup", "description": "Look up.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
            ),
            AIResponse(content="Found it.", finish_reason="stop"),
        ]
    )

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        await work.run()
        return "found"

    kit.register_channel(Agent("agent", provider=provider, tools=[tool], tool_handler=lookup))
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)


async def sample(case: str, grace: float = GRACE_S) -> float:
    """Seconds one close of a kit running *case*'s work takes."""
    work = _Work(ignores=case.endswith("-ignores"))
    kit = RoomKit(close_grace=grace)
    kit.register_channel(_Transport("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    if case.startswith("hook"):
        _hook(kit, work)
    else:
        await _tool(kit, work)
    message = InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body="Go."))
    call = asyncio.ensure_future(kit.process_inbound(message, room_id="r"))
    try:
        await asyncio.wait_for(work.started.wait(), _START_BOUND_S)
        started = time.perf_counter()
        await kit.close()
        return time.perf_counter() - started
    finally:
        work.release()
        await asyncio.gather(call, return_exceptions=True)
        await kit.close()


def measure(
    cases: tuple[str, ...] = CASES, samples: dict[str, int] | None = None
) -> Callable[[], dict[str, list[float]]]:
    """One pass: every case's samples, each in a fresh event loop's kit."""
    counts = samples or SAMPLES

    def run() -> dict[str, list[float]]:
        return {case: [asyncio.run(sample(case)) for _ in range(counts[case])] for case in cases}

    return run
