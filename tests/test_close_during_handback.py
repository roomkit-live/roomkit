"""``kit.close()`` cuts a background hand-back under way, on every door that
hands one back (RMK-514, RFC §23.3 step 8).

The work is done and the notified agent is answering the result when the
framework closes: its turn is cancelled, nothing of it is stored after the
close, and ``close()`` does not wait for it. A delegation run by the task
runner (``delegate(wait=False, notify=...)``) is cut as a strategy's
background run is: a supervisor's per-worker or strategy tool, an
asynchronous Loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.orchestration.strategies.loop import _VoiceLoopServer
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.orchestration.strategies.supervisor._inject_per_worker import _PerWorkerToolServer
from roomkit.orchestration.strategies.supervisor._inject_strategy import _StrategyToolServer
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import ToolCallContext, _current_tool_call
from tests.test_framework import SimpleChannel


class _Answering(MockAIProvider):
    """The notified agent: it answers the result until cancelled."""

    def __init__(self) -> None:
        super().__init__(streaming=True)
        self.started = asyncio.Event()
        self.cancelled = False

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        self.started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return AIResponse(content="Thanks, worker.")


async def _kit() -> tuple[RoomKit, Agent, Agent, _Answering]:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    answering = _Answering()
    boss = Agent("boss", provider=answering)
    worker = Agent("worker", provider=MockAIProvider(responses=["APPROVED"], streaming=True))
    kit.register_channel(boss)
    kit.register_channel(worker)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "boss", category=ChannelCategory.INTELLIGENCE)
    return kit, boss, worker, answering


async def _task_runner(kit: RoomKit, boss: Agent, worker: Agent) -> None:
    await kit.delegate("r", "worker", "Find it.", notify="boss")


async def test_the_delegation_still_ends_when_its_hand_back_is_cut() -> None:
    """The task's end goes on past the cut: its waiters wake on its result."""
    kit, boss, worker, answering = await _kit()
    handle = await kit.delegate("r", "worker", "Find it.", notify="boss")
    await asyncio.wait_for(answering.started.wait(), 5)

    await kit.close()
    result = await handle.wait(timeout=1)

    assert (str(result.status), result.output) == ("completed", "APPROVED")


async def _per_worker(kit: RoomKit, boss: Agent, worker: Agent) -> None:
    server = _PerWorkerToolServer(
        kit,
        boss,
        {"delegate_to_worker": "worker"},
        wait=False,
        share_channels=[],
        task_timeout=5.0,
    )
    await server.serve("r", "delegate_to_worker", {"task": "Find it."})


async def _strategy(kit: RoomKit, boss: Agent, worker: Agent) -> None:
    server = _StrategyToolServer(
        kit,
        boss,
        [worker],
        Supervisor(boss, [worker], strategy="parallel")._strategy,
        share_channels=[],
        async_delivery=True,
        task_timeout=10.0,
        max_revisions=1,
    )
    await server.serve("r", "delegate_workers", {"task": "Find it."})


async def _loop(kit: RoomKit, boss: Agent, worker: Agent) -> None:
    reviewer = Agent("rev", provider=MockAIProvider(responses=["APPROVED"]))
    kit.register_channel(reviewer)
    server = _VoiceLoopServer(kit, worker, [reviewer], None, 1)
    # The boss's tool call starts the loop: it is who is told.
    call = ToolCallContext(tool_call_id="x", channel_id="boss", room_id="r")
    token = _current_tool_call.set(call)
    try:
        await server.serve("r", "delegate_loop", {"task": "Find it."})
    finally:
        _current_tool_call.reset(token)


_DOORS: dict[str, Any] = {
    "task-runner": _task_runner,
    "per-worker": _per_worker,
    "strategy": _strategy,
    "loop": _loop,
}


@pytest.mark.parametrize("door", list(_DOORS))
async def test_close_cuts_a_hand_back_under_way(door: str) -> None:
    kit, boss, worker, answering = await _kit()
    await _DOORS[door](kit, boss, worker)
    await asyncio.wait_for(answering.started.wait(), 5)

    began = time.monotonic()
    await kit.close()
    took = time.monotonic() - began

    assert answering.cancelled
    assert took < 5
    stored = [
        event
        for event in await kit.store.list_events("r")
        if event.type == EventType.MESSAGE and event.source.channel_id == "boss"
    ]
    assert stored == []
