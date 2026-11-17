"""The real plan publication and tracked hook keep their ordering boundaries."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, PlanUpdatedEvent
from roomkit.models import ChannelType
from roomkit.realtime import EphemeralEvent, EphemeralEventType, InMemoryRealtime
from tests.planning_support import answer, message, plan, room

_TASKS = [{"title": "Follow up", "status": "pending"}]


async def test_a_failing_plan_observer_ignores_channel_type_filters_and_keeps_the_plan(
    streaming: bool,
) -> None:
    kit, _, model = await room([plan(_TASKS), answer(), answer()], streaming)
    seen: list[PlanUpdatedEvent] = []
    heard = asyncio.Event()

    @kit.hook(HookTrigger.ON_PLAN_UPDATED, channel_types={ChannelType.SMS})
    async def observe(event: PlanUpdatedEvent, context: Any) -> None:
        seen.append(event)
        heard.set()
        raise RuntimeError("observer failed")

    try:
        result = await kit.process_inbound(message())
        await asyncio.wait_for(heard.wait(), 3)
        assert result.error is None and len(model.calls) == 2
        assert len(seen) == 1 and seen[0].tasks == _TASKS
        await kit.process_inbound(message("Continue."))
        assert "Follow up" in str(model.calls[-1].messages[-1].content)
        assert "## Current Task Plan" in str(model.calls[-1].messages[-1].content)
    finally:
        await kit.close()


class PlanPublication(InMemoryRealtime):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = fail
        self.plans: list[EphemeralEvent] = []

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        if event.type == EphemeralEventType.CUSTOM and event.data.get("type") == "plan_updated":
            self.plans.append(event)
            self.entered.set()
            await self.release.wait()
            if self.fail:
                raise RuntimeError("plan publisher failed")
        await super().publish_to_room(room_id, event)


@pytest.mark.parametrize("fail", [False, True])
async def test_a_plan_hook_observes_stored_state_without_waiting_for_the_owner_publication(
    fail: bool, streaming: bool
) -> None:
    realtime = PlanPublication(fail=fail)
    kit, channel, model = await room([plan(_TASKS), answer()], streaming, realtime=realtime)
    seen: list[PlanUpdatedEvent] = []
    heard = asyncio.Event()
    before = datetime.now(UTC)

    @kit.hook(HookTrigger.ON_PLAN_UPDATED, execution=HookExecution.ASYNC)
    async def observe(event: PlanUpdatedEvent, context: Any) -> None:
        assert context.room.id == event.room_id == "r"
        seen.append(event)
        heard.set()

    turn = asyncio.create_task(kit.process_inbound(message()))
    try:
        await asyncio.wait_for(realtime.entered.wait(), 3)
        # The ordering seam observes the real public call's state, not a
        # second simulated planner. Public hook/publication assertions follow.
        assert channel._planner is not None
        assert channel._planner.plan_for("r") == _TASKS
        await asyncio.wait_for(heard.wait(), 3)
        assert len(seen) == 1 and not realtime.release.is_set()
        [published] = realtime.plans
        assert (published.room_id, published.channel_id, published.user_id) == (
            "r",
            "agent",
            "agent",
        )
        assert published.data == {"type": "plan_updated", "tasks": _TASKS}
        realtime.release.set()
        result = await asyncio.wait_for(turn, 3)
        await asyncio.wait_for(heard.wait(), 3)
        assert result.error is None
        assert len(seen) == 1 and seen[0].tasks == _TASKS
        assert seen[0].channel_id == "agent"
        assert before <= seen[0].timestamp <= datetime.now(UTC)
        assert len(model.calls) == 2
    finally:
        realtime.release.set()
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await kit.close()


@pytest.mark.parametrize("waiter", ["drain", "close"])
async def test_a_retained_plan_observer_does_not_block_the_round_but_is_drained(
    streaming: bool, waiter: str
) -> None:
    kit, _, model = await room([plan(_TASKS), answer()], streaming)
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @kit.hook(HookTrigger.ON_PLAN_UPDATED, execution=HookExecution.ASYNC)
    async def observe(event: PlanUpdatedEvent, context: Any) -> None:
        entered.set()
        await release.wait()
        finished.set()

    turn = asyncio.create_task(kit.process_inbound(message()))
    drain: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        result = await asyncio.wait_for(turn, 3)
        assert result.error is None and len(model.calls) == 2
        drain = asyncio.create_task(
            kit.close() if waiter == "close" else kit.hook_engine.drain_observers()
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(drain), 0.05)
        assert not finished.is_set()
        release.set()
        await asyncio.wait_for(drain, 3)
        assert finished.is_set()
    finally:
        release.set()
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        if drain is not None:
            await asyncio.wait_for(drain, 3)
        await kit.close()
