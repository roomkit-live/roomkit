"""The shared room path of a delegation consumes a named entry only once."""

from __future__ import annotations

import asyncio

from roomkit import HookResult, HookTrigger, RoomKit
from roomkit.models import InboundResult, RoomContext, RoomEvent
from roomkit.testing import EchoAgent
from tests.test_framework import SimpleChannel
from tests.test_reentrance import _body, _message


async def test_a_shared_delegated_answer_charges_each_adopted_message_once() -> None:
    kit = RoomKit(max_chain_depth=2)
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(EchoAgent("worker", "Found it."))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    answers: list[InboundResult] = []
    child_rooms: list[str] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def enter(event: RoomEvent, context: RoomContext) -> HookResult:
        if _body(event) == "Found it." and context.room.id != "r" and not child_rooms:
            child_rooms.append(context.room.id)
            # The buffered answer uses one of twenty passes. These messages
            # cannot ask the worker for another answer at this depth.
            for _ in range(20):
                answers.append(
                    await kit.process_inbound(
                        _message("nested").model_copy(update={"chain_depth": 2}),
                        room_id=context.room.id,
                    )
                )
        return HookResult.allow()

    try:
        task = await asyncio.wait_for(
            kit.delegate("r", "worker", "Find it.", wait=True, share_channels=["sms"]), 5
        )
        assert child_rooms == [task.child_room_id]
        assert len(answers) == 20
        assert [answer.blocked for answer in answers] == [False] * 19 + [True]
        assert all(answer.event is not None for answer in answers[:19])
        assert answers[-1].reason == "reentry_loop_cap"
        assert answers[-1].event is None
    finally:
        await kit.close()
