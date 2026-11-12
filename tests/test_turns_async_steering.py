"""Awaited channel steering runs beside turn work and ends with its executor."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import RoomKit
from roomkit.models import (
    Cancel,
    ChannelBinding,
    ChannelCategory,
    ChannelOutput,
    ChannelType,
    InboundMessage,
    RoomContext,
    RoomEvent,
    SteeringDirective,
    TextContent,
)
from tests.test_framework import SimpleChannel


class AwaitedSteering(SimpleChannel):
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, *, fails: bool = False) -> None:
        super().__init__("agent", ChannelType.AI)
        self.fails = fails
        self.started = asyncio.Event()
        self.advance = asyncio.Event()
        self.progressed = asyncio.Event()
        self.finish = asyncio.Event()
        self.steering = asyncio.Event()
        self.answer = asyncio.Event()
        self.stopped = asyncio.Event()

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.started.set()
        await self.advance.wait()
        self.progressed.set()
        await self.finish.wait()
        return ChannelOutput.empty()

    async def steer(self, directive: SteeringDirective, *, turn_id: str) -> int:
        self.steering.set()
        try:
            await self.answer.wait()
            if self.fails:
                raise RuntimeError("steering write refused")
            return 1
        finally:
            self.stopped.set()


async def _start(channel: AwaitedSteering) -> tuple[RoomKit, asyncio.Task[object]]:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="room")
    for name in ("sms", "agent"):
        await kit.attach_channel("room", name)
    running = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="alice", content=TextContent(body="hi")),
            room_id="room",
        )
    )
    await asyncio.wait_for(channel.started.wait(), 5)
    return kit, running


@pytest.mark.parametrize("fails", [False, True])
async def test_awaited_steering_keeps_work_running_and_reports_its_result(fails: bool) -> None:
    channel = AwaitedSteering(fails=fails)
    kit, running = await _start(channel)
    steering = asyncio.create_task(kit.steer("room", "agent", Cancel()))
    try:
        await asyncio.wait_for(channel.steering.wait(), 5)
        assert not steering.done()
        channel.advance.set()
        await asyncio.wait_for(channel.progressed.wait(), 5)
        assert not steering.done()
        channel.answer.set()
        assert await asyncio.wait_for(steering, 5) == (0 if fails else 1)
        channel.finish.set()
        await asyncio.wait_for(running, 5)
    finally:
        channel.answer.set()
        channel.advance.set()
        channel.finish.set()
        await kit.close()
        await asyncio.gather(steering, running, return_exceptions=True)


async def test_executor_close_stops_an_awaited_steering_write() -> None:
    channel = AwaitedSteering()
    kit, running = await _start(channel)
    steering = asyncio.create_task(kit.steer("room", "agent", Cancel()))
    try:
        await asyncio.wait_for(channel.steering.wait(), 5)
        await asyncio.wait_for(kit.turn_executor.close(), 5)
        await asyncio.wait_for(channel.stopped.wait(), 5)
        assert not channel.answer.is_set()
    finally:
        channel.answer.set()
        channel.advance.set()
        channel.finish.set()
        await kit.close()
        await asyncio.gather(steering, running, return_exceptions=True)
