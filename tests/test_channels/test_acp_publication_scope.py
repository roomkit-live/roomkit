"""ACP setters called from another channel cannot borrow its publication scope."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import RoomEvent, TextContent
from roomkit.realtime.base import EphemeralEvent
from roomkit.realtime.memory import InMemoryRealtime
from roomkit.tools import current_response_metadata, tool_turn_context
from tests.test_channels.test_acp import _channel
from tests.test_channels.test_acp_usage import _update
from tests.test_framework import AILikeChannel, SimpleChannel


class ObservedRealtime(InMemoryRealtime):
    def __init__(self) -> None:
        super().__init__()
        self.published: list[tuple[str, EphemeralEvent]] = []

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        self.published.append((room_id, event))
        await super().publish_to_room(room_id, event)


@pytest.mark.parametrize("target", ["caller-room", "other-room"])
async def test_a_setter_cannot_publish_as_another_channels_turn(
    tmp_path: Path, target: str
) -> None:
    acp, _, _ = _channel(tmp_path, emit_updates=False)

    class Caller(AILikeChannel):
        async def on_event(
            self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
        ) -> ChannelOutput:
            values = await acp.set_config_option(target, "model", "sonnet")
            assert values["model"] == "sonnet"
            return ChannelOutput.empty()

    realtime = ObservedRealtime()
    kit = RoomKit(realtime=realtime)
    kit.register_channel(acp)
    kit.register_channel(Caller("caller"))
    kit.register_channel(SimpleChannel("sms"))
    try:
        for room_id in ("caller-room", "other-room"):
            await kit.create_room(room_id=room_id)
        for channel_id in ("sms", "caller"):
            await kit.attach_channel("caller-room", channel_id)
        result = await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="alice", content=TextContent(body="go")),
            room_id="caller-room",
        )
        assert result.error is None
    finally:
        await kit.close()
    assert [
        event for _, event in realtime.published if event.data.get("type") == "acp_config_options"
    ] == []


async def test_a_subscriber_can_change_configuration_during_the_opening_announcement(
    tmp_path: Path,
) -> None:
    acp, _, _ = _channel(tmp_path, emit_updates=False)

    class ReentrantRealtime(InMemoryRealtime):
        armed = True

        async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
            if self.armed and event.data.get("type") == "acp_config_options":
                self.armed = False
                assert await acp.set_config_option(room_id, "model", "opus") == {"model": "opus"}
            await super().publish_to_room(room_id, event)

    kit = RoomKit(realtime=ReentrantRealtime())
    kit.register_channel(acp)
    try:
        values = await acp.set_config_option("room-1", "model", "sonnet")
        assert values == {"model": "opus"}
        assert acp.session_config("room-1") == {"model": "opus"}
        assert acp.room_state("room-1")["config"] == {"model": "opus"}
    finally:
        await kit.close()


async def test_an_idle_callback_does_not_borrow_its_transports_record(tmp_path: Path) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    records: list[bool] = []

    class ContextRealtime(InMemoryRealtime):
        async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
            records.append(current_response_metadata() is not None)
            await super().publish_to_room(room_id, event)

    kit = RoomKit(realtime=ContextRealtime())
    kit.register_channel(channel)
    try:
        await channel.set_config_option("room-1", "model", "sonnet")
        records.clear()
        with tool_turn_context(room_id="foreign"):
            assert current_response_metadata() is not None
            await connection.client.session_update("session-1", _update())
            assert current_response_metadata() is not None
        assert records == [False]
    finally:
        await kit.close()


async def test_a_callback_child_keeps_its_turns_narrow_answer_scope(tmp_path: Path) -> None:
    channel, _, _ = _channel(tmp_path)
    realtime = ObservedRealtime()
    kit = RoomKit(realtime=realtime)
    kit.register_channel(channel)
    kit.register_channel(SimpleChannel("sms"))
    changed = asyncio.Event()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def change_from_child(event, context) -> None:
        values = await asyncio.create_task(channel.set_config_option("room-1", "model", "sonnet"))
        assert values["model"] == "sonnet"
        changed.set()

    try:
        await kit.create_room(room_id="room-1")
        for name in ("sms", channel.channel_id):
            await kit.attach_channel("room-1", name)
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id="alice",
                content=TextContent(body="go"),
                response_visibility="intelligence",
            ),
            room_id="room-1",
        )
        assert result.error is None
        await asyncio.wait_for(changed.wait(), 5)
    finally:
        await kit.close()
    assert [event for _, event in realtime.published if event.type.value == "custom"] == []
