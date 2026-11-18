"""ACP setters called from another channel cannot borrow its publication scope."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

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


@pytest.mark.parametrize("publish_text", [False, True])
async def test_a_directly_wired_acp_prompt_publishes_thinking_and_tools_in_order(
    tmp_path: Path, publish_text: bool
) -> None:
    from roomkit.realtime import EphemeralEventType as Kind
    from tests.test_turns_wiring import _directly_wired

    agent, _, _ = _channel(
        tmp_path, publish_thinking_text=publish_text, publish_tool_call_text=publish_text
    )
    async with _directly_wired(agent) as (executor, call):
        backend = executor.wiring.env.realtime()
        assert backend is not None
        published: asyncio.Queue[EphemeralEvent] = asyncio.Queue()

        async def record(event: EphemeralEvent) -> None:
            published.put_nowait(event)

        await backend.subscribe("room:r1", record)
        await call()
        expected = [
            Kind.THINKING_START,
            Kind.THINKING_DELTA,
            Kind.TOOL_CALL_START,
            Kind.TOOL_CALL_END,
            Kind.THINKING_END,
        ]
        observed = []
        while len(observed) < len(expected):
            event = await asyncio.wait_for(published.get(), timeout=2)
            if event.type not in expected:
                continue
            assert event.room_created_at is not None
            assert event.channel_id == agent.channel_id
            if event.type == Kind.THINKING_DELTA:
                assert ("thinking" in event.data) is publish_text
            elif event.type == Kind.TOOL_CALL_START:
                assert ("arguments" in event.data["tool_calls"][0]) is publish_text
            elif event.type == Kind.TOOL_CALL_END:
                assert ("result" in event.data["tool_calls"][0]) is publish_text
            observed.append(event.type)
        assert observed == expected


@pytest.mark.parametrize("access", ["private_trigger", "revoked", "missing", "old_incarnation"])
async def test_a_direct_prompt_checks_its_room_and_live_binding_on_each_publication(
    tmp_path: Path, access: str
) -> None:
    from roomkit.models.context import RoomContext
    from roomkit.realtime import EphemeralEventType as Kind
    from roomkit.turns import TurnExecutor
    from tests.test_channels.test_acp_session_setup import _request

    channel, _, _ = _channel(tmp_path)

    class Revocable(ObservedRealtime):
        async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
            await super().publish_to_room(room_id, event)
            if access == "revoked" and event.type == Kind.THINKING_DELTA:
                await kit.set_visibility(room_id, channel.channel_id, "intelligence")

    realtime = Revocable()
    kit = RoomKit(realtime=realtime)
    executor = TurnExecutor(lambda _: None)
    try:
        room = await kit.create_room(room_id="direct")
        kit.register_channel(channel.declaration())
        binding = await kit.attach_channel(room.id, channel.channel_id)
        if access == "missing":
            await kit.detach_channel(room.id, channel.channel_id)
        context = RoomContext(room=room, bindings=[binding])
        if access == "old_incarnation":
            from datetime import timedelta

            context.room = room.model_copy(
                update={"created_at": room.created_at - timedelta(seconds=1)}
            )
        event = RoomEvent.model_validate(
            _request(room.id).model_dump()
            | {
                "response_visibility": "intelligence" if access == "private_trigger" else None,
            }
        )
        executor.use_wiring(kit.turn_executor.wiring.env)
        executor.wire(channel)
        finished = asyncio.Event()

        @executor.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
        async def reported(event: Any, ctx: RoomContext) -> None:
            finished.set()

        realtime.published.clear()
        output = await channel.on_event(event, binding, context)
        assert output.response_stream is not None
        async for _ in output.response_stream:
            pass
        await asyncio.wait_for(finished.wait(), 2)
        activity = [
            (e.type.value, e.data.get("type"))
            for _, e in realtime.published
            if e.channel_id == channel.channel_id
            and e.type.value not in {"presence_offline", "presence_online"}
        ]
        expected = (
            [("custom", "acp_config_options"), ("thinking_start", None), ("thinking_delta", None)]
            if access == "revoked"
            else []
        )
        assert activity == expected
    finally:
        await channel.close()
        await executor.close()
        await kit.close()
