"""Integration: Sync hook blocks sensitive content and injects notices."""

from __future__ import annotations

from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType, HookTrigger
from roomkit.models.event import (
    EventSource,
    RoomEvent,
    SystemContent,
    TextContent,
)
from roomkit.models.hook import HookResult, InjectedEvent


class TestSINScanning:
    async def test_block_sensitive_content(self) -> None:
        """Hook blocks messages containing SIN-like patterns."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="sin_scanner")
        async def scan(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if isinstance(event.content, TextContent) and "123-456-789" in event.content.body:
                notice = RoomEvent(
                    room_id=event.room_id,
                    source=EventSource(
                        channel_id="system",
                        channel_type=ChannelType.WEBHOOK,
                    ),
                    content=SystemContent(
                        body="Message blocked: potential SIN detected",
                        code="block_notice",
                    ),
                    type=EventType.SYSTEM,
                )
                return HookResult.block(
                    "SIN detected",
                    injected=[InjectedEvent(event=notice)],
                )
            return HookResult.allow()

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="My SIN is 123-456-789"),
        )
        result = await kit.process_inbound(msg)
        assert result.blocked
        assert result.reason == "SIN detected"

    async def test_allow_clean_content(self) -> None:
        """Clean messages pass through the scanner."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="sin_scanner")
        async def scan(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if isinstance(event.content, TextContent) and "123-456-789" in event.content.body:
                return HookResult.block("SIN detected")
            return HookResult.allow()

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Hello, how are you?"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked

    async def test_redact_sensitive_content(self) -> None:
        """Hook modifies message to redact sensitive content."""
        kit = RoomKit()
        ws1 = WebSocketChannel("ws1")
        ws2 = WebSocketChannel("ws2")
        ws2_received: list[RoomEvent] = []

        async def ws2_send(conn_id: str, event: RoomEvent) -> None:
            ws2_received.append(event)

        ws2.register_connection("conn2", ws2_send)

        kit.register_channel(ws1)
        kit.register_channel(ws2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ws2")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="redactor")
        async def redact(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if isinstance(event.content, TextContent) and "secret" in event.content.body.lower():
                modified = event.model_copy(update={"content": TextContent(body="[REDACTED]")})
                return HookResult.modify(modified)
            return HookResult.allow()

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="The secret code is XYZ"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked
        assert result.event is not None
        assert isinstance(result.event.content, TextContent)
        assert result.event.content.body == "[REDACTED]"
        assert len(ws2_received) == 1
        assert isinstance(ws2_received[0].content, TextContent)
        assert ws2_received[0].content.body == "[REDACTED]"

    async def test_inject_notice_on_block(self) -> None:
        """Injected events are stored even when message is blocked."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="blocker_with_notice")
        async def block_with_notice(event: RoomEvent, ctx: RoomContext) -> HookResult:
            notice = RoomEvent(
                room_id=event.room_id,
                source=EventSource(
                    channel_id="system",
                    channel_type=ChannelType.WEBHOOK,
                ),
                content=SystemContent(body="Blocked for compliance"),
                type=EventType.SYSTEM,
            )
            return HookResult.block(
                "compliance",
                injected=[InjectedEvent(event=notice)],
            )

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="anything"),
        )
        result = await kit.process_inbound(msg)
        assert result.blocked
        events = await kit.store.list_events("r1")
        # RFC §4.2: both the blocked original event (audit) and the injected notice are stored
        assert len(events) >= 2
        blocked_events = [e for e in events if e.status.value == "blocked"]
        system_events = [e for e in events if e.type == EventType.SYSTEM]
        assert len(blocked_events) >= 1  # Blocked original
        assert len(system_events) >= 1  # Injected notice
