"""Integration: Human-to-AI conversation with response."""

from __future__ import annotations

from roomkit.channels import SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.mock import MockAIProvider


class TestHumanAI:
    async def test_sms_to_ai_response(self) -> None:
        """SMS message triggers AI response that is stored."""
        kit = RoomKit()
        ai_provider = MockAIProvider(responses=["I can help with that!"])
        sms = SMSChannel("sms1")
        ai = AIChannel("ai1", provider=ai_provider, system_prompt="Be helpful")

        kit.register_channel(sms)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="What is the weather?"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked

        events = await kit.store.list_events("r1")
        # Original event + AI response
        assert len(events) >= 2
        ai_response = [e for e in events if e.source.channel_id == "ai1"]
        assert len(ai_response) >= 1

    async def test_websocket_to_ai_response(self) -> None:
        """WebSocket message triggers AI response delivered back to WebSocket."""
        kit = RoomKit()
        ai_provider = MockAIProvider(responses=["Sure, here's the answer"])
        ws = WebSocketChannel("ws1")
        ai = AIChannel("ai1", provider=ai_provider)
        ws_received: list[RoomEvent] = []

        async def ws_send(conn_id: str, event: RoomEvent) -> None:
            ws_received.append(event)

        ws.register_connection("conn1", ws_send)

        kit.register_channel(ws)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Help me"),
        )
        await kit.process_inbound(msg)
        assert len(ai_provider.calls) == 1

    async def test_ai_context_includes_history(self) -> None:
        """AI provider receives conversation history."""
        kit = RoomKit()
        ai_provider = MockAIProvider(responses=["Reply 1", "Reply 2"])
        sms = SMSChannel("sms1")
        ai = AIChannel("ai1", provider=ai_provider)

        kit.register_channel(sms)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        for body in ["First message", "Second message"]:
            msg = InboundMessage(
                channel_id="sms1",
                sender_id="user1",
                content=TextContent(body=body),
            )
            await kit.process_inbound(msg)

        assert len(ai_provider.calls) == 2
        # Second call should have more context
        second_call = ai_provider.calls[1]
        assert len(second_call.messages) >= 2

    async def test_send_event_delivers_streaming_ai_response(self) -> None:
        """A directly-injected event that wakes a STREAMING intelligence channel
        gets its response consumed + persisted, not silently dropped (regression:
        send_event omitted the streaming-response drain the inbound path runs)."""
        kit = RoomKit()
        sms = SMSChannel("sms1")
        ai = AIChannel(
            "ai1", provider=MockAIProvider(responses=["Streamed reply"], streaming=True)
        )
        kit.register_channel(sms)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        await kit.send_event(room_id="r1", channel_id="sms1", content=TextContent(body="hi"))

        events = await kit.store.list_events("r1")
        ai_replies = [
            e for e in events if e.source.channel_id == "ai1" and getattr(e.content, "body", "")
        ]
        assert len(ai_replies) == 1
        assert ai_replies[0].content.body == "Streamed reply"
