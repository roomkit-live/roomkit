"""Tests for Messenger channel."""

from __future__ import annotations

from roomkit.channels import MessengerChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.providers.messenger.mock import MockMessengerProvider
from tests.conftest import make_event


class TestMessengerChannel:
    async def test_handle_inbound(self) -> None:
        ch = MessengerChannel("msg1")
        ctx = RoomContext(room=Room(id="r1"))
        msg = InboundMessage(
            channel_id="msg1",
            sender_id="sender-123",
            content=TextContent(body="hello"),
        )
        event = await ch.handle_inbound(msg, ctx)
        assert event.source.channel_type == ChannelType.MESSENGER

    async def test_deliver_with_provider(self) -> None:
        provider = MockMessengerProvider()
        ch = MessengerChannel("msg1", provider=provider)
        binding = ChannelBinding(
            channel_id="msg1",
            room_id="r1",
            channel_type=ChannelType.MESSENGER,
            metadata={"facebook_user_id": "user-456"},
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(channel_id="msg1", channel_type=ChannelType.MESSENGER)
        await ch.deliver(event, binding, ctx)
        assert len(provider.sent) == 1
        assert provider.sent[0]["to"] == "user-456"

    async def test_capabilities(self) -> None:
        ch = MessengerChannel("msg1")
        caps = ch.capabilities()
        assert ChannelMediaType.TEXT in caps.media_types
        assert ChannelMediaType.TEMPLATE in caps.media_types
        assert caps.max_length == 2000
        assert caps.supports_quick_replies is True
