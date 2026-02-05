"""Tests for SMS channel."""

from __future__ import annotations

from roomkit.channels import SMSChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.providers.sms.mock import MockSMSProvider
from tests.conftest import make_event


class TestSMSChannel:
    async def test_handle_inbound(self) -> None:
        ch = SMSChannel("sms1")
        ctx = RoomContext(room=Room(id="r1"))
        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        event = await ch.handle_inbound(msg, ctx)
        assert event.room_id == "r1"
        assert event.source.channel_type == ChannelType.SMS

    async def test_deliver_with_provider(self) -> None:
        provider = MockSMSProvider()
        ch = SMSChannel("sms1", provider=provider, from_number="+15550001111")
        binding = ChannelBinding(
            channel_id="sms1",
            room_id="r1",
            channel_type=ChannelType.SMS,
            metadata={"phone_number": "+15559999999"},
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(channel_id="sms1")
        await ch.deliver(event, binding, ctx)
        assert len(provider.sent) == 1

    async def test_capabilities(self) -> None:
        ch = SMSChannel("sms1")
        caps = ch.capabilities()
        assert ChannelMediaType.TEXT in caps.media_types
        assert caps.max_length == 1600
