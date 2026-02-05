"""Tests for WhatsApp channel."""

from __future__ import annotations

from roomkit.channels import WhatsAppChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.providers.whatsapp.mock import MockWhatsAppProvider
from tests.conftest import make_event


class TestWhatsAppChannel:
    async def test_handle_inbound(self) -> None:
        ch = WhatsAppChannel("wa1")
        ctx = RoomContext(room=Room(id="r1"))
        msg = InboundMessage(
            channel_id="wa1",
            sender_id="+15551234567",
            content=TextContent(body="hello"),
        )
        event = await ch.handle_inbound(msg, ctx)
        assert event.source.channel_type == ChannelType.WHATSAPP

    async def test_deliver_with_provider(self) -> None:
        provider = MockWhatsAppProvider()
        ch = WhatsAppChannel("wa1", provider=provider)
        binding = ChannelBinding(
            channel_id="wa1",
            room_id="r1",
            channel_type=ChannelType.WHATSAPP,
            metadata={"phone_number": "+15559999999"},
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(channel_id="wa1", channel_type=ChannelType.WHATSAPP)
        await ch.deliver(event, binding, ctx)
        assert len(provider.sent) == 1

    async def test_capabilities(self) -> None:
        ch = WhatsAppChannel("wa1")
        caps = ch.capabilities()
        assert ChannelMediaType.TEMPLATE in caps.media_types
        assert caps.supports_templates is True
