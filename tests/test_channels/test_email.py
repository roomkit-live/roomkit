"""Tests for Email channel."""

from __future__ import annotations

from roomkit.channels import EmailChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.providers.email.mock import MockEmailProvider
from tests.conftest import make_event


class TestEmailChannel:
    async def test_handle_inbound(self) -> None:
        ch = EmailChannel("email1")
        ctx = RoomContext(room=Room(id="r1"))
        msg = InboundMessage(
            channel_id="email1",
            sender_id="user@example.com",
            content=TextContent(body="hello"),
        )
        event = await ch.handle_inbound(msg, ctx)
        assert event.source.channel_type == ChannelType.EMAIL

    async def test_deliver_with_provider(self) -> None:
        provider = MockEmailProvider()
        ch = EmailChannel("email1", provider=provider, from_address="noreply@test.com")
        binding = ChannelBinding(
            channel_id="email1",
            room_id="r1",
            channel_type=ChannelType.EMAIL,
            metadata={"email_address": "user@example.com", "subject": "Test"},
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(channel_id="email1", channel_type=ChannelType.EMAIL)
        await ch.deliver(event, binding, ctx)
        assert len(provider.sent) == 1

    async def test_capabilities(self) -> None:
        ch = EmailChannel("email1")
        caps = ch.capabilities()
        assert ChannelMediaType.RICH in caps.media_types
        assert caps.supports_threading is True
