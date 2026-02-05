"""Tests for mock providers."""

from __future__ import annotations

from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, IdentificationStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.identity import Identity
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIContext, AIMessage
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.email.mock import MockEmailProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.providers.whatsapp.mock import MockWhatsAppProvider


def _make_event() -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="ch1", channel_type=ChannelType.SMS),
        content=TextContent(body="test"),
    )


class TestMockSMSProvider:
    async def test_send_records(self) -> None:
        provider = MockSMSProvider()
        result = await provider.send(_make_event(), to="+15551234567")
        assert result.success is True
        assert len(provider.sent) == 1
        assert provider.sent[0]["to"] == "+15551234567"


class TestMockEmailProvider:
    async def test_send_records(self) -> None:
        provider = MockEmailProvider()
        result = await provider.send(_make_event(), to="test@example.com", subject="Hello")
        assert result.success is True
        assert len(provider.sent) == 1
        assert provider.sent[0]["subject"] == "Hello"


class TestMockAIProvider:
    async def test_generate_returns_response(self) -> None:
        provider = MockAIProvider(responses=["Hi!", "Bye!"])
        ctx = AIContext(messages=[AIMessage(role="user", content="hello")])
        r1 = await provider.generate(ctx)
        assert r1.content == "Hi!"
        r2 = await provider.generate(ctx)
        assert r2.content == "Bye!"
        assert len(provider.calls) == 2

    async def test_round_robin(self) -> None:
        provider = MockAIProvider(responses=["A", "B"])
        ctx = AIContext(messages=[AIMessage(role="user", content="x")])
        results = [await provider.generate(ctx) for _ in range(4)]
        contents = [r.content for r in results]
        assert contents == ["A", "B", "A", "B"]


class TestMockWhatsAppProvider:
    async def test_send_records(self) -> None:
        provider = MockWhatsAppProvider()
        result = await provider.send(_make_event(), to="+15551234567")
        assert result.success is True
        assert len(provider.sent) == 1


class TestMockIdentityResolver:
    async def test_resolve_known(self) -> None:
        identity = Identity(id="user1", display_name="Alice")
        resolver = MockIdentityResolver(mapping={"sender1": identity})
        msg = InboundMessage(
            channel_id="ch1",
            sender_id="sender1",
            content=TextContent(body="hi"),
        )
        ctx = RoomContext(room=Room(id="r1"))
        result = await resolver.resolve(msg, ctx)
        assert result.status == IdentificationStatus.IDENTIFIED
        assert result.identity is not None
        assert result.identity.display_name == "Alice"

    async def test_resolve_unknown(self) -> None:
        resolver = MockIdentityResolver()
        msg = InboundMessage(
            channel_id="ch1",
            sender_id="unknown",
            content=TextContent(body="hi"),
        )
        ctx = RoomContext(room=Room(id="r1"))
        result = await resolver.resolve(msg, ctx)
        assert result.status == IdentificationStatus.UNKNOWN
