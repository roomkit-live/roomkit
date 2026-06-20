"""InboundMessage.visibility — deliver to transports without waking the agent.

A proactive notification posted into a room (e.g. a Telegram HITL prompt) must
reach the external transport so the human sees it, but must NOT trigger the
room's intelligence channel into replying. ``visibility="transport"`` on the
inbound expresses exactly that.
"""

from __future__ import annotations

from types import SimpleNamespace

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    Visibility,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.core.visibility import visibility_allows
from roomkit.providers.ai.mock import MockAIProvider


def _binding(category: ChannelCategory, channel_id: str) -> SimpleNamespace:
    """Minimal stand-in exposing the two attributes ``visibility_allows`` reads."""
    return SimpleNamespace(category=category, channel_id=channel_id)


class TestVisibilityAllows:
    """Scope-keyword resolution shared by the broadcast router and streaming."""

    _transport = _binding(ChannelCategory.TRANSPORT, "sms")
    _intelligence = _binding(ChannelCategory.INTELLIGENCE, "ai")

    def test_all_reaches_every_binding(self) -> None:
        assert visibility_allows(Visibility.ALL, self._transport)
        assert visibility_allows(Visibility.ALL, self._intelligence)

    def test_none_reaches_nothing(self) -> None:
        assert not visibility_allows(Visibility.NONE, self._transport)
        assert not visibility_allows(Visibility.NONE, self._intelligence)

    def test_internal_reaches_nothing(self) -> None:
        # Internal-scoped events reach no channel — they live only in stored
        # room history (delegation, system, handoff events).
        assert not visibility_allows(Visibility.INTERNAL, self._transport)
        assert not visibility_allows(Visibility.INTERNAL, self._intelligence)

    def test_transport_and_intelligence_scopes(self) -> None:
        assert visibility_allows(Visibility.TRANSPORT, self._transport)
        assert not visibility_allows(Visibility.TRANSPORT, self._intelligence)
        assert visibility_allows(Visibility.INTELLIGENCE, self._intelligence)
        assert not visibility_allows(Visibility.INTELLIGENCE, self._transport)

    def test_channel_id_specs(self) -> None:
        assert visibility_allows("sms", self._transport)
        assert not visibility_allows("other", self._transport)
        assert visibility_allows("sms,ai", self._transport)
        assert visibility_allows("sms, ai", self._intelligence)
        assert not visibility_allows("x,y", self._transport)


async def _kit() -> tuple[RoomKit, MockAIProvider]:
    kit = RoomKit()
    provider = MockAIProvider(responses=["Hello!"])
    kit.register_channel(WebSocketChannel("ws-user"))
    kit.register_channel(AIChannel("ai-bot", provider=provider))
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "ws-user")
    await kit.attach_channel("room-1", "ai-bot", category=ChannelCategory.INTELLIGENCE)
    return kit, provider


class TestInboundVisibility:
    async def test_default_visibility_triggers_the_agent(self) -> None:
        kit, provider = await _kit()
        await kit.process_inbound(
            InboundMessage(channel_id="ws-user", sender_id="u1", content=TextContent(body="Hi")),
            room_id="room-1",
        )
        assert len(provider.calls) == 1  # the agent generated a reply

    async def test_transport_visibility_does_not_trigger_the_agent(self) -> None:
        kit, provider = await _kit()
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="u1",
                content=TextContent(body="Notification"),
                visibility="transport",
            ),
            room_id="room-1",
        )
        assert len(provider.calls) == 0  # intelligence channel was skipped
        # The message is still stored (the transport/human sees it).
        events = await kit.store.list_events("room-1")
        assert any(
            isinstance(e.content, TextContent) and e.content.body == "Notification" for e in events
        )

    async def test_internal_visibility_reaches_no_channel_but_is_stored(self) -> None:
        kit, provider = await _kit()
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="u1",
                content=TextContent(body="Internal"),
                visibility=Visibility.INTERNAL,
            ),
            room_id="room-1",
        )
        assert len(provider.calls) == 0  # not delivered to the intelligence channel
        events = await kit.store.list_events("room-1")
        assert any(
            isinstance(e.content, TextContent) and e.content.body == "Internal" for e in events
        )
