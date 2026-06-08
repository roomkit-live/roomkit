"""Integration: regenerate_response re-runs the agent on the last user message."""

from __future__ import annotations

from roomkit.channels import SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import TextContent
from roomkit.providers.ai.mock import MockAIProvider


def _user_messages(events: list, transport_id: str) -> list:
    return [
        e
        for e in events
        if e.type == EventType.MESSAGE and e.source.channel_id == transport_id
    ]


def _ai_messages(events: list, ai_id: str) -> list:
    return [
        e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == ai_id
    ]


class TestRegenerate:
    async def _kit_with_turn(self, *, streaming: bool) -> tuple[RoomKit, MockAIProvider]:
        kit = RoomKit()
        ai_provider = MockAIProvider(
            responses=["First answer", "Second answer"], streaming=streaming
        )
        sms = SMSChannel("sms1")
        ai = AIChannel("ai1", provider=ai_provider)
        kit.register_channel(sms)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1",
                sender_id="user1",
                content=TextContent(body="What is the weather?"),
            )
        )
        return kit, ai_provider

    async def test_regenerate_adds_response_without_duplicating_user_message(self) -> None:
        kit, ai_provider = await self._kit_with_turn(streaming=False)

        before = await kit.store.list_events("r1")
        users_before = _user_messages(before, "sms1")
        assert len(users_before) == 1
        assert len(ai_provider.calls) == 1

        result = await kit.regenerate_response("r1")
        assert result is not None

        after = await kit.store.list_events("r1")
        # The user's message is untouched — same id, no duplicate.
        users_after = _user_messages(after, "sms1")
        assert len(users_after) == 1
        assert users_after[0].id == users_before[0].id
        # A fresh AI response was generated and stored.
        assert len(ai_provider.calls) == 2
        assert len(_ai_messages(after, "ai1")) == 2

    async def test_regenerate_context_includes_user_message(self) -> None:
        kit, ai_provider = await self._kit_with_turn(streaming=False)

        await kit.regenerate_response("r1")

        # The regenerated call must see the user's last message as the final
        # user turn (retrieve excludes the trigger by id, _build_context re-adds it).
        second_call = ai_provider.calls[1]
        user_turns = [m for m in second_call.messages if m.role == "user"]
        assert any("weather" in str(m.content) for m in user_turns)
        # Exactly one copy of the user's message — not double-counted.
        weather_turns = [m for m in user_turns if "weather" in str(m.content)]
        assert len(weather_turns) == 1

    async def test_regenerate_streaming_provider(self) -> None:
        kit, ai_provider = await self._kit_with_turn(streaming=True)

        assert len(ai_provider.calls) == 1
        result = await kit.regenerate_response("r1")
        assert result is not None

        after = await kit.store.list_events("r1")
        assert len(_user_messages(after, "sms1")) == 1
        assert len(ai_provider.calls) == 2
        assert len(_ai_messages(after, "ai1")) == 2

    async def test_regenerate_with_no_prior_response(self) -> None:
        """Case where the first turn produced no answer (error / server cut)."""
        kit = RoomKit()
        ai_provider = MockAIProvider(responses=["Recovered answer"])
        sms = SMSChannel("sms1")
        kit.register_channel(sms)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        # User message lands with no intelligence channel attached → no response.
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1",
                sender_id="user1",
                content=TextContent(body="Are you there?"),
            )
        )
        assert len(ai_provider.calls) == 0

        # Agent comes online; regenerate produces the missing answer.
        ai = AIChannel("ai1", provider=ai_provider)
        kit.register_channel(ai)
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        result = await kit.regenerate_response("r1")
        assert result is not None
        assert len(ai_provider.calls) == 1

        after = await kit.store.list_events("r1")
        assert len(_user_messages(after, "sms1")) == 1
        assert len(_ai_messages(after, "ai1")) == 1

    async def test_regenerate_empty_room_returns_none(self) -> None:
        kit = RoomKit()
        sms = SMSChannel("sms1")
        kit.register_channel(sms)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        assert await kit.regenerate_response("r1") is None
