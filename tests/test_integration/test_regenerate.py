"""Integration: regenerate_response re-runs the agent on the last user message."""

from __future__ import annotations

import pytest

from roomkit.channels import SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import RoomClosedError, RoomNotFoundError
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    EventStatus,
    EventType,
    HookTrigger,
    RoomStatus,
)
from roomkit.models.event import TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult
from roomkit.providers.ai.mock import MockAIProvider


def _user_messages(events: list, transport_id: str) -> list:
    return [
        e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == transport_id
    ]


def _ai_messages(events: list, ai_id: str) -> list:
    return [e for e in events if e.type == EventType.MESSAGE and e.source.channel_id == ai_id]


async def _kit_with_turn(
    *, streaming: bool, turns: int = 1, responses: list[str] | None = None
) -> tuple[RoomKit, MockAIProvider]:
    """A room with an SMS transport, an AI channel and *turns* completed turns."""
    kit = RoomKit()
    ai_provider = MockAIProvider(
        responses=responses or ["First answer", "Second answer"], streaming=streaming
    )
    sms = SMSChannel("sms1")
    ai = AIChannel("ai1", provider=ai_provider)
    kit.register_channel(sms)
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    for i in range(turns):
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1",
                sender_id="user1",
                content=TextContent(body=f"What is the weather? ({i})"),
            )
        )
    return kit, ai_provider


class TestRegenerate:
    async def test_regenerate_adds_response_without_duplicating_user_message(self) -> None:
        kit, ai_provider = await _kit_with_turn(streaming=False)

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
        kit, ai_provider = await _kit_with_turn(streaming=False)

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
        kit, ai_provider = await _kit_with_turn(streaming=True)

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


class TestRegenerateStatusGuard:
    """RFC §5.1: a room whose status refuses new events refuses a regenerate
    before the agent runs — the pipeline's gate never saw a re-broadcast."""

    async def test_a_closed_room_is_refused_before_the_agent_runs(self) -> None:
        kit, ai_provider = await _kit_with_turn(streaming=False)
        refused: list[FrameworkEvent] = []

        @kit.on("room_refused_event")
        async def on_refused(fe: FrameworkEvent) -> None:
            refused.append(fe)

        await kit.close_room("r1")
        before = await kit.store.list_events("r1")
        trigger = _user_messages(before, "sms1")[-1]

        result = await kit.regenerate_response("r1")

        assert result is not None
        assert result.blocked is True
        assert result.reason == "room_closed"
        # The first turn's call only: no generation was paid for the refusal.
        assert len(ai_provider.calls) == 1
        # Nothing written, not even an audit record (§5.1).
        assert await kit.store.list_events("r1") == before
        assert len(refused) == 1
        assert refused[0].room_id == "r1"
        assert refused[0].event_id == trigger.id
        assert refused[0].data == {
            "status": str(RoomStatus.CLOSED),
            "operation": "regenerate",
            "event_type": str(EventType.MESSAGE),
        }

    async def test_an_archived_room_is_refused_too(self) -> None:
        kit, ai_provider = await _kit_with_turn(streaming=True)
        await kit.archive_room("r1")

        result = await kit.regenerate_response("r1")

        assert result is not None
        assert result.blocked is True
        assert result.reason == "room_closed"
        assert len(ai_provider.calls) == 1

    async def test_a_closed_room_with_nothing_to_regenerate_is_still_refused(self) -> None:
        """The status is the verdict, not the tail: a closed room says
        ``room_closed`` whether or not a trigger exists in it."""
        kit = RoomKit()
        kit.register_channel(SMSChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.close_room("r1")
        refused: list[FrameworkEvent] = []

        @kit.on("room_refused_event")
        async def on_refused(fe: FrameworkEvent) -> None:
            refused.append(fe)

        result = await kit.regenerate_response("r1")

        assert result is not None
        assert result.blocked is True
        assert result.reason == "room_closed"
        # No trigger to name: event_id and event_type are null, the contract
        # keeps its keys (RFC §8.2).
        assert len(refused) == 1
        assert refused[0].event_id is None
        assert refused[0].data == {
            "status": str(RoomStatus.CLOSED),
            "operation": "regenerate",
            "event_type": None,
        }


class TestRegenerateTriggerId:
    """``trigger_id`` makes the call a compare-and-regenerate: the host names
    the message it prepared for, and a selection that moved is refused."""

    async def test_a_message_landing_in_between_refuses_the_regenerate(self) -> None:
        kit, ai_provider = await _kit_with_turn(streaming=False)
        prepared = await kit.regenerate_target("r1")
        assert prepared is not None
        # Between the host's read and its regenerate a new message lands, and
        # the pipeline answers it. A regenerate would answer it a second time.
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="user1", content=TextContent(body="And on Saturdays?")
            )
        )
        calls_before = len(ai_provider.calls)
        before = await kit.store.list_events("r1")

        result = await kit.regenerate_response("r1", trigger_id=prepared.id)

        assert result is not None
        assert result.blocked is True
        assert result.reason == "trigger_moved"
        assert result.event is None
        # No generation paid, nothing written.
        assert len(ai_provider.calls) == calls_before
        assert await kit.store.list_events("r1") == before

    async def test_the_named_trigger_regenerates_as_usual(self) -> None:
        kit, ai_provider = await _kit_with_turn(streaming=False)
        prepared = await kit.regenerate_target("r1")
        assert prepared is not None

        result = await kit.regenerate_response("r1", trigger_id=prepared.id)

        assert result is not None and result.event is not None
        assert result.blocked is False
        assert result.event.id == prepared.id
        assert len(ai_provider.calls) == 2
        assert len(_ai_messages(await kit.store.list_events("r1"), "ai1")) == 2

    async def test_a_closed_room_is_refused_before_the_trigger_is_compared(self) -> None:
        kit, ai_provider = await _kit_with_turn(streaming=False)
        await kit.close_room("r1")

        result = await kit.regenerate_response("r1", trigger_id="not-the-trigger")

        assert result is not None
        assert result.blocked is True
        assert result.reason == "room_closed"
        assert len(ai_provider.calls) == 1

    async def test_nothing_left_to_replay_is_a_moved_trigger(self) -> None:
        """The host named a trigger and the selection is now empty (its source
        can no longer write): the compare fails the same way. The host asked
        about *that* message, and the answer is that it no longer is the one."""
        kit, ai_provider = await _kit_with_turn(streaming=False)
        prepared = await kit.regenerate_target("r1")
        assert prepared is not None
        await kit.set_access("r1", "sms1", Access.READ_ONLY)

        result = await kit.regenerate_response("r1", trigger_id=prepared.id)

        assert result is not None
        assert result.blocked is True
        assert result.reason == "trigger_moved"
        assert len(ai_provider.calls) == 1

    async def test_without_a_trigger_id_the_selection_is_whatever_it_is(self) -> None:
        kit, _ai_provider = await _kit_with_turn(streaming=False)
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="user1", content=TextContent(body="And on Saturdays?")
            )
        )

        result = await kit.regenerate_response("r1")

        assert result is not None and result.event is not None
        assert result.blocked is False
        assert result.event.content.body == "And on Saturdays?"


class TestRegenerateTarget:
    """``regenerate_target`` answers with the primitive's own selection."""

    async def test_target_is_the_event_regenerate_response_replays(self) -> None:
        # Thirty turns: sixty message events with the agent's answers
        # interleaved. Then lifecycle noise a host writes on the transport
        # after the trigger — system events, not messages — which the
        # conversation never holds and the selection must not pick.
        kit, _ai_provider = await _kit_with_turn(streaming=False, turns=30, responses=["ok"])
        for i in range(3):
            await kit.send_event(
                "r1", "sms1", TextContent(body=f"reconnected {i}"), event_type=EventType.SYSTEM
            )
        events = await kit.store.list_events("r1", limit=500)
        assert len(events) > 50
        last_user = _user_messages(events, "sms1")[-1]
        assert last_user.content.body == "What is the weather? (29)"
        assert max(e.index for e in events) > last_user.index

        target = await kit.regenerate_target("r1")

        assert target is not None
        assert target.id == last_user.id

        result = await kit.regenerate_response("r1")

        assert result is not None
        assert result.event is not None
        assert result.event.id == target.id

    async def test_target_is_none_when_the_source_binding_cannot_write(self) -> None:
        kit, _ai_provider = await _kit_with_turn(streaming=False)
        await kit.mute("r1", "sms1")

        assert await kit.regenerate_target("r1") is None
        assert await kit.regenerate_response("r1") is None

    async def test_target_is_none_when_the_source_is_read_only(self) -> None:
        kit, _ai_provider = await _kit_with_turn(streaming=False)
        await kit.set_access("r1", "sms1", Access.READ_ONLY)

        assert await kit.regenerate_target("r1") is None
        assert await kit.regenerate_response("r1") is None

    async def test_a_blocked_message_is_never_the_trigger(self) -> None:
        """A message a hook refused is stored BLOCKED and never broadcast
        (RFC §10.1 step 10): the room did not answer it, so a regenerate does
        not answer it either."""
        kit, ai_provider = await _kit_with_turn(streaming=False)

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def refuse_spam(event, context) -> HookResult:
            if "SPAM" in getattr(event.content, "body", ""):
                return HookResult.block("spam")
            return HookResult.allow()

        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="user1", content=TextContent(body="SPAM"))
        )
        events = await kit.store.list_events("r1")
        assert events[-1].status == EventStatus.BLOCKED
        accepted = _user_messages(events, "sms1")[0]

        target = await kit.regenerate_target("r1")
        result = await kit.regenerate_response("r1")

        assert target is not None and target.id == accepted.id
        assert result is not None and result.event is not None
        assert result.event.id == accepted.id
        # The regenerated turn answers the accepted message, not the refused one.
        assert ai_provider.calls[-1].messages[-1].content == accepted.content.body

    async def test_a_paused_room_still_has_a_target(self) -> None:
        """PAUSED accepts events (RFC §5.1): the guard stops at CLOSED and ARCHIVED."""
        kit, _ai_provider = await _kit_with_turn(streaming=False)
        room = await kit.store.get_room("r1")
        assert room is not None
        await kit.store.update_room(room.model_copy(update={"status": RoomStatus.PAUSED}))

        assert await kit.regenerate_target("r1") is not None

    async def test_target_is_none_without_a_transport_message(self) -> None:
        kit = RoomKit()
        kit.register_channel(SMSChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        assert await kit.regenerate_target("r1") is None

    async def test_target_raises_on_a_room_that_refuses_writes(self) -> None:
        """An accessor returning an event has no way to hand back a refusal:
        it raises, on ``send_event``'s reasoning, so a host learns the room
        is closed before it acts on a trigger."""
        kit, _ai_provider = await _kit_with_turn(streaming=False)
        await kit.close_room("r1")

        with pytest.raises(RoomClosedError):
            await kit.regenerate_target("r1")

    async def test_target_raises_on_an_unknown_room(self) -> None:
        kit = RoomKit()

        with pytest.raises(RoomNotFoundError):
            await kit.regenerate_target("nope")
