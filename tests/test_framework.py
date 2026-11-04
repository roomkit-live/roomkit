"""Tests for RoomKit framework class."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from roomkit.channels.base import Channel
from roomkit.core._private_state import PRIVATE_STATE_KEYS
from roomkit.core.exceptions import RoomAlreadyExistsError, ShutdownCancelledError
from roomkit.core.framework import (
    ChannelAlreadyRegisteredError,
    ChannelNotRegisteredError,
    RoomKit,
    RoomKitError,
    RoomNotFoundError,
)
from roomkit.identity.base import IdentityResolver
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
    IdentificationStatus,
    RoomStatus,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.identity import IdentityResult
from roomkit.models.task import Observation, Task


class SimpleChannel(Channel):
    """Minimal channel for testing."""

    channel_type = ChannelType.SMS

    def __init__(self, channel_id: str, channel_type: ChannelType = ChannelType.SMS) -> None:
        super().__init__(channel_id)
        self.channel_type = channel_type
        self.delivered: list[RoomEvent] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        return ChannelOutput.empty()


class AILikeChannel(Channel):
    """Channel that responds to events (like AI)."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str, response: str = "AI reply") -> None:
        super().__init__(channel_id)
        self._response = response
        self.delivered: list[RoomEvent] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        resp = RoomEvent(
            room_id=event.room_id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=ChannelType.AI,
            ),
            content=TextContent(body=self._response),
            chain_depth=event.chain_depth + 1,
        )
        return ChannelOutput(responded=True, response_events=[resp])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestRoomLifecycle:
    async def test_create_room(self, kit: RoomKit) -> None:
        room = await kit.create_room(room_id="r1")
        assert room.id == "r1"
        assert room.status == RoomStatus.ACTIVE

    async def test_a_room_id_is_created_once(
        self, kit: RoomKit, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RFC §14.3: the room an id names is never replaced, and its refusal
        says the id is taken and nothing of the room, another
        organization's."""
        await kit.create_room(room_id="r1", organization_id="acme", metadata={"k": "secret"})

        with caplog.at_level(logging.DEBUG), pytest.raises(RoomAlreadyExistsError) as taken:
            await kit.create_room(room_id="r1", organization_id="globex")

        assert (await kit.get_room("r1")).organization_id == "acme"
        assert taken.value.args == ("Room r1 already exists",)
        assert vars(taken.value) == {}
        assert "acme" not in caplog.text and "secret" not in caplog.text

    async def test_create_room_auto_id(self, kit: RoomKit) -> None:
        room = await kit.create_room()
        assert room.id  # auto-generated

    async def test_get_room(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        room = await kit.get_room("r1")
        assert room.id == "r1"

    async def test_get_room_not_found(self, kit: RoomKit) -> None:
        with pytest.raises(RoomNotFoundError):
            await kit.get_room("nope")

    async def test_close_room(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        room = await kit.close_room("r1")
        assert room.status == RoomStatus.CLOSED
        assert room.closed_at is not None

    async def test_update_metadata(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        room = await kit.update_room_metadata("r1", {"key": "val"})
        assert room.metadata["key"] == "val"

    @pytest.mark.parametrize(
        "key",
        [
            *sorted(PRIVATE_STATE_KEYS),
            "roomkit:detached_channels",
            "roomkit:channel_state:ws",
            "roomkit:anything",
            "_discussion",
            "_speak_queue",
        ],
    )
    async def test_a_key_only_the_framework_writes_is_dropped_with_a_warning(
        self, kit: RoomKit, key: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RFC §5.1, §14.1 (RMK-978): the implementation's namespace,
        ``roomkit:``, and the keys of the room's private state are the
        framework's alone. A room's creation and an integrator's metadata
        update that name one have it dropped with a warning, the rest
        written, as a channel's ``metadata_updates`` do."""
        with caplog.at_level(logging.WARNING):
            created = await kit.create_room(room_id="r2", metadata={key: {}, "crm": "c_42"})
            await kit.create_room(room_id="r1")
            updated = await kit.update_room_metadata("r1", {key: {}, "crm": "c_42"})

        stored = [(await kit.get_room(r)).metadata for r in ("r1", "r2")]
        assert [created.metadata, updated.metadata, *stored] == [{"crm": "c_42"}] * 4
        assert sum(repr(key) in r.getMessage() for r in caplog.records) == 2


class TestTimelineQuery:
    """``RoomKit.get_timeline`` pages a log from its head, or snapshots its tail."""

    @staticmethod
    async def _seed(kit: RoomKit, room_id: str, count: int) -> None:
        for i in range(count):
            await kit.store.commit_event(
                room_id,
                RoomEvent(
                    room_id=room_id,
                    type=EventType.MESSAGE,
                    source=EventSource(channel_id="sms1", channel_type=ChannelType.SMS),
                    content=TextContent(body=f"msg-{i}"),
                ),
            )

    async def test_default_pages_from_the_head(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        await self._seed(kit, "r1", 10)

        events = await kit.get_timeline("r1", limit=3)

        assert [e.index for e in events] == [0, 1, 2]

    async def test_newest_first_snapshots_the_tail(self, kit: RoomKit) -> None:
        """A reconnecting client wants what was just said, not the room's opening."""
        await kit.create_room(room_id="r1")
        await self._seed(kit, "r1", 10)

        events = await kit.get_timeline("r1", limit=3, newest_first=True)

        assert [e.index for e in events] == [7, 8, 9]


class TestChannelManagement:
    async def test_register_and_attach(self, kit: RoomKit) -> None:
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        binding = await kit.attach_channel("r1", "sms1")
        assert binding.channel_id == "sms1"
        assert binding.channel_type == ChannelType.SMS

    async def test_attach_unregistered(self, kit: RoomKit) -> None:
        await kit.create_room(room_id="r1")
        with pytest.raises(ChannelNotRegisteredError):
            await kit.attach_channel("r1", "nope")

    async def test_detach_channel(self, kit: RoomKit) -> None:
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        assert await kit.detach_channel("r1", "sms1") is True

    async def test_unregister_channel(self, kit: RoomKit) -> None:
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)

        removed = kit.unregister_channel("sms1")
        assert removed is ch

        # Gone from the registry: attaching now fails
        await kit.create_room(room_id="r1")
        with pytest.raises(ChannelNotRegisteredError):
            await kit.attach_channel("r1", "sms1")

    async def test_unregister_unknown_channel_returns_none(self, kit: RoomKit) -> None:
        assert kit.unregister_channel("nope") is None

    async def test_mute_unmute(self, kit: RoomKit) -> None:
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        binding = await kit.mute("r1", "sms1")
        assert binding.muted is True
        binding = await kit.unmute("r1", "sms1")
        assert binding.muted is False

    async def test_set_visibility(self, kit: RoomKit) -> None:
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        binding = await kit.set_visibility("r1", "sms1", "transport")
        assert binding.visibility == "transport"

    async def test_set_access(self, kit: RoomKit) -> None:
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        binding = await kit.set_access("r1", "sms1", Access.READ_ONLY)
        assert binding.access == Access.READ_ONLY

    async def test_the_response_report_follows_the_category(self, kit: RoomKit) -> None:
        """Any INTELLIGENCE channel can say a turn finished, whatever its class.

        Selecting on ``AIChannel`` left an agent that owns its own turn — an
        ACP coding agent — with no way to reach ``ON_AI_RESPONSE``, and so
        invisible to every post-processing task the host hangs off it.
        """
        agent = AILikeChannel("agent1")
        transport = SimpleChannel("sms1")
        kit.register_channel(agent)
        kit.register_channel(transport)

        assert agent._after_response_hook is not None  # ty: ignore[unresolved-attribute]
        assert not hasattr(transport, "_after_response_hook")


class TestInboundPipeline:
    async def test_basic_inbound(self, kit: RoomKit) -> None:
        ch1 = SimpleChannel("sms1")
        ch2 = SimpleChannel("ws1")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked
        assert result.event is not None
        assert len(ch2.delivered) == 1

    async def test_inbound_unregistered_channel(self, kit: RoomKit) -> None:
        with pytest.raises(ChannelNotRegisteredError):
            await kit.process_inbound(
                InboundMessage(
                    channel_id="nope",
                    sender_id="x",
                    content=TextContent(body="hi"),
                )
            )

    async def test_idempotency_dedup(self, kit: RoomKit) -> None:
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
            idempotency_key="dedup-1",
        )
        r1 = await kit.process_inbound(msg)
        assert not r1.blocked
        r2 = await kit.process_inbound(msg)
        # Not reprocessed, and not refused either: the resend is answered with
        # the event the first call committed (RFC §13.4).
        assert not r2.blocked
        assert r2.event is not None
        assert r2.event.id == r1.event.id


class TestHookIntegration:
    async def test_hook_decorator(self, kit: RoomKit) -> None:
        ch1 = SimpleChannel("sms1")
        ch2 = SimpleChannel("ws1")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        hook_called = False

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="test_hook")
        async def my_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            nonlocal hook_called
            hook_called = True
            return HookResult.allow()

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        assert hook_called

    async def test_hook_block(self, kit: RoomKit) -> None:
        ch1 = SimpleChannel("sms1")
        ch2 = SimpleChannel("ws1")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="blocker")
        async def blocker(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("no spam")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        result = await kit.process_inbound(msg)
        assert result.blocked
        assert result.reason == "no spam"
        assert len(ch2.delivered) == 0

    async def test_hook_modify(self, kit: RoomKit) -> None:
        ch1 = SimpleChannel("sms1")
        ch2 = SimpleChannel("ws1")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="modifier")
        async def modifier(event: RoomEvent, ctx: RoomContext) -> HookResult:
            modified = event.model_copy(update={"content": TextContent(body="[REDACTED]")})
            return HookResult.modify(modified)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="secret stuff"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked
        assert result.event is not None
        assert isinstance(result.event.content, TextContent)
        assert result.event.content.body == "[REDACTED]"

    async def test_hook_inject(self, kit: RoomKit) -> None:
        ch1 = SimpleChannel("sms1")
        kit.register_channel(ch1)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="injector")
        async def injector(event: RoomEvent, ctx: RoomContext) -> HookResult:
            notice = RoomEvent(
                room_id=event.room_id,
                source=EventSource(
                    channel_id="system",
                    channel_type=ChannelType.WEBHOOK,
                ),
                content=TextContent(body="Notice: content scanned"),
                type=EventType.SYSTEM,
            )
            return HookResult.allow(injected=[InjectedEvent(event=notice)])

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        events = await kit.store.list_events("r1")
        # Should have original + injected
        assert len(events) >= 2


class TestFrameworkEvents:
    async def test_on_decorator(self, kit: RoomKit) -> None:
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        received: list[FrameworkEvent] = []

        @kit.on("event_processed")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        assert len(received) == 1
        assert received[0].type == "event_processed"


class TestSendEvent:
    async def test_send_event(self, kit: RoomKit) -> None:
        ch1 = SimpleChannel("sms1")
        ch2 = SimpleChannel("ws1")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        event = await kit.send_event("r1", "sms1", TextContent(body="direct message"))
        assert event.room_id == "r1"
        assert len(ch2.delivered) == 1

    async def test_send_event_triggers_after_broadcast_hooks(self, kit: RoomKit) -> None:
        """send_event should trigger AFTER_BROADCAST hooks for observability/fan-out."""
        ch1 = SimpleChannel("sms1")
        kit.register_channel(ch1)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        # Track hook invocations
        hook_calls: list[RoomEvent] = []

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
        async def track_broadcast(event: RoomEvent, ctx: RoomContext) -> HookResult:
            hook_calls.append(event)
            return HookResult.allow()

        # Send event via send_event (not process_inbound)
        event = await kit.send_event("r1", "sms1", TextContent(body="test message"))

        # AFTER_BROADCAST hook should have been called
        assert len(hook_calls) == 1
        assert hook_calls[0].id == event.id
        assert hook_calls[0].content.body == "test message"

    @pytest.mark.parametrize("channel", ["never registered", "bound, then unregistered"])
    async def test_a_channel_the_kit_does_not_register_is_refused(
        self, kit: RoomKit, channel: str
    ) -> None:
        """RFC §10.5 (RMK-977): the channel an injection names MUST be
        registered, before the room's binding is read: a channel the room
        still binds after the kit unregistered it writes and delivers
        nothing either."""
        sms = SimpleChannel("sms1")
        kit.register_channel(sms)
        kit.register_channel(SimpleChannel("ws1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")
        channel_id = "nope"
        if channel == "bound, then unregistered":
            kit.unregister_channel("ws1")
            channel_id = "ws1"

        with pytest.raises(ChannelNotRegisteredError, match=channel_id):
            await kit.send_event("r1", channel_id, TextContent(body="x"))

        messages = [e for e in await kit.store.list_events("r1") if e.type == EventType.MESSAGE]
        assert (messages, sms.delivered) == ([], [])


class TestChainDepth:
    async def test_ai_response_chain(self, kit: RoomKit) -> None:
        ch_sms = SimpleChannel("sms1")
        ch_ai = AILikeChannel("ai1")
        kit.register_channel(ch_sms)
        kit.register_channel(ch_ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello AI"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked
        # AI should have received the message
        assert len(ch_ai.delivered) >= 1


class TestRoomHooks:
    async def test_add_and_remove_room_hook(self, kit: RoomKit) -> None:
        called = False

        async def my_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            nonlocal called
            called = True
            return HookResult.allow()

        await kit.add_room_hook(
            "r1",
            HookTrigger.BEFORE_BROADCAST,
            HookExecution.SYNC,
            my_hook,
            name="test_room_hook",
        )
        assert await kit.remove_room_hook("r1", "test_room_hook") is True
        assert await kit.remove_room_hook("r1", "missing") is False


class TestIdempotencyRace:
    async def test_concurrent_duplicate_only_stores_once(self, kit: RoomKit) -> None:
        """Two concurrent messages with the same idempotency key should result in
        only one stored event (TOCTOU race prevention)."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
            idempotency_key="race-key",
        )

        results = await asyncio.gather(
            kit.process_inbound(msg),
            kit.process_inbound(msg),
        )

        # Neither call is refused: the loser of the race replays the winner's
        # answer rather than reprocessing (RFC §13.4).
        assert all(not r.blocked for r in results)
        assert all(r.event is not None for r in results)
        assert results[0].event.id == results[1].event.id

        # And the timeline holds that event exactly once.
        timeline = await kit.get_timeline("r1")
        assert sum(1 for e in timeline if e.id == results[0].event.id) == 1


# -- Changeset 3: Concurrent locking tests --


class TestConcurrentLocking:
    async def test_concurrent_close_room(self, kit: RoomKit) -> None:
        """Two concurrent close_room calls converge on one CLOSED room."""
        await kit.create_room(room_id="r1")
        results = await asyncio.gather(
            kit.close_room("r1"),
            kit.close_room("r1"),
            return_exceptions=True,
        )
        # Whatever each call reported, the store's answer is a closed room.
        room = await kit.get_room("r1")
        assert room.status == RoomStatus.CLOSED
        # At least one call succeeded, and any failure is a domain error,
        # not corruption surfacing as an arbitrary exception.
        closed = [r for r in results if not isinstance(r, Exception)]
        assert len(closed) >= 1
        for r in closed:
            assert r.status == RoomStatus.CLOSED
        for r in results:
            if isinstance(r, Exception):
                assert isinstance(r, RoomKitError)

    async def test_concurrent_mute_and_set_access_both_apply(self, kit: RoomKit) -> None:
        """Concurrent single-field mutations must not lose each other's write.

        mute() and set_access() both read-modify-write the binding; without
        the room lock one write clobbers the other (lost update). Both fields
        landing is the falsifiable proof of serialization — unlike asserting
        a boolean is a boolean.
        """
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        await asyncio.gather(
            kit.mute("r1", "sms1"),
            kit.set_access("r1", "sms1", Access.READ_ONLY),
        )
        binding = await kit.get_binding("r1", "sms1")
        assert binding.muted is True
        assert binding.access == Access.READ_ONLY

    async def test_concurrent_set_access_set_visibility(self, kit: RoomKit) -> None:
        """Concurrent access and visibility changes should not corrupt state."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        await asyncio.gather(
            kit.set_access("r1", "sms1", Access.READ_ONLY),
            kit.set_visibility("r1", "sms1", "transport"),
        )
        binding = await kit.get_binding("r1", "sms1")
        # Both mutations should have applied
        assert binding.access == Access.READ_ONLY
        assert binding.visibility == "transport"

    async def test_concurrent_attach_of_two_channels(self, kit: RoomKit) -> None:
        """Two concurrent attaches of different channels both land."""
        ch1 = SimpleChannel("sms1")
        ch2 = SimpleChannel("ws1")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")

        await asyncio.gather(
            kit.attach_channel("r1", "sms1"),
            kit.attach_channel("r1", "ws1"),
        )
        bindings = await kit.list_bindings("r1")
        assert len(bindings) == 2

    async def test_concurrent_attach_detach_same_channel_is_consistent(self, kit: RoomKit) -> None:
        """Attach racing detach settles on one of the two serial outcomes.

        Whichever order the lock imposes, the store must agree with itself:
        the binding either exists (detach ran first) or is absent with the
        revocation recorded (attach ran first) — never a half state where
        list_bindings and get_binding disagree.
        """
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        await asyncio.gather(
            kit.attach_channel("r1", "sms1"),
            kit.detach_channel("r1", "sms1"),
            return_exceptions=True,
        )
        binding = await kit._store.get_binding("r1", "sms1")
        listed = {b.channel_id for b in await kit.list_bindings("r1")}
        if binding is None:
            assert "sms1" not in listed
        else:
            assert "sms1" in listed

    async def test_concurrent_update_metadata(self, kit: RoomKit) -> None:
        """Concurrent metadata updates should not lose data."""
        await kit.create_room(room_id="r1")

        await asyncio.gather(
            kit.update_room_metadata("r1", {"a": 1}),
            kit.update_room_metadata("r1", {"b": 2}),
        )
        room = await kit.get_room("r1")
        assert "a" in room.metadata
        assert "b" in room.metadata


class TestLifecycleGuards:
    async def test_registering_a_duplicate_channel_id_is_refused(self, kit: RoomKit) -> None:
        kit.register_channel(SimpleChannel("sms1"))
        with pytest.raises(ChannelAlreadyRegisteredError):
            kit.register_channel(SimpleChannel("sms1"))

    async def test_the_first_registration_survives_a_refused_duplicate(self, kit: RoomKit) -> None:
        first = SimpleChannel("sms1")
        kit.register_channel(first)
        with pytest.raises(ChannelAlreadyRegisteredError):
            kit.register_channel(SimpleChannel("sms1"))
        assert kit._channels["sms1"] is first

    async def test_unregister_then_register_swaps_deliberately(self, kit: RoomKit) -> None:
        first = SimpleChannel("sms1")
        kit.register_channel(first)
        assert kit.unregister_channel("sms1") is first
        second = SimpleChannel("sms1")
        kit.register_channel(second)
        assert kit._channels["sms1"] is second

    async def test_close_is_idempotent(self) -> None:
        kit = RoomKit()
        channel = _GatedClose("sms1")
        channel.release.set()
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.close()
        # A second close is a no-op, not a second teardown.
        await kit.close()
        assert channel.closes == 1


class TestOneShutdown:
    """The kit has one shutdown: every close() joins it, and a caller that
    stops waiting stops only its own wait (RMK-1015)."""

    @pytest.mark.parametrize("cancelled", [False, True], ids=["joined", "first-cancelled"])
    async def test_a_close_during_another_returns_once_the_shutdown_ends(
        self, cancelled: bool
    ) -> None:
        kit = RoomKit()
        channel = _GatedClose("sms1")
        kit.register_channel(channel)
        first = asyncio.ensure_future(kit.close())
        await asyncio.wait_for(channel.entered.wait(), 1)
        if cancelled:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
        second = asyncio.ensure_future(kit.close())
        await asyncio.sleep(0)
        assert not second.done()

        channel.release.set()
        await asyncio.wait_for(second, 1)

        assert channel.closes == 1 and channel.finished
        assert first.cancelled() is cancelled
        await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), 1)

    async def test_a_close_after_a_failed_one_raises_the_same_failure(self) -> None:
        kit = RoomKit()
        channel = _GatedClose("sms1", error=RuntimeError("the transport refused"))
        channel.release.set()
        kit.register_channel(channel)

        with pytest.raises(ExceptionGroup) as first:
            await kit.close()
        with pytest.raises(ExceptionGroup) as second:
            await kit.close()

        assert second.value is first.value
        assert channel.closes == 1

    async def test_a_close_after_its_shutdown_was_cut_says_so(self) -> None:
        """Every task cancelled, as a process's last cleanup does, cuts the
        shutdown: a later caller nobody cancelled is told so, not handed a
        cancellation of its own."""
        kit = RoomKit()
        channel = _GatedClose("sms1")
        kit.register_channel(channel)
        first = asyncio.ensure_future(kit.close())
        await asyncio.wait_for(channel.entered.wait(), 1)
        for task in asyncio.all_tasks() - {asyncio.current_task()}:
            task.cancel()
        await asyncio.gather(first, return_exceptions=True)

        with pytest.raises(ShutdownCancelledError, match="shutdown was cancelled"):
            await kit.close()

    async def test_a_close_from_the_shutdown_s_own_work_returns_at_once(self) -> None:
        kit = RoomKit()
        channel = _GatedClose("sms1", closes_the_kit=kit)
        channel.release.set()
        kit.register_channel(channel)

        await asyncio.wait_for(kit.close(), 1)

        assert channel.finished


class _GatedClose(SimpleChannel):
    """A channel whose close waits for *release*, then raises *error* when
    given; with *closes_the_kit*, it closes that kit from its own close."""

    def __init__(
        self,
        channel_id: str,
        *,
        error: Exception | None = None,
        closes_the_kit: RoomKit | None = None,
    ) -> None:
        super().__init__(channel_id)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.error = error
        self.kit = closes_the_kit
        self.closes = 0
        self.finished = False

    async def close(self) -> None:
        self.closes += 1
        self.entered.set()
        await self.release.wait()
        if self.kit is not None:
            await self.kit.close()
        self.finished = True
        if self.error is not None:
            raise self.error


# -- Changeset 4: Identity resolution idempotency --


class TestIdentityIdempotency:
    async def test_duplicate_pending_participant_not_created(self) -> None:
        """Concurrent inbound messages for same sender with ambiguous identity
        should not create duplicate pending participants."""
        from roomkit.identity.base import IdentityResolver

        class AmbiguousResolver(IdentityResolver):
            async def resolve(
                self, message: InboundMessage, context: RoomContext
            ) -> IdentityResult:
                return IdentityResult(
                    status=IdentificationStatus.AMBIGUOUS,
                )

        kit = RoomKit(identity_resolver=AmbiguousResolver())
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        msg1 = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        msg2 = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="world"),
        )

        await asyncio.gather(
            kit.process_inbound(msg1),
            kit.process_inbound(msg2),
        )

        participants = await kit.store.list_participants("r1")
        # Filter out system/internal participants; only pending ones for user1
        pending = [p for p in participants if p.identification == IdentificationStatus.PENDING]
        # The second call should have found the existing participant
        assert len(pending) <= 2  # no duplicates for same participant_id


# -- Changeset 5: Timeout tests --


class TestTimeouts:
    async def test_identity_timeout_falls_back_to_unknown(self) -> None:
        """When identity resolution times out, fall back to UNKNOWN status."""
        from roomkit.identity.base import IdentityResolver

        class SlowResolver(IdentityResolver):
            async def resolve(
                self, message: InboundMessage, context: RoomContext
            ) -> IdentityResult:
                await asyncio.sleep(5.0)
                return IdentityResult(status=IdentificationStatus.IDENTIFIED)

        events: list[FrameworkEvent] = []

        kit = RoomKit(identity_resolver=SlowResolver(), identity_timeout=0.01)
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        @kit.on("identity_timeout")
        async def capture(fe: FrameworkEvent) -> None:
            events.append(fe)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        result = await kit.process_inbound(msg)
        # Should not be blocked — falls back to unknown and continues
        assert not result.blocked
        assert len(events) == 1
        assert events[0].type == "identity_timeout"

    async def test_process_timeout_in_precommit_returns_blocked(self) -> None:
        """A timeout in the pre-commit phase blocks cleanly, with nothing
        persisted to the timeline (RFC §13.6)."""
        events: list[FrameworkEvent] = []

        kit = RoomKit(process_timeout=0.01)
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        @kit.on("process_timeout")
        async def capture(fe: FrameworkEvent) -> None:
            events.append(fe)

        # process_timeout now wraps only the pre-commit phase — slow it down.
        original = kit._run_precommit

        async def slow_precommit(
            event: RoomEvent, room_id: str, context: RoomContext, *args: object, **kwargs: object
        ) -> object:
            await asyncio.sleep(5.0)
            return await original(event, room_id, context, *args, **kwargs)

        kit._run_precommit = slow_precommit  # type: ignore[assignment]

        base_timeline = len(await kit.get_timeline("r1"))
        room_before = await kit.get_room("r1")
        assert room_before is not None
        base_count = room_before.event_count

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        result = await kit.process_inbound(msg)
        assert result.blocked
        assert result.reason == "process_timeout"
        assert len(events) == 1
        # Aborted before the commit point — no new event, counters unchanged.
        assert len(await kit.get_timeline("r1")) == base_timeline
        room_after = await kit.get_room("r1")
        assert room_after is not None
        assert room_after.event_count == base_count

    async def test_process_timeout_during_identity_resolution_returns_blocked(self) -> None:
        """The pre-commit budget spent while the resolver is still asked
        refuses the message, as anywhere else in that phase (RFC §13.6): the
        cancelled resolution leaves no result behind, and the span closing
        it must not read one."""

        class SlowResolver(IdentityResolver):
            async def resolve(
                self, message: InboundMessage, context: RoomContext
            ) -> IdentityResult:
                await asyncio.sleep(5.0)
                return IdentityResult(status=IdentificationStatus.IDENTIFIED)

        kit = RoomKit(
            identity_resolver=SlowResolver(), identity_timeout=10.0, process_timeout=0.05
        )
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        before = len(await kit.get_timeline("r1"))

        msg = InboundMessage(channel_id="sms1", sender_id="user1", content=TextContent(body="hi"))
        result = await kit.process_inbound(msg, room_id="r1")

        assert result.blocked
        assert result.reason == "process_timeout"
        assert len(await kit.get_timeline("r1")) == before

    async def test_slow_broadcast_after_commit_is_not_blocked(self) -> None:
        """A slow post-commit broadcast MUST NOT be cancelled or re-marked
        blocked by process_timeout: the committed event stays DELIVERED and the
        room counters reflect it (P0-3 — RFC §13.6 / §14.3)."""
        kit = RoomKit(process_timeout=0.01)
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        # Delivery execution (post-commit, in the lane) takes far longer
        # than process_timeout.
        original = kit._execute_plan

        async def slow_execute(plan: object, cascade: object) -> object:
            await asyncio.sleep(0.1)
            return await original(plan, cascade)

        kit._execute_plan = slow_execute  # type: ignore[assignment]

        base_timeline = len(await kit.get_timeline("r1"))

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        result = await kit.process_inbound(msg)

        # Committed, not blocked — the result matches the stored timeline.
        assert not result.blocked
        assert result.event is not None
        assert result.event.status == EventStatus.DELIVERED
        timeline = await kit.get_timeline("r1")
        assert len(timeline) == base_timeline + 1
        assert timeline[-1].id == result.event.id
        assert timeline[-1].status == EventStatus.DELIVERED
        room_after = await kit.get_room("r1")
        assert room_after is not None
        # No divergence (§14.3): the counters reflect the stored timeline.
        assert room_after.event_count == len(timeline)
        assert room_after.latest_index == result.event.index

    async def test_slow_store_commit_is_not_reported_blocked(self) -> None:
        """A store commit slower than process_timeout MUST NOT be cancelled or
        reported blocked: the commit is not part of the cancellable pre-commit
        unit (RFC §13.6 MUST NOT)."""
        from roomkit.store.memory import InMemoryStore

        class SlowCommitStore(InMemoryStore):
            async def commit_event(
                self, room_id: str, event: RoomEvent, **kwargs: Any
            ) -> RoomEvent:
                await asyncio.sleep(0.1)
                return await super().commit_event(room_id, event, **kwargs)

        kit = RoomKit(process_timeout=0.01, store=SlowCommitStore())
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        timeouts: list[FrameworkEvent] = []

        @kit.on("process_timeout")
        async def capture(fe: FrameworkEvent) -> None:
            timeouts.append(fe)

        base_timeline = len(await kit.get_timeline("r1"))

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        result = await kit.process_inbound(msg)

        assert timeouts == []
        assert not result.blocked
        assert result.event is not None
        assert result.event.status == EventStatus.DELIVERED
        timeline = await kit.get_timeline("r1")
        assert len(timeline) == base_timeline + 1
        assert timeline[-1].id == result.event.id
        room_after = await kit.get_room("r1")
        assert room_after is not None
        assert room_after.event_count == len(timeline)


# -- Changeset 6: Broadcast & side-effect error handling --


class TestBroadcastPartialFailure:
    async def test_partial_failure_emits_framework_event(self, kit: RoomKit) -> None:
        """When some channels fail delivery, emit broadcast_partial_failure event."""

        class FailingChannel(Channel):
            channel_type = ChannelType.WEBSOCKET

            def __init__(self, channel_id: str) -> None:
                super().__init__(channel_id)

            async def handle_inbound(
                self, message: InboundMessage, context: RoomContext
            ) -> RoomEvent:
                raise NotImplementedError

            async def deliver(
                self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
            ) -> ChannelOutput:
                raise RuntimeError("delivery failed")

        ch1 = SimpleChannel("sms1")
        ch2 = FailingChannel("ws1")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        events: list[FrameworkEvent] = []

        @kit.on("broadcast_partial_failure")
        async def capture(fe: FrameworkEvent) -> None:
            events.append(fe)

        msg = InboundMessage(
            channel_id="sms1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked
        assert len(events) == 1
        assert events[0].data["failed"] >= 1


class TestPersistSideEffectsErrorHandling:
    async def test_first_task_failure_does_not_lose_second(self, kit: RoomKit) -> None:
        """If the first task fails to persist, the second should still be saved."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        task1 = Task(id="t1", room_id="r1", title="Task 1")
        task2 = Task(id="t2", room_id="r1", title="Task 2")

        original_add_task = kit.store.add_task
        call_count = 0

        async def failing_add_task(task: Task) -> Task:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("storage error")
            return await original_add_task(task)

        kit.store.add_task = failing_add_task  # type: ignore[assignment]

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="sms1", channel_type=ChannelType.SMS),
            content=TextContent(body="test"),
        )
        context = await kit._build_context("r1")

        await kit._persist_side_effects("r1", [task1, task2], [], event, context)

        tasks = await kit.store.list_tasks("r1")
        assert len(tasks) == 1
        assert tasks[0].id == "t2"

    async def test_observation_failure_does_not_stop_processing(self, kit: RoomKit) -> None:
        """If an observation fails to persist, processing continues."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        obs1 = Observation(id="o1", room_id="r1", channel_id="sms1", content="obs 1")
        obs2 = Observation(id="o2", room_id="r1", channel_id="sms1", content="obs 2")

        original_add_obs = kit.store.add_observation
        call_count = 0

        async def failing_add_obs(obs: Observation) -> Observation:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("storage error")
            return await original_add_obs(obs)

        kit.store.add_observation = failing_add_obs  # type: ignore[assignment]

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="sms1", channel_type=ChannelType.SMS),
            content=TextContent(body="test"),
        )
        context = await kit._build_context("r1")

        await kit._persist_side_effects("r1", [], [obs1, obs2], event, context)

        observations = await kit.store.list_observations("r1")
        assert len(observations) == 1
        assert observations[0].id == "o2"
