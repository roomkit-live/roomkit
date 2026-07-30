"""Tests for the full identity resolution pipeline (RFC §11)."""

from __future__ import annotations

import pytest

from roomkit.channels.base import Channel
from roomkit.channels.cli import CLIChannel
from roomkit.channels.conference import ConferenceChannel
from roomkit.channels.voice import VoiceChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import IdentityNotFoundError, ParticipantNotFoundError, RoomKit
from roomkit.identity.base import IdentityResolver
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelDirection,
    ChannelType,
    HookTrigger,
    IdentificationStatus,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.identity import Identity, IdentityHookResult, IdentityResult
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from tests.test_framework import SimpleChannel

# ---- Resolvers ----


class AmbiguousResolver(IdentityResolver):
    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        return IdentityResult(
            status=IdentificationStatus.AMBIGUOUS,
            candidates=[
                Identity(id="id1", display_name="Alice"),
                Identity(id="id2", display_name="Bob"),
            ],
        )


class UnknownResolver(IdentityResolver):
    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        return IdentityResult(status=IdentificationStatus.UNKNOWN)


class RejectedResolver(IdentityResolver):
    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        return IdentityResult(status=IdentificationStatus.REJECTED, message="not allowed")


class CountingIdentityResolver(IdentityResolver):
    """Answers nothing, remembers every address it was handed."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        self.calls.append(message.sender_id or "")
        return IdentityResult(status=IdentificationStatus.UNKNOWN)


# ---- Helpers ----


class _AddressedChannel(SimpleChannel):
    """A channel that attributes its events, as every transport channel does.

    ``SimpleChannel`` leaves ``source.participant_id`` unset, which no real
    transport does (see ``TransportChannel.handle_inbound``) — and the roster is
    keyed on exactly that field.
    """

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        event = await super().handle_inbound(message, context)
        return event.model_copy(
            update={
                "source": event.source.model_copy(
                    update={"participant_id": message.sender_id or None}
                )
            }
        )


class _NamingChannel(_AddressedChannel):
    """A channel whose senders the room names, as a conference's are.

    Declared on the class, which is where a channel states it: an instance
    attribute would read as something an integrator flips per channel.
    """

    sender_is_participant = True


async def _setup_addressed_room(kit: RoomKit) -> _AddressedChannel:
    channel = _AddressedChannel("sms")
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    return channel


def _addressed_msg(*, channel_id: str = "sms", sender: str = "+15551234") -> InboundMessage:
    return InboundMessage(
        channel_id=channel_id,
        sender_id=sender,
        content=TextContent(body="hello"),
    )


async def _setup_room(
    kit: RoomKit,
) -> tuple[SimpleChannel, SimpleChannel]:
    """Create a room with two channels attached."""
    ch1 = SimpleChannel("sms1")
    ch2 = SimpleChannel("sms2")
    kit.register_channel(ch1)
    kit.register_channel(ch2)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "sms2")
    return ch1, ch2


def _msg(sender: str = "user1") -> InboundMessage:
    return InboundMessage(
        channel_id="sms1",
        sender_id=sender,
        content=TextContent(body="hello"),
    )


# ---- Tests: Ambiguous identity hooks ----


class TestAmbiguousIdentityHooks:
    async def test_hook_resolves_identity(self) -> None:
        """Identity hook resolves ambiguous → participant_id set."""
        alice = Identity(id="id1", display_name="Alice")
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.resolved(alice)

        result = await kit.process_inbound(_msg())
        assert result.event is not None
        assert result.event.source.participant_id == "id1"

    async def test_hook_challenges(self) -> None:
        """Identity hook challenges → message blocked, reason is challenge."""
        from roomkit.models.hook import InjectedEvent

        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)

        challenge_event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="system", channel_type="webhook"),
            content=TextContent(body="Please identify yourself"),
        )

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def challenge(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.challenge(
                inject=InjectedEvent(event=challenge_event, target_channel_ids=["sms1"]),
                message="Identify yourself",
            )

        result = await kit.process_inbound(_msg())
        assert result.blocked is True
        assert result.reason == "identity_challenge_sent"

    async def test_hook_rejects(self) -> None:
        """Identity hook rejects → message blocked."""
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def reject(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.reject("Not allowed")

        result = await kit.process_inbound(_msg())
        assert result.blocked is True
        assert result.reason == "Not allowed"

    async def test_hook_pending_creates_participant(self) -> None:
        """No hook resolution → pending participant created."""
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)

        result = await kit.process_inbound(_msg())
        # Not blocked — message still processed
        assert result.event is not None

        # Verify a system event for participant_joined_pending was stored
        events = await kit.store.list_events("r1")
        sys_events = [
            e
            for e in events
            if hasattr(e.content, "code") and e.content.code == "participant_joined_pending"
        ]
        assert len(sys_events) >= 1

    async def test_hook_returns_none_falls_through(self) -> None:
        """Identity hook returns None → pending participant created."""
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def no_op(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult | None:
            return None

        result = await kit.process_inbound(_msg())
        assert result.event is not None


# ---- Tests: Unknown identity hooks ----


class TestUnknownIdentityHooks:
    async def test_hook_rejects_unknown(self) -> None:
        """ON_IDENTITY_UNKNOWN hook rejects → message blocked."""
        kit = RoomKit(identity_resolver=UnknownResolver())
        await _setup_room(kit)

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def reject(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.reject("Unknown sender rejected")

        result = await kit.process_inbound(_msg())
        assert result.blocked is True
        assert result.reason == "Unknown sender rejected"

    async def test_hook_identifies_unknown(self) -> None:
        """ON_IDENTITY_UNKNOWN hook identifies → participant_id set."""
        identity = Identity(id="found_id", display_name="Found User")
        kit = RoomKit(identity_resolver=UnknownResolver())
        await _setup_room(kit)

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def identify(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.resolved(identity)

        result = await kit.process_inbound(_msg())
        assert result.event is not None
        assert result.event.source.participant_id == "found_id"

    async def test_no_hook_passes_through(self) -> None:
        """No identity hook for unknown → message still processes."""
        kit = RoomKit(identity_resolver=UnknownResolver())
        await _setup_room(kit)

        result = await kit.process_inbound(_msg())
        assert result.event is not None
        assert result.blocked is False


# ---- Tests: Mock resolver enhancements ----


class TestMockResolverEnhanced:
    async def test_ambiguous_result(self) -> None:
        """MockIdentityResolver returns AMBIGUOUS for ambiguous mappings."""
        alice = Identity(id="a1", display_name="Alice")
        bob = Identity(id="b1", display_name="Bob")
        resolver = MockIdentityResolver(ambiguous={"user1": [alice, bob]})
        msg = InboundMessage(channel_id="ch1", sender_id="user1", content=TextContent(body="hi"))
        ctx = RoomContext(room=Room(id="r1"))
        result = await resolver.resolve(msg, ctx)
        assert result.status == IdentificationStatus.AMBIGUOUS
        assert len(result.candidates) == 2

    async def test_unknown_status_configurable(self) -> None:
        """MockIdentityResolver unknown_status is configurable."""
        resolver = MockIdentityResolver(unknown_status=IdentificationStatus.PENDING)
        msg = InboundMessage(channel_id="ch1", sender_id="nobody", content=TextContent(body="hi"))
        ctx = RoomContext(room=Room(id="r1"))
        result = await resolver.resolve(msg, ctx)
        assert result.status == IdentificationStatus.PENDING

    async def test_exact_match_takes_priority(self) -> None:
        """Exact match in mapping takes priority over ambiguous."""
        alice = Identity(id="a1", display_name="Alice")
        bob = Identity(id="b1", display_name="Bob")
        resolver = MockIdentityResolver(
            mapping={"user1": alice},
            ambiguous={"user1": [alice, bob]},
        )
        msg = InboundMessage(channel_id="ch1", sender_id="user1", content=TextContent(body="hi"))
        ctx = RoomContext(room=Room(id="r1"))
        result = await resolver.resolve(msg, ctx)
        assert result.status == IdentificationStatus.IDENTIFIED
        assert result.identity is not None
        assert result.identity.id == "a1"


# ---- Tests: resolve_participant (manual identity resolution) ----


class TestResolveParticipant:
    async def test_resolves_pending_participant(self) -> None:
        """resolve_participant updates all fields."""
        kit = RoomKit()
        await kit.create_room(room_id="r1")
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.attach_channel("r1", "sms1")

        # Create a pending participant
        p = Participant(
            id="p1",
            room_id="r1",
            channel_id="sms1",
            identification=IdentificationStatus.PENDING,
            candidates=["id1", "id2"],
        )
        await kit.store.add_participant(p)

        # Create identity in store
        identity = Identity(id="id1", display_name="Alice")
        await kit.store.create_identity(identity)

        # Resolve
        result = await kit.resolve_participant("r1", "p1", "id1", resolved_by="advisor1")
        assert result.identification == IdentificationStatus.IDENTIFIED
        assert result.identity_id == "id1"
        assert result.resolved_by == "advisor1"
        assert result.resolved_at is not None
        assert result.candidates is None
        assert result.display_name == "Alice"

    async def test_resolve_emits_system_event(self) -> None:
        """resolve_participant emits PARTICIPANT_IDENTIFIED system event."""
        kit = RoomKit()
        await kit.create_room(room_id="r1")
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.attach_channel("r1", "sms1")

        p = Participant(
            id="p1",
            room_id="r1",
            channel_id="sms1",
            identification=IdentificationStatus.PENDING,
        )
        await kit.store.add_participant(p)
        await kit.store.create_identity(Identity(id="id1", display_name="Alice"))

        await kit.resolve_participant("r1", "p1", "id1")

        events = await kit.store.list_events("r1")
        id_events = [
            e
            for e in events
            if hasattr(e.content, "code") and e.content.code == "participant_identified"
        ]
        assert len(id_events) >= 1

    async def test_resolve_updates_binding(self) -> None:
        """resolve_participant updates the binding participant_id."""
        kit = RoomKit()
        await kit.create_room(room_id="r1")
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)
        await kit.attach_channel("r1", "sms1")

        p = Participant(
            id="p1",
            room_id="r1",
            channel_id="sms1",
            identification=IdentificationStatus.PENDING,
        )
        await kit.store.add_participant(p)
        await kit.store.create_identity(Identity(id="id1", display_name="Alice"))

        await kit.resolve_participant("r1", "p1", "id1")

        binding = await kit.store.get_binding("r1", "sms1")
        assert binding is not None
        assert binding.participant_id == "id1"

    async def test_resolve_nonexistent_participant_raises(self) -> None:
        """resolve_participant raises for unknown participant."""
        kit = RoomKit()
        await kit.create_room(room_id="r1")

        with pytest.raises(ParticipantNotFoundError, match="Participant"):
            await kit.resolve_participant("r1", "nope", "id1")

    async def test_resolve_nonexistent_identity_raises(self) -> None:
        """resolve_participant raises for unknown identity."""
        kit = RoomKit()
        await kit.create_room(room_id="r1")

        p = Participant(id="p1", room_id="r1", channel_id="sms1")
        await kit.store.add_participant(p)

        with pytest.raises(IdentityNotFoundError, match="Identity"):
            await kit.resolve_participant("r1", "p1", "no_such_id")


# ---- Tests: Identity hook filtering ----


class TestIdentityHookFiltering:
    """Tests for identity hook channel_types/channel_ids/directions filtering."""

    async def test_filter_by_channel_type_matches(self) -> None:
        """Identity hook with matching channel_type filter is invoked."""
        alice = Identity(id="id1", display_name="Alice")
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        called = []

        @kit.identity_hook(
            HookTrigger.ON_IDENTITY_AMBIGUOUS,
            channel_types={ChannelType.SMS},
        )
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            called.append(event)
            return IdentityHookResult.resolved(alice)

        result = await kit.process_inbound(_msg())
        assert len(called) == 1
        assert result.event is not None
        assert result.event.source.participant_id == "id1"

    async def test_filter_by_channel_type_skipped(self) -> None:
        """Identity hook with non-matching channel_type filter is skipped."""
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        called = []

        @kit.identity_hook(
            HookTrigger.ON_IDENTITY_AMBIGUOUS,
            channel_types={ChannelType.EMAIL},  # SMS channel won't match
        )
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            called.append(event)
            return IdentityHookResult.reject("Should not be called")

        result = await kit.process_inbound(_msg())
        assert len(called) == 0
        # Message should still process (pending participant created)
        assert result.event is not None

    async def test_filter_by_channel_id_matches(self) -> None:
        """Identity hook with matching channel_id filter is invoked."""
        alice = Identity(id="id1", display_name="Alice")
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        called = []

        @kit.identity_hook(
            HookTrigger.ON_IDENTITY_AMBIGUOUS,
            channel_ids={"sms1"},
        )
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            called.append(event)
            return IdentityHookResult.resolved(alice)

        result = await kit.process_inbound(_msg())
        assert len(called) == 1
        assert result.event is not None

    async def test_filter_by_channel_id_skipped(self) -> None:
        """Identity hook with non-matching channel_id filter is skipped."""
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        called = []

        @kit.identity_hook(
            HookTrigger.ON_IDENTITY_AMBIGUOUS,
            channel_ids={"other_channel"},
        )
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            called.append(event)
            return IdentityHookResult.reject("Should not be called")

        result = await kit.process_inbound(_msg())
        assert len(called) == 0
        assert result.event is not None

    async def test_filter_by_direction_matches(self) -> None:
        """Identity hook with matching direction filter is invoked."""
        alice = Identity(id="id1", display_name="Alice")
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        called = []

        @kit.identity_hook(
            HookTrigger.ON_IDENTITY_AMBIGUOUS,
            directions={ChannelDirection.INBOUND},
        )
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            called.append(event)
            return IdentityHookResult.resolved(alice)

        result = await kit.process_inbound(_msg())
        assert len(called) == 1
        assert result.event is not None

    async def test_filter_by_direction_skipped(self) -> None:
        """Identity hook with non-matching direction filter is skipped."""
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        called = []

        @kit.identity_hook(
            HookTrigger.ON_IDENTITY_AMBIGUOUS,
            directions={ChannelDirection.OUTBOUND},  # Inbound won't match
        )
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            called.append(event)
            return IdentityHookResult.reject("Should not be called")

        result = await kit.process_inbound(_msg())
        assert len(called) == 0
        assert result.event is not None

    async def test_multiple_filters_all_must_match(self) -> None:
        """All specified filters must match for hook to be invoked."""
        alice = Identity(id="id1", display_name="Alice")
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        called = []

        @kit.identity_hook(
            HookTrigger.ON_IDENTITY_AMBIGUOUS,
            channel_types={ChannelType.SMS},
            channel_ids={"sms1"},
            directions={ChannelDirection.INBOUND},
        )
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            called.append(event)
            return IdentityHookResult.resolved(alice)

        result = await kit.process_inbound(_msg())
        assert len(called) == 1
        assert result.event is not None

    async def test_multiple_filters_one_fails(self) -> None:
        """Hook is skipped if any filter doesn't match."""
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        called = []

        @kit.identity_hook(
            HookTrigger.ON_IDENTITY_AMBIGUOUS,
            channel_types={ChannelType.SMS},  # Matches
            channel_ids={"sms1"},  # Matches
            directions={ChannelDirection.OUTBOUND},  # Does NOT match (inbound)
        )
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            called.append(event)
            return IdentityHookResult.reject("Should not be called")

        result = await kit.process_inbound(_msg())
        assert len(called) == 0
        assert result.event is not None

    async def test_no_filters_matches_all(self) -> None:
        """Identity hook without filters matches all events (backward compat)."""
        alice = Identity(id="id1", display_name="Alice")
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        called = []

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            called.append(event)
            return IdentityHookResult.resolved(alice)

        result = await kit.process_inbound(_msg())
        assert len(called) == 1
        assert result.event is not None


# ---- Tests: Identity result backfill (address/channel_type) ----


class TestIdentityResultBackfill:
    """Tests for automatic backfill of address/channel_type in IdentityResult."""

    async def test_address_backfilled_from_sender_id(self) -> None:
        """IdentityResult.address is backfilled from message.sender_id."""
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        captured_result: list[IdentityResult] = []

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def capture(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult | None:
            captured_result.append(id_result)
            return None  # Fall through to pending

        await kit.process_inbound(_msg(sender="+14185551234"))

        assert len(captured_result) == 1
        assert captured_result[0].address == "+14185551234"

    async def test_channel_type_backfilled_from_channel(self) -> None:
        """IdentityResult.channel_type is backfilled from channel."""
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)
        captured_result: list[IdentityResult] = []

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def capture(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult | None:
            captured_result.append(id_result)
            return None

        await kit.process_inbound(_msg())

        assert len(captured_result) == 1
        assert captured_result[0].channel_type == "sms"

    async def test_resolver_provided_address_not_overwritten(self) -> None:
        """If resolver sets address, it is not overwritten."""

        class AddressSettingResolver(IdentityResolver):
            async def resolve(
                self, message: InboundMessage, context: RoomContext
            ) -> IdentityResult:
                return IdentityResult(
                    status=IdentificationStatus.AMBIGUOUS,
                    candidates=[Identity(id="id1", display_name="Alice")],
                    address="custom-address",
                )

        kit = RoomKit(identity_resolver=AddressSettingResolver())
        await _setup_room(kit)
        captured_result: list[IdentityResult] = []

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def capture(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult | None:
            captured_result.append(id_result)
            return None

        await kit.process_inbound(_msg(sender="+14185551234"))

        assert len(captured_result) == 1
        assert captured_result[0].address == "custom-address"  # Not overwritten


# ---- Tests: Participant creation for resolved identity ----


class TestResolvedIdentityParticipant:
    """Tests for automatic participant creation when identity is resolved."""

    async def test_hook_resolved_creates_participant(self) -> None:
        """When hook returns resolved(), a participant record is created."""
        alice = Identity(id="alice-id", display_name="Alice")
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.resolved(alice)

        result = await kit.process_inbound(_msg())
        assert result.event is not None
        assert result.event.source.participant_id == "alice-id"

        # Verify participant record was created
        participant = await kit.store.get_participant("r1", "alice-id")
        assert participant is not None
        assert participant.display_name == "Alice"
        assert participant.identification == IdentificationStatus.IDENTIFIED
        assert participant.identity_id == "alice-id"

    async def test_unknown_hook_resolved_creates_participant(self) -> None:
        """When ON_IDENTITY_UNKNOWN hook resolves, a participant is created."""
        bob = Identity(id="bob-id", display_name="Bob")
        kit = RoomKit(identity_resolver=UnknownResolver())
        await _setup_room(kit)

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.resolved(bob)

        result = await kit.process_inbound(_msg())
        assert result.event is not None

        # Verify participant record was created
        participant = await kit.store.get_participant("r1", "bob-id")
        assert participant is not None
        assert participant.display_name == "Bob"
        assert participant.identification == IdentificationStatus.IDENTIFIED

    async def test_direct_identified_creates_participant(self) -> None:
        """When resolver returns IDENTIFIED directly, a participant is created."""

        class DirectIdentifiedResolver(IdentityResolver):
            async def resolve(
                self, message: InboundMessage, context: RoomContext
            ) -> IdentityResult:
                return IdentityResult(
                    status=IdentificationStatus.IDENTIFIED,
                    identity=Identity(id="charlie-id", display_name="Charlie"),
                )

        kit = RoomKit(identity_resolver=DirectIdentifiedResolver())
        await _setup_room(kit)

        result = await kit.process_inbound(_msg())
        assert result.event is not None
        assert result.event.source.participant_id == "charlie-id"

        # Verify participant record was created
        participant = await kit.store.get_participant("r1", "charlie-id")
        assert participant is not None
        assert participant.display_name == "Charlie"
        assert participant.identification == IdentificationStatus.IDENTIFIED

    async def test_resolved_participant_idempotent(self) -> None:
        """Resolving same identity twice doesn't duplicate participant."""
        alice = Identity(id="alice-id", display_name="Alice")
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.resolved(alice)

        # Process two messages from same identity
        await kit.process_inbound(_msg())
        await kit.process_inbound(_msg())

        # Should still only have one participant
        participants = await kit.store.list_participants("r1")
        alice_participants = [p for p in participants if p.id == "alice-id"]
        assert len(alice_participants) == 1

    async def test_resolved_emits_system_event(self) -> None:
        """Resolving identity emits participant_joined_identified system event."""
        alice = Identity(id="alice-id", display_name="Alice")
        kit = RoomKit(identity_resolver=AmbiguousResolver())
        await _setup_room(kit)

        @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.resolved(alice)

        await kit.process_inbound(_msg())

        # Verify system event was emitted
        events = await kit.store.list_events("r1")
        sys_events = [
            e
            for e in events
            if hasattr(e.content, "code") and e.content.code == "participant_joined_identified"
        ]
        assert len(sys_events) == 1
        assert sys_events[0].content.data["participant_id"] == "alice-id"


class TestIdentityChannelTypesFiltering:
    """Test identity_channel_types parameter for filtering identity resolution by channel."""

    @pytest.mark.asyncio
    async def test_resolver_skipped_for_excluded_channel_type(self) -> None:
        """Identity resolution is skipped for channels not in identity_channel_types."""
        # Track if resolver was called
        resolver_calls: list[str] = []

        class TrackingResolver(IdentityResolver):
            async def resolve(
                self, message: InboundMessage, context: RoomContext
            ) -> IdentityResult:
                resolver_calls.append(message.channel_id)
                return IdentityResult(status=IdentificationStatus.UNKNOWN)

        # Only resolve identity for SMS channels
        kit = RoomKit(
            identity_resolver=TrackingResolver(),
            identity_channel_types={ChannelType.SMS},
        )

        # Register SMS and WebSocket channels
        sms_ch = SimpleChannel("sms-ch", channel_type=ChannelType.SMS)
        ws_ch = SimpleChannel("ws-ch", channel_type=ChannelType.WEBSOCKET)
        kit.register_channel(sms_ch)
        kit.register_channel(ws_ch)

        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms-ch")
        await kit.attach_channel("r1", "ws-ch")

        # Process SMS message - resolver should be called
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms-ch", sender_id="sender1", content=TextContent(body="hello")
            )
        )
        assert "sms-ch" in resolver_calls

        # Process WebSocket message - resolver should NOT be called
        resolver_calls.clear()
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-ch", sender_id="sender2", content=TextContent(body="hello")
            )
        )
        assert "ws-ch" not in resolver_calls

    @pytest.mark.asyncio
    async def test_resolver_runs_for_all_when_no_filter(self) -> None:
        """Without identity_channel_types, resolver runs for all channels."""
        resolver_calls: list[str] = []

        class TrackingResolver(IdentityResolver):
            async def resolve(
                self, message: InboundMessage, context: RoomContext
            ) -> IdentityResult:
                resolver_calls.append(message.channel_id)
                return IdentityResult(status=IdentificationStatus.UNKNOWN)

        # No channel type filter
        kit = RoomKit(identity_resolver=TrackingResolver())

        sms_ch = SimpleChannel("sms-ch", channel_type=ChannelType.SMS)
        ws_ch = SimpleChannel("ws-ch", channel_type=ChannelType.WEBSOCKET)
        kit.register_channel(sms_ch)
        kit.register_channel(ws_ch)

        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms-ch")
        await kit.attach_channel("r1", "ws-ch")

        await kit.process_inbound(
            InboundMessage(channel_id="sms-ch", sender_id="s1", content=TextContent(body="hi"))
        )
        await kit.process_inbound(
            InboundMessage(channel_id="ws-ch", sender_id="s2", content=TextContent(body="hi"))
        )

        # Both should have called the resolver
        assert "sms-ch" in resolver_calls
        assert "ws-ch" in resolver_calls

    @pytest.mark.asyncio
    async def test_no_pending_participant_for_excluded_channel(self) -> None:
        """Excluded channels don't create pending participants."""
        kit = RoomKit(
            identity_resolver=UnknownResolver(),
            identity_channel_types={ChannelType.SMS},  # Only SMS
        )

        ws_ch = SimpleChannel("ws-ch", channel_type=ChannelType.WEBSOCKET)
        kit.register_channel(ws_ch)

        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws-ch")

        # Process WebSocket message
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-ch", sender_id="ws-user", content=TextContent(body="hello")
            )
        )

        # No participant_joined_pending system event should be emitted
        events = await kit.store.list_events("r1")
        pending_events = [
            e
            for e in events
            if hasattr(e.content, "code") and e.content.code == "participant_joined_pending"
        ]
        assert len(pending_events) == 0


class TestTheRestrictionIsAskedOnceAndAnsweredEverywhere:
    """``identity_enabled_for`` is the single answer to "resolve this channel?".

    Public because the inbound pipeline is not the only caller: a conference
    resolves an arriving participant on a path with no inbound message, and a
    restriction the integrator configured has to reach that path too. A copy of
    the test inline is a copy that can drift out of agreement with this one.
    """

    def test_no_resolver_means_no_channel_type_is_enabled(self) -> None:
        kit = RoomKit()

        assert kit.identity_enabled_for(ChannelType.SMS) is False
        assert kit.identity_enabled_for(ChannelType.CONFERENCE) is False

    def test_no_restriction_means_every_channel_type_is_enabled(self) -> None:
        kit = RoomKit(identity_resolver=UnknownResolver())

        assert kit.identity_enabled_for(ChannelType.SMS) is True
        assert kit.identity_enabled_for(ChannelType.CONFERENCE) is True

    def test_a_restriction_answers_for_the_types_it_names_and_against_the_rest(self) -> None:
        kit = RoomKit(
            identity_resolver=UnknownResolver(),
            identity_channel_types={ChannelType.SMS},
        )

        assert kit.identity_enabled_for(ChannelType.SMS) is True
        assert kit.identity_enabled_for(ChannelType.CONFERENCE) is False


class TestSendersTheResolverIsNotAskedAbout:
    """Two senders carry no question, and asking anyway is not free (RFC §11.6).

    A resolver maps an address to an Identity. Where there is no address to map
    — because the room already answered, or because the channel names its own
    senders — the call costs a lookup per message and re-fires
    ``ON_IDENTITY_UNKNOWN``, so a hook written to refuse unknown senders refuses
    one the room knows.
    """

    async def test_a_sender_the_room_has_identified_is_not_looked_up_again(self) -> None:
        resolver = CountingIdentityResolver()
        kit = RoomKit(identity_resolver=resolver)
        await _setup_addressed_room(kit)
        await kit.store.add_participant(
            Participant(
                id="+15551234",
                room_id="r1",
                channel_id="sms",
                identification=IdentificationStatus.IDENTIFIED,
                identity_id="alice-id",
            )
        )

        result = await kit.process_inbound(_addressed_msg())

        assert resolver.calls == []
        assert not result.blocked
        assert result.event is not None
        assert result.event.source.participant_id == "+15551234"

    async def test_a_pending_sender_is_still_resolved_and_still_refusable(self) -> None:
        """The room having a record is not the room having an answer: PENDING is
        exactly the case a resolver may still settle, and a hook may still want
        to challenge or refuse.
        """
        resolver = CountingIdentityResolver()
        kit = RoomKit(identity_resolver=resolver)
        await _setup_addressed_room(kit)
        await kit.store.add_participant(
            Participant(
                id="+15551234",
                room_id="r1",
                channel_id="sms",
                identification=IdentificationStatus.PENDING,
            )
        )

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def refuse(
            event: RoomEvent, context: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.reject("unknown sender")

        result = await kit.process_inbound(_addressed_msg())

        assert resolver.calls == ["+15551234"]
        assert result.blocked
        assert result.reason == "unknown sender"

    async def test_a_channel_that_names_its_own_senders_is_not_resolved(self) -> None:
        """No roster record involved: the declaration alone settles it, which is
        what a conference needs — its minted participants are PENDING.
        """
        resolver = CountingIdentityResolver()
        kit = RoomKit(identity_resolver=resolver)
        await _setup_addressed_room(kit)
        kit.register_channel(_NamingChannel("named"))
        await kit.attach_channel("r1", "named")

        await kit.process_inbound(_addressed_msg(channel_id="named"))
        assert resolver.calls == []

        # Contrast: the same sender on a channel that does carry addresses.
        await kit.process_inbound(_addressed_msg())
        assert resolver.calls == ["+15551234"]


class TestWhichChannelsNameTheirOwnSenders:
    """Who declares ``sender_is_participant``, and who deliberately does not.

    Declared wrongly it costs a channel its resolution entirely — every address
    it carries stays unresolved, silently — so the roster is asserted here
    rather than left to a review. A channel belongs on the naming side when the
    framework itself chooses the ``sender_id``; on the addressed side when the
    value arrives from the remote network or from the integrator, which is where
    ``identity_channel_types`` (RFC §11.4) is the lever instead.
    """

    @pytest.mark.parametrize("channel_class", [CLIChannel, ConferenceChannel])
    def test_a_channel_that_names_its_senders_declares_it(
        self, channel_class: type[Channel]
    ) -> None:
        assert channel_class.sender_is_participant is True

    @pytest.mark.parametrize("channel_class", [SimpleChannel, VoiceChannel, WebSocketChannel])
    def test_a_channel_that_carries_addresses_leaves_the_default(
        self, channel_class: type[Channel]
    ) -> None:
        assert channel_class.sender_is_participant is False

    async def test_a_websocket_sender_is_resolved_as_before(self) -> None:
        """A WebSocket carries whatever its integrator puts on ``sender_id``,
        which may well be an address — RFC §11.4 leaves that call to
        ``identity_channel_types``, not to the channel.
        """
        resolver = CountingIdentityResolver()
        kit = RoomKit(identity_resolver=resolver)
        await kit.create_room(room_id="r1")
        kit.register_channel(WebSocketChannel("ws"))
        await kit.attach_channel("r1", "ws")

        await kit.process_inbound(
            InboundMessage(
                channel_id="ws", sender_id="+15551234", content=TextContent(body="hello")
            )
        )

        assert resolver.calls == ["+15551234"]
