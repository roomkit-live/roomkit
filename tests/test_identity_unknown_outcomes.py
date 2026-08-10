"""Every identity outcome is acted on, whoever reaches it (RFC §11.3, §10.1 step 5).

The AMBIGUOUS path honoured all four answers an ``IdentityHookResult`` can give.
The UNKNOWN path honoured two: ``challenge()`` and ``pending()`` fell through it
and the message was processed as though no hook had spoken — the security-shaped
half, since a hook that says "make this sender prove who they are" was answered
by letting the sender straight in.

A *resolver* returning ``CHALLENGE_SENT`` was dropped the same way, by the
dispatch that had no branch for it at all.
"""

from __future__ import annotations

from roomkit.core.framework import RoomKit
from roomkit.identity.base import IdentityResolver
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import HookTrigger, IdentificationStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import InjectedEvent
from roomkit.models.identity import Identity, IdentityHookResult, IdentityResult
from tests.test_framework import SimpleChannel


class UnknownResolver(IdentityResolver):
    """Never names the sender — the case ON_IDENTITY_UNKNOWN exists for."""

    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        return IdentityResult(status=IdentificationStatus.UNKNOWN)


class ChallengingResolver(IdentityResolver):
    """Has already put a challenge to the sender out of band."""

    def __init__(self, message: str | None = None) -> None:
        self._message = message

    async def resolve(self, msg: InboundMessage, context: RoomContext) -> IdentityResult:
        return IdentityResult(
            status=IdentificationStatus.CHALLENGE_SENT,
            challenge_type="sms_code",
            message=self._message,
        )


async def _room(resolver: IdentityResolver) -> RoomKit:
    kit = RoomKit(identity_resolver=resolver)
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    return kit


def _msg() -> InboundMessage:
    return InboundMessage(
        channel_id="sms1", sender_id="+15550001111", content=TextContent(body="hello")
    )


def _challenge_event() -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="system", channel_type="webhook"),
        content=TextContent(body="Who are you?"),
    )


class TestTheUnknownHookCanChallenge:
    async def test_the_message_is_held(self) -> None:
        kit = await _room(UnknownResolver())

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def challenge(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.challenge(
                inject=InjectedEvent(event=_challenge_event(), target_channel_ids=["sms1"]),
                message="Identify yourself",
            )

        result = await kit.process_inbound(_msg())

        assert result.blocked is True
        assert result.reason == "identity_challenge_sent"
        assert result.event is None

    async def test_the_challenge_reaches_the_sender(self) -> None:
        """Holding the message is only half of it — the question must go out."""
        kit = await _room(UnknownResolver())
        channel = kit.get_channel("sms1")

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def challenge(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.challenge(
                inject=InjectedEvent(event=_challenge_event(), target_channel_ids=["sms1"])
            )

        await kit.process_inbound(_msg())

        assert [e.content.body for e in channel.delivered] == ["Who are you?"]

    async def test_the_sender_message_is_not_in_the_timeline(self) -> None:
        kit = await _room(UnknownResolver())

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def challenge(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.challenge(
                inject=InjectedEvent(event=_challenge_event(), target_channel_ids=["sms1"])
            )

        await kit.process_inbound(_msg())
        timeline = await kit.get_timeline("r1")

        assert "hello" not in [getattr(e.content, "body", None) for e in timeline]


class TestTheUnknownHookCanPend:
    async def test_a_pending_participant_is_created(self) -> None:
        kit = await _room(UnknownResolver())

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def pend(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.pending(display_name="Maybe Alice")

        result = await kit.process_inbound(_msg())
        participants = await kit.store.list_participants("r1")

        assert result.blocked is False
        assert len(participants) == 1
        assert participants[0].identification is IdentificationStatus.PENDING

    async def test_the_hooks_candidates_are_carried(self) -> None:
        """A hook that pends with candidates is naming who it might be."""
        kit = await _room(UnknownResolver())
        alice = Identity(id="id1", display_name="Alice")
        bob = Identity(id="id2", display_name="Bob")

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def pend(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.pending(candidates=[alice, bob])

        await kit.process_inbound(_msg())
        participants = await kit.store.list_participants("r1")

        assert participants[0].candidates == ["id1", "id2"]


class TestTheUnknownHookStillRejectsAndIdentifies:
    """The two outcomes that already worked keep working."""

    async def test_reject_blocks_with_its_reason(self) -> None:
        kit = await _room(UnknownResolver())

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def reject(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.reject("Not allowed")

        result = await kit.process_inbound(_msg())

        assert result.blocked is True
        assert result.reason == "Not allowed"

    async def test_resolved_stamps_the_participant(self) -> None:
        kit = await _room(UnknownResolver())
        alice = Identity(id="id1", display_name="Alice")

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def resolve(
            event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.resolved(alice)

        result = await kit.process_inbound(_msg())

        assert result.event is not None
        assert result.event.source.participant_id == "id1"

    async def test_no_hook_at_all_leaves_the_message_alone(self) -> None:
        """An UNKNOWN sender nobody objects to is still processed."""
        kit = await _room(UnknownResolver())

        result = await kit.process_inbound(_msg())

        assert result.blocked is False
        assert result.event is not None


class TestAResolverThatAlreadyChallenged:
    async def test_the_message_is_blocked(self) -> None:
        kit = await _room(ChallengingResolver())

        result = await kit.process_inbound(_msg())

        assert result.blocked is True
        assert result.reason == "identity_challenge_sent"

    async def test_the_resolvers_message_becomes_the_reason(self) -> None:
        kit = await _room(ChallengingResolver("code sent by sms"))

        result = await kit.process_inbound(_msg())

        assert result.reason == "code sent by sms"

    async def test_nothing_is_stored(self) -> None:
        kit = await _room(ChallengingResolver())

        await kit.process_inbound(_msg())
        timeline = await kit.get_timeline("r1")

        assert "hello" not in [getattr(e.content, "body", None) for e in timeline]
