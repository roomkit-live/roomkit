"""Identifying the conference participant the framework did not name (RFC §12.10.2).

The rule these exercise is a MUST with a stated purpose: a phone participant
dialling into a conference must reach the same Identity it would have reached by
texting the room. Two things have to be true for that, and they pull in opposite
directions — the resolver must see the caller's *number*, and everything that
attributes media must go on seeing the *backend's* identity. So the suite checks
both ends: what the resolver was handed, and what the room ended up with.
"""

from __future__ import annotations

import asyncio

from roomkit import (
    CONFERENCE_ADDRESS_KEYS,
    CONFERENCE_METADATA_KEY,
    CONFERENCE_UNASSERTED_METADATA_KEY,
    MockConferenceBackend,
    RoomKit,
    TrackKind,
)
from roomkit.channels._conference_identity import ConferenceIdentity
from roomkit.channels._conference_metadata import ASSERTED_KEY
from roomkit.channels.conference import ConferenceChannel
from roomkit.conference.models import ConferenceParticipant, ConferenceTrack
from roomkit.identity.base import IdentityResolver
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, HookExecution, HookTrigger, IdentificationStatus
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.identity import Identity, IdentityHookResult, IdentityResult
from roomkit.models.participant import Participant
from roomkit.voice.stt.mock import MockSTTProvider
from tests.conference.lane_audio import drain, say
from tests.test_framework import SimpleChannel

ROOM = "room-1"
DIAL_IN = "sip_15551234"
NUMBER = "+15551234"

ALICE = Identity(id="user-42", display_name="Alice", channel_addresses={"sms": [NUMBER]})


class RecordingResolver(IdentityResolver):
    """Answers nothing, remembers everything it was asked."""

    def __init__(self, identity: Identity | None = None) -> None:
        self.seen: list[InboundMessage] = []
        self._identity = identity

    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        self.seen.append(message)
        if self._identity is None:
            return IdentityResult(status=IdentificationStatus.UNKNOWN)
        return IdentityResult(status=IdentificationStatus.IDENTIFIED, identity=self._identity)


class AmbiguousResolver(IdentityResolver):
    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        return IdentityResult(
            status=IdentificationStatus.AMBIGUOUS,
            candidates=[Identity(id="user-42"), Identity(id="user-43")],
        )


class RejectingResolver(IdentityResolver):
    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        return IdentityResult(status=IdentificationStatus.REJECTED)


class SlowResolver(IdentityResolver):
    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        await asyncio.sleep(5)
        return IdentityResult(status=IdentificationStatus.IDENTIFIED, identity=ALICE)


class BrokenResolver(IdentityResolver):
    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        raise RuntimeError("the directory is down")


async def _kit_with_channel(
    resolver: IdentityResolver | None = None,
    *,
    identity_timeout: float = 10.0,
    identity_channel_types: set[ChannelType] | None = None,
    **channel_kwargs: object,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    backend = MockConferenceBackend()
    channel = ConferenceChannel("conf", backend=backend, **channel_kwargs)  # type: ignore[arg-type]
    kit = RoomKit(
        identity_resolver=resolver,
        identity_timeout=identity_timeout,
        identity_channel_types=identity_channel_types,
    )
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    return kit, channel, backend


async def _participant(kit: RoomKit, participant_id: str = DIAL_IN) -> Participant:
    record = await kit.store.get_participant(ROOM, participant_id)
    assert record is not None
    return record


def _asserted(record: Participant) -> dict[str, object]:
    """The attributes the SFU vouched for, as the record keeps them."""
    return record.metadata[CONFERENCE_METADATA_KEY][ASSERTED_KEY]


async def _utter(
    backend: MockConferenceBackend,
    channel: ConferenceChannel,
    track: ConferenceTrack,
    *,
    times: int,
) -> None:
    """Speak *times* separate utterances on a track, letting each land."""
    for _ in range(times):
        await say(backend, track)
        await drain(channel, track.id)


def _transcripts(events: list[RoomEvent]) -> list[str]:
    return [event.content.body for event in events if isinstance(event.content, TextContent)]


def _refusing_unknown_senders(kit: RoomKit) -> list[str]:
    """Register the refusal pattern an SMS deployment writes, and watch it.

    Returns the list it appends an address to on every firing, so a test can
    assert on what it was asked about as well as on what survived it.
    """
    seen: list[str] = []

    @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
    async def refuse(
        event: RoomEvent, context: RoomContext, id_result: IdentityResult
    ) -> IdentityHookResult:
        seen.append(id_result.address or "")
        return IdentityHookResult.reject("unknown sender")

    return seen


class TestWhatTheResolverIsHanded:
    async def test_the_caller_number_is_what_the_resolver_matches_on(self) -> None:
        """The point of the rule: resolvers key on the address a person is
        reachable at, and the backend's identity is not one.
        """
        resolver = RecordingResolver()
        kit, _, backend = await _kit_with_channel(resolver)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        assert [message.sender_id for message in resolver.seen] == [NUMBER]

    async def test_the_opaque_identity_travels_with_it(self) -> None:
        """Not thrown away: a resolver wanting more than the address has the
        backend's identity and the provider's attributes.
        """
        resolver = RecordingResolver()
        _, _, backend = await _kit_with_channel(resolver)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER, "sip.callID": "abc"}
        )

        message = resolver.seen[-1]
        assert message.external_id == DIAL_IN
        assert message.metadata["sip.callID"] == "abc"

    async def test_an_arrival_without_an_address_is_not_resolved(self) -> None:
        """Explicitly not "resolve on the opaque identity": no resolver could
        match it, and a lookup that cannot succeed reads as one that did.
        """
        resolver = RecordingResolver()
        kit, _, backend = await _kit_with_channel(resolver)

        await backend.simulate_participant_joined(ROOM, DIAL_IN)

        assert resolver.seen == []
        assert (await _participant(kit)).identification is IdentificationStatus.UNKNOWN

    async def test_the_number_that_was_dialled_is_not_the_caller(self) -> None:
        """Every dial-in reaches the same trunk number. Reading it would
        identify all of them as one person.
        """
        resolver = RecordingResolver()
        _, _, backend = await _kit_with_channel(resolver)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.trunkPhoneNumber": "+18005551212"}
        )

        assert resolver.seen == []

    async def test_a_provider_key_wins_over_a_generic_one(self) -> None:
        resolver = RecordingResolver()
        _, _, backend = await _kit_with_channel(resolver)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"from_number": "+19999999999", "sip.phoneNumber": NUMBER}
        )

        assert resolver.seen[0].sender_id == NUMBER

    async def test_a_non_string_attribute_is_not_an_address(self) -> None:
        """An integer has lost the leading ``+`` on the way in, and handing the
        resolver ``15551234`` silently fails to match ``+15551234``.
        """
        resolver = RecordingResolver()
        _, _, backend = await _kit_with_channel(resolver)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"phone_number": 15551234}
        )

        assert resolver.seen == []

    async def test_configured_keys_replace_the_defaults(self) -> None:
        resolver = RecordingResolver()
        _, _, backend = await _kit_with_channel(resolver, identity_address_keys=("x-caller",))

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"x-caller": NUMBER, "sip.phoneNumber": "+19999999999"}
        )

        assert resolver.seen[0].sender_id == NUMBER

    async def test_the_defaults_lead_with_the_provider_key(self) -> None:
        assert CONFERENCE_ADDRESS_KEYS[0] == "sip.phoneNumber"
        assert "sip.trunkPhoneNumber" not in CONFERENCE_ADDRESS_KEYS

    async def test_a_sip_header_name_is_not_a_default_key(self) -> None:
        """``from`` says nothing about where the value under it came from, and
        a caller writing its own is the impersonation this list must not enable.
        """
        assert "from" not in CONFERENCE_ADDRESS_KEYS


class TestWhoPutTheAddressThere:
    """Provenance before specificity (RFC §12.10.2).

    An attribute a participant's own client supplied is a claim about itself.
    Resolving on it hands the caller whichever Identity that number belongs to
    — someone else's — and writes the victim's ``identity_id`` onto the record
    every later attribution reads. So the framework believes the SFU, and
    nobody else, unless the integrator says otherwise.
    """

    async def test_an_address_the_participant_supplied_is_not_resolved(self) -> None:
        """The impersonation, refused: the caller number is there, and nothing
        vouches for it.
        """
        resolver = RecordingResolver(ALICE)
        kit, _, backend = await _kit_with_channel(resolver)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, client_metadata={"sip.phoneNumber": NUMBER}
        )

        assert resolver.seen == []
        record = await _participant(kit)
        assert record.identification is IdentificationStatus.UNKNOWN
        assert record.identity_id is None

    async def test_a_backend_that_cannot_tell_resolves_nothing(self) -> None:
        """``asserted_metadata`` left null is a statement, and this is what it
        says: no attribute here can be told from one the client wrote.
        """
        resolver = RecordingResolver(ALICE)
        kit, _, backend = await _kit_with_channel(resolver)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}, asserts_provenance=False
        )

        assert resolver.seen == []
        assert (await _participant(kit)).identification is IdentificationStatus.UNKNOWN

    async def test_an_integrator_can_widen_it_deliberately(self) -> None:
        """A closed client fleet, or provenance established elsewhere. The
        policy is the integrator's; what the framework owes is that the safe
        reading holds unconfigured.
        """
        resolver = RecordingResolver(ALICE)
        kit, _, backend = await _kit_with_channel(
            resolver, identity_trusts_unasserted_metadata=True
        )

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}, asserts_provenance=False
        )

        assert [message.sender_id for message in resolver.seen] == [NUMBER]
        assert (await _participant(kit)).identity_id == ALICE.id

    async def test_provenance_outranks_specificity(self) -> None:
        """An attacker chooses the key and never the provenance, so the generic
        key the SFU asserted beats the provider's own key it did not.
        """
        resolver = RecordingResolver()
        _, _, backend = await _kit_with_channel(resolver)

        await backend.simulate_participant_joined(
            ROOM,
            DIAL_IN,
            metadata={"caller_id": NUMBER},
            client_metadata={"sip.phoneNumber": "+19999999999"},
        )

        assert resolver.seen[0].sender_id == NUMBER

    async def test_what_was_claimed_reaches_the_resolver_nested(self) -> None:
        """Still handed over — a resolver that wants it can have it — but never
        where a resolver reading the provider's key would meet it by accident.
        """
        resolver = RecordingResolver()
        _, _, backend = await _kit_with_channel(resolver)

        await backend.simulate_participant_joined(
            ROOM,
            DIAL_IN,
            metadata={"sip.phoneNumber": NUMBER},
            client_metadata={"nickname": "bob", "phone_number": "+19999999999"},
        )

        message = resolver.seen[0]
        assert message.metadata["sip.phoneNumber"] == NUMBER
        assert "phone_number" not in message.metadata
        assert message.metadata[CONFERENCE_UNASSERTED_METADATA_KEY] == {
            "nickname": "bob",
            "phone_number": "+19999999999",
        }

    async def test_trusting_the_unasserted_flattens_what_it_trusts(self) -> None:
        """An integrator who said to read them gets them where they read them."""
        resolver = RecordingResolver()
        _, _, backend = await _kit_with_channel(resolver, identity_trusts_unasserted_metadata=True)

        await backend.simulate_participant_joined(
            ROOM,
            DIAL_IN,
            metadata={"sip.phoneNumber": NUMBER},
            client_metadata={"nickname": "bob"},
        )

        assert resolver.seen[0].metadata["nickname"] == "bob"


class TestWhatTheRoomEndsUpWith:
    async def test_a_dial_in_is_identified_on_arrival(self) -> None:
        """Without waiting for the caller to speak — someone can sit through a
        whole meeting listening.
        """
        kit, _, backend = await _kit_with_channel(MockIdentityResolver({NUMBER: ALICE}))

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        record = await _participant(kit)
        assert record.identification is IdentificationStatus.IDENTIFIED
        assert record.identity_id == ALICE.id
        assert record.display_name == "Alice"

    async def test_the_record_is_still_keyed_on_the_backend_identity(self) -> None:
        """The correlation of RFC 12.10.2 rule 2: the Identity is linked to the
        participant, never substituted for it, or the transcript and the
        recording would name the same person differently.
        """
        kit, _, backend = await _kit_with_channel(MockIdentityResolver({NUMBER: ALICE}))

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        records = await kit.store.list_participants(ROOM)
        assert [record.id for record in records] == [DIAL_IN]
        assert records[0].external_id == DIAL_IN

    async def test_an_arrival_records_the_conference_among_the_channels_reached(self) -> None:
        """RFC §5.5: a channel that reaches a participant records itself.

        A record the conference created seeds the list; a record it did not —
        one an integrator admitted through another channel — must gain the
        conference too, or ``connected_via`` is incomplete exactly in the
        cross-channel case it exists for.
        """
        kit, _, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "ws:alice", "p-alice")

        await backend.simulate_participant_joined(ROOM, "p-alice")

        record = await _participant(kit, "p-alice")
        assert record.channel_id == "ws:alice"  # an arrival is not a join
        assert record.connected_via == ["ws:alice", "conf"]

    async def test_a_dial_in_seeds_the_channels_reached_with_the_conference(self) -> None:
        kit, _, backend = await _kit_with_channel()

        await backend.simulate_participant_joined(ROOM, DIAL_IN)

        assert (await _participant(kit)).connected_via == ["conf"]

    async def test_a_dial_in_reaches_the_same_identity_as_a_text(self) -> None:
        """The purpose the RFC states, end to end: one resolver, one number, one
        Identity, whether the person dialled in or wrote.
        """
        resolver = MockIdentityResolver({NUMBER: ALICE})
        kit, _, backend = await _kit_with_channel(resolver)
        sms = SimpleChannel("sms")
        kit.register_channel(sms)
        await kit.attach_channel(ROOM, "sms")

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )
        await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id=NUMBER, content=TextContent(body="hello")),
            room_id=ROOM,
        )

        dialled_in = await _participant(kit)
        texted = await kit.store.get_participant(ROOM, ALICE.id)
        assert texted is not None
        assert dialled_in.identity_id == texted.identity_id == ALICE.id

    async def test_an_ambiguous_result_records_its_candidates(self) -> None:
        kit, _, backend = await _kit_with_channel(AmbiguousResolver())

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        record = await _participant(kit)
        assert record.identification is IdentificationStatus.PENDING
        assert record.candidates == ["user-42", "user-43"]

    async def test_a_rejection_leaves_the_participant_unknown(self) -> None:
        """A rejection answers a message by refusing it. There is no message
        here, and the SFU has already let the caller in.
        """
        kit, _, backend = await _kit_with_channel(RejectingResolver())

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        assert (await _participant(kit)).identification is IdentificationStatus.UNKNOWN

    async def test_identification_precedes_the_join_hook(self) -> None:
        """Observers must not read an unknown that changes underneath them."""
        kit, _, backend = await _kit_with_channel(MockIdentityResolver({NUMBER: ALICE}))
        seen: list[str | None] = []

        @kit.hook(HookTrigger.ON_CONFERENCE_PARTICIPANT_JOINED, execution=HookExecution.ASYNC)
        async def observe(event: object, context: object) -> None:
            record = await kit.store.get_participant(ROOM, DIAL_IN)
            seen.append(None if record is None else record.identity_id)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        assert seen == [ALICE.id]

    async def test_a_participant_the_framework_named_is_not_re_resolved(self) -> None:
        """A minted participant already has the record its integrator created.
        Resolving over it would replace what they know with a guess.
        """
        resolver = RecordingResolver(ALICE)
        kit, _, backend = await _kit_with_channel(resolver)
        await kit.store.add_participant(
            Participant(
                id="p-alice",
                room_id=ROOM,
                channel_id="conf",
                display_name="Alice from the CRM",
                identification=IdentificationStatus.PENDING,
            )
        )

        await backend.simulate_participant_joined(
            ROOM, "p-alice", metadata={"sip.phoneNumber": NUMBER}
        )

        assert resolver.seen == []
        record = await _participant(kit, "p-alice")
        assert record.identification is IdentificationStatus.PENDING
        assert record.display_name == "Alice from the CRM"

    async def test_transcription_stays_attributed_to_the_backend_identity(self) -> None:
        """Identifying the caller must not move the transcript onto the
        Identity: the recording and the interruption allowlist stayed on the
        backend's identity, and one human under two identifiers is the failure
        rule 2 forbids.
        """
        kit, channel, backend = await _kit_with_channel(
            MockIdentityResolver({NUMBER: ALICE}), stt=MockSTTProvider(transcripts=["bonjour"])
        )
        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )
        track = await backend.simulate_track_published(ROOM, DIAL_IN, TrackKind.AUDIO)
        await backend.subscribe_track(backend.bots[0], track.id)

        await say(backend, track)
        await drain(channel, track.id)

        spoken = [
            event
            for event in await kit.store.list_events(ROOM)
            if getattr(event.content, "body", None) == "bonjour"
        ]
        assert [event.source.participant_id for event in spoken] == [DIAL_IN]
        assert [record.id for record in await kit.store.list_participants(ROOM)] == [DIAL_IN]


class TestWhenResolutionDoesNotAnswer:
    async def test_a_slow_resolver_does_not_hold_the_arrival(self) -> None:
        """RFC §11.5, applied where the message is an arrival: treat it as
        unknown, say so, and let the participant in.
        """
        kit, _, backend = await _kit_with_channel(SlowResolver(), identity_timeout=0.01)
        timeouts: list[object] = []

        @kit.on("identity_timeout")
        async def observe(event: object) -> None:
            timeouts.append(event)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        assert (await _participant(kit)).identification is IdentificationStatus.UNKNOWN
        assert len(timeouts) == 1

    async def test_a_resolver_that_raises_does_not_keep_someone_out(self) -> None:
        kit, _, backend = await _kit_with_channel(BrokenResolver())

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        assert (await _participant(kit)).identification is IdentificationStatus.UNKNOWN

    async def test_no_resolver_configured_records_the_participant_as_before(self) -> None:
        kit, _, backend = await _kit_with_channel()

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        record = await _participant(kit)
        assert record.identification is IdentificationStatus.UNKNOWN
        assert _asserted(record) == {"sip.phoneNumber": NUMBER}


class TestWhatTheConfigurationExcludes:
    """``identity_channel_types`` reaches the arrival path too.

    An integrator restricting resolution to SMS does it to stop addresses
    leaving for the resolver — a contractual limit, a data-processing one. A
    dial-in's caller number is precisely such an address, so the arrival is
    gated by the same configuration as an inbound message: what the public
    parameter promises is what every path that could disclose an address obeys.
    """

    async def test_a_restriction_that_excludes_conferences_stops_the_arrival_lookup(
        self,
    ) -> None:
        resolver = RecordingResolver(ALICE)
        kit, _, backend = await _kit_with_channel(
            resolver, identity_channel_types={ChannelType.SMS}
        )

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        assert resolver.seen == []
        assert (await _participant(kit)).identification is IdentificationStatus.UNKNOWN

    async def test_the_participant_still_joins_and_keeps_its_attributes(self) -> None:
        """Not resolving is not refusing. The excluded arrival is the same
        arrival, minus the lookup — the roster records it, and the address stays
        where the provider put it for anything downstream that wants it.
        """
        kit, _, backend = await _kit_with_channel(
            RecordingResolver(ALICE), identity_channel_types={ChannelType.SMS}
        )

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        record = await _participant(kit)
        assert record.id == DIAL_IN
        assert record.identity_id is None
        assert _asserted(record) == {"sip.phoneNumber": NUMBER}

    async def test_a_restriction_that_names_conferences_resolves_as_before(self) -> None:
        resolver = RecordingResolver(ALICE)
        kit, _, backend = await _kit_with_channel(
            resolver, identity_channel_types={ChannelType.CONFERENCE}
        )

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        assert [message.sender_id for message in resolver.seen] == [NUMBER]
        record = await _participant(kit)
        assert record.identification is IdentificationStatus.IDENTIFIED
        assert record.identity_id == ALICE.id

    async def test_no_restriction_resolves_as_before(self) -> None:
        resolver = RecordingResolver(ALICE)
        kit, _, backend = await _kit_with_channel(resolver, identity_channel_types=None)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        assert [message.sender_id for message in resolver.seen] == [NUMBER]
        assert (await _participant(kit)).identity_id == ALICE.id

    async def test_resolve_refuses_on_its_own_and_not_only_behind_active(self) -> None:
        """The arrival path asks ``active`` first, so the guard in ``resolve``
        is never what stops it there. It is there for the caller that does not
        ask: a configuration saying no has to say no to every one of them.
        """
        resolver = RecordingResolver(ALICE)
        kit, _, _ = await _kit_with_channel(resolver, identity_channel_types={ChannelType.SMS})
        identity = ConferenceIdentity("conf")
        identity.set_framework(kit)

        arrival = ConferenceParticipant(
            participant_id=DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )

        result = await identity.resolve(ROOM, arrival)

        assert result is None
        assert resolver.seen == []


class TestWhatAnUtteranceDoesNotAskAgain:
    """A conference resolves on arrival, and speaking asks nothing new.

    An utterance reaches the inbound pipeline carrying the identity its track
    was published under — the Room ``Participant.id``, never an address. Running
    a resolver over it is the "resolve on the opaque identity" rule 3 rules out,
    and it fails the way rule 3 says it would: nothing matches, so every
    utterance comes back UNKNOWN.

    Which would be noise, if ``ON_IDENTITY_UNKNOWN`` did not also let a hook
    refuse the sender. Registered globally — as an SMS deployment does — that
    hook silently deletes the transcripts of someone the framework identified
    when they dialled in.
    """

    async def test_a_refusal_of_unknown_senders_does_not_eat_the_transcripts(self) -> None:
        """The failure this exists to prevent, end to end."""
        kit, channel, backend = await _kit_with_channel(
            MockIdentityResolver({NUMBER: ALICE}), stt=MockSTTProvider(transcripts=["bonjour"])
        )
        refusals = _refusing_unknown_senders(kit)

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )
        track = await backend.simulate_track_published(ROOM, DIAL_IN, TrackKind.AUDIO)
        await _utter(backend, channel, track, times=3)

        assert refusals == []
        assert _transcripts(await kit.store.list_events(ROOM)) == ["bonjour"] * 3

    async def test_speaking_three_times_asks_the_resolver_once(self) -> None:
        """Once, on arrival. A resolver reads a CRM at an integrator's, and a
        query per sentence per participant is the cost of asking again.
        """
        resolver = RecordingResolver(ALICE)
        _, channel, backend = await _kit_with_channel(
            resolver, stt=MockSTTProvider(transcripts=["bonjour"])
        )

        await backend.simulate_participant_joined(
            ROOM, DIAL_IN, metadata={"sip.phoneNumber": NUMBER}
        )
        track = await backend.simulate_track_published(ROOM, DIAL_IN, TrackKind.AUDIO)
        await _utter(backend, channel, track, times=3)

        assert [message.sender_id for message in resolver.seen] == [NUMBER]

    async def test_a_participant_the_framework_named_is_never_resolved(self) -> None:
        """The minted case, which is the same defect with a different id:
        ``sender_id`` is then the ``Participant.id`` the integrator chose, which
        no resolver knows any better than a backend's identity — and
        ``ensure_participant`` leaves it PENDING, so nothing about the record
        says "already answered".
        """
        resolver = RecordingResolver(ALICE)
        kit, channel, backend = await _kit_with_channel(
            resolver, stt=MockSTTProvider(transcripts=["bonjour"])
        )
        refusals = _refusing_unknown_senders(kit)

        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice", TrackKind.AUDIO)
        await _utter(backend, channel, track, times=3)

        assert (await _participant(kit, "p-alice")).identification is IdentificationStatus.PENDING
        assert resolver.seen == []
        assert refusals == []
        assert _transcripts(await kit.store.list_events(ROOM)) == ["bonjour"] * 3
