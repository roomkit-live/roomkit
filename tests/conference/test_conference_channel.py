"""ConferenceChannel behaviour (RFC §12.10.4).

Written before the channel existed. Each test here corresponds to a normative
rule, and several of them exist because the rule is not obvious: the bot must
ignore itself, an unconsumed track must never be subscribed, and a participant
the framework did not name must still reach identity resolution through the
address its provider attached.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from roomkit import (
    CONFERENCE_METADATA_KEY,
    ConferenceCapability,
    ConferenceGrants,
    ConferenceInterruptionConfig,
    ConferenceInterruptionScope,
    ConferenceParticipant,
    MockConferenceBackend,
    RoomKit,
    TrackKind,
)
from roomkit.channels._conference_metadata import (
    MAX_ATTRIBUTES,
    MAX_KEY_CHARS,
    MAX_VALUE_CHARS,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.core.exceptions import (
    ConferenceAlreadyAttachedError,
    ParticipantNotAdmittedError,
    ParticipantNotFoundError,
    RoomNotAttachedError,
)
from roomkit.identity.base import IdentityResolver
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.enums import (
    Access,
    ChannelMediaType,
    ChannelType,
    HookExecution,
    HookTrigger,
    IdentificationStatus,
    ParticipantStatus,
)
from roomkit.models.identity import Identity
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from tests.conference.lane_audio import drain, say
from tests.conference.test_conference_races import _settle

ROOM = "room-1"


class _EchoDuringJoinBackend(MockConferenceBackend):
    """Reports the bot as a participant from inside ``join_as_bot``.

    The SELF-1 window, scripted: the bot is in the conference and being
    announced before the channel has a session to recognise it by.
    """

    async def join_as_bot(  # type: ignore[no-untyped-def]
        self, room_id, identity, grants
    ):
        await self.simulate_participant_joined(room_id, identity)
        return await super().join_as_bot(room_id, identity, grants)


class _JoinedEarlierBackend(MockConferenceBackend):
    """Reports a bot that has been in the conference for a known while."""

    AGO_MS = 90_000

    async def join_as_bot(self, room_id, identity, grants):  # type: ignore[no-untyped-def]
        bot = await super().join_as_bot(room_id, identity, grants)
        bot.joined_at = datetime.now(UTC) - timedelta(milliseconds=self.AGO_MS)
        return bot


class _PreAttributesBackend(MockConferenceBackend):
    """A backend written before `mint_access` grew an ``attributes`` argument.

    Stands for every integrator's own backend on the day this parameter
    landed: it keeps working, and only a caller that actually asks for
    attributes meets its refusal.
    """

    async def mint_access(  # type: ignore[no-untyped-def,override]
        self,
        room_id,
        identity,
        grants,
        *,
        display_name=None,
    ):
        return await super().mint_access(room_id, identity, grants, display_name=display_name)


class _NaiveClockBackend(MockConferenceBackend):
    """Reaches for ``datetime.now()`` rather than ``datetime.now(UTC)``."""

    async def join_as_bot(self, room_id, identity, grants):  # type: ignore[no-untyped-def]
        bot = await super().join_as_bot(room_id, identity, grants)
        bot.joined_at = datetime.now()  # noqa: DTZ005 — the mistake under test
        return bot


async def _kit_with_channel(
    backend: MockConferenceBackend | None = None,
    *,
    resolver: IdentityResolver | None = None,
    **channel_kwargs: object,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    backend = backend or MockConferenceBackend()
    # A need is what arms the lazy join (RMK-75): a channel with nothing to
    # consume or say never joins. These tests exercise the join's mechanics,
    # not which need armed it, so a recognizer stands in unless the test
    # brought a need of its own. Pure transport has its own file.
    if not {"stt", "tts", "recording", "realtime"} & channel_kwargs.keys():
        channel_kwargs["stt"] = MockSTTProvider()
    channel = ConferenceChannel("conf", backend=backend, **channel_kwargs)  # type: ignore[arg-type]
    kit = RoomKit(identity_resolver=resolver)
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    return kit, channel, backend


class TestChannelIdentity:
    async def test_channel_type_is_conference(self) -> None:
        _, channel, _ = await _kit_with_channel()

        assert channel.channel_type is ChannelType.CONFERENCE

    async def test_the_binding_announces_audio_alone(self) -> None:
        """Vision is a SHOULD (RFC §12.10.11), so an audio-only conference
        conforms — announcing a media type nothing carries does not.

        Asserted on the binding rather than on ``capabilities()``: the binding
        is the copy routing and transcoding read, and an integrator asking what
        a conference carries reads it too.
        """
        kit, channel, _ = await _kit_with_channel()

        (binding,) = await kit.list_bindings(ROOM)

        assert binding.capabilities.media_types == [ChannelMediaType.AUDIO]
        assert channel.info()["vision_configured"] is False


class TestLifecycle:
    async def test_attach_ensures_the_sfu_room(self) -> None:
        _, _, backend = await _kit_with_channel()

        assert ROOM in backend.rooms

    async def test_bot_join_is_lazy(self) -> None:
        """Nothing connects until there is someone to listen to — a room can be
        created long before anyone confers.
        """
        _, _, backend = await _kit_with_channel()

        assert backend.bots == []

    async def test_first_participant_brings_the_bot_in(self) -> None:
        _, _, backend = await _kit_with_channel()

        await backend.simulate_participant_joined(ROOM, "p-alice")

        assert len(backend.bots) == 1

    async def test_detach_leaves_without_closing_the_room_by_default(self) -> None:
        kit, _, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")

        await kit.detach_channel(ROOM, "conf")

        assert backend.bots == []
        assert ROOM in backend.rooms

    async def test_detach_closes_the_room_when_configured(self) -> None:
        kit, _, backend = await _kit_with_channel(close_room_on_detach=True)
        await backend.simulate_participant_joined(ROOM, "p-alice")

        await kit.detach_channel(ROOM, "conf")

        assert ROOM not in backend.rooms

    async def test_detach_without_a_bot_session_is_harmless(self) -> None:
        """The lazy join may never have happened."""
        kit, _, backend = await _kit_with_channel()

        await kit.detach_channel(ROOM, "conf")

        assert backend.bots == []

    @staticmethod
    async def _end_data(kit: RoomKit, backend: MockConferenceBackend) -> list[dict[str, object]]:
        """Join, detach, and hand back what `conference_ended` carried."""
        seen: list[dict[str, object]] = []

        @kit.on("conference_ended")
        async def _ended(event) -> None:  # type: ignore[no-untyped-def]
            seen.append(event.data)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await kit.detach_channel(ROOM, "conf")
        return seen

    async def test_the_end_reports_how_long_the_bot_was_in_the_conference(self) -> None:
        """RFC §8.2 has `conference_ended` carry `duration_ms`. It is what an
        operator reads to know how long the framework was actually in the
        meeting, which is not the same as how long the meeting ran.

        Measured against a session that joined a known while ago, because a test
        that only asks for a non-negative number is satisfied by a channel that
        always reports nothing at all.
        """
        kit, _, backend = await _kit_with_channel(_JoinedEarlierBackend())

        seen = await self._end_data(kit, backend)

        assert len(seen) == 1
        duration = seen[0]["duration_ms"]
        assert isinstance(duration, int)
        assert _JoinedEarlierBackend.AGO_MS <= duration < _JoinedEarlierBackend.AGO_MS + 5_000

    async def test_a_backend_clock_without_a_timezone_does_not_cost_the_end(self) -> None:
        """`duration_ms` subtracts `joined_at` from an aware now, and Python
        refuses to subtract a naive datetime from an aware one. Raised inside the
        teardown, that TypeError is logged by the hook engine and the detach
        still reports success — so the conference would simply never be
        announced as over, with nothing but a stack trace to say why.
        """
        kit, _, backend = await _kit_with_channel(_NaiveClockBackend())

        seen = await self._end_data(kit, backend)

        assert len(seen) == 1
        assert isinstance(seen[0]["duration_ms"], int)

    async def test_detach_without_a_bot_session_still_closes_the_room(self) -> None:
        """A room nobody ever conferred in is the ordinary case for a meeting
        that was cancelled, and the detach still owes it the same teardown.

        The hook engine logs what a lifecycle hook raises rather than surfacing
        it, so a teardown that fails half way reports a clean detach — and
        leaves the conference room open on the SFU behind it.
        """
        kit, _, backend = await _kit_with_channel(close_room_on_detach=True)

        assert await kit.detach_channel(ROOM, "conf") is True
        assert ROOM not in backend.rooms

    async def test_a_detach_with_no_bot_raises_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        kit, _, _ = await _kit_with_channel()

        with caplog.at_level(logging.ERROR):
            await kit.detach_channel(ROOM, "conf")

        assert [r for r in caplog.records if r.exc_info] == []


class TestBotSelfExclusion:
    async def test_bot_echo_creates_no_participant_record(self) -> None:
        kit, _, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        bot = backend.bots[0]

        await backend.simulate_bot_echo(bot)

        participants = await kit.store.list_participants(ROOM)
        assert [p.id for p in participants] == ["p-alice"]

    async def test_bot_is_excluded_before_its_session_is_registered(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Backends announce participants while the connection is being
        established, the bot's own among them. Recognising it only once a
        session exists leaves a window where the bot is taken for a human.

        And it is a window the channel knows it is in, which is what keeps this
        apart from an identity collision: the echo is expected here, so it is
        excluded without a word.
        """
        kit, _, backend = await _kit_with_channel(_EchoDuringJoinBackend(), bot_identity="roomkit")

        with caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"):
            await backend.simulate_participant_joined(ROOM, "p-alice")

        participants = await kit.store.list_participants(ROOM)
        assert [p.id for p in participants] == ["p-alice"]
        assert caplog.records == []

    async def test_simultaneous_arrivals_open_one_bot_session(self) -> None:
        """Participants arriving together is how a meeting starts; two joins
        would publish the AI on two tracks.
        """
        import asyncio

        _, _, backend = await _kit_with_channel()

        await asyncio.gather(
            backend.simulate_participant_joined(ROOM, "p-alice"),
            backend.simulate_participant_joined(ROOM, "p-bob"),
        )

        assert len(backend.bots) == 1

    async def test_bot_own_track_is_never_subscribed(self) -> None:
        """Otherwise the bot's TTS returns through a lane, the STT transcribes
        the AI's own speech, and the AI answers itself.
        """
        _, _, backend = await _kit_with_channel(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        bot = backend.bots[0]

        echoed = await backend.simulate_bot_echo(bot)

        assert echoed.id not in backend.subscriptions


class TestSelectiveSubscription:
    async def test_audio_track_is_subscribed_when_stt_is_configured(self) -> None:
        _, _, backend = await _kit_with_channel(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")

        track = await backend.simulate_track_published(ROOM, "p-alice", TrackKind.AUDIO)

        assert track.id in backend.subscriptions

    async def test_audio_track_is_not_subscribed_without_a_consumer(self) -> None:
        """A speaking channel is in the meeting for its voice, not for the
        audio: with no stt and no recording, nothing reads what a subscription
        would deliver.
        """
        _, _, backend = await _kit_with_channel(tts=MockTTSProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")

        track = await backend.simulate_track_published(ROOM, "p-alice", TrackKind.AUDIO)

        assert track.id not in backend.subscriptions

    async def test_video_track_is_not_subscribed_without_vision(self) -> None:
        """No video frame reaches the process unless something consumes it —
        this is what keeps a ten-person meeting affordable.
        """
        _, _, backend = await _kit_with_channel(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")

        track = await backend.simulate_track_published(ROOM, "p-alice", TrackKind.VIDEO)

        assert track.id not in backend.subscriptions

    async def test_screen_share_is_not_subscribed_without_vision(self) -> None:
        _, _, backend = await _kit_with_channel(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")

        track = await backend.simulate_track_published(ROOM, "p-alice", TrackKind.SCREEN_SHARE)

        assert track.id not in backend.subscriptions

    async def test_unpublishing_tears_the_lane_down(self) -> None:
        _, channel, backend = await _kit_with_channel(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await backend.simulate_track_unpublished(track.id)

        assert track.id not in channel.active_lanes


class TestParticipantRecords:
    async def test_joining_participant_becomes_a_room_participant(self) -> None:
        kit, _, backend = await _kit_with_channel()

        await backend.simulate_participant_joined(ROOM, "p-alice")

        participants = await kit.store.list_participants(ROOM)
        assert [p.id for p in participants] == ["p-alice"]

    async def test_provider_address_reaches_the_participant_record(self) -> None:
        """A phone participant should reach the same identity it would have
        reached over SMS, so its caller number must survive the crossing — under
        the conference's own key, with the provenance the SFU gave it.
        """
        kit, _, backend = await _kit_with_channel()

        await backend.simulate_participant_joined(
            ROOM, "sip_15551234", metadata={"sip.phoneNumber": "+15551234"}
        )

        participant = (await kit.store.list_participants(ROOM))[0]
        assert participant.external_id == "sip_15551234"
        conference = participant.metadata[CONFERENCE_METADATA_KEY]
        assert conference["asserted"] == {"sip.phoneNumber": "+15551234"}


class TestArrivalWithoutABot:
    """An SFU that refuses the bot must not take the arrival with it.

    RFC §12.10.4 makes recording a participant an unconditional MUST (step 2)
    and the bot join a lazy SHOULD (step 1), which is the order of dependence:
    the person is in the meeting whether or not the framework got its own
    session in. Refusing `join_as_bot` is also the likeliest thing a conference
    does wrong — it is the first network call of the whole meeting — and the
    roster is what the disclosure obligations of §17.7 are read from.
    """

    async def test_a_refused_join_still_puts_the_participant_on_the_roster(self) -> None:
        kit, _, backend = await _kit_with_channel()
        backend.fail("join_as_bot")

        await backend.simulate_participant_joined(ROOM, "p-alice")

        assert [p.id for p in await kit.store.list_participants(ROOM)] == ["p-alice"]

    async def test_a_refused_join_still_identifies_the_arrival(self) -> None:
        """Identification happens on the way into the record, so losing the
        arrival loses the answer with it — the dial-in that would have been
        Alice comes back as nobody.
        """
        alice = Identity(id="user-42", display_name="Alice")
        kit, _, backend = await _kit_with_channel(
            resolver=MockIdentityResolver({"+15551234": alice})
        )
        backend.fail("join_as_bot")

        await backend.simulate_participant_joined(
            ROOM, "sip_15551234", metadata={"sip.phoneNumber": "+15551234"}
        )

        participant = (await kit.store.list_participants(ROOM))[0]
        assert participant.identification is IdentificationStatus.IDENTIFIED
        assert participant.identity_id == "user-42"

    async def test_a_refused_join_still_announces_the_arrival(self) -> None:
        kit, _, backend = await _kit_with_channel()
        fired: list[str] = []
        emitted: list[str] = []

        @kit.hook(HookTrigger.ON_CONFERENCE_PARTICIPANT_JOINED, execution=HookExecution.ASYNC)
        async def _joined(event: object, ctx: object) -> None:
            fired.append("participant_joined")

        @kit.on("conference_participant_joined")
        async def _emitted(event: object) -> None:
            emitted.append("participant_joined")

        backend.fail("join_as_bot")
        await backend.simulate_participant_joined(ROOM, "p-alice")

        assert fired == ["participant_joined"]
        assert emitted == ["participant_joined"]

    async def test_a_refused_join_is_reported_as_no_bot_in_the_room(self) -> None:
        """The disclosure answer must be the true one: nothing of the framework
        is in that conference, and `info()` is where an integrator asks.
        """
        _, channel, backend = await _kit_with_channel(stt=MockSTTProvider())
        backend.fail("join_as_bot")

        await backend.simulate_participant_joined(ROOM, "p-alice")

        room_info = channel.info()["rooms"][ROOM]
        assert room_info["bot_present"] is False
        assert room_info["bot_session_id"] is None
        assert room_info["stt_active"] is False

    async def test_a_conference_that_never_started_is_not_announced(self) -> None:
        """`conference_started` carries the bot session an integrator acts on.
        Announcing one that does not exist is worse than announcing nothing.
        """
        kit, _, backend = await _kit_with_channel()
        announced: list[str] = []

        @kit.on("conference_started")
        async def _started(event: object) -> None:
            announced.append("started")

        backend.fail("join_as_bot")
        await backend.simulate_participant_joined(ROOM, "p-alice")

        assert announced == []

    async def test_a_refused_join_says_so(self, caplog: pytest.LogCaptureFixture) -> None:
        """Swallowed, not silenced. The exception no longer reaches the
        backend's emission loop, which was the only thing logging it, so a
        conference running without transcription or an AI voice would otherwise
        do it without a word.
        """
        _, _, backend = await _kit_with_channel()
        backend.fail("join_as_bot")

        with caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"):
            await backend.simulate_participant_joined(ROOM, "p-alice")

        said = [
            r for r in caplog.records if r.name == "roomkit.channels.conference" and r.exc_info
        ]
        assert len(said) == 1
        assert ROOM in said[0].getMessage()

    async def test_a_later_arrival_brings_the_bot_in(self) -> None:
        """The failure is not sticky: no session was recorded, so the next
        arrival opens one — which is how a conference recovers from an SFU that
        was briefly unreachable.
        """
        kit, _, backend = await _kit_with_channel()
        backend.fail("join_as_bot", times=1)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        assert backend.bots == []

        await backend.simulate_participant_joined(ROOM, "p-bob")

        assert len(backend.bots) == 1
        roster = await kit.store.list_participants(ROOM)
        assert sorted(p.id for p in roster) == ["p-alice", "p-bob"]


class TestTranscription:
    async def test_speech_becomes_an_event_attributed_to_its_speaker(self) -> None:
        """Attribution comes from track identity, which is why a conference
        needs no diarization. The lane's own behaviour is covered in
        test_conference_lane.py.
        """
        kit, channel, backend = await _kit_with_channel(
            stt=MockSTTProvider(transcripts=["bonjour"])
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.subscribe_track(backend.bots[0], track.id)

        await say(backend, track)
        await drain(channel, track.id)

        events = await kit.store.list_events(ROOM)
        spoken = [e for e in events if getattr(e.content, "body", None) == "bonjour"]
        assert len(spoken) == 1
        assert spoken[0].source.participant_id == "p-alice"
        assert spoken[0].metadata["conference_track_id"] == track.id

    async def test_two_speakers_produce_separately_attributed_events(self) -> None:
        _, channel, backend = await _kit_with_channel(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")

        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")

        assert alice.id in channel.active_lanes
        assert bob.id in channel.active_lanes
        assert channel.active_lanes[alice.id] != channel.active_lanes[bob.id]


class TestRosterAfterDeparture:
    async def test_leaving_marks_the_participant_as_left(self) -> None:
        """Firing a hook is not enough: a participant left behind as active
        makes the roster lie to everything that reads it.
        """
        kit, _, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")

        await backend.simulate_participant_left(ROOM, "p-alice")

        participant = (await kit.store.list_participants(ROOM))[0]
        assert participant.status is ParticipantStatus.LEFT

    async def test_re_entry_reactivates_the_participant(self) -> None:
        kit, _, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_left(ROOM, "p-alice")

        await backend.simulate_participant_joined(ROOM, "p-alice")

        participant = (await kit.store.list_participants(ROOM))[0]
        assert participant.status is ParticipantStatus.ACTIVE


class TestDetachedChannelStaysOut:
    async def test_a_participant_joining_after_detach_brings_no_bot_back(self) -> None:
        """Detaching leaves the conference running for the humans in it, so
        callbacks keep arriving. Acting on them would reconnect a bot nobody
        asked for.
        """
        kit, _, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await kit.detach_channel(ROOM, "conf")

        await backend.simulate_participant_joined(ROOM, "p-bob")

        assert backend.bots == []

    async def test_a_track_published_after_detach_is_not_subscribed(self) -> None:
        kit, _, backend = await _kit_with_channel(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await kit.detach_channel(ROOM, "conf")

        track = await backend.simulate_track_published(ROOM, "p-bob")

        assert track.id not in backend.subscriptions


class TestUnsupportedConfiguration:
    async def test_stt_on_an_encrypted_conference_is_refused(self) -> None:
        """RFC 12.10.2: an implementation offering E2EE either admits the bot as
        a key holder or refuses media intelligence. There is no key-holder
        contract in ConferenceBackend, so the bot would subscribe to ciphertext
        and transcribe noise while the configuration read as if it worked.
        """
        with pytest.raises(ValueError, match="end-to-end encrypted"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(capabilities=ConferenceCapability.E2EE),
                stt=MockSTTProvider(),
                e2ee=True,
            )

    async def test_encryption_without_media_intelligence_is_allowed(self) -> None:
        """The refusal is about what the bot would do with the media, not about
        encryption itself: a conference the framework only orchestrates is fine.
        """
        channel = ConferenceChannel(
            "conf",
            backend=MockConferenceBackend(capabilities=ConferenceCapability.E2EE),
            e2ee=True,
        )

        assert channel.info()["e2ee"] is True

    @pytest.mark.parametrize("bound", [0, -1])
    async def test_an_unbounded_lane_queue_is_refused(self, bound: int) -> None:
        """asyncio.Queue treats maxsize<=0 as unbounded, so accepting it would
        turn the lane's documented backpressure into unbounded growth.
        """
        with pytest.raises(ValueError, match="at least 1"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                max_queued_frames=bound,
            )


class TestDisclosureSurface:
    """RFC 17.7: the framework mandates no announcement, but an integrator must
    be able to ask what the bot is and what it is doing with the media — at any
    time, not only when the channel was configured.

    The question a disclosure obligation asks is "is *this meeting* being
    transcribed", so the answer is per conference. A channel serving three rooms
    is configured once and behaves differently in each.
    """

    async def test_info_names_the_bot_and_what_was_configured(self) -> None:
        _, channel, _ = await _kit_with_channel(stt=MockSTTProvider(), bot_identity="notetaker")

        info = channel.info()

        assert info["bot_identity"] == "notetaker"
        assert info["bot_hidden"] is False
        assert info["stt_configured"] is True
        assert info["vision_configured"] is False
        assert info["recording_configured"] is False

    async def test_info_reports_the_bot_arriving_and_leaving(self) -> None:
        kit, channel, backend = await _kit_with_channel(stt=MockSTTProvider())
        assert channel.info()["rooms"][ROOM]["bot_present"] is False

        await backend.simulate_participant_joined(ROOM, "p-alice")
        assert channel.info()["rooms"][ROOM]["bot_present"] is True

        await kit.detach_channel(ROOM, "conf")
        assert ROOM not in channel.info()["rooms"]

    async def test_stt_is_not_active_until_a_track_is_carried(self) -> None:
        """A configured recognizer is not a running one. Nothing is transcribed
        until there is a bot in the room and a lane carrying somebody's track.
        """
        _, channel, backend = await _kit_with_channel(stt=MockSTTProvider())
        assert channel.info()["rooms"][ROOM]["stt_active"] is False

        await backend.simulate_participant_joined(ROOM, "p-alice")
        assert channel.info()["rooms"][ROOM]["stt_active"] is False

        await backend.simulate_track_published(ROOM, "p-alice")
        assert channel.info()["rooms"][ROOM]["stt_active"] is True

    async def test_closing_the_binding_stops_reporting_stt_as_active(self) -> None:
        """A binding closed to Access.NONE stops collection in that room alone,
        which is exactly the difference a channel-wide flag cannot express.
        """
        kit, channel, backend = await _kit_with_channel(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_track_published(ROOM, "p-alice")
        assert channel.info()["rooms"][ROOM]["stt_active"] is True

        await kit.set_access(ROOM, "conf", Access.NONE)
        await asyncio.sleep(0)

        room_info = channel.info()["rooms"][ROOM]
        assert room_info["collecting"] is False
        assert room_info["stt_active"] is False

    async def test_a_channel_with_no_recognizer_never_reports_stt_active(self) -> None:
        _, channel, backend = await _kit_with_channel()

        await backend.simulate_participant_joined(ROOM, "p-alice")

        assert channel.info()["rooms"][ROOM]["stt_active"] is False

    async def test_info_reports_a_hidden_bot_as_hidden(self) -> None:
        _, channel, _ = await _kit_with_channel(bot_grants=ConferenceGrants.observer())

        assert channel.info()["bot_hidden"] is True


class TestHooks:
    async def test_participant_and_track_hooks_fire(self) -> None:
        kit, _, backend = await _kit_with_channel(stt=MockSTTProvider())
        fired: list[str] = []

        @kit.hook(HookTrigger.ON_CONFERENCE_PARTICIPANT_JOINED, execution=HookExecution.ASYNC)
        async def _joined(event: object, ctx: object) -> None:
            fired.append("participant_joined")

        @kit.hook(HookTrigger.ON_CONFERENCE_TRACK_PUBLISHED, execution=HookExecution.ASYNC)
        async def _published(event: object, ctx: object) -> None:
            fired.append("track_published")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_track_published(ROOM, "p-alice")

        assert "participant_joined" in fired
        assert "track_published" in fired

    async def test_screen_share_fires_the_existing_trigger(self) -> None:
        kit, _, backend = await _kit_with_channel()
        fired: list[str] = []

        @kit.hook(HookTrigger.ON_SCREEN_SHARE_STARTED, execution=HookExecution.ASYNC)
        async def _shared(event: object, ctx: object) -> None:
            fired.append("screen_share")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_track_published(ROOM, "p-alice", TrackKind.SCREEN_SHARE)

        assert fired == ["screen_share"]


class TestInterruptionPolicy:
    async def test_scope_none_means_the_bot_is_never_interrupted(self) -> None:
        _, channel, _ = await _kit_with_channel(
            interruption=ConferenceInterruptionConfig(scope=ConferenceInterruptionScope.NONE)
        )

        assert channel.may_interrupt("p-alice") is False

    async def test_scope_any_lets_anyone_interrupt(self) -> None:
        _, channel, _ = await _kit_with_channel()

        assert channel.may_interrupt("p-alice") is True

    async def test_allowlist_scope_admits_only_listed_participants(self) -> None:
        _, channel, _ = await _kit_with_channel(
            interruption=ConferenceInterruptionConfig(
                scope=ConferenceInterruptionScope.ALLOWLIST,
                allowlist=["p-moderator"],
            )
        )

        assert channel.may_interrupt("p-moderator") is True
        assert channel.may_interrupt("p-alice") is False


class TestBotGrants:
    """What the bot asks the SFU for is derived, not defaulted.

    The permissive ConferenceGrants defaults exist because the framework cannot
    know what a human will do. It knows what it configured the bot to do.
    """

    async def _joined_with(self, **channel_kwargs: object) -> ConferenceGrants:
        _, _, backend = await _kit_with_channel(**channel_kwargs)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        join = next(call for call in backend.calls if call.method == "join_as_bot")
        grants: ConferenceGrants = join.args["grants"]
        return grants

    async def test_a_bot_with_no_synthesizer_may_not_publish_audio(self) -> None:
        grants = await self._joined_with()

        assert grants.publish_audio is False

    async def test_a_bot_with_a_synthesizer_may_publish_audio(self) -> None:
        grants = await self._joined_with(tts=MockTTSProvider())

        assert grants.publish_audio is True

    async def test_the_bot_never_asks_for_video_or_a_screen_it_does_not_have(self) -> None:
        grants = await self._joined_with(tts=MockTTSProvider())

        assert grants.publish_video is False
        assert grants.publish_screen_share is False

    async def test_a_bot_that_consumes_tracks_subscribes(self) -> None:
        grants = await self._joined_with(stt=MockSTTProvider())

        assert grants.subscribe is True

    async def test_a_bot_with_nothing_to_consume_does_not_subscribe(self) -> None:
        """A speaking channel with no recognizer subscribes to no track at all
        (``_consumes``), so the grant is permission to receive every
        participant's media for nobody to read.
        """
        grants = await self._joined_with(tts=MockTTSProvider())

        assert grants.subscribe is False

    async def test_explicit_grants_win_over_the_derivation(self) -> None:
        grants = await self._joined_with(bot_grants=ConferenceGrants())

        assert grants.publish_screen_share is True


async def _join_settled(channel: ConferenceChannel) -> None:
    """Wait out the room's background work — the mint's spawned join among it."""
    room = channel._room(ROOM)
    while room.tasks:
        await asyncio.wait(list(room.tasks), timeout=5.0)


class TestMintBootstrapsTheBot:
    """A credential going out is what makes the first join happen (RMK-68).

    Against a real SFU, presence is observable only through a connection: no
    participant or track callback fires before the bot holds one, so no
    callback can make the *first* join happen (RFC §12.10.3). The mint is the
    one trigger the backend cannot withhold. No test here touches
    ``simulate_*`` before asserting on the join — that absence is the point:
    silence is exactly what a real SFU gives a channel that has not joined yet.
    """

    async def test_a_mint_alone_brings_the_bot_in(self) -> None:
        """The main use case of a conference: humans speak, the AI listens.

        Without this trigger the bot only ever entered behind a backend
        callback or a delivery — so a meeting the framework never spoke into
        was never joined, and never transcribed.
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        announced: list[str] = []

        @kit.on("conference_started")
        async def _started(event: object) -> None:
            announced.append("started")

        await channel.mint_access(ROOM, "p-alice")
        await _join_settled(channel)

        assert len(backend.bots) == 1
        assert announced == ["started"]

    async def test_the_join_never_delays_or_fails_the_mint(self) -> None:
        """The credential belongs to the participant whether or not the
        framework got its own session into the room (RFC §12.10.4).
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        backend.fail("join_as_bot", times=1)

        access = await channel.mint_access(ROOM, "p-alice")
        await _join_settled(channel)

        assert access.token
        assert backend.bots == []

    async def test_a_later_mint_tries_the_join_again(self) -> None:
        """Nothing is retried on a timer; the next need finds ``room.bot``
        unset and tries again — and a second admission is a next need.
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await kit.ensure_participant(ROOM, "conf", "p-bob")
        backend.fail("join_as_bot", times=1)
        await channel.mint_access(ROOM, "p-alice")
        await _join_settled(channel)

        await channel.mint_access(ROOM, "p-bob")
        await _join_settled(channel)

        assert len(backend.bots) == 1

    async def test_a_second_mint_does_not_join_twice(self) -> None:
        """Two admissions, one bot: a second join would publish the AI on two
        tracks.
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await kit.ensure_participant(ROOM, "conf", "p-bob")

        await channel.mint_access(ROOM, "p-alice")
        await _join_settled(channel)
        await channel.mint_access(ROOM, "p-bob")
        await _join_settled(channel)

        assert len([c for c in backend.calls if c.method == "join_as_bot"]) == 1

    async def test_a_refused_mint_starts_no_join(self) -> None:
        """A refused credential admits nobody, so nobody is about to arrive."""
        _, channel, backend = await _kit_with_channel()

        with pytest.raises(ParticipantNotFoundError):
            await channel.mint_access(ROOM, "p-typo")
        await _join_settled(channel)

        assert backend.bots == []
        assert not [c for c in backend.calls if c.method == "join_as_bot"]

    async def test_a_detach_racing_the_spawned_join_leaves_no_bot(self) -> None:
        """The join is a room task: a detach cancels it or takes its bot out —
        either interleaving ends with nobody left in the meeting.
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        await channel.mint_access(ROOM, "p-alice")
        await kit.detach_channel(ROOM, "conf")
        await _settle(channel)

        assert backend.bots == []

    async def test_a_failed_join_after_a_mint_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Swallowed, not silenced: a conference running without transcription
        or an AI voice must not do it without a word.
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        backend.fail("join_as_bot")

        with caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"):
            await channel.mint_access(ROOM, "p-alice")
            await _join_settled(channel)

        assert any("after minting" in record.message for record in caplog.records)


class TestAttachResumesALiveConference:
    """An attach over a conference already underway is itself a first need (RMK-71).

    The mint bootstraps a conference nobody has been admitted to yet; it
    cannot resume one already running. A channel restarted mid-meeting
    re-attaches above participants an earlier life admitted, and every other
    trigger is out of reach: the re-join supervisor died with the process, no
    callback can arrive without a connection (RFC §12.10.3), and the humans
    already in the room may never mint again nor be delivered to. So the
    attach probes the conference's occupancy — ``list_participants()`` is
    control-plane and needs no connection — and a non-empty answer starts the
    lazy join (RFC §12.10.4 step 1).
    """

    @staticmethod
    def _occupied(backend: MockConferenceBackend, *identities: str) -> None:
        """Put participants in the SFU's conference before the channel attaches.

        Written straight into the backend's state, not simulated through its
        callbacks: this is what a restart leaves behind — people connected to
        a conference nobody on this side of it has ever observed.
        """
        backend.participants[ROOM] = {
            identity: ConferenceParticipant(participant_id=identity) for identity in identities
        }

    async def test_an_attach_over_a_live_conference_brings_the_bot_in(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The service restarted mid-meeting; the meeting must get its bot back.

        And it says why: a bot joining a meeting where nobody minted,
        delivered or spoke looks spontaneous unless the probe names its
        reason.
        """
        backend = MockConferenceBackend()
        self._occupied(backend, "p-alice")
        channel = ConferenceChannel("conf", backend=backend, stt=MockSTTProvider())
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        announced: list[str] = []

        @kit.on("conference_started")
        async def _started(event: object) -> None:
            announced.append("started")

        with caplog.at_level(logging.INFO, logger="roomkit.channels.conference"):
            await kit.attach_channel(ROOM, "conf")
            await _join_settled(channel)

        assert len(backend.bots) == 1
        assert announced == ["started"]
        assert any("already in room" in record.message for record in caplog.records)

    async def test_an_attach_over_an_empty_conference_stays_lazy(self) -> None:
        """A room nobody confers in costs one control-plane call and nothing
        more (RFC §12.10.4 step 1)."""
        _, channel, backend = await _kit_with_channel()
        await _join_settled(channel)

        assert len([c for c in backend.calls if c.method == "list_participants"]) == 1
        assert not [c for c in backend.calls if c.method == "join_as_bot"]
        assert backend.bots == []

    async def test_a_conference_holding_only_a_stale_bot_is_not_occupied(self) -> None:
        """A restart can leave the previous process's bot sitting in the SFU;
        a session an earlier life left behind is not occupancy.
        """
        backend = MockConferenceBackend()
        self._occupied(backend, "ai-bot")
        _, channel, backend = await _kit_with_channel(backend, bot_identity="ai-bot")
        await _join_settled(channel)

        assert backend.bots == []
        assert not [c for c in backend.calls if c.method == "join_as_bot"]

    async def test_a_failed_probe_never_fails_the_attach(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The probe's failure is never the attach's: the binding stands, and
        the lazy join remains for the next mint, delivery or arrival.
        """
        backend = MockConferenceBackend()
        self._occupied(backend, "p-alice")
        backend.fail("list_participants")

        with caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"):
            kit, channel, _ = await _kit_with_channel(backend)
            await _join_settled(channel)

        assert channel._room(ROOM).attached
        assert backend.bots == []
        assert any("could not ask who is in" in record.message for record in caplog.records)

    async def test_a_failed_join_after_a_positive_probe_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Swallowed, not silenced: a meeting resuming untranscribed must not
        do it without a word.
        """
        backend = MockConferenceBackend()
        self._occupied(backend, "p-alice")
        backend.fail("join_as_bot")

        with caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"):
            _, channel, _ = await _kit_with_channel(backend)
            await _join_settled(channel)

        assert backend.bots == []
        assert any("could not bring its bot in" in record.message for record in caplog.records)

    async def test_a_detach_racing_the_probe_leaves_no_bot(self) -> None:
        """The probe is a room task: a detach cancels it or takes its bot out —
        either interleaving ends with nobody left in the meeting.
        """
        backend = MockConferenceBackend()
        self._occupied(backend, "p-alice")
        kit, channel, _ = await _kit_with_channel(backend)

        await kit.detach_channel(ROOM, "conf")
        await _settle(channel)

        assert backend.bots == []


class TestSpeechEdges:
    """The lanes announce the VAD's utterance boundaries (RMK-73).

    ``ON_SPEECH_START`` / ``ON_SPEECH_END`` per participant and track — the
    real-time "who is speaking right now" a management interface reads (RFC
    §12.10.4). The SFU's dominant-speaker signal cannot say that nobody is
    speaking, and the transcription arrives only after the recognizer's
    round trip.
    """

    async def test_a_lane_announces_both_edges_of_an_utterance(self) -> None:
        kit, channel, backend = await _kit_with_channel(stt=MockSTTProvider())
        edges: list[tuple[str, str, str]] = []

        @kit.hook(HookTrigger.ON_SPEECH_START)
        async def started(event: Any, ctx: Any) -> None:
            data = event.content.data
            edges.append(("start", data["participant_id"], data["track_id"]))

        @kit.hook(HookTrigger.ON_SPEECH_END)
        async def ended(event: Any, ctx: Any) -> None:
            data = event.content.data
            edges.append(("end", data["participant_id"], data["track_id"]))

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await say(backend, track)
        await drain(channel, track.id)

        assert edges == [("start", "p-alice", track.id), ("end", "p-alice", track.id)]

    async def test_the_end_is_announced_before_the_transcription(self) -> None:
        """ "They stopped speaking" is true the moment the VAD closes the
        utterance; recognition is a round trip that has not happened yet.
        """
        kit, channel, backend = await _kit_with_channel(stt=MockSTTProvider())
        order: list[str] = []

        @kit.hook(HookTrigger.ON_SPEECH_END)
        async def ended(event: Any, ctx: Any) -> None:
            order.append("end")

        @kit.hook(HookTrigger.AFTER_BROADCAST)
        async def heard(event: Any, ctx: Any) -> None:
            order.append("heard")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await say(backend, track)
        await drain(channel, track.id)
        await _settle(channel)

        assert order == ["end", "heard"]


class TestConnectionQualityRelay:
    """The SFU's view of a participant's connection reaches its hook (RMK-73).

    Not collection — no media is read to relay it — so it is not gated by the
    binding's collection state, exactly like the active-speaker signal (RFC
    §12.10.4). A quality bar in a management interface is the consumer.
    """

    async def test_the_quality_report_reaches_its_hook(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        seen: list[tuple[str, str]] = []

        @kit.hook(HookTrigger.ON_CONNECTION_QUALITY_CHANGED)
        async def quality(event: Any, ctx: Any) -> None:
            data = event.content.data
            seen.append((data["participant_id"], data["quality"]))

        await backend.simulate_connection_quality(ROOM, "p-alice", "poor")
        await _settle(channel)

        assert seen == [("p-alice", "poor")]

    async def test_a_detached_room_relays_nothing(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        seen: list[str] = []

        @kit.hook(HookTrigger.ON_CONNECTION_QUALITY_CHANGED)
        async def quality(event: Any, ctx: Any) -> None:
            seen.append(event.content.data["participant_id"])

        await kit.detach_channel(ROOM, "conf")
        await _settle(channel)
        await backend.simulate_connection_quality(ROOM, "p-alice", "poor")

        assert seen == []


class TestTrackMuteRelay:
    """A publisher's mute reaches its hooks, kind included (RFC §12.10.4).

    Presence, not media: most clients express a camera toggle as a muted
    VIDEO track rather than an unpublish, so microphone and camera
    indicators both read from this pair and the track's kind — no
    subscription required, which is what keeps a camera indicator free on
    a channel that consumes no video.
    """

    async def test_a_mute_and_unmute_reach_their_hooks(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        seen: list[tuple[str, str, str]] = []

        @kit.hook(HookTrigger.ON_CONFERENCE_TRACK_MUTED)
        async def muted(event: Any, ctx: Any) -> None:
            data = event.content.data
            seen.append(("muted", data["participant_id"], data["kind"]))

        @kit.hook(HookTrigger.ON_CONFERENCE_TRACK_UNMUTED)
        async def unmuted(event: Any, ctx: Any) -> None:
            data = event.content.data
            seen.append(("unmuted", data["participant_id"], data["kind"]))

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice", TrackKind.VIDEO)
        await backend.simulate_track_muted(track.id)
        await backend.simulate_track_unmuted(track.id)
        await _settle(channel)

        assert seen == [("muted", "p-alice", "video"), ("unmuted", "p-alice", "video")]

    async def test_a_moderation_mute_is_reported_like_any_other(self) -> None:
        """The SFU observes its own moderation; the room hears one story."""
        kit, channel, backend = await _kit_with_channel()
        seen: list[str] = []

        @kit.hook(HookTrigger.ON_CONFERENCE_TRACK_MUTED)
        async def muted(event: Any, ctx: Any) -> None:
            seen.append(event.content.data["track_id"])

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.mute_track(ROOM, track.id)
        await _settle(channel)

        assert seen == [track.id]

    async def test_the_bots_own_track_mute_is_not_relayed(self) -> None:
        kit, channel, backend = await _kit_with_channel(tts=MockTTSProvider())
        seen: list[str] = []

        @kit.hook(HookTrigger.ON_CONFERENCE_TRACK_MUTED)
        async def muted(event: Any, ctx: Any) -> None:
            seen.append(event.content.data["track_id"])

        await backend.simulate_participant_joined(ROOM, "p-alice")
        echoed = await backend.simulate_bot_echo(backend.bots[0])
        await backend.simulate_track_muted(echoed.id)
        await _settle(channel)

        assert seen == []

    async def test_a_detached_room_relays_no_mute(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        seen: list[str] = []

        @kit.hook(HookTrigger.ON_CONFERENCE_TRACK_MUTED)
        async def muted(event: Any, ctx: Any) -> None:
            seen.append(event.content.data["track_id"])

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await kit.detach_channel(ROOM, "conf")
        await _settle(channel)
        await backend.simulate_track_muted(track.id)

        assert seen == []


class TestDisplayNameRidesTheCredential:
    """The name the room gave a participant travels with the mint (RMK-73).

    Presentation, never identity (RFC §12.10.3): attribution rides the
    participant id alone. The SFU renders the name, reports it back on its
    participants, and a roster record that has none takes it — which is how
    a roster rebuilt from the join's catch-up gets its names back after a
    restart.
    """

    async def test_the_mint_carries_the_rooms_name(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice", display_name="Alice")

        await channel.mint_access(ROOM, "p-alice")

        mint = [c for c in backend.calls if c.method == "mint_access"][-1]
        assert mint.args["display_name"] == "Alice"

    async def test_a_nameless_record_mints_nameless(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        await channel.mint_access(ROOM, "p-alice")

        mint = [c for c in backend.calls if c.method == "mint_access"][-1]
        assert mint.args["display_name"] is None

    async def test_a_reported_name_fills_an_empty_roster_record(self) -> None:
        """A dial-in — or a roster rebuilt from catch-up after a restart —
        arrives carrying the SFU's name, with no record of its own to meet.
        """
        kit, channel, backend = await _kit_with_channel()

        await backend.simulate_participant_joined(ROOM, "p-dialin", display_name="Bob Landry")

        participant = await kit.store.get_participant(ROOM, "p-dialin")
        assert participant is not None
        assert participant.display_name == "Bob Landry"

    async def test_a_reported_name_never_overwrites_the_integrators(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice", display_name="Alice")

        await backend.simulate_participant_joined(ROOM, "p-alice", display_name="alice2")

        participant = await kit.store.get_participant(ROOM, "p-alice")
        assert participant is not None
        assert participant.display_name == "Alice"


class TestMintedAttributesRideTheCredential:
    """The identity is no longer the only field that travels (RMK-110).

    An integrator whose own clients must be told *who* is behind a channel
    identity had nowhere to put it, so the pressure was to encode meaning in
    the identity itself — which is the separation `identity_id` exists to keep.
    What travels now is what the caller passed, and nothing else.
    """

    async def test_the_mint_carries_what_the_caller_passed(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        await channel.mint_access(ROOM, "p-alice", attributes={"app.user": "user-42"})

        mint = [c for c in backend.calls if c.method == "mint_access"][-1]
        assert mint.args["attributes"] == {"app.user": "user-42"}

    async def test_a_mint_that_asks_for_nothing_carries_nothing(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        await channel.mint_access(ROOM, "p-alice")

        mint = [c for c in backend.calls if c.method == "mint_access"][-1]
        assert mint.args["attributes"] is None

    async def test_the_channel_adds_nothing_the_caller_did_not_ask_for(self) -> None:
        """It knows the participant's ``identity_id`` and could mint it unasked.

        That would solve every host's problem at once and publish the platform
        identity of everyone in the room to every peer of a conference that may
        be pseudonymous. Opt-in per mint is the whole design (RFC §12.10.3).
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        record = await kit.store.get_participant(ROOM, "p-alice")
        assert record is not None
        await kit.store.update_participant(record.model_copy(update={"identity_id": "user-42"}))

        await channel.mint_access(ROOM, "p-alice")

        mint = [c for c in backend.calls if c.method == "mint_access"][-1]
        assert mint.args["attributes"] is None

    async def test_they_come_back_on_the_roster_unasserted(self) -> None:
        """Readable and renderable, and unable to found an identity: they rode
        a token, which is not a thing an SFU established (RFC §12.10.2 rule 1).
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await channel.mint_access(ROOM, "p-alice", attributes={"app.user": "user-42"})

        await backend.simulate_participant_joined(ROOM, "p-alice")

        record = await kit.store.get_participant(ROOM, "p-alice")
        assert record is not None
        provider = record.metadata[CONFERENCE_METADATA_KEY]
        assert provider["unasserted"]["app.user"] == "user-42"
        assert "app.user" not in provider["asserted"]

    async def test_a_backend_from_before_the_parameter_still_mints(self) -> None:
        """The parameter is optional on the way in and on the way down: a
        backend written before it existed is never handed it.
        """
        kit, channel, backend = await _kit_with_channel(_PreAttributesBackend())
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        access = await channel.mint_access(ROOM, "p-alice")

        assert access.token


class TestWhatAMintRefusesToCarry:
    """The bound of RFC §12.10.3: a credential must not carry what the room
    would refuse to persist when the SFU reported it back. It refuses rather
    than truncates, because here the caller is the integrator — and an
    attribute silently missing from a token is met in the browser, later.
    """

    @pytest.mark.parametrize(
        ("attributes", "message"),
        [
            ({f"k{index}": "v" for index in range(MAX_ATTRIBUTES + 1)}, "at most"),
            ({"k" * (MAX_KEY_CHARS + 1): "v"}, "characters"),
            ({"big": "x" * (MAX_VALUE_CHARS + 1)}, "longer than"),
            ({"count": 3}, "carries strings"),
        ],
        ids=["too-many", "long-key", "long-value", "not-a-string"],
    )
    async def test_an_unmintable_attribute_is_refused(
        self, attributes: dict[str, Any], message: str
    ) -> None:
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        with pytest.raises(ValueError, match=message):
            await channel.mint_access(ROOM, "p-alice", attributes=attributes)

    async def test_the_refusal_lands_before_the_backend_is_asked(self) -> None:
        """A mint the channel will refuse over its own argument never reaches
        the SFU, so there is no credential for an operator to wonder about.
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        with pytest.raises(ValueError):
            await channel.mint_access(ROOM, "p-alice", attributes={"count": 3})

        assert [c for c in backend.calls if c.method == "mint_access"] == []


class TestAccessMinting:
    async def test_channel_mints_access_with_its_default_grants(self) -> None:
        kit, channel, backend = await _kit_with_channel(default_grants=ConferenceGrants.observer())
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        access = await channel.mint_access(ROOM, "p-alice")

        assert access.token
        assert backend.calls[-1].args["grants"].hidden is True

    async def test_caller_can_override_the_grants(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")

        await channel.mint_access(ROOM, "p-alice", grants=ConferenceGrants.observer())

        assert backend.calls[-1].args["grants"].hidden is True

    async def test_a_detached_room_mints_nothing(self) -> None:
        """The credential outlives the check, so a room the channel has left
        must not be admitting anyone.
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-alice")
        await kit.detach_channel(ROOM, "conf")

        with pytest.raises(RoomNotAttachedError):
            await channel.mint_access(ROOM, "p-alice")

        assert not [call for call in backend.calls if call.method == "mint_access"]

    async def test_an_unknown_participant_mints_nothing(self) -> None:
        """A mistyped identifier otherwise produces a perfectly valid token for
        an identity with no place in the room (RFC §12.10.2).
        """
        _, channel, backend = await _kit_with_channel()

        with pytest.raises(ParticipantNotFoundError):
            await channel.mint_access(ROOM, "p-typo")

        assert not [call for call in backend.calls if call.method == "mint_access"]

    async def test_a_participant_who_left_may_be_readmitted(self) -> None:
        """Departure is a status on the roster, not a removal, and rejoining a
        conference is ordinary.
        """
        kit, channel, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_left(ROOM, "p-alice")

        await channel.mint_access(ROOM, "p-alice")

        assert (await kit.store.list_participants(ROOM))[0].status is ParticipantStatus.LEFT
        assert backend.calls[-1].method == "mint_access"

    async def test_a_banned_participant_is_refused(self) -> None:
        """``BANNED`` is "removed and blocked" (RFC §5.5), and a conference
        credential is exactly the thing a block has to reach: the SFU honours
        what it minted, and the backend contract offers no revocation.

        Being on the roster was the whole of the check, and a ban leaves the
        record in place — so the one status that means "no" read as "yes".
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-mallory")
        await kit.remove_member(ROOM, "p-mallory", status=ParticipantStatus.BANNED)

        with pytest.raises(ParticipantNotAdmittedError):
            await channel.mint_access(ROOM, "p-mallory")

        assert not [call for call in backend.calls if call.method == "mint_access"]

    async def test_a_ban_is_undone_by_readmitting_them(self) -> None:
        """The refusal is a status, not a sentence: a room that lets someone
        back in mints for them again.
        """
        kit, channel, backend = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-mallory")
        await kit.remove_member(ROOM, "p-mallory", status=ParticipantStatus.BANNED)

        await kit.add_member(ROOM, "conf", "p-mallory")
        await channel.mint_access(ROOM, "p-mallory")

        assert backend.calls[-1].method == "mint_access"

    async def test_a_refused_mint_leaves_no_request_behind(self) -> None:
        """The refusal happens before the backend is asked for anything, so
        there is no in-flight credential for a teardown to have to take back.

        (A ban that lands *during* a mint is a different guarantee, and it is
        held down in test_conference_races.py.)
        """
        kit, channel, _ = await _kit_with_channel()
        await kit.ensure_participant(ROOM, "conf", "p-mallory")
        await kit.remove_member(ROOM, "p-mallory", status=ParticipantStatus.BANNED)

        with pytest.raises(ParticipantNotAdmittedError):
            await channel.mint_access(ROOM, "p-mallory")

        assert channel._room(ROOM).mints == set()


class TestBotIdentityCollision:
    """A participant carrying the bot's identity cannot be told apart from it.

    The channel recognises its own bot by identity, which is what closes the
    window before there is a session (above). The counterpart is that a *human*
    on that identity is excluded too — from the roster, the hooks and the
    transcript. That exclusion stands, because treating them as a human is how
    the AI ends up transcribing itself. What must not stand is doing it in
    silence.
    """

    async def test_the_bot_identity_is_refused_at_the_door(self) -> None:
        """The one place the collision can still be prevented rather than
        reported: the framework names the participants it mints for.
        """
        kit, channel, backend = await _kit_with_channel(bot_identity="roomkit")
        await kit.ensure_participant(ROOM, "conf", "roomkit")

        with pytest.raises(ValueError, match="reserved"):
            await channel.mint_access(ROOM, "roomkit")

        assert not [call for call in backend.calls if call.method == "mint_access"]

    async def test_a_participant_on_the_bot_identity_is_excluded_but_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        kit, _, backend = await _kit_with_channel(bot_identity="roomkit")
        fired: list[str] = []

        @kit.hook(HookTrigger.ON_CONFERENCE_PARTICIPANT_JOINED, execution=HookExecution.ASYNC)
        async def _joined(event: object, ctx: object) -> None:
            fired.append("joined")

        with caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"):
            await backend.simulate_participant_joined(ROOM, "roomkit")
            await backend.simulate_participant_joined(ROOM, "p-alice")

        participants = await kit.store.list_participants(ROOM)
        assert [p.id for p in participants] == ["p-alice"]
        assert fired == ["joined"]
        assert len(caplog.records) == 1
        assert "roomkit" in caplog.records[0].getMessage()

    async def test_the_collision_is_reported_once_per_room(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A participant that keeps being refused goes on publishing. The point
        is to be heard once, not to fill the log with the same line.
        """
        _, _, backend = await _kit_with_channel(stt=MockSTTProvider(), bot_identity="roomkit")

        with caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"):
            await backend.simulate_participant_joined(ROOM, "roomkit")
            await backend.simulate_track_published(ROOM, "roomkit")
            await backend.simulate_track_published(ROOM, "roomkit")

        assert len(caplog.records) == 1


@pytest.mark.parametrize("kind", [TrackKind.VIDEO, TrackKind.SCREEN_SHARE])
async def test_video_kinds_never_subscribe_without_a_consumer(kind: TrackKind) -> None:
    _, _, backend = await _kit_with_channel(stt=MockSTTProvider())
    await backend.simulate_participant_joined(ROOM, "p-alice")

    track = await backend.simulate_track_published(ROOM, "p-alice", kind)

    assert track.id not in backend.subscriptions


class TestABanSurvivesTheSFU:
    """A ban is a decision the room took; the SFU's own events do not undo it.

    A banned participant disconnects — by definition, on their way out — and the
    departure that follows was written straight over the ban as ``LEFT``, which
    is admissible. So the SFU's report of the thing the ban caused became the
    thing that lifted it: an authorization bypass reached by waiting.
    """

    async def test_a_delayed_sfu_departure_does_not_lift_a_ban(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-mallory")
        await kit.remove_member(ROOM, "p-mallory", status=ParticipantStatus.BANNED)

        await backend.simulate_participant_left(ROOM, "p-mallory")

        participant = await kit.store.get_participant(ROOM, "p-mallory")
        assert participant is not None
        assert participant.status is ParticipantStatus.BANNED
        with pytest.raises(ParticipantNotAdmittedError):
            await channel.mint_access(ROOM, "p-mallory")

    async def test_a_delayed_sfu_arrival_does_not_lift_a_ban(self) -> None:
        """The other direction: the SFU reporting them connected is not the room
        changing its mind either.
        """
        kit, channel, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-mallory")
        await kit.remove_member(ROOM, "p-mallory", status=ParticipantStatus.BANNED)

        await backend.simulate_participant_joined(ROOM, "p-mallory")

        participant = await kit.store.get_participant(ROOM, "p-mallory")
        assert participant is not None
        assert participant.status is ParticipantStatus.BANNED
        with pytest.raises(ParticipantNotAdmittedError):
            await channel.mint_access(ROOM, "p-mallory")

    async def test_an_ordinary_departure_is_still_recorded(self) -> None:
        """And the guard is not "never write": a participant who simply leaves
        is still marked as having left.
        """
        kit, _, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")

        await backend.simulate_participant_left(ROOM, "p-alice")

        participant = await kit.store.get_participant(ROOM, "p-alice")
        assert participant is not None
        assert participant.status is ParticipantStatus.LEFT


class TestOneConferencePerRoom:
    """RFC 12.10.1 principle 2: a conference maps 1:1 to a Room, both ways.
    A second conference channel is a second bot, a second transcription of
    every utterance and a second AI voice — so the attach refuses it.
    """

    async def test_a_second_conference_channel_is_refused(self) -> None:
        backend_a = MockConferenceBackend()
        backend_b = MockConferenceBackend()
        kit = RoomKit()
        kit.register_channel(ConferenceChannel("conf-a", backend=backend_a, stt=MockSTTProvider()))
        kit.register_channel(ConferenceChannel("conf-b", backend=backend_b, stt=MockSTTProvider()))
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf-a")

        with pytest.raises(ConferenceAlreadyAttachedError) as refusal:
            await kit.attach_channel(ROOM, "conf-b")

        assert "conf-a" in str(refusal.value)
        # The refusal came before anything reached the second backend: no
        # duplicate SFU room was created for the same RoomKit room.
        assert ROOM not in backend_b.rooms
        assert [b.channel_id for b in await kit.store.list_bindings(ROOM)] == ["conf-a"]

    async def test_reattaching_the_same_conference_channel_stays_ordinary(self) -> None:
        kit = RoomKit()
        kit.register_channel(
            ConferenceChannel("conf", backend=MockConferenceBackend(), stt=MockSTTProvider())
        )
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await kit.attach_channel(ROOM, "conf")

        assert [b.channel_id for b in await kit.store.list_bindings(ROOM)] == ["conf"]

    async def test_a_detached_conference_frees_the_slot(self) -> None:
        backend_b = MockConferenceBackend()
        kit = RoomKit()
        kit.register_channel(
            ConferenceChannel("conf-a", backend=MockConferenceBackend(), stt=MockSTTProvider())
        )
        kit.register_channel(ConferenceChannel("conf-b", backend=backend_b, stt=MockSTTProvider()))
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf-a")
        await kit.detach_channel(ROOM, "conf-a")

        await kit.attach_channel(ROOM, "conf-b")

        assert ROOM in backend_b.rooms


class TestASpontaneousBotDisconnect:
    """RFC 12.10.3: a backend that observes the SFU ending the bot's session
    without a leave() reports it, and the channel treats the report as the
    session's end in fact — off the books, announced, re-joinable.
    """

    async def test_the_loss_comes_off_the_books_and_is_announced(self) -> None:
        kit, channel, backend = await _kit_with_channel()
        ended: list[dict[str, object]] = []

        @kit.on("conference_ended")
        async def _ended(event) -> None:  # type: ignore[no-untyped-def]
            ended.append(event.data)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        bot = channel._room(ROOM).bot
        assert bot is not None

        await backend.simulate_bot_disconnected(bot, "signalling connection lost")

        assert channel.info()["rooms"][ROOM]["bot_present"] is False
        assert len(ended) == 1

    async def test_the_next_need_rejoins_lazily(self) -> None:
        _, channel, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        first = channel._room(ROOM).bot
        assert first is not None

        await backend.simulate_bot_disconnected(first)
        assert backend.bots == []

        await backend.simulate_participant_joined(ROOM, "p-bob")

        assert len(backend.bots) == 1
        second = channel._room(ROOM).bot
        assert second is not None
        assert second.id != first.id

    async def test_the_lost_sessions_recordings_are_finalized(self) -> None:
        from roomkit import ConferenceRecordingConfig
        from roomkit.recorder.mock import MockMediaRecorder
        from tests.conference.lane_audio import speech_frame
        from tests.conference.test_conference_races import _until

        recorder = MockMediaRecorder()
        backend = MockConferenceBackend()
        channel = ConferenceChannel(
            "conf",
            backend=backend,
            recording=ConferenceRecordingConfig(),
            recorder=recorder,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await _until(lambda: recorder.chunks != [])
        bot = channel._room(ROOM).bot
        assert bot is not None

        await backend.simulate_bot_disconnected(bot)

        assert len(recorder.results) == 1, "the lost session left its recording open"

    async def test_a_stale_report_corrects_nothing(self) -> None:
        """A session already replaced by a re-join is not the current bot; a
        report about it must not take the new session off the books.
        """
        _, channel, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        first = channel._room(ROOM).bot
        assert first is not None
        await backend.simulate_bot_disconnected(first)
        await backend.simulate_participant_joined(ROOM, "p-bob")
        second = channel._room(ROOM).bot
        assert second is not None

        await backend.simulate_bot_disconnected(first, "a very late report")

        assert channel._room(ROOM).bot is second
        assert channel.info()["rooms"][ROOM]["bot_present"] is True


class _GatedLeaveBackend(MockConferenceBackend):
    """Holds leave() open so a teardown can be caught mid-flight."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.leaving = asyncio.Event()

    async def leave(self, bot):  # type: ignore[no-untyped-def]
        self.leaving.set()
        await self.gate.wait()
        return await super().leave(bot)


class TestTheConferenceSlotOutlivesTheBinding:
    """RFC 12.10.4: the one-conference reservation holds for as long as the
    previous channel still holds the room — a session in the meeting, a
    teardown still running — not merely while its binding exists. A detach
    removes the binding first, which is exactly the window this closes.
    """

    async def test_a_session_left_behind_by_a_detach_holds_the_slot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A detach whose leave() the budget abandoned leaves its bot in the
        meeting, on the leaving ledger. The binding is long gone; the slot is
        not free until the session is.
        """
        from roomkit.channels import _conference_activity as activity_module

        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        monkeypatch.setattr(activity_module, "CANCEL_GRACE_S", 0.05)
        backend_a = _GatedLeaveBackend()
        backend_b = MockConferenceBackend()
        kit = RoomKit()
        kit.register_channel(ConferenceChannel("conf-a", backend=backend_a, stt=MockSTTProvider()))
        kit.register_channel(ConferenceChannel("conf-b", backend=backend_b, stt=MockSTTProvider()))
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf-a")
        await backend_a.simulate_participant_joined(ROOM, "p-alice")

        try:
            await asyncio.wait_for(kit.detach_channel(ROOM, "conf-a"), timeout=5.0)

            assert backend_a.bots != [], "the leave was supposed to be abandoned mid-flight"
            with pytest.raises(ConferenceAlreadyAttachedError) as refusal:
                await kit.attach_channel(ROOM, "conf-b")
            assert "conf-a" in str(refusal.value)
            assert ROOM not in backend_b.rooms
        finally:
            backend_a.gate.set()

    async def test_a_deferred_teardown_holds_the_slot_too(self) -> None:
        """A detach issued from inside an announcement defers its destructive
        phase onto a task; the room has no binding and no visible detach in
        flight, and the old bot is still in the meeting.
        """
        backend_a = MockConferenceBackend()
        backend_b = MockConferenceBackend()
        kit = RoomKit()
        channel_a = ConferenceChannel("conf-a", backend=backend_a, stt=MockSTTProvider())
        kit.register_channel(channel_a)
        kit.register_channel(ConferenceChannel("conf-b", backend=backend_b, stt=MockSTTProvider()))
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf-a")
        refusals: list[BaseException] = []
        done = asyncio.Event()

        @kit.on("conference_started")
        async def _swap(event: object) -> None:
            if done.is_set():
                return
            done.set()
            await kit.detach_channel(ROOM, "conf-a")
            # The teardown is deferred behind this very announcement, and the
            # old bot has not left. Waiting here would deadlock; the refusal
            # is what makes the retry safe.
            try:
                await kit.attach_channel(ROOM, "conf-b")
            except ConferenceAlreadyAttachedError as error:
                refusals.append(error)

        await backend_a.simulate_participant_joined(ROOM, "p-alice")
        await _settle(channel_a)

        assert len(refusals) == 1
        assert ROOM not in backend_b.rooms
        # After the deferred teardown has ended, the slot is genuinely free.
        await kit.attach_channel(ROOM, "conf-b")
        assert ROOM in backend_b.rooms


class TestTheReconnectSupervisor:
    """A lost bot cannot manufacture its own "next need": the dead session was
    what received the frames and the events. The supervisor is that need —
    bounded, backed off, and standing down the moment anything contradicts it.
    """

    async def test_the_bot_rejoins_without_any_new_external_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from roomkit.channels import _conference_session as session_module
        from tests.conference.test_conference_races import _until

        monkeypatch.setattr(session_module, "REJOIN_DELAYS_S", (0.01,))
        _, channel, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        first = channel._room(ROOM).bot
        assert first is not None

        await backend.simulate_bot_disconnected(first, "connection lost")

        # No arrival, no delivery, no track event — the supervisor alone.
        await _until(lambda: len(backend.bots) == 1)
        assert channel.info()["rooms"][ROOM]["bot_present"] is True

    async def test_failed_attempts_back_off_and_then_stand_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from roomkit.channels import _conference_session as session_module
        from tests.conference.test_conference_races import _until

        monkeypatch.setattr(session_module, "REJOIN_DELAYS_S", (0.01, 0.01))
        _, channel, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        first = channel._room(ROOM).bot
        assert first is not None
        # Exactly the supervisor's two attempts fail; the SFU then recovers.
        backend.fail("join_as_bot", RuntimeError("SFU still down"), times=2)

        await backend.simulate_bot_disconnected(first, "connection lost")
        await _until(lambda: not channel._room(ROOM).tasks)

        assert backend.bots == []
        # The lazy join is still the fallback once the SFU recovers.
        await backend.simulate_participant_joined(ROOM, "p-bob")
        assert len(backend.bots) == 1

    async def test_a_detach_stands_the_supervisor_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from roomkit.channels import _conference_session as session_module

        monkeypatch.setattr(session_module, "REJOIN_DELAYS_S", (0.01,))
        kit, channel, backend = await _kit_with_channel()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        first = channel._room(ROOM).bot
        assert first is not None

        await backend.simulate_bot_disconnected(first, "connection lost")
        await kit.detach_channel(ROOM, "conf")
        await asyncio.sleep(0.05)

        assert backend.bots == []
