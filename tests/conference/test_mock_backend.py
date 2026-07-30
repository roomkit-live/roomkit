"""MockConferenceBackend behaviour (RFC §12.10.3).

The mock is the only backend the specification requires, and it earns its place
by refusing to do things a real SFU would not: it delivers no frame for a track
nobody subscribed to, and it refuses operations the declared capabilities do not
cover. Those refusals are what make the framework's own rules testable.
"""

from __future__ import annotations

import pytest

from roomkit import (
    ConferenceCapability,
    ConferenceCapabilityError,
    ConferenceGrants,
    MockConferenceBackend,
    TrackKind,
)
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk

ROOM = "room-1"


def _frame() -> AudioFrame:
    return AudioFrame(data=b"\x00\x00")


async def _joined_bot(backend: MockConferenceBackend):
    await backend.ensure_room(ROOM)
    return await backend.join_as_bot(ROOM, "roomkit", ConferenceGrants())


class TestSelectiveSubscription:
    async def test_unsubscribed_track_delivers_no_frame(self) -> None:
        """A real SFU forwards nothing to a subscriber that did not ask. A mock
        that delivered anyway would make selective subscription untestable.
        """
        backend = MockConferenceBackend()
        await _joined_bot(backend)
        seen: list[str] = []
        backend.on_track_audio(lambda track, frame: seen.append(track.id))

        track = await backend.simulate_track_published(ROOM, "p-alice")
        delivered = await backend.simulate_audio(track, _frame())

        assert delivered is False
        assert seen == []
        assert backend.dropped_frames == [track.id]

    async def test_subscribed_track_delivers(self) -> None:
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        seen: list[str] = []
        backend.on_track_audio(lambda track, frame: seen.append(track.id))

        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.subscribe_track(bot, track.id)
        delivered = await backend.simulate_audio(track, _frame())

        assert delivered is True
        assert seen == [track.id]

    async def test_unsubscribing_stops_delivery(self) -> None:
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        seen: list[str] = []
        backend.on_track_audio(lambda track, frame: seen.append(track.id))

        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.subscribe_track(bot, track.id)
        await backend.simulate_audio(track, _frame())
        await backend.unsubscribe_track(bot, track.id)
        await backend.simulate_audio(track, _frame())

        assert len(seen) == 1

    async def test_video_obeys_the_same_rule(self) -> None:
        backend = MockConferenceBackend()
        await _joined_bot(backend)
        seen: list[str] = []
        backend.on_track_video(lambda track, frame: seen.append(track.id))

        track = await backend.simulate_track_published(ROOM, "p-alice", TrackKind.VIDEO)
        delivered = await backend.simulate_video(track, VideoFrame(data=b"", codec="raw_rgb24"))

        assert delivered is False
        assert seen == []

    async def test_unpublishing_drops_the_subscription(self) -> None:
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)

        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.subscribe_track(bot, track.id)
        await backend.simulate_track_unpublished(track.id)

        assert backend.subscriptions == set()


class TestBotEcho:
    async def test_echo_reports_the_bot_as_a_participant_and_publisher(self) -> None:
        """Some SFUs report the bot back through its own callbacks. Without
        self-exclusion the framework would create a participant record for its
        own bot and transcribe the AI's own speech.
        """
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        participants: list[str] = []
        tracks: list[str] = []
        backend.on_participant_joined(lambda room, p: participants.append(p.participant_id))
        backend.on_track_published(lambda room, t: tracks.append(t.participant_id))

        echoed = await backend.simulate_bot_echo(bot)

        assert participants == [bot.identity]
        assert tracks == [bot.identity]
        assert echoed.participant_id == bot.identity


class TestCapabilityGating:
    async def test_unmute_without_capability_is_refused(self) -> None:
        """Unmuting someone else's microphone is a privacy decision, and SFUs
        commonly refuse it unless explicitly enabled server-side.
        """
        backend = MockConferenceBackend()

        with pytest.raises(ConferenceCapabilityError, match="REMOTE_UNMUTE"):
            await backend.unmute_track(ROOM, "tr-1")

    async def test_unmute_with_capability_works(self) -> None:
        backend = MockConferenceBackend(capabilities=ConferenceCapability.REMOTE_UNMUTE)
        bot = await _joined_bot(backend)
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.subscribe_track(bot, track.id)

        await backend.mute_track(ROOM, track.id)
        assert backend.tracks[track.id].muted is True

        await backend.unmute_track(ROOM, track.id)
        assert backend.tracks[track.id].muted is False

    async def test_muting_never_needs_a_capability(self) -> None:
        backend = MockConferenceBackend()
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await backend.mute_track(ROOM, track.id)

        assert backend.tracks[track.id].muted is True

    async def test_bot_video_without_capability_is_refused(self) -> None:
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)

        with pytest.raises(ConferenceCapabilityError, match="VIDEO_PUBLISH"):
            await backend.publish_video(bot, VideoFrame(data=b"", codec="raw_rgb24"))

    async def test_e2ee_without_capability_is_refused(self) -> None:
        """With E2EE active the bot cannot decode, so a configuration asking
        for it against a backend that does not support it must fail loudly.
        """
        backend = MockConferenceBackend()

        with pytest.raises(ConferenceCapabilityError, match="E2EE"):
            await backend.ensure_room(ROOM, e2ee=True)

    async def test_e2ee_with_capability_is_recorded_on_the_room(self) -> None:
        backend = MockConferenceBackend(capabilities=ConferenceCapability.E2EE)

        await backend.ensure_room(ROOM, e2ee=True)

        assert backend.rooms[ROOM]["e2ee"] is True


class TestControlPlane:
    async def test_ensure_room_is_idempotent(self) -> None:
        backend = MockConferenceBackend()

        await backend.ensure_room(ROOM)
        await backend.ensure_room(ROOM)

        assert list(backend.rooms) == [ROOM]

    async def test_mint_access_returns_usable_credentials(self) -> None:
        backend = MockConferenceBackend()

        access = await backend.mint_access(ROOM, "p-alice", ConferenceGrants.observer())

        assert access.token
        assert access.url.startswith("wss://")
        assert backend.calls[-1].method == "mint_access"
        assert backend.calls[-1].args["grants"].hidden is True

    async def test_participants_are_listed_and_removed(self) -> None:
        backend = MockConferenceBackend()
        await backend.ensure_room(ROOM)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        assert len(await backend.list_participants(ROOM)) == 2

        await backend.remove_participant(ROOM, "p-bob")
        assert [p.participant_id for p in await backend.list_participants(ROOM)] == ["p-alice"]

    async def test_provider_attributes_survive_to_the_participant(self) -> None:
        """The caller number of a dial-in is what identity resolution consumes,
        so the backend must carry provider attributes through.
        """
        backend = MockConferenceBackend()
        await backend.ensure_room(ROOM)

        participant = await backend.simulate_participant_joined(
            ROOM, "sip_15551234", metadata={"sip.phoneNumber": "+15551234"}
        )

        assert participant.metadata["sip.phoneNumber"] == "+15551234"
        assert participant.asserted_metadata == {"sip.phoneNumber": "+15551234"}

    async def test_what_the_client_supplied_is_surfaced_but_not_vouched_for(self) -> None:
        """Both bags reach the channel — the second is what a participant said
        about itself, and the mock is explicit about which is which.
        """
        backend = MockConferenceBackend()
        await backend.ensure_room(ROOM)

        participant = await backend.simulate_participant_joined(
            ROOM,
            "sip_15551234",
            metadata={"sip.phoneNumber": "+15551234"},
            client_metadata={"nickname": "bob"},
        )

        assert participant.metadata == {"sip.phoneNumber": "+15551234", "nickname": "bob"}
        assert participant.asserted_metadata == {"sip.phoneNumber": "+15551234"}

    async def test_a_backend_can_say_it_cannot_tell_the_two_apart(self) -> None:
        backend = MockConferenceBackend()
        await backend.ensure_room(ROOM)

        participant = await backend.simulate_participant_joined(
            ROOM,
            "sip_15551234",
            metadata={"sip.phoneNumber": "+15551234"},
            asserts_provenance=False,
        )

        assert participant.metadata["sip.phoneNumber"] == "+15551234"
        assert participant.asserted_metadata is None

    async def test_leaving_participant_is_announced_once(self) -> None:
        backend = MockConferenceBackend()
        await backend.ensure_room(ROOM)
        left: list[str] = []
        backend.on_participant_left(lambda room, p: left.append(p.participant_id))

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_left(ROOM, "p-alice")
        await backend.simulate_participant_left(ROOM, "p-alice")

        assert left == ["p-alice"]


class TestPublishing:
    async def test_published_audio_is_recorded_for_assertions(self) -> None:
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)

        await backend.publish_audio(bot, AudioChunk(data=b"\x00\x00"))
        await backend.publish_audio(bot, AudioChunk(data=b"\x00\x00", is_final=True))

        assert len(backend.published_audio) == 2
        assert backend.published_audio[-1].is_final is True

    async def test_tracks_carry_room_and_publisher(self) -> None:
        backend = MockConferenceBackend()
        await backend.ensure_room(ROOM)

        track = await backend.simulate_track_published(ROOM, "p-alice")

        assert track.room_id == ROOM
        assert track.participant_id == "p-alice"
        assert track.kind is TrackKind.AUDIO


class TestEphemeralSignals:
    async def test_active_speaker_and_quality_reach_subscribers(self) -> None:
        backend = MockConferenceBackend()
        speakers: list[str] = []
        quality: list[tuple[str, str]] = []
        backend.on_active_speaker_changed(lambda room, pid: speakers.append(pid))
        backend.on_connection_quality(lambda room, pid, q: quality.append((pid, q)))

        await backend.simulate_active_speaker(ROOM, "p-alice")
        await backend.simulate_connection_quality(ROOM, "p-bob", "poor")

        assert speakers == ["p-alice"]
        assert quality == [("p-bob", "poor")]
