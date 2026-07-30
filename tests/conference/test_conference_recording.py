"""Framework-mode conference recording (RFC §12.10.8, §12.11).

The rule these tests exist for is attribution: a conference recording answers
"who said what", so every track is recorded on its own and carries the
participant who published it. One recording per track is what makes that
survive a meeting people join and leave — a single recording holding every
track would have to admit one mid-write, which the usual containers refuse.

The bot is in that list. It is the only track that resembles the outbound
direction of a call, and the temptation is to mix it into the others; what the
AI said is part of what was said, and it is recorded like anyone else's.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from roomkit import (
    ConferenceRecordingConfig,
    ConferenceRecordingMode,
    ConferenceRecordingStarted,
    ConferenceRecordingStopped,
    MockConferenceBackend,
    MockTrackFormat,
    RoomKit,
    TrackKind,
)
from roomkit.channels import _conference_activity as activity_module
from roomkit.channels.conference import ConferenceChannel
from roomkit.conference.models import ConferenceTrack
from roomkit.models.enums import Access, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from tests.conference.lane_audio import drain, drain_recordings, say, speech_frame
from tests.conference.test_conference_outbound import _Source
from tests.conference.test_conference_races import _settle, _until

ROOM = "room-1"


async def _recording_conference(
    *,
    recorder: MockMediaRecorder | None = None,
    **channel_kwargs: object,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend, MockMediaRecorder]:
    backend = MockConferenceBackend()
    recorder = recorder if recorder is not None else MockMediaRecorder()
    channel = ConferenceChannel(
        "conf",
        backend=backend,
        recorder=recorder,
        recording=ConferenceRecordingConfig(),
        **channel_kwargs,  # type: ignore[arg-type]
    )
    kit = RoomKit()
    kit.register_channel(channel)
    kit.register_channel(_Source("src", ChannelType.AI))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    await kit.attach_channel(ROOM, "src")
    return kit, channel, backend, recorder


def _tracks_of(recorder: MockMediaRecorder, participant_id: str) -> list[str]:
    return [t.id for t in recorder.tracks if t.participant_id == participant_id]


class TestPerTrackRecordings:
    async def test_each_track_gets_its_own_recording(self) -> None:
        """One `on_recording_start` per track, so a file is one participant's
        media rather than a mix nobody can take apart afterwards.
        """
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")

        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(bob, speech_frame())
        await drain_recordings(channel)

        assert len(recorder.handles) == 2
        assert {t.id for t in recorder.tracks} == {alice.id, bob.id}

    async def test_every_track_carries_the_participant_that_published_it(self) -> None:
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")

        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(bob, speech_frame())
        await drain_recordings(channel)

        assert _tracks_of(recorder, "p-alice") == [alice.id]
        assert _tracks_of(recorder, "p-bob") == [bob.id]
        assert all(t.channel_id == "conf" for t in recorder.tracks)
        assert all(t.kind == TrackKind.AUDIO.value for t in recorder.tracks)

    async def test_one_speaker_never_lands_in_another_speakers_recording(self) -> None:
        """The attribution has to hold frame by frame, not only at the header:
        a recording that mixed two speakers would be worthless as evidence.
        """
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")

        for _ in range(3):
            await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(bob, speech_frame())
        await drain_recordings(channel)

        by_track = [c.track_id for c in recorder.chunks]
        assert by_track.count(alice.id) == 3
        assert by_track.count(bob.id) == 1

    async def test_a_participant_arriving_late_records_from_their_own_start(self) -> None:
        """The reason recordings are per track rather than one per room: a
        container fixes its streams at the first write, so a late arrival added
        to an already-writing recording is media that is never written.
        """
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        for _ in range(5):
            await backend.simulate_audio(alice, speech_frame())

        await backend.simulate_participant_joined(ROOM, "p-carol")
        carol = await backend.simulate_track_published(ROOM, "p-carol")
        await backend.simulate_audio(carol, speech_frame())
        await drain_recordings(channel)

        assert len(recorder.handles) == 2
        assert _tracks_of(recorder, "p-carol") == [carol.id]
        assert [c.track_id for c in recorder.chunks].count(carol.id) == 1

    async def test_a_silent_track_produces_no_recording(self) -> None:
        """A recording opens on a track's first frame, so a participant who
        joins muted leaves no empty file behind.
        """
        _, _, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_track_published(ROOM, "p-alice")

        assert recorder.handles == []


class TestTrackFormat:
    """RFC §12.10.8 and §12.11: a recording says what its track carries.

    Participants negotiate their format with the SFU separately and nothing
    obliges them to agree, so a conference of three may carry three. A recording
    that assumed one would describe a stereo track as mono and an 8-bit track as
    16-bit — and PCM carrying no description of itself, the file would open,
    play wrong and report nothing. That is the worst way for a recording of a
    meeting to fail, because it may be opened months later.
    """

    @pytest.mark.parametrize(
        ("audio_format", "codec"),
        [
            (MockTrackFormat(sample_rate=8_000, channels=1, sample_width=1), "pcm_s8"),
            (MockTrackFormat(sample_rate=16_000, channels=1, sample_width=2), "pcm_s16le"),
            (MockTrackFormat(sample_rate=48_000, channels=2, sample_width=2), "pcm_s16le"),
            (MockTrackFormat(sample_rate=48_000, channels=2, sample_width=4), "pcm_s32le"),
        ],
        ids=["dial_in_8bit", "default", "stereo", "stereo_32bit"],
    )
    async def test_the_recording_describes_the_format_the_track_carries(
        self, audio_format: MockTrackFormat, codec: str
    ) -> None:
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice", audio_format=audio_format)

        await backend.simulate_audio(alice, backend.frame_for(alice))
        await drain_recordings(channel)

        assert len(recorder.tracks) == 1
        assert recorder.tracks[0].codec == codec
        assert recorder.tracks[0].sample_rate == audio_format.sample_rate
        assert recorder.tracks[0].channels == audio_format.channels

    async def test_two_participants_need_not_have_negotiated_the_same_format(self) -> None:
        """The format belongs to the track, not to the conference: a phone
        dial-in and a studio microphone in one meeting are recorded as what each
        of them actually sent.
        """
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(
            ROOM,
            "p-alice",
            audio_format=MockTrackFormat(sample_rate=8_000, channels=1, sample_width=1),
        )
        bob = await backend.simulate_track_published(
            ROOM,
            "p-bob",
            audio_format=MockTrackFormat(sample_rate=48_000, channels=2, sample_width=2),
        )

        await backend.simulate_audio(alice, backend.frame_for(alice))
        await backend.simulate_audio(bob, backend.frame_for(bob))
        await drain_recordings(channel)

        described = {
            track.id: (track.codec, track.sample_rate, track.channels) for track in recorder.tracks
        }
        assert described == {
            alice.id: ("pcm_s8", 8_000, 1),
            bob.id: ("pcm_s16le", 48_000, 2),
        }

    async def test_the_bot_track_carries_the_format_it_published(self) -> None:
        """The bot's audio arrives as an AudioChunk, which names its own codec.
        It is a track like any other and describes itself like any other.
        """
        kit, channel, _, recorder = await _recording_conference(tts=MockTTSProvider())

        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))
        await drain_recordings(channel)

        bot = [track for track in recorder.tracks if track.participant_id == "roomkit"]
        assert len(bot) == 1
        assert (bot[0].codec, bot[0].sample_rate, bot[0].channels) == ("pcm_s16le", 16_000, 1)

    async def test_the_opening_report_carries_the_format(self) -> None:
        """ON_RECORDING_STARTED is where an integrator learns what a track was
        recorded as; the sample rate alone is half a description.
        """
        kit, channel, backend, _ = await _recording_conference()
        started: list[ConferenceRecordingStarted] = []

        @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
        async def _started(event: ConferenceRecordingStarted, ctx: object) -> None:
            started.append(event)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(
            ROOM,
            "p-alice",
            audio_format=MockTrackFormat(sample_rate=8_000, channels=2, sample_width=1),
        )
        await backend.simulate_audio(alice, backend.frame_for(alice))
        await drain_recordings(channel)

        assert len(started) == 1
        assert (started[0].sample_rate, started[0].channels, started[0].codec) == (
            8_000,
            2,
            "pcm_s8",
        )


class TestFormatChangedMidTrack:
    """A recording is opened on one format, and a container fixes its streams at
    the first write — so a frame that renegotiated has nowhere honest to go.

    Written anyway it is decoded as the header claims and the file plays wrong
    while reporting nothing; refused, it leaves a gap, which is a defect anyone
    can see. RFC §12.10.8 requires the loss to be exposed rather than only
    logged.
    """

    @staticmethod
    def _stereo_like(frame: AudioFrame) -> AudioFrame:
        """The same bytes, arriving as stereo. The publisher renegotiated."""
        return AudioFrame(data=frame.data, sample_rate=frame.sample_rate, channels=2)

    async def test_a_frame_in_another_format_is_not_written(self) -> None:
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(alice, self._stereo_like(speech_frame()))
        await drain_recordings(channel)

        assert [c.track_id for c in recorder.chunks] == [alice.id]
        # And the recording still describes what it was opened on.
        assert recorder.tracks[0].channels == 1

    async def test_refused_frames_are_counted_as_loss(self) -> None:
        _, channel, backend, _ = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        await backend.simulate_audio(alice, speech_frame())
        for _ in range(3):
            await backend.simulate_audio(alice, self._stereo_like(speech_frame()))
        await drain_recordings(channel)

        assert channel.info()["rooms"][ROOM]["recording_dropped_frames"] == 3

    async def test_a_track_republished_in_a_new_format_is_recorded_in_it(self) -> None:
        """Refusing is not giving up on the track: unpublishing closes the
        recording, and what comes back is a new one, opened on whatever it
        carries now.
        """
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_track_unpublished(alice.id)

        again = await backend.simulate_track_published(
            ROOM,
            "p-alice",
            audio_format=MockTrackFormat(sample_rate=48_000, channels=2, sample_width=2),
        )
        await backend.simulate_audio(again, backend.frame_for(again))
        await drain_recordings(channel)

        assert len(recorder.handles) == 2
        assert [track.channels for track in recorder.tracks] == [2]


class TestRecordingWithoutTranscription:
    async def test_frames_are_recorded_with_no_stt_configured(self) -> None:
        """Recording is a consumer in its own right. With no STT there is no
        pipeline and therefore no lane, and a channel that routed frames by
        lane alone would subscribe the tracks and drop every frame.
        """
        _, channel, backend, recorder = await _recording_conference()
        assert channel.active_lanes == {}
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        assert alice.id in backend.subscriptions
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        assert [c.track_id for c in recorder.chunks] == [alice.id]

    async def test_recording_and_transcription_share_one_subscription(self) -> None:
        _, channel, backend, recorder = await _recording_conference(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        sent = await say(backend, alice)
        await drain(channel, alice.id)
        await drain_recordings(channel)

        assert len(backend.subscriptions) == 1
        assert alice.id in channel.active_lanes
        assert len([c for c in recorder.chunks if c.track_id == alice.id]) == sent


class TestBotTrack:
    async def test_what_the_bot_said_is_recorded_on_a_track_of_its_own(self) -> None:
        kit, channel, backend, recorder = await _recording_conference(tts=MockTTSProvider())

        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))
        await drain_recordings(channel)

        bot_tracks = _tracks_of(recorder, "roomkit")
        assert len(bot_tracks) == 1
        assert bot_tracks[0] == f"bot:{backend.bots[0].id}"

    async def test_the_bot_is_never_mixed_into_a_participants_recording(self) -> None:
        kit, channel, backend, recorder = await _recording_conference(tts=MockTTSProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))
        await drain_recordings(channel)

        bot_track_id = f"bot:{backend.bots[0].id}"
        assert {c.track_id for c in recorder.chunks} == {alice.id, bot_track_id}
        assert len(recorder.handles) == 2

    async def test_a_conference_without_a_synthesizer_records_no_bot_track(self) -> None:
        kit, _, _, recorder = await _recording_conference()

        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))

        assert recorder.handles == []


class TestRecordingLifecycle:
    async def test_unpublishing_a_track_finalizes_only_its_recording(self) -> None:
        _, _, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")
        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(bob, speech_frame())

        await backend.simulate_track_unpublished(alice.id)

        assert len(recorder.results) == 1
        assert _tracks_of(recorder, "p-alice") == []
        assert _tracks_of(recorder, "p-bob") == [bob.id]

    async def test_detaching_finalizes_every_recording_the_room_held(self) -> None:
        kit, _, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")
        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(bob, speech_frame())

        await kit.detach_channel(ROOM, "conf")

        assert len(recorder.results) == 2
        assert recorder.tracks == []

    async def test_closing_the_channel_finalizes_and_releases_the_recorder(self) -> None:
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        await channel.close()

        assert len(recorder.results) == 1
        assert recorder.closed is True

    async def test_a_wedged_announcement_does_not_hold_the_close(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The stop announcements run hooks and read the store, neither of
        which is the channel's own, so ``close()`` gives them the drain budget
        and goes on (RFC 12.10.4). The recordings themselves are finalized
        either way; what the budget abandons is the notification, and the log
        says which.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        kit, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        async def wedged(room_id: str) -> object:
            await asyncio.Event().wait()

        monkeypatch.setattr(kit, "_build_context", wedged)

        with caplog.at_level(logging.ERROR):
            await asyncio.wait_for(channel.close(), timeout=2.0)

        assert channel._backend_closed is True
        assert len(recorder.results) == 1
        assert recorder.closed is True
        assert "without announcing" in caplog.text

    async def test_a_shared_recorder_is_finalized_but_not_released(self) -> None:
        """``close_providers=False`` is a caller saying it owns the provider.
        The recordings this channel opened are still its own to finish: a
        container nobody finalized is not a recording.
        """
        _, channel, backend, recorder = await _recording_conference(close_providers=False)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        await channel.close()

        assert len(recorder.results) == 1
        assert recorder.closed is False


class TestCollectionGate:
    async def test_closing_the_binding_stops_recording(self) -> None:
        """Recording is collection, and the binding gate is what an integrator
        closes to stop a room being collected from at all.
        """
        kit, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)
        before = len(recorder.chunks)

        await kit.set_access(ROOM, "conf", Access.NONE)
        await asyncio.sleep(0)

        assert alice.id not in backend.subscriptions
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)
        assert len(recorder.chunks) == before

    async def test_closing_the_binding_releases_subscriptions_with_no_lanes(self) -> None:
        """The unsubscription cannot be driven off the lanes: a channel that
        records without transcribing has none, and the tracks would stay
        subscribed for a room nobody is allowed to collect from.
        """
        kit, channel, backend, _ = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_track_published(ROOM, "p-alice")
        assert channel.active_lanes == {}
        assert len(backend.subscriptions) == 1

        await kit.set_access(ROOM, "conf", Access.NONE)
        await asyncio.sleep(0)

        assert backend.subscriptions == set()


class TestDisclosureSurface:
    async def test_recording_is_reported_per_room(self) -> None:
        """RFC §17.7: an integrator must be able to ask whether recording is
        running on *this* conference, not what the channel was built with.
        """
        _, channel, backend, _ = await _recording_conference()
        assert channel.info()["recording_configured"] is True
        assert channel.info()["rooms"][ROOM]["recording_active"] is False

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_track_published(ROOM, "p-alice")

        assert channel.info()["rooms"][ROOM]["recording_active"] is True

    async def test_a_closed_binding_reports_recording_stopped(self) -> None:
        kit, channel, backend, _ = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_track_published(ROOM, "p-alice")

        await kit.set_access(ROOM, "conf", Access.NONE)
        await asyncio.sleep(0)

        assert channel.info()["rooms"][ROOM]["recording_active"] is False

    async def test_a_channel_without_a_recorder_reports_nothing_recording(self) -> None:
        backend = MockConferenceBackend()
        channel = ConferenceChannel("conf", backend=backend, stt=MockSTTProvider())
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_track_published(ROOM, "p-alice")

        assert channel.info()["recording_configured"] is False
        assert channel.info()["rooms"][ROOM]["recording_active"] is False


class TestRefusedConfiguration:
    async def test_a_configuration_without_a_recorder_is_refused(self) -> None:
        """Accepting it would subscribe the audio tracks and write nothing —
        collecting more than the conference needs, for no output.
        """
        with pytest.raises(ValueError, match="needs a recorder"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                recording=ConferenceRecordingConfig(),
            )

    async def test_a_recorder_with_no_configuration_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no recording configuration"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                recorder=MockMediaRecorder(),
            )

    async def test_egress_mode_is_refused_by_name(self) -> None:
        """RFC §12.10.8 specifies egress and gives it no result contract, and
        ConferenceBackend has no egress surface to delegate to. The difference
        between "recorded by the SFU" and "not recorded" is not one an
        integrator should have to discover.
        """
        with pytest.raises(ValueError, match="egress"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                recorder=MockMediaRecorder(),
                recording=ConferenceRecordingConfig(mode=ConferenceRecordingMode.EGRESS),
            )

    async def test_recording_an_encrypted_conference_is_refused(self) -> None:
        """RFC §12.10.2, the same key-holder gap that refuses STT: the bot
        receives ciphertext, so the file would hold noise while reading as
        evidence.
        """
        with pytest.raises(ValueError, match="end-to-end encrypted"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                recorder=MockMediaRecorder(),
                recording=ConferenceRecordingConfig(),
                e2ee=True,
            )


class TestStrayFrames:
    async def test_a_frame_for_an_unsubscribed_track_is_not_recorded(self) -> None:
        """A backend that forwards more than it was asked for must not decide
        what this conference collects.
        """
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        # Never published, so never subscribed — the shape of a track the
        # channel has no business consuming.
        stray = ConferenceTrack(
            id="stray", room_id=ROOM, participant_id="p-bob", kind=TrackKind.AUDIO
        )

        await backend.simulate_audio(alice, speech_frame())
        await channel._on_track_audio(stray, speech_frame())
        await drain_recordings(channel)

        assert [c.track_id for c in recorder.chunks] == [alice.id]


class TestEventOrdering:
    async def test_a_recording_is_finalized_before_the_conference_is_announced_over(
        self,
    ) -> None:
        """The recordings close inside the teardown, before the end is
        announced: an observer told the conference is over must not then find
        its recording still open.
        """
        kit, _, backend, recorder = await _recording_conference()
        finalized_at_end: list[int] = []

        @kit.on("conference_ended")
        async def _ended(event) -> None:  # type: ignore[no-untyped-def]
            finalized_at_end.append(len(recorder.results))

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        await kit.detach_channel(ROOM, "conf")

        assert finalized_at_end == [1]

    async def test_a_recording_is_reported_before_the_conference_is_announced_over(
        self,
    ) -> None:
        """Where the file went is part of what the conference was, so it is
        said before the announcement that the conference is over.
        """
        kit, _, backend, _ = await _recording_conference()
        order: list[str] = []

        @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
        async def _stopped(event: object, ctx: object) -> None:
            order.append("recording_stopped")

        @kit.on("conference_ended")
        async def _ended(event) -> None:  # type: ignore[no-untyped-def]
            order.append("conference_ended")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        await kit.detach_channel(ROOM, "conf")

        assert order == ["recording_stopped", "conference_ended"]


class TestRecordingResultReported:
    """RFC §12.10.8: a recording that says nothing about where it was written
    leaves an integrator with files on a disk and no way to find them. One
    report per track, at the two moments a track's recording has.
    """

    async def test_a_finished_recording_says_where_it_was_written(self) -> None:
        kit, _, backend, recorder = await _recording_conference()
        stopped: list[ConferenceRecordingStopped] = []

        @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
        async def _stopped(event: ConferenceRecordingStopped, ctx: object) -> None:
            stopped.append(event)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        await backend.simulate_track_unpublished(alice.id)

        assert len(stopped) == 1
        assert stopped[0].url == recorder.results[0].url
        assert stopped[0].url != ""
        assert stopped[0].id == recorder.results[0].id
        assert stopped[0].track_id == alice.id
        assert stopped[0].participant_id == "p-alice"
        assert stopped[0].room_id == ROOM

    async def test_a_track_that_stayed_silent_reports_nothing(self) -> None:
        """The recording opens on the first frame, so a participant who
        published a track and never spoke leaves no file — and a report of a
        file that does not exist is worse than no report.
        """
        kit, _, backend, _ = await _recording_conference()
        fired: list[str] = []

        @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
        async def _started(event: object, ctx: object) -> None:
            fired.append("started")

        @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
        async def _stopped(event: object, ctx: object) -> None:
            fired.append("stopped")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_track_unpublished(alice.id)

        assert fired == []

    async def test_an_opening_is_reported_once_however_many_frames_follow(self) -> None:
        kit, channel, backend, _ = await _recording_conference()
        started: list[ConferenceRecordingStarted] = []

        @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
        async def _started(event: ConferenceRecordingStarted, ctx: object) -> None:
            started.append(event)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        assert len(started) == 1
        assert started[0].track_id == alice.id
        assert started[0].participant_id == "p-alice"
        assert started[0].kind == TrackKind.AUDIO.value

    async def test_detaching_reports_every_recording_it_finalized(self) -> None:
        kit, _, backend, _ = await _recording_conference()
        stopped: list[ConferenceRecordingStopped] = []

        @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
        async def _stopped(event: ConferenceRecordingStopped, ctx: object) -> None:
            stopped.append(event)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")
        await backend.simulate_audio(alice, speech_frame())
        await backend.simulate_audio(bob, speech_frame())

        await kit.detach_channel(ROOM, "conf")

        assert {e.track_id for e in stopped} == {alice.id, bob.id}
        assert {e.participant_id for e in stopped} == {"p-alice", "p-bob"}

    async def test_closing_the_channel_reports_what_was_still_open(self) -> None:
        """The last moment anything can be said about those files."""
        kit, channel, backend, _ = await _recording_conference()
        stopped: list[ConferenceRecordingStopped] = []

        @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
        async def _stopped(event: ConferenceRecordingStopped, ctx: object) -> None:
            stopped.append(event)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        await channel.close()

        assert [e.track_id for e in stopped] == [alice.id]

    async def test_the_bot_track_reports_its_own_recording(self) -> None:
        """What the AI said is part of what was said, and its file is found
        the same way anyone else's is.
        """
        kit, channel, backend, _ = await _recording_conference(tts=MockTTSProvider())
        started: list[ConferenceRecordingStarted] = []

        @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
        async def _started(event: ConferenceRecordingStarted, ctx: object) -> None:
            started.append(event)

        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))
        await drain_recordings(channel)

        bot_started = [e for e in started if e.participant_id == "roomkit"]
        assert len(bot_started) == 1
        assert bot_started[0].track_id == f"bot:{backend.bots[0].id}"

    async def test_the_framework_event_carries_the_location(self) -> None:
        """The cross-channel echo of the hook — a status bus subscriber learns
        where the file went without registering a hook of its own.
        """
        kit, _, backend, recorder = await _recording_conference()
        events: list[object] = []

        @kit.on("recording_stopped")
        async def _stopped(event) -> None:  # type: ignore[no-untyped-def]
            events.append(event)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        await backend.simulate_track_unpublished(alice.id)

        assert len(events) == 1
        assert events[0].channel_id == "conf"  # type: ignore[attr-defined]
        assert events[0].room_id == ROOM  # type: ignore[attr-defined]
        assert events[0].data["track_id"] == alice.id  # type: ignore[attr-defined]
        assert events[0].data["url"] == recorder.results[0].url  # type: ignore[attr-defined]

    async def test_detaching_from_an_opening_handler_is_deferred(self) -> None:
        """A disclosure policy that ends a meeting rather than be recorded is
        ordinary integrator code. The announcement is registered as room
        activity, so the detach it triggers recognises itself as nested and
        finishes afterwards — rather than announcing the recording's end from
        inside the announcement of its start.

        Made from the recording's own task, which is where the recorder is
        asked to open it: waited for on the handler having run rather than on a
        drain, since the detach it performs takes the recording out of the
        collection a drain would look in.
        """
        kit, channel, backend, _ = await _recording_conference()
        inside: list[bool] = []

        @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
        async def _started(event: ConferenceRecordingStarted, ctx: object) -> None:
            await kit.detach_channel(ROOM, "conf")
            # Still inside the announcement: the bot is out of the channel's
            # books, but has not left the conference yet.
            inside.append(backend.bots != [])

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await _until(lambda: inside != [])
        await _settle(channel)

        assert inside == [True]
        assert backend.bots == []

    async def test_a_handler_that_raises_does_not_stop_the_closing(self) -> None:
        """The file is already written and the teardown has a bot to remove
        from a conference afterwards. Integrator code cannot take that down.
        """
        kit, _, backend, recorder = await _recording_conference()

        @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
        async def _stopped(event: object, ctx: object) -> None:
            raise RuntimeError("archival failed")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        await kit.detach_channel(ROOM, "conf")

        assert len(recorder.results) == 1
        assert backend.bots == []
