"""Regression: one pipeline, many speakers, one detection state each.

`VoiceChannel` holds a single `AudioPipeline` for every session on the channel —
up to ten with `AudioBridge` — and a conference lane will be another stream
through the same stages. Before the stages took a stream key, they shared one
state, so Alice's silence closed Bob's utterance.

The reproduction from the ticket: one `AudioPipeline`, two `VoiceSession`, one
sequenced `MockVADProvider`. Bob used to receive `SPEECH_END` on his very first
frame because Alice's frames had already advanced the sequence for him.
"""

from __future__ import annotations

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline.aec.mock import MockAECProvider
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider


def _session(sid: str) -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id=f"p-{sid}", channel_id="c1")


def _frame(value: int = 1000) -> AudioFrame:
    return AudioFrame(
        data=value.to_bytes(2, "little", signed=True) * 160,
        sample_rate=16000,
        channels=1,
        sample_width=2,
    )


def _sequence() -> list[VADEvent | None]:
    return [
        VADEvent(type=VADEventType.SPEECH_START),
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"utterance", duration_ms=100.0),
        VADEvent(type=VADEventType.SPEECH_START),
    ]


class TestBridgedSessionsHaveIndependentVAD:
    """Two bridged sessions on one pipeline — the ticket's reproduction."""

    def test_each_session_walks_its_own_vad_sequence(self) -> None:
        vad = MockVADProvider(events=_sequence())
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        seen: list[tuple[str, VADEventType]] = []
        pipeline.on_vad_event(lambda s, e: seen.append((s.id, e.type)))

        alice, bob = _session("alice"), _session("bob")

        # Alice speaks first and gets two frames in before Bob says anything.
        pipeline.process_inbound(alice, _frame())
        pipeline.process_inbound(bob, _frame())
        pipeline.process_inbound(alice, _frame())

        # Bob's first frame must be his SPEECH_START, not Alice's SPEECH_END.
        assert seen == [
            ("alice", VADEventType.SPEECH_START),
            ("bob", VADEventType.SPEECH_START),
            ("alice", VADEventType.SPEECH_END),
        ]

    def test_one_speaker_leaving_does_not_reset_the_other(self) -> None:
        vad = MockVADProvider(events=_sequence())
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        seen: list[tuple[str, VADEventType]] = []
        pipeline.on_vad_event(lambda s, e: seen.append((s.id, e.type)))

        alice, bob = _session("alice"), _session("bob")

        pipeline.process_inbound(alice, _frame())
        pipeline.process_inbound(bob, _frame())

        # Alice hangs up mid-conversation.
        pipeline.on_session_ended(alice)

        # Bob keeps going from where he was: his second event, not his first.
        pipeline.process_inbound(bob, _frame())

        assert seen == [
            ("alice", VADEventType.SPEECH_START),
            ("bob", VADEventType.SPEECH_START),
            ("bob", VADEventType.SPEECH_END),
        ]

    def test_a_returning_speaker_starts_clean(self) -> None:
        vad = MockVADProvider(events=_sequence())
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        seen: list[tuple[str, VADEventType]] = []
        pipeline.on_vad_event(lambda s, e: seen.append((s.id, e.type)))

        alice = _session("alice")
        pipeline.process_inbound(alice, _frame())
        pipeline.on_session_ended(alice)
        pipeline.process_inbound(alice, _frame())

        # Not resumed halfway through a stale sequence.
        assert seen == [
            ("alice", VADEventType.SPEECH_START),
            ("alice", VADEventType.SPEECH_START),
        ]


class TestDepartedSpeakersDoNotAccumulate:
    """A room that runs for hours must not hold every speaker who ever joined.

    Stage state is per stream and some of it is native memory, so a missed
    release is a C-side leak rather than a red test.
    """

    def test_session_end_releases_stage_state(self) -> None:
        vad = MockVADProvider(events=_sequence())
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad, aec=aec))

        for name in ("alice", "bob", "carol"):
            session = _session(name)
            pipeline.process_inbound(session, _frame())
            pipeline.on_session_ended(session)

        assert aec.reset_streams == ["alice", "bob", "carol"]
        assert vad.reset_count == 3

    def test_pipeline_reset_releases_streams_still_open(self) -> None:
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))

        pipeline.process_inbound(_session("alice"), _frame())
        pipeline.process_inbound(_session("bob"), _frame())

        pipeline.reset()

        assert sorted(aec.reset_streams) == ["alice", "bob"]


class TestConferenceLanesHaveIndependentVAD:
    """Two conference lanes through one pipeline.

    Lanes do not build an `AudioPipeline` yet — that is RMK-8, which this
    change unblocks. What is verifiable today, and what RMK-8 depends on, is
    that two arbitrary stream identities on one pipeline stay independent.
    A lane will be exactly that: another key.
    """

    def test_two_lanes_walk_their_own_vad_sequences(self) -> None:
        vad = MockVADProvider(events=_sequence())
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        seen: list[tuple[str, VADEventType]] = []
        pipeline.on_vad_event(lambda s, e: seen.append((s.id, e.type)))

        lane_a, lane_b = _session("lane-a"), _session("lane-b")

        for _ in range(2):
            pipeline.process_inbound(lane_a, _frame())
        pipeline.process_inbound(lane_b, _frame())

        assert seen == [
            ("lane-a", VADEventType.SPEECH_START),
            ("lane-a", VADEventType.SPEECH_END),
            ("lane-b", VADEventType.SPEECH_START),
        ]


class TestAECReferenceIsRoutedPerStream:
    """Each stream's echo canceller is fed its own playback, not everyone's."""

    def test_outbound_reference_carries_the_session_key(self) -> None:
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))

        alice, bob = _session("alice"), _session("bob")
        pipeline.process_outbound(alice, _frame())
        pipeline.process_outbound(bob, _frame())

        assert aec.reference_streams == ["alice", "bob"]

    def test_capture_and_reference_agree_on_the_key(self) -> None:
        aec = MockAECProvider()
        pipeline = AudioPipeline(AudioPipelineConfig(aec=aec))

        alice = _session("alice")
        pipeline.process_inbound(alice, _frame())
        pipeline.process_outbound(alice, _frame())

        # Same canceller: the reference must reach the state the mic feeds.
        assert aec.streams == ["alice"]
        assert aec.reference_streams == ["alice"]
