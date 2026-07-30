"""The stream-keyed inbound entry point (RFC §12.3, §12.10.4).

``process_inbound`` is written in terms of a VoiceSession, and so is every
callback it fans out to. A conference lane has no session, and fabricating one
per track is the workaround that keying the stages on a stream identity exists
to remove — so the pipeline exposes a pull-style entry point instead.

What matters is that it is the *same* chain: one implementation of the stage
ordering, two ways in.
"""

from __future__ import annotations

from typing import Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline.agc.mock import MockAGCProvider
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.denoiser.mock import MockDenoiserProvider
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider


def _frame() -> AudioFrame:
    return AudioFrame(data=b"\x00\x00" * 160)


def _session(session_id: str = "sess-1") -> VoiceSession:
    return VoiceSession(
        id=session_id, channel_id="voice", room_id="room-1", participant_id="p-alice"
    )


class TestStreamEntryPoint:
    def test_it_returns_what_the_stages_produced(self) -> None:
        events: list[VADEvent | None] = [
            VADEvent(type=VADEventType.SPEECH_START),
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"parole"),
        ]
        agc, denoiser = MockAGCProvider(), MockDenoiserProvider()
        pipeline = AudioPipeline(
            AudioPipelineConfig(agc=agc, denoiser=denoiser, vad=MockVADProvider(events=events))
        )

        first = pipeline.process_inbound_stream("track-1", _frame())
        second = pipeline.process_inbound_stream("track-1", _frame())

        assert first.vad_event is not None
        assert first.vad_event.type is VADEventType.SPEECH_START
        assert second.vad_event is not None
        assert second.vad_event.audio_bytes == b"parole"
        assert second.frame.metadata["agc"] == agc.name
        assert second.frame.metadata["denoiser"] == denoiser.name

    def test_it_fires_no_session_callbacks(self) -> None:
        """The callbacks are typed on a VoiceSession. Handing them a stream id
        would be a type lie the moment anyone read ``session.metadata``.
        """
        seen: list[Any] = []
        pipeline = AudioPipeline(
            AudioPipelineConfig(
                vad=MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
            )
        )
        pipeline.on_vad_event(lambda session, event: seen.append(("vad", session)))
        pipeline.on_processed_frame(lambda session, frame: seen.append(("frame", session)))
        pipeline.on_speech_frame(lambda session, frame: seen.append(("speech", session)))

        pipeline.process_inbound_stream("track-1", _frame())

        assert seen == []

    def test_a_session_caller_still_gets_its_callbacks(self) -> None:
        """The extraction must not have cost the session path its fanout."""
        seen: list[Any] = []
        pipeline = AudioPipeline(
            AudioPipelineConfig(
                vad=MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
            )
        )
        pipeline.on_vad_event(lambda session, event: seen.append((session.id, event.type)))

        pipeline.process_inbound(_session(), _frame())

        assert seen == [("sess-1", VADEventType.SPEECH_START)]

    def test_streams_and_sessions_keep_separate_stage_state(self) -> None:
        """A lane and a voice session on the same pipeline are two speakers."""
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"a"),
            ]
        )
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))

        lane = pipeline.process_inbound_stream("track-1", _frame())
        pipeline.process_inbound(_session(), _frame())
        lane_next = pipeline.process_inbound_stream("track-1", _frame())

        assert lane.vad_event is not None
        assert lane.vad_event.type is VADEventType.SPEECH_START
        assert lane_next.vad_event is not None
        assert lane_next.vad_event.type is VADEventType.SPEECH_END

    def test_release_stream_resets_the_stages_for_that_stream_only(self) -> None:
        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START), None])
        pipeline = AudioPipeline(AudioPipelineConfig(vad=vad))
        pipeline.process_inbound_stream("track-1", _frame())
        pipeline.process_inbound_stream("track-2", _frame())

        pipeline.release_stream("track-1")

        # track-1 starts its sequence over; track-2 carries on where it was.
        assert pipeline.process_inbound_stream("track-1", _frame()).vad_event is not None
        assert pipeline.process_inbound_stream("track-2", _frame()).vad_event is None
