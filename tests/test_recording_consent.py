"""ON_RECORDING_STARTED is a consent point on every path (RFC §17.6).

"Implementations that support audio recording MUST provide a mechanism for
recording consent management" and the hook SHOULD fire before any audio is
captured. The conference path did that; the voice pipeline announced *after*
starting the recorder, so the first frames were tapped while the announcement
was still in flight, and the room-level recorder announced nothing at all.
"""

from __future__ import annotations

import asyncio

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import (
    AudioPipeline,
    AudioPipelineConfig,
    MockAudioRecorder,
    MockVADProvider,
    RecordingConfig,
)


def _session() -> VoiceSession:
    return VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="voice-1")


def _frame() -> AudioFrame:
    return AudioFrame(data=b"\x01\x02" * 160, sample_rate=16000)


class TestVoicePipelineWaitsForItsAnnouncement:
    async def test_no_frame_is_tapped_before_the_announcement_returns(self) -> None:
        recorder = MockAudioRecorder()
        released = asyncio.Event()

        async def slow_announcement(session: VoiceSession, handle: object) -> None:
            await released.wait()

        pipeline = AudioPipeline(
            AudioPipelineConfig(
                vad=MockVADProvider(),
                recorder=recorder,
                recording_config=RecordingConfig(),
            )
        )
        pipeline.on_recording_started(slow_announcement)

        session = _session()
        pipeline.on_session_active(session)
        await asyncio.sleep(0)

        # Audio arriving while the announcement is being heard reaches no
        # recorder: pre-consent audio is not captured.
        pipeline.process_inbound(session, _frame())
        assert recorder.inbound_frames == []

        released.set()
        await asyncio.sleep(0.01)

        pipeline.process_inbound(session, _frame())
        assert len(recorder.inbound_frames) == 1

    async def test_a_refusal_during_the_announcement_records_nothing(self) -> None:
        """Refusing means stopping the recording from inside the hook."""
        recorder = MockAudioRecorder()
        pipeline = AudioPipeline(
            AudioPipelineConfig(
                vad=MockVADProvider(),
                recorder=recorder,
                recording_config=RecordingConfig(),
            )
        )
        session = _session()

        async def refuse(sess: VoiceSession, handle: object) -> None:
            pipeline.on_session_ended(sess)

        pipeline.on_recording_started(refuse)
        pipeline.on_session_active(session)
        await asyncio.sleep(0.01)

        pipeline.process_inbound(session, _frame())
        assert recorder.inbound_frames == []

    def test_without_a_subscriber_recording_is_not_delayed(self) -> None:
        """Nobody is listening, so there is no announcement to wait for —
        deferring would drop the first frames of every session for nothing."""
        recorder = MockAudioRecorder()
        pipeline = AudioPipeline(
            AudioPipelineConfig(
                vad=MockVADProvider(),
                recorder=recorder,
                recording_config=RecordingConfig(),
            )
        )
        session = _session()
        pipeline.on_session_active(session)

        pipeline.process_inbound(session, _frame())
        assert len(recorder.inbound_frames) == 1


class TestRoomLevelRecordingAnnounces:
    async def test_a_room_recorder_fires_the_hook(self) -> None:
        from roomkit.recorder.base import MediaRecordingConfig, RoomRecorderBinding
        from roomkit.recorder.mock import MockMediaRecorder

        kit = RoomKit()
        announced: list[object] = []

        @kit.hook(HookTrigger.ON_RECORDING_STARTED, HookExecution.ASYNC)
        async def on_started(event, ctx):  # noqa: ANN001
            announced.append(event)

        recorder = MockMediaRecorder()
        await kit.create_room(
            room_id="r1",
            recorders=[RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())],
        )
        await asyncio.sleep(0.01)

        assert len(announced) == 1
        assert announced[0].room_id == "r1"
        # No participant session behind a room-level recording.
        assert announced[0].session is None
