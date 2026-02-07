"""Mock pipeline providers for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.voice.pipeline.aec_provider import AECProvider
from roomkit.voice.pipeline.agc_provider import AGCProvider
from roomkit.voice.pipeline.backchannel_detector import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
)
from roomkit.voice.pipeline.denoiser_provider import DenoiserProvider
from roomkit.voice.pipeline.diarization_provider import DiarizationProvider, DiarizationResult
from roomkit.voice.pipeline.dtmf_detector import DTMFDetector, DTMFEvent
from roomkit.voice.pipeline.recorder import (
    AudioRecorder,
    RecordingConfig,
    RecordingHandle,
    RecordingResult,
)
from roomkit.voice.pipeline.turn_detector import TurnContext, TurnDecision, TurnDetector
from roomkit.voice.pipeline.vad_provider import VADEvent, VADProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import VoiceSession


class MockVADProvider(VADProvider):
    """Mock VAD provider that returns a preconfigured sequence of events.

    Example:
        from roomkit.voice.pipeline.vad_provider import VADEvent, VADEventType

        events = [
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio"),
        ]
        vad = MockVADProvider(events=events)
    """

    def __init__(self, events: list[VADEvent | None] | None = None) -> None:
        self._events = events or []
        self._index = 0
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockVADProvider"

    def process(self, frame: AudioFrame) -> VADEvent | None:
        self.frames.append(frame)
        if self._index < len(self._events):
            event = self._events[self._index]
            self._index += 1
            return event
        return None

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class MockDenoiserProvider(DenoiserProvider):
    """Mock denoiser that passes frames through unchanged.

    Tracks processed frames for test assertions.
    """

    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockDenoiserProvider"

    def process(self, frame: AudioFrame) -> AudioFrame:
        self.frames.append(frame)
        return frame

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class MockDiarizationProvider(DiarizationProvider):
    """Mock diarization provider that returns a preconfigured sequence of results.

    Example:
        results = [
            DiarizationResult(speaker_id="speaker_0", confidence=0.9, is_new_speaker=True),
            DiarizationResult(speaker_id="speaker_0", confidence=0.95, is_new_speaker=False),
        ]
        diarizer = MockDiarizationProvider(results=results)
    """

    def __init__(self, results: list[DiarizationResult | None] | None = None) -> None:
        self._results = results or []
        self._index = 0
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockDiarizationProvider"

    def process(self, frame: AudioFrame) -> DiarizationResult | None:
        self.frames.append(frame)
        if self._index < len(self._results):
            result = self._results[self._index]
            self._index += 1
            return result
        return None

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class MockAGCProvider(AGCProvider):
    """Mock AGC provider that passes frames through unchanged."""

    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockAGCProvider"

    def process(self, frame: AudioFrame) -> AudioFrame:
        self.frames.append(frame)
        return frame

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class MockAECProvider(AECProvider):
    """Mock AEC provider that passes frames through unchanged."""

    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.reference_frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockAECProvider"

    def process(self, frame: AudioFrame) -> AudioFrame:
        self.frames.append(frame)
        return frame

    def feed_reference(self, frame: AudioFrame) -> None:
        self.reference_frames.append(frame)

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class MockDTMFDetector(DTMFDetector):
    """Mock DTMF detector that returns a preconfigured sequence of events."""

    def __init__(self, events: list[DTMFEvent | None] | None = None) -> None:
        self._events = events or []
        self._index = 0
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockDTMFDetector"

    def process(self, frame: AudioFrame) -> DTMFEvent | None:
        self.frames.append(frame)
        if self._index < len(self._events):
            event = self._events[self._index]
            self._index += 1
            return event
        return None

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class MockAudioRecorder(AudioRecorder):
    """Mock audio recorder that tracks calls."""

    def __init__(self) -> None:
        self.started: list[tuple[str, RecordingConfig]] = []
        self.stopped: list[RecordingHandle] = []
        self.inbound_frames: list[tuple[str, AudioFrame]] = []
        self.outbound_frames: list[tuple[str, AudioFrame]] = []
        self.reset_count = 0
        self.closed = False
        self._next_id = 0

    @property
    def name(self) -> str:
        return "MockAudioRecorder"

    def start(self, session: VoiceSession, config: RecordingConfig) -> RecordingHandle:
        self._next_id += 1
        handle = RecordingHandle(
            recording_id=f"rec_{self._next_id}",
            session_id=session.id,
            path=f"/tmp/recording_{self._next_id}.wav",
        )
        self.started.append((session.id, config))
        return handle

    def stop(self, handle: RecordingHandle) -> RecordingResult:
        self.stopped.append(handle)
        return RecordingResult(
            recording_id=handle.recording_id,
            path=handle.path,
            duration_ms=1000.0,
            size_bytes=32000,
        )

    def tap_inbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        self.inbound_frames.append((handle.recording_id, frame))

    def tap_outbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        self.outbound_frames.append((handle.recording_id, frame))

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class MockTurnDetector(TurnDetector):
    """Mock turn detector that returns a preconfigured sequence of decisions."""

    def __init__(self, decisions: list[TurnDecision] | None = None) -> None:
        self._decisions = decisions or []
        self._index = 0
        self.evaluations: list[TurnContext] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockTurnDetector"

    def evaluate(self, context: TurnContext) -> TurnDecision:
        self.evaluations.append(context)
        if self._index < len(self._decisions):
            decision = self._decisions[self._index]
            self._index += 1
            return decision
        return TurnDecision(is_complete=True, confidence=1.0, reason="default")

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class MockBackchannelDetector(BackchannelDetector):
    """Mock backchannel detector that returns a preconfigured sequence of decisions."""

    def __init__(self, decisions: list[BackchannelDecision] | None = None) -> None:
        self._decisions = decisions or []
        self._index = 0
        self.evaluations: list[BackchannelContext] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockBackchannelDetector"

    def evaluate(self, context: BackchannelContext) -> BackchannelDecision:
        self.evaluations.append(context)
        if self._index < len(self._decisions):
            decision = self._decisions[self._index]
            self._index += 1
            return decision
        return BackchannelDecision(is_backchannel=False, confidence=1.0)

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True
