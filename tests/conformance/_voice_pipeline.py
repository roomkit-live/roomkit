"""Scripted pipeline stages for RFC §12.3 voice, each call they receive traced.

Every scripted stage logs each call it receives (its stage name, the call, the
stream key, a copy of the frame as it arrived) into one trace per channel,
which ``voice.read`` returns as ``stages``.
"""

from __future__ import annotations

import base64
import enum
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.voice import AudioFrame, VoiceSession
from roomkit.voice.pipeline import (
    AudioFormat,
    AudioPipelineConfig,
    AudioPipelineContract,
    AudioPostProcessor,
    AudioRecorder,
    DTMFEvent,
    MockAECProvider,
    MockAGCProvider,
    MockDenoiserProvider,
    MockDiarizationProvider,
    MockDTMFDetector,
    MockTurnDetector,
    MockVADProvider,
    RecordingConfig,
    RecordingHandle,
    RecordingMode,
    RecordingResult,
    VADConfig,
)

if TYPE_CHECKING:
    from roomkit.voice.pipeline import DiarizationResult, TurnDecision, VADEvent

Trace = list[dict[str, Any]]


def _plain(value: Any) -> Any:
    """A JSON copy of a frame's metadata value: what a scenario can name."""
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def frame_record(frame: AudioFrame) -> dict[str, Any]:
    """The frame as a stage received it, copied before later stages change it."""
    return {
        "data": base64.b64encode(frame.data).decode("ascii"),
        "sample_rate": frame.sample_rate,
        "channels": frame.channels,
        "sample_width": frame.sample_width,
        "timestamp_ms": frame.timestamp_ms,
        "metadata": _plain(frame.metadata),
    }


def _output(frame: AudioFrame, spec: Mapping[str, Any]) -> AudioFrame:
    """The frame a scripted stage returns: its scripted bytes, the format unchanged."""
    if "output" not in spec:
        return frame
    return AudioFrame(
        data=base64.b64decode(spec["output"], validate=True),
        sample_rate=frame.sample_rate,
        channels=frame.channels,
        sample_width=frame.sample_width,
        timestamp_ms=frame.timestamp_ms,
        metadata=dict(frame.metadata),
    )


def _log(
    trace: Trace,
    stage: str,
    call: str,
    stream: str | None = None,
    frame: AudioFrame | None = None,
    **extra: Any,
) -> None:
    """Append one call to the trace: its stage, its name, its key and frame if any."""
    record: dict[str, Any] = {"stage": stage, "call": call}
    if stream is not None:
        record["stream"] = stream
    if frame is not None:
        record["frame"] = frame_record(frame)
    trace.append(record | extra)


class _TracedAEC(MockAECProvider):
    def __init__(self, trace: Trace, spec: Mapping[str, Any]) -> None:
        super().__init__()
        self.trace = trace
        self.spec = spec

    @property
    def name(self) -> str:
        return "aec"

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        _log(self.trace, "aec", "process", stream, frame)
        return _output(super().process(frame, stream), self.spec)

    def feed_reference(self, frame: AudioFrame, stream: str) -> None:
        _log(self.trace, "aec", "feed_reference", stream, frame)
        super().feed_reference(frame, stream)

    def set_active(self, active: bool) -> None:
        _log(self.trace, "aec", "set_active", active=active)
        super().set_active(active)

    def set_stream_active(self, stream: str, active: bool) -> None:
        _log(self.trace, "aec", "set_stream_active", stream, active=active)
        super().set_stream_active(stream, active)

    def reset(self, stream: str) -> None:
        _log(self.trace, "aec", "reset", stream)
        super().reset(stream)

    def close(self) -> None:
        _log(self.trace, "aec", "close")
        super().close()


class _TracedAGC(MockAGCProvider):
    def __init__(self, trace: Trace, spec: Mapping[str, Any]) -> None:
        super().__init__()
        self.trace = trace
        self.spec = spec

    @property
    def name(self) -> str:
        return "agc"

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        _log(self.trace, "agc", "process", stream, frame)
        return _output(super().process(frame, stream), self.spec)

    def reset(self, stream: str) -> None:
        _log(self.trace, "agc", "reset", stream)
        super().reset(stream)

    def close(self) -> None:
        _log(self.trace, "agc", "close")
        super().close()


class _TracedDenoiser(MockDenoiserProvider):
    def __init__(self, trace: Trace, spec: Mapping[str, Any]) -> None:
        super().__init__()
        self.trace = trace
        self.spec = spec

    @property
    def name(self) -> str:
        return "denoiser"

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        _log(self.trace, "denoiser", "process", stream, frame)
        return _output(super().process(frame, stream), self.spec)

    def reset(self, stream: str) -> None:
        _log(self.trace, "denoiser", "reset", stream)
        super().reset(stream)

    def close(self) -> None:
        _log(self.trace, "denoiser", "close")
        super().close()


class _TracedPostProcessor(AudioPostProcessor):
    def __init__(self, trace: Trace, spec: Mapping[str, Any]) -> None:
        self.trace = trace
        self.spec = spec

    @property
    def name(self) -> str:
        return str(self.spec["name"])

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        _log(self.trace, self.name, "process", stream, frame)
        return _output(frame, self.spec)

    def reset(self, stream: str) -> None:
        _log(self.trace, self.name, "reset", stream)

    def close(self) -> None:
        _log(self.trace, self.name, "close")


class _TracedRecorder(AudioRecorder):
    """Records nothing anywhere: each call is only logged."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    @property
    def name(self) -> str:
        return "recorder"

    def start(self, session: VoiceSession, config: RecordingConfig) -> RecordingHandle:
        _log(self.trace, "recorder", "start", session.id)
        return RecordingHandle(id=uuid4().hex, session_id=session.id)

    def stop(self, handle: RecordingHandle) -> RecordingResult:
        _log(self.trace, "recorder", "stop", handle.session_id)
        handle.state = "stopped"
        return RecordingResult(id=handle.id)

    def tap_inbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        _log(self.trace, "recorder", "tap_inbound", handle.session_id, frame)

    def tap_outbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        _log(self.trace, "recorder", "tap_outbound", handle.session_id, frame)

    def reset(self) -> None:
        _log(self.trace, "recorder", "reset")

    def close(self) -> None:
        _log(self.trace, "recorder", "close")


class TracedVAD(MockVADProvider):
    def __init__(self, trace: Trace, events: list[VADEvent | None]) -> None:
        super().__init__(events)
        self.trace = trace

    def process(self, frame: AudioFrame, stream: str) -> VADEvent | None:
        _log(self.trace, "vad", "process", stream, frame)
        return super().process(frame, stream)

    def configure(self, config: VADConfig) -> None:
        _log(self.trace, "vad", "configure", config=_plain(asdict(config)))
        super().configure(config)

    def reset(self, stream: str) -> None:
        _log(self.trace, "vad", "reset", stream)
        super().reset(stream)

    def close(self) -> None:
        _log(self.trace, "vad", "close")
        super().close()


class TracedDTMF(MockDTMFDetector):
    def __init__(self, trace: Trace, events: list[DTMFEvent | None]) -> None:
        super().__init__(events)
        self.trace = trace

    def process(self, frame: AudioFrame, stream: str) -> DTMFEvent | None:
        _log(self.trace, "dtmf", "process", stream, frame)
        return super().process(frame, stream)

    def reset(self, stream: str) -> None:
        _log(self.trace, "dtmf", "reset", stream)
        super().reset(stream)

    def close(self) -> None:
        _log(self.trace, "dtmf", "close")
        super().close()


class TracedDiarization(MockDiarizationProvider):
    def __init__(self, trace: Trace, results: list[DiarizationResult | None]) -> None:
        super().__init__(results)
        self.trace = trace

    def process(self, frame: AudioFrame, stream: str) -> DiarizationResult | None:
        _log(self.trace, "diarization", "process", stream, frame)
        return super().process(frame, stream)

    def reset(self, stream: str) -> None:
        _log(self.trace, "diarization", "reset", stream)
        super().reset(stream)

    def close(self) -> None:
        _log(self.trace, "diarization", "close")
        super().close()


class TracedTurnDetector(MockTurnDetector):
    def __init__(self, trace: Trace, decisions: list[TurnDecision]) -> None:
        super().__init__(decisions)
        self.trace = trace

    def reset(self) -> None:
        _log(self.trace, "turn_detector", "reset")
        super().reset()

    def close(self) -> None:
        _log(self.trace, "turn_detector", "close")
        super().close()


def _format(spec: Mapping[str, Any] | None) -> AudioFormat:
    return AudioFormat(**spec) if spec is not None else AudioFormat()


def add_pipeline_stages(
    pipeline: AudioPipelineConfig, spec: Mapping[str, Any], trace: Trace
) -> None:
    """Set the scripted stages and the format contract ``voice.open`` names."""
    if "aec" in spec:
        pipeline.aec = _TracedAEC(trace, spec["aec"])
    if "agc" in spec:
        pipeline.agc = _TracedAGC(trace, spec["agc"])
    if "denoiser" in spec:
        pipeline.denoiser = _TracedDenoiser(trace, spec["denoiser"])
    pipeline.postprocessors = [
        _TracedPostProcessor(trace, p) for p in spec.get("postprocessors", [])
    ]
    if "recorder" in spec:
        pipeline.recorder = _TracedRecorder(trace)
        mode = spec["recorder"].get("mode", RecordingMode.BOTH.value)
        pipeline.recording_config = RecordingConfig(mode=RecordingMode(mode))
    if "vad_config" in spec:
        pipeline.vad_config = VADConfig(**spec["vad_config"])
    if "contract" in spec:
        contract = spec["contract"]
        pipeline.contract = AudioPipelineContract(
            transport_inbound_format=_format(contract.get("transport_inbound_format")),
            transport_outbound_format=_format(contract.get("transport_outbound_format")),
            internal_format=_format(contract.get("internal_format")),
        )


def read_stages(trace: Trace, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The traced calls ``voice.read`` returns: of the named stages and calls only."""
    return [
        call
        for call in trace
        if ("stages" not in spec or call["stage"] in spec["stages"])
        and ("calls" not in spec or call["call"] in spec["calls"])
    ]


STAGE_ARGUMENTS = {
    "aec",
    "agc",
    "denoiser",
    "postprocessors",
    "recorder",
    "vad_config",
    "contract",
}
READ_ARGUMENTS = {"stages", "calls"}
