"""What the audio pipeline reports about itself.

Instrumentation only — spans, per-stage timings, frame and byte counters.  It
is kept apart from :mod:`roomkit.voice.pipeline.engine` because none of it
touches the audio: the engine calls in, the audio never depends on the answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.telemetry.base import Attr, SpanKind
from roomkit.voice.base import VoiceCapability

if TYPE_CHECKING:
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.voice.pipeline.config import AudioPipelineConfig

_METRIC_INTERVAL_FRAMES = 500
"""How many frames one stream processes between two metric emissions."""


def active_stage_names(
    config: AudioPipelineConfig,
    *,
    resampling: bool,
    backend_capabilities: VoiceCapability,
) -> str:
    """Name the inbound stages that will actually run, in pipeline order.

    Reported once per speech segment as ``pipeline.stages``, so a trace shows
    what the audio went through rather than what was configured — AEC and AGC
    are configured but skipped when the backend cancels and levels natively.
    """
    native_aec = VoiceCapability.NATIVE_AEC in backend_capabilities
    native_agc = VoiceCapability.NATIVE_AGC in backend_capabilities
    stages = (
        ("resampler", resampling),
        ("dtmf", config.dtmf is not None),
        ("aec", config.aec is not None and not native_aec),
        (
            "agc",
            (config.agc is not None or config.agc_config is not None) and not native_agc,
        ),
        ("denoiser", config.denoiser is not None),
        ("vad", config.vad is not None),
        ("diarization", config.diarization is not None),
    )
    return ",".join(name for name, runs in stages if runs)


class _PipelineTelemetry:
    """Spans and counters for one :class:`AudioPipeline`.

    Every method is a cheap no-op when no provider is configured — the engine
    calls these on the media path, once or more per frame, and an unobserved
    pipeline must not pay for the observation it isn't making.

    State is keyed by stream, like the stages themselves: :meth:`release` frees
    one stream's worth when its track goes away.
    """

    def __init__(
        self,
        provider: TelemetryProvider | None,
        stages: str,
    ) -> None:
        self._provider = provider
        self._stages = stages
        # Open speech-segment spans (stream -> span_id)
        self._segment_spans: dict[str, str] = {}
        # Cumulative per-stage time inside the open segment (stream -> {stage: ns})
        self._stage_timings: dict[str, dict[str, int]] = {}
        # Frames seen since the segment opened (stream -> count)
        self._segment_frames: dict[str, int] = {}
        # Span the segments hang under (stream -> VOICE_SESSION span_id)
        self._parent_spans: dict[str, str] = {}
        self._frames: dict[str, int] = {}
        self._bytes: dict[str, int] = {}

    def set_parent_span(self, stream: str, span_id: str) -> None:
        """Set the parent span (VOICE_SESSION) that segment spans hang under."""
        self._parent_spans[stream] = span_id

    def in_segment(self, stream: str) -> bool:
        """Whether a speech segment is open — the only time stages are timed."""
        return stream in self._segment_spans

    def start_segment(self, stream: str) -> None:
        """Open a speech-segment span; timings and frames accrue against it."""
        if self._provider is None:
            return
        self._segment_spans[stream] = self._provider.start_span(
            SpanKind.PIPELINE_SPEECH_SEGMENT,
            "pipeline.speech_segment",
            parent_id=self._parent_spans.get(stream),
            session_id=stream,
            attributes={Attr.PIPELINE_STAGES: self._stages},
        )
        self._stage_timings[stream] = {}
        self._segment_frames[stream] = 0

    def end_segment(self, stream: str) -> None:
        """Close the open segment span, reporting its frames and stage timings."""
        span = self._segment_spans.pop(stream, None)
        if span is None or self._provider is None:
            return
        attributes: dict[str, Any] = {
            Attr.PIPELINE_FRAMES: self._segment_frames.pop(stream, 0),
        }
        for stage, ns in self._stage_timings.pop(stream, {}).items():
            attributes[f"pipeline.{stage}_ms"] = round(ns / 1_000_000, 2)
        self._provider.end_span(span, attributes=attributes)

    def count_frame(self, stream: str) -> None:
        """Count one frame against the open segment."""
        frames = self._segment_frames.get(stream)
        if frames is not None:
            self._segment_frames[stream] = frames + 1

    def add_stage_time(self, stream: str, stage: str, elapsed_ns: int) -> None:
        """Add one stage's time to what the open segment has accumulated."""
        timings = self._stage_timings.get(stream)
        if timings is not None:
            timings[stage] = timings.get(stage, 0) + elapsed_ns

    def record_frame(self, stream: str, size: int) -> None:
        """Count a processed frame, emitting the periodic pipeline metrics.

        Counters are per stream so concurrent streams don't contend for one.
        """
        if self._provider is None:
            return
        frames = self._frames.get(stream, 0) + 1
        self._frames[stream] = frames
        total_bytes = self._bytes.get(stream, 0) + size
        self._bytes[stream] = total_bytes
        if frames % _METRIC_INTERVAL_FRAMES != 0:
            return
        self._provider.record_metric(
            "roomkit.pipeline.frame_count",
            float(frames),
            attributes={"session_id": stream},
        )
        self._provider.record_metric(
            "roomkit.pipeline.bytes_processed",
            float(total_bytes),
            unit="bytes",
            attributes={"session_id": stream},
        )

    def release(self, stream: str) -> None:
        """Drop everything one stream held, closing its open segment first."""
        self.end_segment(stream)
        self._parent_spans.pop(stream, None)
        self._stage_timings.pop(stream, None)
        self._segment_frames.pop(stream, None)
        self._frames.pop(stream, None)
        self._bytes.pop(stream, None)

    def clear(self) -> None:
        """Drop every stream's state, abandoning open spans rather than ending them.

        This backs ``AudioPipeline.reset()``, which restarts the pipeline
        wholesale.  Ending the spans would report segments that no longer
        describe anything: their frames and timings die with the reset.
        """
        self._segment_spans.clear()
        self._stage_timings.clear()
        self._segment_frames.clear()
        self._parent_spans.clear()
        self._frames.clear()
        self._bytes.clear()
