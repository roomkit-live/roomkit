"""Pipeline debug taps — diagnostic audio capture at stage boundaries (RFC §12.3.15).

Captures audio at every processing stage into separate WAV files, allowing
developers to compare the signal before and after each transformation::

    config = AudioPipelineConfig(
        denoiser=denoiser,
        vad=vad,
        debug_taps=PipelineDebugTaps(output_dir="./debug_audio/"),
    )

Output files are numbered by pipeline order::

    debug_audio/
      {session_id}_01_raw.wav
      {session_id}_02_post_aec.wav
      {session_id}_03_post_agc.wav
      {session_id}_04_post_denoiser.wav
      {session_id}_05_post_vad_speech_001.wav
      {session_id}_06_outbound_raw.wav
      {session_id}_07_outbound_final.wav
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger("roomkit.voice.pipeline.debug_taps")

# Stage name → (numeric prefix, label)
_STAGE_ORDER: dict[str, tuple[str, str]] = {
    "raw": ("01", "raw"),
    "post_aec": ("02", "post_aec"),
    "post_agc": ("03", "post_agc"),
    "post_denoiser": ("04", "post_denoiser"),
    "post_vad_speech": ("05", "post_vad_speech"),
    "outbound_raw": ("06", "outbound_raw"),
    "outbound_final": ("07", "outbound_final"),
}

ALL_STAGES = list(_STAGE_ORDER.keys())


@dataclass
class PipelineDebugTaps:
    """Configuration for pipeline debug audio capture.

    Attributes:
        output_dir: Directory for debug WAV files.
        stages: Which stages to capture — a list of stage names, or
            ``["all"]`` to capture every stage boundary. Valid names:
            ``raw``, ``post_aec``, ``post_agc``, ``post_denoiser``,
            ``post_vad_speech``, ``outbound_raw``, ``outbound_final``.
        session_scoped: Prefix files with session ID.
    """

    output_dir: str = ""
    """Directory for debug WAV files."""

    stages: list[str] = field(default_factory=lambda: ["all"])
    """Which stages to capture (default: all)."""

    session_scoped: bool = True
    """Prefix files with session ID."""


class _DebugWavWriter:
    """Manages a single WAV writer that opens lazily on first write."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._writer: wave.Wave_write | None = None
        self._bytes_written: int = 0

    def write(self, frame: AudioFrame) -> None:
        """Write audio frame data to the WAV file."""
        if not frame.data:
            return
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = wave.open(str(self._path), "wb")  # noqa: SIM115
            self._writer.setnchannels(frame.channels)
            self._writer.setsampwidth(frame.sample_width)
            self._writer.setframerate(frame.sample_rate)
        self._writer.writeframes(frame.data)
        self._bytes_written += len(frame.data)

    def write_raw(
        self,
        data: bytes,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> None:
        """Write raw PCM bytes (for VAD speech segments)."""
        if not data:
            return
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = wave.open(str(self._path), "wb")  # noqa: SIM115
            self._writer.setnchannels(channels)
            self._writer.setsampwidth(sample_width)
            self._writer.setframerate(sample_rate)
        self._writer.writeframes(data)
        self._bytes_written += len(data)

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class DebugTapSession:
    """Manages debug tap writers for a single voice session."""

    def __init__(self, config: PipelineDebugTaps, session_id: str) -> None:
        self._config = config
        self._session_id = session_id
        self._output_dir = Path(config.output_dir)
        self._enabled_stages = self._resolve_stages(config.stages)
        self._writers: dict[str, _DebugWavWriter] = {}
        self._vad_segment_count: int = 0

        logger.warning(
            "Pipeline debug taps enabled for session %s — stages: %s, output: %s",
            session_id,
            self._enabled_stages,
            self._output_dir,
        )

    @staticmethod
    def _resolve_stages(stages: list[str]) -> set[str]:
        if not stages or "all" in stages:
            return set(ALL_STAGES)
        valid = set()
        for s in stages:
            if s in _STAGE_ORDER:
                valid.add(s)
            else:
                logger.warning("Unknown debug tap stage: %r (ignored)", s)
        return valid

    def _get_writer(self, stage: str) -> _DebugWavWriter:
        """Get or create a writer for the given stage."""
        if stage not in self._writers:
            prefix, label = _STAGE_ORDER[stage]
            if self._config.session_scoped:
                filename = f"{self._session_id}_{prefix}_{label}.wav"
            else:
                filename = f"{prefix}_{label}.wav"
            self._writers[stage] = _DebugWavWriter(self._output_dir / filename)
        return self._writers[stage]

    def tap(self, stage: str, frame: AudioFrame) -> None:
        """Record a frame at the given pipeline stage (if enabled)."""
        if stage not in self._enabled_stages:
            return
        try:
            self._get_writer(stage).write(frame)
        except Exception:
            logger.exception("Debug tap write error at stage %s", stage)

    def tap_vad_speech(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> None:
        """Record accumulated VAD speech segment."""
        if "post_vad_speech" not in self._enabled_stages:
            return
        self._vad_segment_count += 1
        prefix, label = _STAGE_ORDER["post_vad_speech"]
        seg = f"{self._vad_segment_count:03d}"
        if self._config.session_scoped:
            filename = f"{self._session_id}_{prefix}_{label}_{seg}.wav"
        else:
            filename = f"{prefix}_{label}_{seg}.wav"
        writer = _DebugWavWriter(self._output_dir / filename)
        try:
            writer.write_raw(
                audio_bytes,
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
            )
        except Exception:
            logger.exception(
                "Debug tap write error for VAD speech segment %d",
                self._vad_segment_count,
            )
        finally:
            writer.close()

    @property
    def total_bytes_written(self) -> int:
        return sum(w.bytes_written for w in self._writers.values())

    def close(self) -> None:
        """Close all WAV writers."""
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                logger.exception("Error closing debug tap writer")
        self._writers.clear()
        logger.debug(
            "Debug taps closed for session %s — total bytes: %d",
            self._session_id,
            self.total_bytes_written,
        )
