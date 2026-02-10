"""WAV file recorder for debug audio capture."""

from __future__ import annotations

import logging
import re
import struct
import tempfile
import uuid
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from roomkit.voice.pipeline.recorder.base import (
    AudioRecorder,
    RecordingChannelMode,
    RecordingConfig,
    RecordingHandle,
    RecordingMode,
    RecordingResult,
    RecordingTrigger,
)

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger(__name__)

# Pattern for sanitizing session IDs used in filenames
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_\-.]")


def _sanitize_filename_component(value: str) -> str:
    """Strip path separators and special characters from a filename component."""
    return _SAFE_FILENAME_RE.sub("_", value)


@dataclass
class _WavSession:
    """Internal state for an active WAV recording."""

    handle: RecordingHandle
    config: RecordingConfig
    sample_rate: int = 0
    channels: int = 1
    sample_width: int = 2
    output_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()))

    # SEPARATE mode: open wave files
    inbound_writer: wave.Wave_write | None = None
    outbound_writer: wave.Wave_write | None = None

    # MIXED/STEREO mode: byte buffers
    inbound_buf: bytearray = field(default_factory=bytearray)
    outbound_buf: bytearray = field(default_factory=bytearray)

    # Tracking
    inbound_frames: int = 0
    outbound_frames: int = 0

    def _init_format(self, frame: AudioFrame) -> None:
        """Capture format from the first frame seen."""
        if self.sample_rate == 0:
            self.sample_rate = frame.sample_rate
            self.channels = frame.channels
            self.sample_width = frame.sample_width


class WavFileRecorder(AudioRecorder):
    """Debug WAV file recorder using Python's stdlib ``wave`` module.

    Writes raw PCM audio from the pipeline to ``.wav`` files on disk.
    Useful for inspecting audio quality, AEC effectiveness, and
    denoiser behavior.

    Supports three channel modes:

    - **MIXED**: single mono WAV with inbound + outbound averaged together.
    - **SEPARATE**: two WAV files (``*_inbound.wav`` and ``*_outbound.wav``).
    - **STEREO**: single stereo WAV (inbound=left, outbound=right).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _WavSession] = {}

    @property
    def name(self) -> str:
        return "WavFileRecorder"

    def start(self, session: VoiceSession, config: RecordingConfig) -> RecordingHandle:
        if config.trigger == RecordingTrigger.SPEECH_ONLY:
            logger.warning(
                "WavFileRecorder does not support SPEECH_ONLY trigger "
                "(recorder taps run before VAD). Falling back to ALWAYS."
            )

        rec_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        timestamp = now.strftime("%Y%m%dT%H%M%S")
        safe_session_id = _sanitize_filename_component(session.id)
        base_name = f"{safe_session_id}_{timestamp}"

        fallback_dir = Path(tempfile.gettempdir())
        if config.storage:
            # Reject paths with traversal components
            if ".." in Path(config.storage).parts:
                logger.warning(
                    "Suspicious storage path %r contains '..'; falling back to temp directory",
                    config.storage,
                )
                output_dir = fallback_dir
            else:
                output_dir = Path(config.storage).resolve()
        else:
            output_dir = fallback_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.channels == RecordingChannelMode.SEPARATE:
            path = str(output_dir / base_name)
        else:
            path = str(output_dir / f"{base_name}.wav")

        handle = RecordingHandle(
            id=rec_id,
            session_id=session.id,
            state="recording",
            started_at=now,
            path=path,
        )

        ws = _WavSession(handle=handle, config=config, output_dir=output_dir)
        self._sessions[rec_id] = ws

        # For SEPARATE mode, we can't open writers yet — we need the
        # sample format from the first frame.  They'll be opened lazily
        # in tap_inbound / tap_outbound.

        return handle

    def stop(self, handle: RecordingHandle) -> RecordingResult:
        ws = self._sessions.pop(handle.id, None)
        if ws is None:
            return RecordingResult(id=handle.id)

        handle.state = "stopped"
        urls: list[str] = []
        total_size = 0

        if ws.config.channels == RecordingChannelMode.SEPARATE:
            # Close the open wave writers
            for writer, label in [
                (ws.inbound_writer, "inbound"),
                (ws.outbound_writer, "outbound"),
            ]:
                if writer is not None:
                    writer.close()
                    p = Path(f"{ws.handle.path}_{label}.wav")
                    urls.append(str(p))
                    total_size += p.stat().st_size

        elif ws.config.channels == RecordingChannelMode.MIXED:
            path = Path(ws.handle.path)
            self._write_mixed(ws, path)
            if path.exists():
                urls.append(str(path))
                total_size = path.stat().st_size

        elif ws.config.channels == RecordingChannelMode.STEREO:
            path = Path(ws.handle.path)
            self._write_stereo(ws, path)
            if path.exists():
                urls.append(str(path))
                total_size = path.stat().st_size

        duration = 0.0
        # Compute duration from byte lengths
        if ws.sample_rate > 0 and ws.sample_width > 0:
            if ws.config.channels == RecordingChannelMode.SEPARATE:
                # Duration is the longer of the two streams
                in_samples = ws.inbound_frames
                out_samples = ws.outbound_frames
                duration = max(in_samples, out_samples) / ws.sample_rate
            else:
                # Use buffer lengths
                in_samples = len(ws.inbound_buf) // (ws.sample_width * ws.channels)
                out_samples = len(ws.outbound_buf) // (ws.sample_width * ws.channels)
                duration = max(in_samples, out_samples) / ws.sample_rate

        return RecordingResult(
            id=handle.id,
            urls=urls,
            duration_seconds=duration,
            format="wav",
            mode=ws.config.channels,
            size_bytes=total_size,
        )

    def tap_inbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        ws = self._sessions.get(handle.id)
        if ws is None or handle.state != "recording":
            return

        if ws.config.mode == RecordingMode.OUTBOUND_ONLY:
            return

        ws._init_format(frame)

        if ws.config.channels == RecordingChannelMode.SEPARATE:
            if ws.inbound_writer is None:
                ws.inbound_writer = self._open_writer(Path(f"{ws.handle.path}_inbound.wav"), ws)
            ws.inbound_writer.writeframes(frame.data)
            ws.inbound_frames += len(frame.data) // (ws.sample_width * ws.channels)
        else:
            ws.inbound_buf.extend(frame.data)
            ws.inbound_frames += len(frame.data) // (ws.sample_width * ws.channels)

    def tap_outbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        ws = self._sessions.get(handle.id)
        if ws is None or handle.state != "recording":
            return

        if ws.config.mode == RecordingMode.INBOUND_ONLY:
            return

        ws._init_format(frame)

        if ws.config.channels == RecordingChannelMode.SEPARATE:
            if ws.outbound_writer is None:
                ws.outbound_writer = self._open_writer(Path(f"{ws.handle.path}_outbound.wav"), ws)
            ws.outbound_writer.writeframes(frame.data)
            ws.outbound_frames += len(frame.data) // (ws.sample_width * ws.channels)
        else:
            ws.outbound_buf.extend(frame.data)
            ws.outbound_frames += len(frame.data) // (ws.sample_width * ws.channels)

    def reset(self) -> None:
        # Stop all active sessions
        for rec_id in list(self._sessions):
            handle = self._sessions[rec_id].handle
            self.stop(handle)

    def close(self) -> None:
        self.reset()

    # ---- internal helpers ----

    @staticmethod
    def _open_writer(path: Path, ws: _WavSession) -> wave.Wave_write:
        """Open a new WAV file writer with the session's audio format."""
        w = wave.open(str(path), "wb")  # noqa: SIM115
        w.setnchannels(ws.channels)
        w.setsampwidth(ws.sample_width)
        w.setframerate(ws.sample_rate)
        return w

    @staticmethod
    def _write_mixed(ws: _WavSession, path: Path) -> None:
        """Mix inbound + outbound into a single mono WAV."""
        if not ws.inbound_buf and not ws.outbound_buf:
            return

        sw = ws.sample_width
        has_inbound = bool(ws.inbound_buf)
        has_outbound = bool(ws.outbound_buf)

        # If only one direction has data, write it directly (no mixing)
        if has_inbound and not has_outbound:
            data = bytes(ws.inbound_buf)
        elif has_outbound and not has_inbound:
            data = bytes(ws.outbound_buf)
        else:
            # Both directions present — mix by summing with clamp
            max_len = max(len(ws.inbound_buf), len(ws.outbound_buf))
            inb = bytes(ws.inbound_buf).ljust(max_len, b"\x00")
            outb = bytes(ws.outbound_buf).ljust(max_len, b"\x00")

            fmt = "<h" if sw == 2 else "<b"
            sample_count = max_len // sw
            min_val = -(1 << (sw * 8 - 1))
            max_val = (1 << (sw * 8 - 1)) - 1
            mixed = bytearray(max_len)

            for i in range(sample_count):
                offset = i * sw
                a = struct.unpack_from(fmt, inb, offset)[0]
                b = struct.unpack_from(fmt, outb, offset)[0]
                struct.pack_into(fmt, mixed, offset, max(min_val, min(max_val, a + b)))

            data = bytes(mixed)

        with wave.open(str(path), "wb") as w:
            w.setnchannels(ws.channels)
            w.setsampwidth(sw)
            w.setframerate(ws.sample_rate)
            w.writeframes(data)

    @staticmethod
    def _write_stereo(ws: _WavSession, path: Path) -> None:
        """Write inbound (left) + outbound (right) as a stereo WAV."""
        if not ws.inbound_buf and not ws.outbound_buf:
            return

        sw = ws.sample_width
        max_len = max(len(ws.inbound_buf), len(ws.outbound_buf))
        inb = bytes(ws.inbound_buf).ljust(max_len, b"\x00")
        outb = bytes(ws.outbound_buf).ljust(max_len, b"\x00")

        # Interleave: L R L R ...
        sample_count = max_len // sw
        stereo = bytearray(max_len * 2)

        for i in range(sample_count):
            src_offset = i * sw
            dst_offset = i * sw * 2
            stereo[dst_offset : dst_offset + sw] = inb[src_offset : src_offset + sw]
            stereo[dst_offset + sw : dst_offset + sw * 2] = outb[src_offset : src_offset + sw]

        with wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(sw)
            w.setframerate(ws.sample_rate)
            w.writeframes(bytes(stereo))
