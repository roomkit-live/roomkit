"""Writer thread for :class:`WavFileRecorder` — disk I/O off the frame path.

The pipeline taps run on the realtime event loop, once per 20 ms frame.
Everything that touches the filesystem — opening files, ``writeframes``,
spooling, mixing, finalising — happens here instead, on one daemon thread
fed by a bounded queue. A full queue drops the frame (recording is an
observer of the call, never a brake on it) and the drop is counted and
logged. ``stop`` is an op like any other: it queues behind the session's
remaining frames, so a caller that waits on it gets complete files.

MIXED/STEREO/ALL modes spool each direction to a raw PCM file as frames
arrive and mix/interleave at stop — memory stays flat however long the
call runs (the previous design accumulated ~1.9 MB/min/session in RAM and
mixed with a per-sample Python loop at stop, on the event loop).
"""

from __future__ import annotations

import logging
import queue
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from roomkit.voice.pipeline.recorder.base import (
    RecordingChannelMode,
    RecordingConfig,
    RecordingHandle,
    RecordingMode,
    RecordingResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# ~20 s of audio at one 20 ms frame per direction per session, shared
# across sessions. Beyond it, recording lags reality and frames drop.
_MAX_QUEUED_OPS = 1024

# How many drops between WARNING lines, per session.
_DROP_LOG_INTERVAL = 250

# Minimum gap (seconds) before inserting silence. Gaps below this are
# processing jitter, not real silence. Frames typically arrive every
# 20 ms, so 30 ms accommodates jitter without swallowing real pauses.
_SILENCE_GAP_THRESHOLD = 0.03


@dataclass
class _OpOpen:
    session: _WriterSession


@dataclass
class _OpFrame:
    rec_id: str
    inbound: bool
    captured_at: float  # time.monotonic() at tap time, not write time
    data: bytes
    sample_rate: int
    channels: int
    sample_width: int


@dataclass
class _OpStop:
    rec_id: str
    done: threading.Event
    result: list[RecordingResult] = field(default_factory=list)


@dataclass
class _OpShutdown:
    done: threading.Event


_Op = _OpOpen | _OpFrame | _OpStop | _OpShutdown


@dataclass
class _WriterSession:
    """Thread-side state for one active recording. Touched only by the writer."""

    handle: RecordingHandle
    config: RecordingConfig
    output_dir: Path
    sample_rate: int = 0
    channels: int = 1
    sample_width: int = 2

    # SEPARATE mode: wave writers, opened on the first frame per direction.
    inbound_writer: wave.Wave_write | None = None
    outbound_writer: wave.Wave_write | None = None

    # MIXED/STEREO/ALL modes: raw PCM spool files per direction.
    inbound_spool: BinaryIO | None = None
    outbound_spool: BinaryIO | None = None
    inbound_bytes: int = 0
    outbound_bytes: int = 0

    inbound_frames: int = 0
    outbound_frames: int = 0
    last_inbound_ts: float = 0.0
    last_outbound_ts: float = 0.0
    dropped: int = 0

    def spool_path(self, label: str) -> Path:
        return Path(f"{self.handle.path}_{label}.raw")


class WavWriterThread:
    """One daemon thread draining recording ops for every WAV session."""

    def __init__(self) -> None:
        self._queue: queue.Queue[_Op] = queue.Queue(maxsize=_MAX_QUEUED_OPS)
        self._sessions: dict[str, _WriterSession] = {}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Tap-side drop counters, keyed by rec_id — the tap increments them
        # when the queue refuses a frame, so the loss is attributable even
        # though the writer never saw the frame.
        self.tap_drops: dict[str, int] = {}

    # ---- caller side (event loop) ----

    def ensure_started(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="roomkit-wav-recorder", daemon=True
                )
                self._thread.start()

    def submit(self, op: _Op) -> bool:
        """Enqueue an op without blocking. Returns False when refused (full)."""
        try:
            self._queue.put_nowait(op)
            return True
        except queue.Full:
            return False

    def submit_frame(self, op: _OpFrame) -> None:
        """Enqueue a frame; a refusal is counted and periodically logged."""
        if self.submit(op):
            return
        drops = self.tap_drops.get(op.rec_id, 0) + 1
        self.tap_drops[op.rec_id] = drops
        if drops % _DROP_LOG_INTERVAL == 1:
            logger.warning(
                "WAV recorder queue full: dropped %d frame(s) for recording %s "
                "— disk cannot keep up with the call",
                drops,
                op.rec_id,
            )

    def stop_session(self, rec_id: str, *, timeout: float = 10.0) -> RecordingResult | None:
        """Queue a stop behind the session's frames and wait for its result."""
        op = _OpStop(rec_id=rec_id, done=threading.Event())
        self.ensure_started()
        # A stop must not be droppable: without it the session never
        # finalises. Block (briefly) for a slot instead of put_nowait.
        try:
            self._queue.put(op, timeout=timeout)
        except queue.Full:
            logger.error("WAV recorder queue wedged; recording %s left unfinalised", rec_id)
            return None
        if not op.done.wait(timeout):
            logger.error("WAV recorder did not finalise recording %s in %.0fs", rec_id, timeout)
            return None
        self.tap_drops.pop(rec_id, None)
        return op.result[0] if op.result else None

    def shutdown(self, *, timeout: float = 10.0) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is None or not thread.is_alive():
            return
        op = _OpShutdown(done=threading.Event())
        self._queue.put(op, timeout=timeout)
        op.done.wait(timeout)
        thread.join(timeout)

    # ---- writer side (thread) ----

    def _run(self) -> None:
        while True:
            op = self._queue.get()
            try:
                if isinstance(op, _OpOpen):
                    self._handle_open(op.session)
                elif isinstance(op, _OpFrame):
                    self._handle_frame(op)
                elif isinstance(op, _OpStop):
                    self._handle_stop(op)
                else:
                    self._handle_shutdown(op)
                    return
            except Exception:
                logger.exception("WAV writer failed on %s", type(op).__name__)
                if isinstance(op, _OpStop | _OpShutdown):
                    op.done.set()
                    if isinstance(op, _OpShutdown):
                        return

    def _handle_open(self, session: _WriterSession) -> None:
        session.output_dir.mkdir(parents=True, exist_ok=True)
        self._sessions[session.handle.id] = session

    def _handle_shutdown(self, op: _OpShutdown) -> None:
        for rec_id in list(self._sessions):
            stop = _OpStop(rec_id=rec_id, done=threading.Event())
            self._handle_stop(stop)
        op.done.set()

    def _handle_frame(self, op: _OpFrame) -> None:
        ws = self._sessions.get(op.rec_id)
        if ws is None:
            return
        if ws.sample_rate == 0:
            ws.sample_rate = op.sample_rate
            ws.channels = op.channels
            ws.sample_width = op.sample_width

        last_ts = ws.last_inbound_ts if op.inbound else ws.last_outbound_ts
        silence = b""
        if last_ts > 0:
            gap = op.captured_at - last_ts
            bytes_per_sec = ws.sample_rate * ws.sample_width * ws.channels
            frame_duration = len(op.data) / bytes_per_sec if bytes_per_sec else 0.0
            silence_duration = gap - frame_duration
            if silence_duration > _SILENCE_GAP_THRESHOLD:
                n_samples = int(silence_duration * ws.sample_rate)
                silence = b"\x00" * (n_samples * ws.sample_width * ws.channels)

        if ws.config.channels == RecordingChannelMode.SEPARATE:
            writer = self._separate_writer(ws, inbound=op.inbound)
            if silence:
                writer.writeframes(silence)
            writer.writeframes(op.data)
        else:
            spool = self._spool(ws, inbound=op.inbound)
            if silence:
                spool.write(silence)
            spool.write(op.data)
            if op.inbound:
                ws.inbound_bytes += len(silence) + len(op.data)
            else:
                ws.outbound_bytes += len(silence) + len(op.data)

        unit = ws.sample_width * ws.channels
        added = (len(silence) + len(op.data)) // unit if unit else 0
        if op.inbound:
            ws.inbound_frames += added
            ws.last_inbound_ts = op.captured_at
        else:
            ws.outbound_frames += added
            ws.last_outbound_ts = op.captured_at

    def _separate_writer(self, ws: _WriterSession, *, inbound: bool) -> wave.Wave_write:
        writer = ws.inbound_writer if inbound else ws.outbound_writer
        if writer is None:
            label = "inbound" if inbound else "outbound"
            writer = wave.open(f"{ws.handle.path}_{label}.wav", "wb")  # noqa: SIM115
            writer.setnchannels(ws.channels)
            writer.setsampwidth(ws.sample_width)
            writer.setframerate(ws.sample_rate)
            if inbound:
                ws.inbound_writer = writer
            else:
                ws.outbound_writer = writer
        return writer

    def _spool(self, ws: _WriterSession, *, inbound: bool) -> BinaryIO:
        spool = ws.inbound_spool if inbound else ws.outbound_spool
        if spool is None:
            label = "inbound" if inbound else "outbound"
            spool = ws.spool_path(label).open("wb")
            if inbound:
                ws.inbound_spool = spool
            else:
                ws.outbound_spool = spool
        return spool

    def _handle_stop(self, op: _OpStop) -> None:
        ws = self._sessions.pop(op.rec_id, None)
        if ws is None:
            op.result.append(RecordingResult(id=op.rec_id))
            op.done.set()
            return
        try:
            op.result.append(self._finalize(ws))
        finally:
            op.done.set()

    # ---- finalisation ----

    def _finalize(self, ws: _WriterSession) -> RecordingResult:
        urls: list[str] = []
        total_size = 0

        def _account(p: Path) -> None:
            nonlocal total_size
            urls.append(str(p))
            total_size += p.stat().st_size

        mode = ws.config.channels
        if mode == RecordingChannelMode.SEPARATE:
            for writer, label in ((ws.inbound_writer, "inbound"), (ws.outbound_writer, "outbound")):
                if writer is not None:
                    writer.close()
                    _account(Path(f"{ws.handle.path}_{label}.wav"))
        else:
            inbound, outbound = self._read_spools(ws)
            if mode == RecordingChannelMode.ALL:
                for label, buf in (("inbound", inbound), ("outbound", outbound)):
                    if buf:
                        p = Path(f"{ws.handle.path}_{label}.wav")
                        self._write_wav(ws, buf, p, channels=ws.channels)
                        _account(p)
                mixed = _mix(inbound, outbound, ws.sample_width)
                if mixed:
                    p = Path(f"{ws.handle.path}_mixed.wav")
                    self._write_wav(ws, mixed, p, channels=ws.channels)
                    _account(p)
            elif mode == RecordingChannelMode.MIXED:
                mixed = _mix(inbound, outbound, ws.sample_width)
                if mixed:
                    p = Path(ws.handle.path)
                    self._write_wav(ws, mixed, p, channels=ws.channels)
                    _account(p)
            elif mode == RecordingChannelMode.STEREO:
                stereo = _interleave(inbound, outbound, ws.sample_width)
                if stereo:
                    p = Path(ws.handle.path)
                    self._write_wav(ws, stereo, p, channels=2)
                    _account(p)

        duration = 0.0
        if ws.sample_rate > 0:
            duration = max(ws.inbound_frames, ws.outbound_frames) / ws.sample_rate

        if ws.dropped:
            logger.warning(
                "Recording %s finished with %d dropped frame(s)", ws.handle.id, ws.dropped
            )

        return RecordingResult(
            id=ws.handle.id,
            urls=urls,
            duration_seconds=duration,
            format="wav",
            mode=mode,
            size_bytes=total_size,
        )

    def _read_spools(self, ws: _WriterSession) -> tuple[bytes, bytes]:
        """Close and read back both spool files, deleting them."""
        out: list[bytes] = []
        for spool, label in ((ws.inbound_spool, "inbound"), (ws.outbound_spool, "outbound")):
            if spool is None:
                out.append(b"")
                continue
            spool.close()
            path = ws.spool_path(label)
            out.append(path.read_bytes())
            path.unlink(missing_ok=True)
        return out[0], out[1]

    def _write_wav(self, ws: _WriterSession, data: bytes, path: Path, *, channels: int) -> None:
        if not data:
            return
        with wave.open(str(path), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(ws.sample_width)
            w.setframerate(ws.sample_rate)
            w.writeframes(data)


def _numpy() -> Any | None:
    try:
        from roomkit.voice.utils import _get_np

        return _get_np()
    except ImportError:
        return None


def _mix(inbound: bytes, outbound: bytes, sample_width: int) -> bytes:
    """Sum both directions with clamping into a single mono stream."""
    if not inbound:
        return outbound
    if not outbound:
        return inbound
    max_len = max(len(inbound), len(outbound))
    inb = inbound.ljust(max_len, b"\x00")
    outb = outbound.ljust(max_len, b"\x00")

    np = _numpy() if sample_width == 2 else None
    if np is not None:
        a = np.frombuffer(inb, dtype="<i2").astype(np.int32)
        b = np.frombuffer(outb, dtype="<i2").astype(np.int32)
        return np.clip(a + b, -32768, 32767).astype("<i2").tobytes()

    fmt = "<h" if sample_width == 2 else "<b"
    min_val = -(1 << (sample_width * 8 - 1))
    max_val = (1 << (sample_width * 8 - 1)) - 1
    mixed = bytearray(max_len)
    for offset in range(0, max_len, sample_width):
        a_val = struct.unpack_from(fmt, inb, offset)[0]
        b_val = struct.unpack_from(fmt, outb, offset)[0]
        struct.pack_into(fmt, mixed, offset, max(min_val, min(max_val, a_val + b_val)))
    return bytes(mixed)


def _interleave(inbound: bytes, outbound: bytes, sample_width: int) -> bytes:
    """Interleave inbound (left) and outbound (right) into stereo frames."""
    if not inbound and not outbound:
        return b""
    max_len = max(len(inbound), len(outbound))
    inb = inbound.ljust(max_len, b"\x00")
    outb = outbound.ljust(max_len, b"\x00")

    np = _numpy() if sample_width == 2 else None
    if np is not None:
        left = np.frombuffer(inb, dtype="<i2")
        right = np.frombuffer(outb, dtype="<i2")
        return np.column_stack((left, right)).tobytes()

    sample_count = max_len // sample_width
    stereo = bytearray(max_len * 2)
    for i in range(sample_count):
        src = i * sample_width
        dst = i * sample_width * 2
        stereo[dst : dst + sample_width] = inb[src : src + sample_width]
        stereo[dst + sample_width : dst + sample_width * 2] = outb[src : src + sample_width]
    return bytes(stereo)


def make_frame_op(
    rec_id: str,
    *,
    inbound: bool,
    data: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
    clock: Callable[[], float] = time.monotonic,
) -> _OpFrame:
    return _OpFrame(
        rec_id=rec_id,
        inbound=inbound,
        captured_at=clock(),
        data=data,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


__all__ = [
    "RecordingMode",
    "WavWriterThread",
    "_OpOpen",
    "_WriterSession",
    "make_frame_op",
]
