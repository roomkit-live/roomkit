"""WAV file recorder for debug audio capture."""

from __future__ import annotations

import logging
import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from roomkit.voice.pipeline.recorder._wav_writer import (
    WavWriterThread,
    _OpOpen,
    _WriterSession,
    make_frame_op,
)
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


class WavFileRecorder(AudioRecorder):
    """Debug WAV file recorder using Python's stdlib ``wave`` module.

    Writes raw PCM audio from the pipeline to ``.wav`` files on disk.
    Useful for inspecting audio quality, AEC effectiveness, and
    denoiser behavior.

    Supports three channel modes:

    - **MIXED**: single mono WAV with inbound + outbound averaged together.
    - **SEPARATE**: two WAV files (``*_inbound.wav`` and ``*_outbound.wav``).
    - **STEREO**: single stereo WAV (inbound=left, outbound=right).

    The taps run on the realtime frame path, so they only enqueue: all
    disk I/O — file opens, writes, spooling, mixing — happens on a
    dedicated writer thread (:mod:`._wav_writer`). ``stop()`` queues
    behind the session's remaining frames and waits, so the files it
    reports are complete when it returns.
    """

    def __init__(self) -> None:
        self._writer = WavWriterThread()
        # Tap-side view: enough to gate a frame without touching writer
        # state. Keyed by recording id; removed by stop().
        self._active: dict[str, tuple[RecordingHandle, RecordingConfig]] = {}

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

        if config.channels in (RecordingChannelMode.SEPARATE, RecordingChannelMode.ALL):
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

        self._active[rec_id] = (handle, config)
        self._writer.ensure_started()
        self._writer.submit(
            _OpOpen(_WriterSession(handle=handle, config=config, output_dir=output_dir))
        )
        return handle

    def stop(self, handle: RecordingHandle) -> RecordingResult:
        entry = self._active.pop(handle.id, None)
        if entry is None:
            return RecordingResult(id=handle.id)
        handle.state = "stopped"
        result = self._writer.stop_session(handle.id)
        if result is None:
            # The writer could not finalise (wedged disk); report what is
            # known rather than pretending the files exist.
            return RecordingResult(id=handle.id, format="wav", mode=entry[1].channels)
        return result

    def tap_inbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        self._tap(handle, frame, inbound=True)

    def tap_outbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        self._tap(handle, frame, inbound=False)

    def _tap(self, handle: RecordingHandle, frame: AudioFrame, *, inbound: bool) -> None:
        entry = self._active.get(handle.id)
        if entry is None or handle.state != "recording":
            return
        config = entry[1]
        skip = RecordingMode.OUTBOUND_ONLY if inbound else RecordingMode.INBOUND_ONLY
        if config.mode == skip:
            return
        self._writer.submit_frame(
            make_frame_op(
                handle.id,
                inbound=inbound,
                data=frame.data,
                sample_rate=frame.sample_rate,
                channels=frame.channels,
                sample_width=frame.sample_width,
            )
        )

    def reset(self) -> None:
        # Stop all active sessions
        for rec_id in list(self._active):
            handle = self._active[rec_id][0]
            self.stop(handle)

    def close(self) -> None:
        self.reset()
        self._writer.shutdown()
