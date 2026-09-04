"""WAV and PCM helpers for the voice test bench.

Stdlib only (``wave``, ``array``, ``math``): the bench runs wherever the
default test suite runs, with no numpy. :class:`PCMAudio` is the decoded form
every helper here speaks — the samples plus the format needed to frame them —
and :func:`pcm_frames` turns one into the 20 ms :class:`AudioFrame` sequence a
:class:`~roomkit.voice.backends.base.VoiceBackend` delivers.

Not the recorder: :class:`~roomkit.voice.pipeline.recorder.WavFileRecorder`
writes a live call from a pipeline tap on its own thread, off the frame path.
These are the synchronous helpers a test fixture reads a corpus file and
writes a capture with.
"""

from __future__ import annotations

import math
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from roomkit.voice.audio_frame import AudioFrame

DEFAULT_SAMPLE_RATE = 16000
"""The bench's native rate: the corpus is stored at it, the mock STT/TTS speak it."""


@dataclass(frozen=True)
class PCMAudio:
    """Decoded PCM audio: the samples and the format needed to frame them.

    ``sample_width`` is bytes per sample; 16-bit signed little-endian PCM
    (``sample_width=2``) is what every WAV the bench reads or writes carries.
    Two clips of the same format concatenate with ``+``.
    """

    data: bytes
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")
        if self.sample_width not in (1, 2, 4):
            raise ValueError(f"sample_width must be 1, 2, or 4, got {self.sample_width}")
        if len(self.data) % self.frame_align != 0:
            raise ValueError(
                f"data length ({len(self.data)}) must be divisible by "
                f"sample_width * channels ({self.frame_align})"
            )

    @property
    def frame_align(self) -> int:
        """Bytes per sample frame: one sample of every channel."""
        return self.sample_width * self.channels

    @property
    def duration_ms(self) -> float:
        return len(self.data) / self.frame_align / self.sample_rate * 1000.0

    def frame_bytes(self, frame_ms: int) -> int:
        """Bytes in one *frame_ms* frame at this format."""
        return self.sample_rate * frame_ms // 1000 * self.frame_align

    def __add__(self, other: PCMAudio) -> PCMAudio:
        if (self.sample_rate, self.channels, self.sample_width) != (
            other.sample_rate,
            other.channels,
            other.sample_width,
        ):
            raise ValueError("cannot concatenate PCM audio of different formats")
        return PCMAudio(
            data=self.data + other.data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
        )


def read_wav(path: str | Path) -> PCMAudio:
    """Read a whole WAV file into a :class:`PCMAudio`."""
    with wave.open(str(path), "rb") as wav:
        return PCMAudio(
            data=wav.readframes(wav.getnframes()),
            sample_rate=wav.getframerate(),
            channels=wav.getnchannels(),
            sample_width=wav.getsampwidth(),
        )


def write_wav(path: str | Path, audio: PCMAudio) -> Path:
    """Write *audio* as a WAV file, creating the parent directory if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(audio.channels)
        wav.setsampwidth(audio.sample_width)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(audio.data)
    return target


def pcm_frames(audio: PCMAudio, *, frame_ms: int = 20) -> list[AudioFrame]:
    """Cut *audio* into *frame_ms* frames, timestamped from zero.

    The last frame is padded with silence to a whole frame: a backend always
    delivers full frames, and so does a real transport.
    """
    size = audio.frame_bytes(frame_ms)
    if size <= 0:
        raise ValueError(f"frame_ms must cover at least one sample, got {frame_ms}")
    frames: list[AudioFrame] = []
    for i, start in enumerate(range(0, len(audio.data), size)):
        chunk = audio.data[start : start + size]
        if len(chunk) < size:
            chunk = chunk + bytes(size - len(chunk))
        frames.append(
            AudioFrame(
                data=chunk,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                sample_width=audio.sample_width,
                timestamp_ms=float(i * frame_ms),
            )
        )
    return frames


def silence(duration_ms: int, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> PCMAudio:
    """*duration_ms* of digital silence, 16-bit mono."""
    samples = sample_rate * duration_ms // 1000
    return PCMAudio(data=bytes(samples * 2), sample_rate=sample_rate)


def tone(
    duration_ms: int,
    *,
    frequency_hz: float = 440.0,
    amplitude: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> PCMAudio:
    """A sine tone, 16-bit mono: loud enough for an energy VAD, cheap to make.

    *amplitude* is a fraction of full scale (0 to 1).
    """
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError(f"amplitude must be between 0 and 1, got {amplitude}")
    samples = sample_rate * duration_ms // 1000
    peak = amplitude * 32767
    step = 2.0 * math.pi * frequency_hz / sample_rate
    pcm = array("h", (int(peak * math.sin(step * i)) for i in range(samples)))
    return PCMAudio(data=pcm.tobytes(), sample_rate=sample_rate)
