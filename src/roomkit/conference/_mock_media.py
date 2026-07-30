"""Media records and PCM synthesis for MockConferenceBackend.

Three things the mock has to be able to say about media, and one it has to be
able to produce:

- what format a track was published in, because participants each negotiate
  their own with the SFU and nothing obliges them to agree;
- which chunks belong to which utterance on which bot's track, because a flat
  list of chunks makes two answers talking over each other invisible, and a
  record that ignored the track would blame one room for another's;
- how long a frame took to deliver, because RFC section 12.10.4 says lane
  isolation is checkable from outside by "delaying recognition on one track and
  measuring frame delivery on another" — and something has to hold the clock.

The fourth is the audio itself: a frame in a given format, loud enough for a VAD
to call it speech or silent enough to end an utterance.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from roomkit.conference.models import TrackKind
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk

_STRUCT_FORMAT = {1: "b", 2: "h", 4: "i"}
"""Signed little-endian PCM, matching the resamplers' own width mapping."""


@dataclass(frozen=True)
class MockTrackFormat:
    """The audio format a participant negotiated for one track.

    Defaults to what the rest of the framework assumes downstream of format
    normalisation, so a track that declares nothing behaves as before.

    Example::

        dial_in = MockTrackFormat(sample_rate=8_000, channels=1, sample_width=1)
        studio = MockTrackFormat(sample_rate=48_000, channels=2, sample_width=4)
    """

    sample_rate: int = 16_000
    channels: int = 1
    sample_width: int = 2
    """Bytes per sample: 1 (8-bit), 2 (16-bit) or 4 (32-bit), always signed."""

    def __post_init__(self) -> None:
        if self.sample_width == 3:
            raise ValueError(
                "24-bit PCM has no representation in this framework: AudioFrame "
                "accepts sample widths of 1, 2 or 4 bytes, and the resamplers map "
                "only int8, int16 and int32. Use 4-byte samples to exercise a "
                "wide-format publisher."
            )
        if self.sample_width not in _STRUCT_FORMAT:
            raise ValueError(f"sample_width must be 1, 2 or 4, got {self.sample_width}")
        if self.channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")
        if self.sample_rate <= 0 or self.sample_rate > 192_000:
            raise ValueError(f"sample_rate must be between 1 and 192000, got {self.sample_rate}")

    def matches(self, frame: AudioFrame) -> bool:
        """Whether a frame is in this format."""
        return (
            frame.sample_rate == self.sample_rate
            and frame.channels == self.channels
            and frame.sample_width == self.sample_width
        )

    def describe(self) -> str:
        return f"{self.sample_rate} Hz, {self.channels} ch, {self.sample_width * 8}-bit"


@dataclass
class MockUtterance:
    """The chunks published for one utterance on one bot's track.

    An utterance runs until a chunk marks itself final. Two utterances
    published concurrently on the *same* bot therefore land in the same record,
    which is the point: that is what interleaving looks like, and a flat list
    of chunks cannot show it.

    Two bots never share a record. A bot is a track — one per conference room —
    so chunks alternating between two of them are two rooms talking at once,
    which is ordinary, while chunks alternating within one are a single track
    carrying two answers, which is not. A record that could not tell them apart
    would report the first as the second.
    """

    bot_id: str
    """The bot session the chunks were published on, which is to say the track."""

    chunks: list[AudioChunk] = field(default_factory=list)
    complete: bool = False
    """Whether a chunk with ``is_final`` closed it."""

    @property
    def data(self) -> bytes:
        """Everything published for this utterance, in arrival order."""
        return b"".join(chunk.data for chunk in self.chunks)


@dataclass(frozen=True)
class MockDelivery:
    """How long one frame took to reach every subscriber."""

    track_id: str
    kind: TrackKind
    started_at: float
    """Event-loop time the emission began."""

    elapsed: float
    """Seconds the emission took, subscribers included."""


def pcm_frame(
    audio_format: MockTrackFormat,
    *,
    ms: int = 20,
    amplitude: float = 0.25,
) -> AudioFrame:
    """Synthesize one frame of audio in ``audio_format``.

    A square wave, because a test needs energy a VAD will call speech and does
    not need it to sound like anything. ``amplitude`` is a fraction of full
    scale for the format's width; ``0.0`` gives the silence that ends an
    utterance.
    """
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError(f"amplitude must be between 0.0 and 1.0, got {amplitude}")

    samples_per_channel = max(1, audio_format.sample_rate * ms // 1000)
    peak = int(((1 << (audio_format.sample_width * 8 - 1)) - 1) * amplitude)
    wave = [peak if index % 2 == 0 else -peak for index in range(samples_per_channel)]
    interleaved = [value for value in wave for _ in range(audio_format.channels)]

    fmt = _STRUCT_FORMAT[audio_format.sample_width]
    return AudioFrame(
        data=struct.pack(f"<{len(interleaved)}{fmt}", *interleaved),
        sample_rate=audio_format.sample_rate,
        channels=audio_format.channels,
        sample_width=audio_format.sample_width,
    )
