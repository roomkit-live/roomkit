"""AudioFrame data model for inbound audio pipeline processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AudioFrame:
    """A single frame of inbound audio for pipeline processing.

    AudioFrame flows through the audio pipeline stages:
    denoiser -> VAD -> diarization. Each stage may annotate
    the metadata dict with its results.

    This is distinct from AudioChunk, which is used for outbound
    TTS audio streaming.
    """

    data: bytes
    """Raw audio bytes (PCM)."""

    sample_rate: int = 16000
    """Sample rate in Hz."""

    channels: int = 1
    """Number of audio channels."""

    sample_width: int = 2
    """Bytes per sample (2 = 16-bit PCM)."""

    timestamp_ms: float | None = None
    """Timestamp in milliseconds (relative to session start)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Pipeline stages annotate results here."""

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise ValueError("AudioFrame.data must be bytes")
        if self.sample_rate <= 0 or self.sample_rate > 192_000:
            raise ValueError(f"sample_rate must be between 1 and 192000, got {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")
        if self.sample_width not in (1, 2, 4):
            raise ValueError(f"sample_width must be 1, 2, or 4, got {self.sample_width}")
        frame_align = self.sample_width * self.channels
        if len(self.data) % frame_align != 0:
            raise ValueError(
                f"data length ({len(self.data)}) must be divisible by "
                f"sample_width * channels ({frame_align})"
            )
