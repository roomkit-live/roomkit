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
