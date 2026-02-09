"""Turn detection provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnEntry:
    """A single utterance in a turn context."""

    text: str
    """Transcribed text of the utterance."""

    role: str | None = None
    """Conversational role (``"user"`` or ``"assistant"``)."""

    duration_ms: float | None = None
    """Duration of the utterance in milliseconds."""


@dataclass
class TurnContext:
    """Accumulated context for turn detection."""

    conversation_history: list[TurnEntry] = field(default_factory=list)
    """Utterances accumulated so far in this potential turn."""

    silence_duration_ms: float = 0.0
    """Silence duration since last utterance in milliseconds."""

    transcript: str = ""
    """Current transcript text for the latest utterance."""

    is_final: bool = False
    """Whether the transcript is final (not partial)."""

    speech_duration_ms: float = 0.0
    """Duration of the current speech segment in milliseconds."""

    session_id: str = ""
    """The voice session this turn belongs to."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context for the turn detector."""

    audio_bytes: bytes | None = None
    """Raw PCM audio for the speech segment (int16 mono). None in continuous STT mode."""

    audio_sample_rate: int = 16000
    """Sample rate of audio_bytes in Hz."""


@dataclass
class TurnDecision:
    """Decision from a turn detector."""

    is_complete: bool
    """True if the turn is considered complete."""

    confidence: float = 1.0
    """Confidence in the decision (0.0 to 1.0)."""

    reason: str | None = None
    """Human-readable reason for the decision."""

    suggested_wait_ms: float | None = None
    """If incomplete, how long to wait before re-evaluating."""


class TurnDetector(ABC):
    """Abstract base class for turn detection providers.

    Turn detectors evaluate accumulated transcription context
    to decide whether a user's conversational turn is complete
    (i.e. they have finished speaking and expect a response).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'silence_based', 'llm_based')."""
        ...

    @abstractmethod
    def evaluate(self, context: TurnContext) -> TurnDecision:
        """Evaluate whether a turn is complete.

        Args:
            context: Accumulated utterance context.

        Returns:
            A TurnDecision indicating completion status.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
