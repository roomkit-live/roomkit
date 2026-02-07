"""Backchannel detection provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackchannelContext:
    """Context for backchannel detection."""

    transcript: str | None = None
    """The transcribed text to classify."""

    speech_duration_ms: float = 0.0
    """Duration of the utterance in milliseconds."""

    audio_bytes: bytes | None = None
    """Raw audio of the utterance (for acoustic classifiers)."""

    bot_speech_progress: float = 0.0
    """How far into the bot's current utterance (0.0–1.0)."""

    session_id: str = ""
    """The voice session this utterance belongs to."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context for the classifier."""


@dataclass
class BackchannelDecision:
    """Decision from a backchannel detector."""

    is_backchannel: bool
    """True if the utterance is a backchannel (e.g. 'uh-huh', 'yeah')."""

    confidence: float = 1.0
    """Confidence in the decision (0.0 to 1.0)."""

    label: str | None = None
    """Classification label (e.g. ``"acknowledgement"``, ``"filler"``)."""


class BackchannelDetector(ABC):
    """Abstract base class for backchannel detection providers.

    Backchannel detectors classify short utterances as either
    genuine interruptions or backchannels (acknowledgements that
    don't require stopping the current speaker).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @abstractmethod
    def classify(self, context: BackchannelContext) -> BackchannelDecision:
        """Classify whether an utterance is a backchannel.

        Args:
            context: The utterance context to classify.

        Returns:
            A BackchannelDecision indicating classification.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
