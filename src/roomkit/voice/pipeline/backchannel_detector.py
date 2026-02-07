"""Backchannel detection provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BackchannelContext:
    """Context for backchannel detection."""

    text: str
    """The transcribed text to evaluate."""

    duration_ms: float
    """Duration of the utterance in milliseconds."""

    session_id: str = ""
    """The voice session this utterance belongs to."""


@dataclass
class BackchannelDecision:
    """Decision from a backchannel detector."""

    is_backchannel: bool
    """True if the utterance is a backchannel (e.g. 'uh-huh', 'yeah')."""

    confidence: float = 1.0
    """Confidence in the decision (0.0 to 1.0)."""


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
    def evaluate(self, context: BackchannelContext) -> BackchannelDecision:
        """Evaluate whether an utterance is a backchannel.

        Args:
            context: The utterance context to evaluate.

        Returns:
            A BackchannelDecision indicating classification.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
