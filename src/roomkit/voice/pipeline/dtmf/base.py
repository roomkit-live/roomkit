"""DTMF tone detector ABC and redaction configuration (RFC §17.6)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


@dataclass
class DTMFRedaction:
    """How DTMF digits are masked wherever the framework exposes them.

    The defaults mask everything: a redaction that leaks the first four digits
    of a PIN by default would be a worse trap than no redaction at all. The
    RFC's own example ("4111********1111") is the card-number shape — set
    ``keep_first=4`` / ``keep_last=4`` for it, deliberately.
    """

    enabled: bool = True
    """Whether masking is applied. The object exists to turn it on."""

    keep_first: int = 0
    """Leading digits left in the clear."""

    keep_last: int = 0
    """Trailing digits left in the clear."""

    mask_char: str = "*"
    """Character substituted for each masked digit."""

    def __post_init__(self) -> None:
        if self.keep_first < 0 or self.keep_last < 0:
            raise ValueError("keep_first and keep_last must be >= 0")
        if len(self.mask_char) != 1:
            raise ValueError("mask_char must be exactly one character")

    def mask(self, digits: str) -> str:
        """Mask a digit or a sequence of digits.

        A sequence shorter than the digits it would keep in the clear is
        masked entirely — the point of keeping edges is context on a long
        number, not a peephole onto a short secret.
        """
        if not self.enabled or not digits:
            return digits
        if len(digits) <= self.keep_first + self.keep_last:
            return self.mask_char * len(digits)
        masked_len = len(digits) - self.keep_first - self.keep_last
        return (
            digits[: self.keep_first]
            + self.mask_char * masked_len
            + (digits[len(digits) - self.keep_last :] if self.keep_last else "")
        )


@dataclass
class DTMFEvent:
    """A detected DTMF tone."""

    digit: str
    """The DTMF digit ('0'-'9', '*', '#', 'A'-'D')."""

    duration_ms: float
    """Duration of the tone in milliseconds."""

    confidence: float = 1.0
    """Detection confidence (0.0 to 1.0)."""


class DTMFDetector(ABC):
    """Abstract base class for DTMF tone detection providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'goertzel')."""
        ...

    @abstractmethod
    def process(self, frame: AudioFrame, stream: str) -> DTMFEvent | None:
        """Analyse an audio frame for DTMF tones.

        Args:
            frame: The audio frame to analyse.
            stream: Identity of the audio stream this frame belongs to. A
                provider keeps its state per stream: a voice session and a
                conference track are separate speakers, and letting one advance
                the other's detection state makes silence from one close the
                other's utterance.

        Returns:
            A DTMFEvent if a tone was detected, else None.
        """
        ...

    def reset(self, stream: str) -> None:  # noqa: B027
        """Drop a stream's state.

        Called when the stream ends, so a long-running room does not accumulate
        the state of every speaker that ever joined.
        """

    def close(self) -> None:  # noqa: B027
        """Release resources."""
