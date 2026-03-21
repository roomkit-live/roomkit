"""Abstract base class for conversation scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Score:
    """A quality score for an AI response.

    Attributes:
        value: Score between 0.0 (worst) and 1.0 (best).
        dimension: What is being scored (e.g. "relevance", "helpfulness",
            "safety", "accuracy", "coherence").
        reason: Human-readable explanation of the score.
        metadata: Arbitrary metadata (model used for judging, etc.).
    """

    value: float
    dimension: str
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationScorer(ABC):
    """Pluggable quality scorer for AI responses.

    Implement this ABC to evaluate AI response quality.  Scorers are
    invoked by :class:`~roomkit.scoring.ScoringHook` after each AI
    response via the ``ON_AI_RESPONSE`` hook.

    Implementations can be:

    - **LLM-as-judge** — call a separate model to rate the response
    - **Rule-based** — regex/keyword checks, length validation
    - **Heuristic** — latency thresholds, tool usage patterns
    - **Human feedback** — bridge to user rating collection
    """

    @property
    def name(self) -> str:
        """Human-readable scorer name."""
        return type(self).__name__

    @abstractmethod
    async def score(
        self,
        *,
        response_content: str,
        query: str,
        room_id: str,
        channel_id: str,
        usage: dict[str, Any] | None = None,
        thinking: str = "",
    ) -> list[Score]:
        """Score an AI response.

        Args:
            response_content: The AI-generated text.
            query: The user message that triggered the response.
            room_id: Room where the response was generated.
            channel_id: AI channel that generated the response.
            usage: Token usage from the provider.
            thinking: Extended thinking/reasoning (if available).

        Returns:
            A list of :class:`Score` objects, one per dimension.
            Return an empty list to skip scoring for this response.
        """
        ...

    async def close(self) -> None:  # noqa: B027
        """Release resources held by the scorer (optional)."""
