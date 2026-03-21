"""Hook that runs conversation scorers on AI responses.

Usage::

    from roomkit.scoring import ScoringHook, MockScorer

    hook = ScoringHook(scorers=[MockScorer()])
    hook.attach(kit)  # registers AFTER_AI_RESPONSE hook

    # Scores are stored as Observations in the ConversationStore
    # and accessible via hook.recent_scores
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.scoring.base import ConversationScorer, Score

if TYPE_CHECKING:
    from roomkit.core.roomkit import RoomKit

logger = logging.getLogger("roomkit.scoring")


class ScoringHook:
    """Runs conversation scorers on every AI response.

    Attach to a :class:`~roomkit.RoomKit` instance to automatically
    score AI responses via the ``AFTER_AI_RESPONSE`` hook.  Scores are
    stored as :class:`~roomkit.models.task.Observation` objects in the
    conversation store and kept in a bounded in-memory buffer for quick
    access.

    Parameters:
        scorers: List of :class:`ConversationScorer` implementations.
        max_recent: Maximum number of scores to keep in the in-memory buffer.
    """

    def __init__(
        self,
        scorers: list[ConversationScorer],
        *,
        max_recent: int = 100,
    ) -> None:
        self._scorers = scorers
        self._kit: RoomKit | None = None
        self.recent_scores: deque[dict[str, Any]] = deque(maxlen=max_recent)

    def attach(self, kit: RoomKit) -> None:
        """Register the scoring hook on a RoomKit instance."""
        self._kit = kit
        # kit.hook() is a decorator — call it to get the decorator, then apply
        decorator = kit.hook(
            HookTrigger.AFTER_AI_RESPONSE,
            execution=HookExecution.ASYNC,
            name="scoring_hook",
        )
        decorator(self._on_response)

    async def _on_response(self, event: Any, context: Any) -> None:
        """Handle AFTER_AI_RESPONSE hook — run all scorers."""
        from roomkit.models.tool_call import AIResponseEvent

        if not isinstance(event, AIResponseEvent):
            return

        # Extract the last user message as query context
        query = ""
        if hasattr(context, "recent_events") and context.recent_events:
            for e in reversed(context.recent_events):
                if hasattr(e.content, "body") and e.source.channel_id != event.channel_id:
                    query = e.content.body
                    break

        # Run all scorers concurrently
        raw_results = await asyncio.gather(
            *[
                s.score(
                    response_content=event.response_content,
                    query=query,
                    room_id=event.room_id,
                    channel_id=event.channel_id,
                    usage=event.usage,
                    thinking=event.thinking,
                )
                for s in self._scorers
            ],
            return_exceptions=True,
        )

        all_scores: list[tuple[str, Score]] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.warning("Scorer %s failed: %s", self._scorers[i].name, result)
                continue
            for score in result:
                all_scores.append((self._scorers[i].name, score))

        if not all_scores:
            return

        # Store scores as Observations in the ConversationStore
        if self._kit:
            from roomkit.models.task import Observation

            for scorer_name, score in all_scores:
                obs = Observation(
                    id=f"score_{event.room_id}_{score.dimension}_{id(score):x}",
                    room_id=event.room_id,
                    channel_id=event.channel_id,
                    content=f"[{score.dimension}] {score.value:.2f}: {score.reason}",
                    category=f"score:{score.dimension}",
                    confidence=score.value,
                    metadata={
                        "scorer": scorer_name,
                        "dimension": score.dimension,
                        "score_value": score.value,
                        "reason": score.reason,
                        "latency_ms": event.latency_ms,
                        "streaming": event.streaming,
                        **score.metadata,
                    },
                )
                try:
                    await self._kit._store.add_observation(obs)
                except Exception:
                    logger.debug("Failed to store score observation", exc_info=True)

        # Buffer for quick access
        for scorer_name, score in all_scores:
            self.recent_scores.append(
                {
                    "room_id": event.room_id,
                    "channel_id": event.channel_id,
                    "scorer": scorer_name,
                    "dimension": score.dimension,
                    "value": score.value,
                    "reason": score.reason,
                    "latency_ms": event.latency_ms,
                }
            )

    async def close(self) -> None:
        """Close all scorers."""
        for s in self._scorers:
            try:
                await s.close()
            except Exception:
                logger.debug("Scorer %s close failed", s.name, exc_info=True)
