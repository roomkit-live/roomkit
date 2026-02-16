"""Abstract base class for memory providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIMessage


@dataclass
class MemoryResult:
    """Result returned by a memory provider.

    Memory providers can return pre-built AI messages (e.g. summaries,
    synthetic context) and/or raw room events that AIChannel will convert
    using its own content extraction logic (preserving vision support).

    A provider may populate one or both fields. ``messages`` are prepended
    first, then ``events`` are converted and appended.
    """

    messages: list[AIMessage] = field(default_factory=list)
    events: list[RoomEvent] = field(default_factory=list)


class MemoryProvider(ABC):
    """Pluggable memory backend for AI context construction.

    Implement this ABC to control how conversation history is retrieved
    for AI generation.  The library ships with :class:`SlidingWindowMemory`
    (simple last-N events) as the default.

    Lifecycle methods ``ingest``, ``clear``, and ``close`` are concrete
    no-ops so that simple implementations only need to override ``retrieve``.
    """

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return type(self).__name__

    @abstractmethod
    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        """Retrieve context for AI generation.

        Args:
            room_id: The room being processed.
            current_event: The event that triggered AI generation.
            context: Full room context including recent events, bindings,
                and participants.
            channel_id: The AI channel requesting memory (useful when
                multiple AI channels share a room).

        Returns:
            A :class:`MemoryResult` with messages and/or events to include
            in the AI context.
        """
        ...

    async def ingest(  # noqa: B027
        self,
        room_id: str,
        event: RoomEvent,
        *,
        channel_id: str | None = None,
    ) -> None:
        """Ingest an event into memory (optional).

        Stateful providers (e.g. summarization, vector stores) can override
        this to update their internal state as events arrive.  The default
        implementation is a no-op.
        """

    async def clear(self, room_id: str) -> None:  # noqa: B027
        """Clear all stored memory for a room (optional)."""

    async def close(self) -> None:  # noqa: B027
        """Release resources held by the provider (optional)."""
