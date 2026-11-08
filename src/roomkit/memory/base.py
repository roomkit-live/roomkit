"""Abstract base class for memory providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIMessage

# How many recent events a provider that doesn't trim by event count needs from
# ``RoomContext.recent_events``. Token-aware providers (which trim by budget, not
# count) inherit this so the framework loads a full pool for them. Mirrors the
# framework's ``_RECENT_EVENTS_LIMIT`` ceiling, which caps it regardless.
DEFAULT_RECENT_EVENTS_WINDOW = 2_000


@dataclass
class MemoryResult:
    """Result returned by a memory provider.

    Memory providers can return pre-built AI messages (e.g. summaries,
    synthetic context) and/or raw room events that AIChannel will convert
    using its own content extraction logic (preserving vision support).

    A provider may populate any of the fields. ``messages`` are prepended
    first, then ``events`` are converted and appended.

    ``notes`` is what the provider retrieved for the current turn only
    (knowledge passages, say). It changes from one turn to the next, so the
    channel carries it with the turn's notes, after the input, rather than in
    the history a provider caches (RFC §20.2). A provider that wraps another
    and rebuilds its result carries the inner ``notes``
    (``dataclasses.replace`` keeps them).

    ``speakers`` is the labels ``messages`` name participants by, as the
    runtime gives them (a summary's lines, ``Alice``, ``@sms1``): the channel
    counts them toward labelling the turns that follow, so a conversation a
    summary named several people in labels its turns (RFC §6.4).
    """

    messages: list[AIMessage] = field(default_factory=list)
    events: list[RoomEvent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)


class MemoryProvider(ABC):
    """Pluggable memory backend for AI context construction.

    Implement this ABC to control how conversation history is retrieved
    for AI generation.  The library ships with :class:`SlidingWindowMemory`
    (simple last-N events) as the default.

    Lifecycle methods ``ingest``, ``clear``, and ``close`` are concrete
    no-ops, and ``compact`` changes nothing by default, so that simple
    implementations only need to override ``retrieve``.
    """

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return type(self).__name__

    @property
    def recent_events_window(self) -> int:
        """How many recent events this provider may read from ``RoomContext``.

        The framework uses the largest window across a room's bound channels to
        decide how many events to load into ``RoomContext.recent_events`` — so a
        provider that windows to last-N should report N (avoids loading and
        deserialising the whole ceiling every turn). Defaults to the full pool
        for providers that don't trim by event count (e.g. token-aware ones).
        """
        return DEFAULT_RECENT_EVENTS_WINDOW

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

    async def compact(
        self,
        room_id: str,
        history: list[AIMessage],
        *,
        usage: dict[str, int],
        channel_id: str | None = None,
    ) -> list[AIMessage] | None:
        """Compact a turn's history between two of its tool rounds (optional).

        The AI channel calls it after each tool round it ran, before the next
        round's request (RFC §6.4, §20.1): *history* is what the turn sends
        before its input, as the model reads it, and *usage* what the round's
        generation measured (``input_tokens``, ``output_tokens`` and the
        cache counters, disjoint, as the provider reported them; empty when
        it reported none).

        Return ``None`` to leave the turn as it is, the default. A list
        stands for *history* in the next requests: the channel keeps the
        turn's input whole, joins a user text the list ends on to it, and
        stores the long results of the turn's earlier rounds for
        ``read_stored_result``, the last round's left whole. A message of
        the list that is not one of *history*'s has a copy of a runtime mark
        replaced, unless a memory of the runtime built it (a summary).
        An error is logged, and the turn goes on as it was.

        Args:
            room_id: The room of the turn.
            history: The messages before the turn's input.
            usage: The token counters of the round's generation.
            channel_id: The AI channel whose turn it is.

        Returns:
            The history to send in place of *history*, or ``None``.
        """
        return None

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
