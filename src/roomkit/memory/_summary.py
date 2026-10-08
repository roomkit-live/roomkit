"""The summary a summarizing memory puts in place of the events it summarized,
shared by :class:`~roomkit.memory.summarizing.SummarizingMemory` and
:class:`~roomkit.memory.compacting.CompactingMemory` (RFC §6.4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from roomkit._text import CONVERSATION_SUMMARY_TAG, fence, quoted
from roomkit.channels._speaker import several_speakers, turn_labels
from roomkit.memory.token_estimator import extract_event_text
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIMessage

SUMMARY_MARK = "[Conversation summary"
"""How a summary's message opens, whatever header follows: a later summary finds
an earlier one by it, an inner provider's own included."""

SUMMARY_HEADER = f"{SUMMARY_MARK} — earlier messages compacted]"
"""Opens the summary's message."""

EVENT_TEXT_LIMIT = 2000
"""Characters of one event the summarizer reads."""


@dataclass(frozen=True)
class SummaryLines:
    """The lines a summarizer reads for the turns it summarizes, for
    *channel_id*'s agent answering *current* in the room of *context*, with
    *kept*, the turns the memory keeps after the summary: each turn quoted
    on one line after who said it, so no turn can write a line of another
    (RFC §6.4).

    The threshold is the conversation's, over every turn the memory
    retrieved and *current* (a turn with no text, the agent's own and the
    application's instruction aside): when they hold several speakers, a
    participant's line opens with the label the runtime gives its author,
    out of the quote, and ``[assistant]`` names only *channel_id*'s agent
    (another agent speaks under its label, ``@ai2``); otherwise, as in a
    one-to-one conversation, a line reads ``[assistant]`` or ``[user]``."""

    context: RoomContext
    current: RoomEvent | None = None
    channel_id: str | None = None
    kept: tuple[RoomEvent, ...] = ()

    def __call__(self, events: Sequence[RoomEvent]) -> list[str]:
        labels = self._labels(events)
        return [self._line(event, labels.get(event.id)) for event in events]

    def named(self, events: Sequence[RoomEvent]) -> list[str]:
        """The labels the lines for *events* name participants by, for
        :attr:`MemoryResult.speakers`; none when they read ``[user]``."""
        labels = self._labels(events)
        return sorted({label for e in events if (label := labels.get(e.id)) and not self._own(e)})

    def _labels(self, events: Sequence[RoomEvent]) -> dict[str, str | None]:
        """Each turn's label, or none when the turns hold one speaker."""
        current = [self.current] if self.current is not None else []
        turns = list({e.id: e for e in (*events, *self.kept, *current)}.values())
        labels = turn_labels(turns, self.context)
        counted = (labels.get(e.id) for e in turns if self._counts(e))
        return labels if several_speakers(counted) else {}

    def _counts(self, event: RoomEvent) -> bool:
        """Whether *event* counts toward the threshold, as the conversation's
        turns do."""
        if self._own(event) or event.type == EventType.INSTRUCTION:
            return False
        return bool(extract_event_text(event).strip())

    def _own(self, event: RoomEvent) -> bool:
        """Whether *event* is the summarized agent's own turn."""
        if event.source.channel_type != ChannelType.AI:
            return False
        return self.channel_id is None or event.source.channel_id == self.channel_id

    def _line(self, event: RoomEvent, label: str | None) -> str:
        text = quoted(extract_event_text(event), EVENT_TEXT_LIMIT)
        if label is not None and not self._own(event):
            return f"{label}: {text}"
        role = "assistant" if event.source.channel_type == ChannelType.AI else "user"
        return f"[{role}]: {text}"


def summary_message(summary: str) -> AIMessage:
    """The message that stands for the summarized events: :data:`SUMMARY_HEADER`,
    then *summary* fenced as data, a model's rewriting of what people said."""
    return AIMessage(
        role="user", content=f"{SUMMARY_HEADER}\n{fence(CONVERSATION_SUMMARY_TAG, summary)}"
    )


def is_summary(message: AIMessage) -> bool:
    """Whether *message* is a summary, :func:`summary_message`'s or an inner
    provider's."""
    return isinstance(message.content, str) and SUMMARY_MARK in message.content
