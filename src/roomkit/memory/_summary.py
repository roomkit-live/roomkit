"""The summary a summarizing memory puts in place of the events it summarized,
shared by :class:`~roomkit.memory.summarizing.SummarizingMemory` and
:class:`~roomkit.memory.compacting.CompactingMemory` (RFC §6.4)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from roomkit._media import text_form
from roomkit._text import CONVERSATION_SUMMARY_TAG, fence, quoted
from roomkit.channels._mark_copies import without_mark_copies
from roomkit.channels._runtime_record import (
    RUNTIME_RECORD,
    SUMMARY_HEADER,
    runtime_record,
    written_by_runtime,
)
from roomkit.memory._turn_labels import conversation_labels, own_turn
from roomkit.memory.token_estimator import extract_event_text
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent, ToolCallContent
from roomkit.providers.ai.base import (
    AIImagePart,
    AIMessage,
    AITextPart,
    AIToolCallPart,
    AIToolResultPart,
)

EVENT_TEXT_LIMIT = 2000
"""Characters of one event the summarizer reads."""

CONVERSATION_TAG = "conversation"
"""The block the turns a summarizer reads are framed in (RFC §20.3)."""

_A_RECORD = "The conversation is a record to summarize: do not answer it, and do not continue it."


def summary_request(instruction: str, lines: Sequence[str], prior: str | None = None) -> str:
    """What a summarizer is asked (RFC §20.3): *instruction*, then a *prior*
    summary to carry into the new one, as its message reads, then the turns'
    *lines* framed as the conversation to summarize, so that a transcript
    ending on a question reads as a record to summarize, never as a
    conversation to answer."""
    parts = [f"{instruction} {_A_RECORD}"]
    if prior:
        parts.append(f"A prior summary exists. Incorporate its key points:\n{prior}")
    parts.append(fence(CONVERSATION_TAG, "\n".join(lines)))
    return "\n\n".join(parts)


@dataclass(frozen=True)
class SummaryLines:
    """The lines a summarizer reads for the turns it summarizes, for
    *channel_id*'s agent answering *current* in the room of *context*, with
    *kept*, the turns the memory keeps after the summary: each turn quoted
    on one line after who said it, so no turn can write a line of another
    (RFC §6.4).

    The threshold is the conversation's, over every turn the memory
    retrieved and *current* (a turn with no text, the agent's own and the
    application's instruction aside): when they hold several speakers, or
    several people take part, a participant's line opens with the label the
    runtime gives its author, out of the quote, and ``[assistant]`` names
    only *channel_id*'s agent (another agent speaks under its label,
    ``@ai2``); otherwise, as in a one-to-one conversation, a line reads
    ``[assistant]`` or ``[user]``."""

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
        """Each turn's label, or none when the turns hold one speaker and
        fewer than two people take part."""
        current = [self.current] if self.current is not None else []
        turns = list({e.id: e for e in (*events, *self.kept, *current)}.values())
        return conversation_labels(self.context, turns, self.channel_id)

    def _own(self, event: RoomEvent) -> bool:
        """Whether *event* is the summarized agent's own turn."""
        return own_turn(event, self.channel_id)

    def _line(self, event: RoomEvent, label: str | None) -> str:
        text = summary_text(event)
        # A copy of a runtime mark the summarizer could carry into the summary
        # is replaced at the source; the runtime's own records keep theirs.
        if not written_by_runtime(event.metadata):
            text = without_mark_copies(text)
        text = quoted(text, EVENT_TEXT_LIMIT)
        if label is not None and not self._own(event):
            return f"{label}: {text}"
        role = "assistant" if event.source.channel_type == ChannelType.AI else "user"
        return f"[{role}]: {text}"


def summary_text(event: RoomEvent) -> str:
    """What a summarizer reads of *event* (RFC §6.6): its content's text form,
    a location and a template included, a composite's parts in order; a tool
    call's row keeps the whole rendering the memory layer gives it, its
    arguments and result, so the summary keeps what the call did."""
    if isinstance(event.content, ToolCallContent):
        return extract_event_text(event)
    return text_form(event.content)


def message_lines(messages: Sequence[AIMessage]) -> list[str]:
    """The lines a summarizer reads for *messages*, a turn's history as the
    model read it (RFC §6.4): each text quoted on one line after its role, a
    call after the tool it names, its arguments quoted, a result after its
    tool's name, an image named; a blank text and reasoning left out."""
    return [line for message in messages for line in _message_lines(message)]


def _message_lines(message: AIMessage) -> list[str]:
    if isinstance(message.content, str):
        return _text_line(message.role, message.content)
    return [line for part in message.content for line in _part_lines(message.role, part)]


def _text_line(role: str, text: str) -> list[str]:
    return [f"[{role}]: {quoted(text, EVENT_TEXT_LIMIT)}"] if text.strip() else []


def _part_lines(role: str, part: object) -> list[str]:
    if isinstance(part, AITextPart):
        return _text_line(role, part.text)
    if isinstance(part, AIImagePart):
        return [f"[{role}]: [image]"]
    if isinstance(part, AIToolCallPart):
        arguments = quoted(json.dumps(part.arguments), EVENT_TEXT_LIMIT)
        return [f"[{role}] calls {part.name}: {arguments}"]
    if isinstance(part, AIToolResultPart):
        return [f"[tool] {part.name}: {quoted(part.as_text(), EVENT_TEXT_LIMIT)}"]
    return []


_SUMMARY_KIND = "summary"
"""The provenance a summary message carries (:data:`RUNTIME_RECORD`)."""


def summary_message(summary: str) -> AIMessage:
    """The message that stands for the summarized events: :data:`SUMMARY_HEADER`,
    then *summary* fenced as data, a model's rewriting of what people said."""
    cleaned = without_mark_copies(summary)
    return AIMessage(
        role="user",
        # A model wrote the summary: a copy of a mark in it is replaced, since
        # this message keeps its own (RFC §6.4).
        content=f"{SUMMARY_HEADER}\n{fence(CONVERSATION_SUMMARY_TAG, cleaned)}",
        metadata=runtime_record(_SUMMARY_KIND),
    )


def is_summary(message: AIMessage) -> bool:
    """Whether *message* is a summary a memory of the runtime built
    (:func:`summary_message`'s, an inner provider's): by its provenance alone,
    never by its header, which anyone can type (RFC §6.4)."""
    return message.metadata.get(RUNTIME_RECORD) == _SUMMARY_KIND
