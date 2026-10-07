"""The summary a summarizing memory puts in place of the events it summarized,
shared by :class:`~roomkit.memory.summarizing.SummarizingMemory` and
:class:`~roomkit.memory.compacting.CompactingMemory` (RFC §6.4)."""

from __future__ import annotations

from roomkit._text import CONVERSATION_SUMMARY_TAG, fence, quoted
from roomkit.memory.token_estimator import extract_event_text
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import AIMessage

SUMMARY_MARK = "[Conversation summary"
"""How a summary's message opens, whatever header follows: a later summary finds
an earlier one by it, an inner provider's own included."""

SUMMARY_HEADER = f"{SUMMARY_MARK} — earlier messages compacted]"
"""Opens the summary's message."""

EVENT_TEXT_LIMIT = 2000
"""Characters of one event the summarizer reads."""


def summarized_line(event: RoomEvent) -> str:
    """*event* as a summarizer reads it: whether an agent or a user said it, then
    its text as the memory layer reads it, quoted on one line, so no event can
    write a line of another."""
    role = "assistant" if event.source.channel_type == ChannelType.AI else "user"
    return f"[{role}]: {quoted(extract_event_text(event), EVENT_TEXT_LIMIT)}"


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
