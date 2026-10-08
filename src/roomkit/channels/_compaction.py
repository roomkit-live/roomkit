"""What an emergency compaction makes of a turn's messages (RFC §6.4).

A provider that refuses a turn's context as too long gets one compacted
replay. The turn's input and its notes stay whole. When the input falls in
the older half, the history before it is summarized and the long results of
the turn's older tool rounds are stored for re-reading like any large
result, a preview in their place; otherwise the older half of the history is
summarized. Every call keeps its result, and the summary joins the user
message that follows it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._text import quoted
from roomkit.channels._runtime_record import COMPACTION_HEADER as SUMMARY_HEADER
from roomkit.channels._speaker import SPEAKER_KEY, said_by, with_said
from roomkit.channels._tool_eviction import (
    REREAD_TOOL,
    ToolEviction,
    is_eviction_placeholder,
    kept_whole,
)
from roomkit.channels._user_text import LEADING_TEXT, split_leading_text
from roomkit.providers.ai.base import AIMessage, AITextPart, AIToolResultPart
from roomkit.tools.fence import named_blocks

if TYPE_CHECKING:
    from roomkit.channels.ai import _ContentPart

# A result of an older round longer than this is stored at compaction, with a
# preview this long: its placeholder costs less than what it replaces.
_STORED_OVER_CHARS = 2000
_STORED_PREVIEW_CHARS = 1000

# How much of each summarized message the summary quotes.
_SUMMARY_MESSAGE_CHARS = 500
_SUMMARY_PART_CHARS = 200


def compaction_cut(messages: list[AIMessage], turn_input: AIMessage | None) -> tuple[int, int]:
    """Where the summarized messages end, and where the shortened ones end.

    The first half is summarized, never parting a call from its result. When
    the turn's input falls in that half, what precedes the input is
    summarized instead, and the rounds after it, up to the half, shortened.
    """
    half = _past_results(messages, len(messages) // 2)
    at = next((i for i, message in enumerate(messages) if message is turn_input), None)
    if at is None or at >= half:
        return half, half
    return at, half


def _past_results(messages: list[AIMessage], index: int) -> int:
    """*index*, moved past the tool results there, which belong to the call before."""
    while index < len(messages) and messages[index].role == "tool":
        index += 1
    return index


def summary_text(messages: list[AIMessage]) -> str | None:
    """The summary of *messages*, one line each, or ``None`` when there are none."""
    if not messages:
        return None
    lines = [f"[{message.role}]: {_said(message)}" for message in messages]
    return "\n".join([SUMMARY_HEADER, *lines])


def _part_said(text: str) -> str:
    """A part's own words as the summary takes them: its blocks named, cut to a
    part's share."""
    return named_blocks(text)[:_SUMMARY_PART_CHARS]


def _said(message: AIMessage) -> str:
    """What the summary quotes of *message*: its text, cut short and quoted on
    one line after the name the context gave its speaker (RFC §6.4), a
    delimited block named rather than quoted."""
    speaker = message.metadata.get(SPEAKER_KEY) if message.role == "user" else None
    parts = message.content if isinstance(message.content, list) else None
    lead, text = "", message.content if isinstance(message.content, str) else ""
    if parts is not None:
        first = parts[0] if parts else None
        joined_ahead = message.metadata.get(LEADING_TEXT)
        if speaker is not None and isinstance(first, AITextPart) and first.text == joined_ahead:
            lead, parts = first.text, parts[1:]
        # A line per part, so each part's line labels are seen as such; each
        # part's own words named and cut, read back from its line strings.
        text = "\n".join(
            with_said(part.text, speaker, _part_said)
            if isinstance(part, AITextPart)
            else f"[{part.type}]"
            for part in parts
        )
    elif speaker is not None:
        lead, text = split_leading_text(message, text)
    said = said_by(with_said(text, speaker, named_blocks), speaker, _SUMMARY_MESSAGE_CHARS)
    # A summary joined ahead of a labelled turn is quoted apart, so the label
    # stays out of the quote (RFC §6.4); an unlabelled turn reads as one.
    return f"{quoted(named_blocks(lead), _SUMMARY_MESSAGE_CHARS)}\n{said}" if lead else said


def with_results_stored(messages: list[AIMessage], eviction: ToolEviction) -> list[AIMessage]:
    """*messages* with each long tool result stored for re-reading, a preview
    in its place; a message none of whose results changes is kept as it is."""
    return [_with_results_stored(message, eviction) for message in messages]


def _with_results_stored(message: AIMessage, eviction: ToolEviction) -> AIMessage:
    """*message* with its long results stored, or *message* itself when none is."""
    if message.role != "tool" or not isinstance(message.content, list):
        return message
    parts = [_stored(part, eviction) for part in message.content]
    if all(new is old for new, old in zip(parts, message.content, strict=True)):
        return message
    return message.model_copy(update={"content": parts})


def _stored(part: _ContentPart, eviction: ToolEviction) -> _ContentPart:
    """*part* with its result stored when it is a long one a model may page
    through; as it is otherwise.

    A skill's instructions are read whole (``kept_whole``), and a page of
    ``read_stored_result`` already comes from the store.
    """
    if not isinstance(part, AIToolResultPart) or kept_whole(part.name) or part.name == REREAD_TOOL:
        return part
    result = part.result
    if isinstance(result, str):
        if len(result) <= _STORED_OVER_CHARS or is_eviction_placeholder(result):
            return part
        stored = eviction.evict(result, part.tool_call_id, _STORED_PREVIEW_CHARS)
        return part.model_copy(update={"result": stored})
    texts = [p.text for p in result if isinstance(p, AITextPart)]
    if len("\n".join(texts)) <= _STORED_OVER_CHARS or any(map(is_eviction_placeholder, texts)):
        return part
    parts = eviction.evict_parts(result, part.tool_call_id, _STORED_PREVIEW_CHARS)
    return part.model_copy(update={"result": parts})
