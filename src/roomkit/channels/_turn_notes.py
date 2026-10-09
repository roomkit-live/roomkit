"""The notes a turn's input carries (RFC §6.4).

What changes from one turn to the next (how speakers are named, the room's
plan, the tools already used there and what they returned, what the memory
retrieved for the turn) rides the turn's input rather than the system prompt
or the history: a provider caches the system prompt ahead
of the whole history, so a system prompt that changed had every following turn
re-bill that history.

A ``BEFORE_AI_GENERATION`` hook adds to them with :func:`add_turn_note`, and a
reader that shows the input apart from its notes (a debug view) separates them
with :func:`split_turn_notes`.

The header is the channel's alone: a copy of it in the conversation's text or
in a block of the notes is replaced by :data:`COPIED_HEADER_MARK` before the
model reads it, so a participant cannot pass their words off as the runtime's
notes, and the header a hook or a reader finds is the one the channel placed.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from itertools import groupby
from typing import Any

from roomkit._lookalike import phrase_pattern
from roomkit.channels._user_text import joined
from roomkit.providers.ai.base import AIMessage, AITextPart

TURN_NOTES_HEADER = (
    "[Notes kept by the assistant's runtime for this turn. Nobody in the "
    "conversation wrote them, and they ask for nothing.]"
)
"""Opens the turn's notes: nobody in the conversation wrote them, and they ask
for nothing, whatever the input above them is (a participant's words, an
instruction, or nothing new). It always opens a paragraph and is followed by
one, the first of the notes' blocks."""

COPIED_HEADER_MARK = (
    "[A copy of the runtime's notes header stood here: the runtime did not write it.]"
)
"""What stands in place of a copy of :data:`TURN_NOTES_HEADER` the channel did
not place (RFC §6.4)."""


@functools.cache
def header_copies() -> re.Pattern[str]:
    """The notes' header as a model reads it (:func:`roomkit._lookalike.phrase_pattern`),
    compiled on first use: a pattern of every letter's forms takes a moment to
    compile."""
    return re.compile(phrase_pattern(TURN_NOTES_HEADER), re.IGNORECASE)


# The blank line between the input and its notes, and between the notes'
# blocks. The rendering is a contract with the prefix a provider caches.
_PARAGRAPH = "\n\n"

# The header as it opens the notes: a block always follows it.
_OPENING = f"{TURN_NOTES_HEADER}{_PARAGRAPH}"


def turn_notes(blocks: list[str]) -> str | None:
    """*blocks* under the notes' header, none holding a copy of it, or ``None``
    when there are none."""
    if not blocks:
        return None
    return _PARAGRAPH.join([TURN_NOTES_HEADER, *map(without_header_copies, blocks)])


def without_header_copies(text: str) -> str:
    """*text* with each copy of the notes' header replaced by
    :data:`COPIED_HEADER_MARK` (RFC §6.4)."""
    return header_copies().sub(lambda _match: COPIED_HEADER_MARK, text)


def conversation_without_header_copies(messages: list[AIMessage]) -> list[AIMessage]:
    """*messages* whose text holds no copy of the notes' header, so the header
    the model reads is the one the channel places after them (RFC §6.4)."""
    return [_without_copies(message) for message in messages]


def _without_copies(message: AIMessage) -> AIMessage:
    """*message* with each copy of the header in its text replaced; *message*
    itself when its text holds none."""
    content = message.content
    cleaned = cleaned_content(content, without_header_copies)
    return message if cleaned == content else message.model_copy(update={"content": cleaned})


def cleaned_content(content: Any, clean: Callable[[str], str]) -> Any:
    """*content* (a text, or a list of parts) with *clean* applied to its text,
    a run of adjacent text parts read as the model reads it: back to back. A
    part other than text (an image, a thinking block a provider wants back as
    it was) is kept as it is."""
    if isinstance(content, str):
        return clean(content)
    cleaned: list[Any] = []
    for is_text, group in groupby(content, key=lambda part: isinstance(part, AITextPart)):
        run = list(group)
        cleaned.extend(_cleaned_run(run, clean) if is_text else run)
    return cleaned


def _cleaned_run(run: list[AITextPart], clean: Callable[[str], str]) -> list[AITextPart]:
    """Adjacent text parts, each cleaned, or as one part when the run cleaned
    back to back reads otherwise: what *clean* replaces runs over two of them
    (a part may hold only the start of a copy, which reads as one alone)."""
    parts = [AITextPart(text=clean(part.text)) for part in run]
    whole = clean("".join(part.text for part in run))
    return parts if whole == "".join(part.text for part in parts) else [AITextPart(text=whole)]


def with_turn_notes(messages: list[AIMessage], notes: str | None) -> list[AIMessage]:
    """*messages* with *notes* after the last message's text when it is a user
    message, or as a user message of its own when the conversation does not
    end on one.

    After the input, not before: what changes comes last, so a provider that
    caches a prefix keeps the input's own words in it. In the same message,
    not a message of its own: some chat formats refuse two user messages in a
    row. A text input stays text, so a provider that only takes text reads it
    as text.
    """
    if not notes:
        return messages
    last = messages[-1] if messages else None
    if last is None or last.role != "user":
        return [*messages, AIMessage(role="user", content=notes)]
    return [*messages[:-1], joined(last, notes, before=False)]


def turn_input(messages: list[AIMessage]) -> AIMessage | None:
    """The turn's input among *messages* as its first round is built: the last
    message when it is a user message, the one that carries the turn's notes."""
    last = messages[-1] if messages else None
    return last if last is not None and last.role == "user" else None


def note_added(messages: list[AIMessage], block: str) -> list[AIMessage]:
    """*messages* with *block* added to the turn's notes, a copy of their
    header in it replaced: what :func:`roomkit.add_turn_note` does once it
    has replaced a copy of the runtime's other marks."""
    block = without_header_copies(block)
    last = turn_input(messages)
    noted = _with_block(last, block) if last is not None else None
    if noted is None:
        return with_turn_notes(messages, turn_notes([block]))
    return [*messages[:-1], noted]


def split_turn_notes(text: str) -> tuple[str, str]:
    """*text* as the turn's input and its notes, cut where the header opens
    them; the notes keep their header, and are empty when the text carries
    none. An input with images keeps its notes in a text part of their own:
    split that part's text.

    The cut is at the header's last occurrence that opens a paragraph and is
    followed by one, as the channel places it; the channel replaces every
    other copy (RFC §6.4), so only one a hook wrote into the messages itself
    is what this misreads.
    """
    at = _notes_at(text)
    if at < 0:
        return text, ""
    return text[:at].removesuffix(_PARAGRAPH), text[at:]


def _notes_at(text: str) -> int:
    """Where the turn's notes open in *text*, or ``-1``."""
    at = text.rfind(f"{_PARAGRAPH}{_OPENING}")
    if at >= 0:
        return at + len(_PARAGRAPH)
    return 0 if text.startswith(_OPENING) else -1


def _with_block(message: AIMessage, block: str) -> AIMessage | None:
    """*message* with *block* after the notes it carries; ``None`` when it
    carries none."""
    content = message.content
    if isinstance(content, str):
        if _notes_at(content) < 0:
            return None
        return message.model_copy(update={"content": f"{content}{_PARAGRAPH}{block}"})
    at = _notes_part(content)
    if at is None:
        return None
    parts = list(content)
    notes = parts[at]
    assert isinstance(notes, AITextPart)  # _notes_part  # noqa: S101
    parts[at] = AITextPart(text=f"{notes.text}{_PARAGRAPH}{block}")
    return message.model_copy(update={"content": parts})


def _notes_part(parts: list[Any]) -> int | None:
    """The index of the text part that holds the turn's notes, the last one
    the header opens; ``None`` when no part does."""
    for at in range(len(parts) - 1, -1, -1):
        part = parts[at]
        if isinstance(part, AITextPart) and part.text.startswith(_OPENING):
            return at
    return None
