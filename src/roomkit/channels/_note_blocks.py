"""A block added to the turn's notes once the channel built them (RFC §6.4).

Apart from :mod:`~roomkit.channels._turn_notes` because a block is cleaned of
every runtime mark (:mod:`~roomkit.channels._mark_copies`), which reads the
notes' header there.
"""

from __future__ import annotations

from roomkit.channels._mark_copies import without_mark_copies
from roomkit.channels._turn_notes import note_added
from roomkit.providers.ai.base import AIMessage


def add_turn_note(messages: list[AIMessage], block: str) -> list[AIMessage]:
    """*messages* with *block* added to the turn's notes (RFC §6.4).

    The block joins the section the channel opened, under its one header,
    when the turn's input carries it, and opens the section as
    :func:`~roomkit.channels._turn_notes.with_turn_notes` does otherwise.
    Either way the notes read as if they had been assembled at once (the
    same header, the blocks joined by a blank line, a text input still text,
    an input with images keeping its notes in one text part), so the prefix
    a provider caches is the same.

    For a ``BEFORE_AI_GENERATION`` hook::

        event.ai_context.messages = add_turn_note(event.ai_context.messages, block)

    A block quotes others (a passage, a task's progress): a copy of a
    runtime mark in it is replaced, and so is a copy of the notes' header,
    the notes' only mark, which the channel replaces in the conversation
    too, so the header found is the channel's: only one a hook wrote into
    the messages itself is what this misreads. Add notes before anything a
    hook appends to the input: a text input takes the block at its very end.
    """
    return note_added(messages, without_mark_copies(block))
