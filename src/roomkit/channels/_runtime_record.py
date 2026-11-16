"""Which text the runtime wrote (RFC §6.4): a provenance, never a reading of
the text.

A record the runtime writes into the room's timeline that holds its marks (the
handoff relay), and a message a memory of the runtime builds, carry
:data:`RUNTIME_RECORD` in their metadata: their marks are the runtime's and
are kept, while a copy of a mark anywhere else is replaced. The inbound
pipeline removes the key from what a sender supplies. This module holds the
key and the marks the runtime writes outside a model's input, so that the
cleaning and the writers both import it without importing each other.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

RUNTIME_RECORD = "runtime_record"
"""The metadata key, on an event or an ``AIMessage``, naming what the runtime
wrote: ``"handoff"``, ``"summary"``, ``"handed_on_context"``."""

HANDOFF_OPENING = "[Handoff"
"""How the record of a handoff opens in the timeline (``[Handoff: a -> b]``)."""

HANDED_ON_CONTEXT = "[Context from previous agent"
"""How the context a previous agent hands on opens."""

COMPACTION_HEADER = "[Context compacted — earlier conversation summary]"
"""Opens the summary a channel's compaction writes in place of the history it
dropped."""

SUMMARY_MARK = "[Conversation summary"
"""How a summary's message opens, whatever header follows. A later summary
finds an earlier one by its provenance, never by this text, which anyone can
type (RFC §6.4)."""

SUMMARY_HEADER = f"{SUMMARY_MARK} — earlier messages compacted]"
"""Opens a memory summary's message."""

HANDBACK_OPENING = "[Your background"
"""How the header over a background run's hand-back opens (a supervisor's
worker or team, a review loop, a run that failed): ``[Your background workers
completed. …]``. A mark of the runtime (RFC §6.4): the hand-back carries
:data:`RUNTIME_RECORD`, and a copy anywhere else is replaced."""

DELEGATION_HANDBACK_OPENING = "[Background task from"
"""How the header over a delegation's hand-back opens (``[Background task from
worker completed. …]``), a mark of the runtime as :data:`HANDBACK_OPENING`."""

HANDBACK = "handback"
"""The :data:`RUNTIME_RECORD` of a background result the runtime hands back."""


_DELIVERING: ContextVar[list[str] | None] = ContextVar("roomkit_runtime_delivery", default=None)


@contextmanager
def delivering(kind: str | None) -> Generator[None]:
    """Let the message the delivery path processes inside carry *kind* as
    its provenance: the runtime's own text, which the inbound pipeline stamps
    once it removed what the caller's metadata brought
    (:func:`delivered_record`). Taken once, by the first message the
    pipeline reads inside, so nothing a hook sends from there carries it."""
    token = _DELIVERING.set([kind] if kind is not None else None)
    try:
        yield
    finally:
        _DELIVERING.reset(token)


def delivered_record() -> str | None:
    """The provenance :func:`delivering` set for the message being read, once."""
    held = _DELIVERING.get()
    return held.pop() if held else None


def written_by_runtime(metadata: Mapping[str, Any]) -> bool:
    """Whether *metadata* (an event's or a message's) says the runtime wrote it."""
    return isinstance(metadata.get(RUNTIME_RECORD), str)


def runtime_record(kind: str) -> dict[str, str]:
    """The metadata a record of the runtime carries: *kind* names what it is."""
    return {RUNTIME_RECORD: kind}
