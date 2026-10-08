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

from collections.abc import Mapping
from typing import Any

from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent

RUNTIME_RECORD = "runtime_record"
"""The metadata key, on an event or an ``AIMessage``, naming what the runtime
wrote: ``"handoff"``, ``"summary"``, ``"handed_on_context"``."""

HANDOFF_FLAG = "handoff"
"""The metadata flag of the handoff relay, which alone names a relay stored
before :data:`RUNTIME_RECORD` existed; the inbound pipeline removes it from a
``SYSTEM`` event a sender supplies."""

HANDOFF_OPENING = "[Handoff:"
"""How the record of a handoff opens in the timeline (``[Handoff: a -> b]``)."""

HANDED_ON_CONTEXT = "[Context from previous agent"
"""How the context a previous agent hands on opens."""

COMPACTION_HEADER = "[Context compacted — earlier conversation summary]"
"""Opens the summary a channel's compaction writes in place of the history it
dropped."""

SUMMARY_MARK = "[Conversation summary"
"""How a summary's message opens, whatever header follows: a later summary finds
an earlier one by it, an inner provider's own included."""

SUMMARY_HEADER = f"{SUMMARY_MARK} — earlier messages compacted]"
"""Opens a memory summary's message."""


def written_by_runtime(metadata: Mapping[str, Any]) -> bool:
    """Whether *metadata* (an event's or a message's) says the runtime wrote it."""
    return isinstance(metadata.get(RUNTIME_RECORD), str)


def runtime_event(event: RoomEvent) -> bool:
    """Whether the runtime wrote *event*: it says so, or it is a handoff relay
    stored before the key existed, a ``SYSTEM`` event flagged
    :data:`HANDOFF_FLAG`."""
    if written_by_runtime(event.metadata):
        return True
    return event.type == EventType.SYSTEM and event.metadata.get(HANDOFF_FLAG) is True


def runtime_record(kind: str) -> dict[str, str]:
    """The metadata a record of the runtime carries: *kind* names what it is."""
    return {RUNTIME_RECORD: kind}
