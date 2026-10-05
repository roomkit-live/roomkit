"""The calls a text turn announced, each held as a call of its own (RFC §9.3, §12.4).

A provider's id names a call while it is in flight, within its round: a
second call of the round under it is a call of its own, refused, and a call
under it in a later round is a new call. Every record of a call (its one
report, the arguments it ran with, its outcome, who decides it) is the
call's, never its id's, so two calls under one id never share one.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(eq=False)
class AnnouncedCall:
    """One call the turn announced, and what the turn knows of it."""

    call: Any
    marker: Any | None = None
    """Its start marker, on which the arguments it ran with and its end ride."""
    external: bool = False
    """The channel's external handler decides it: one cut before its report
    is reported through that handler."""
    duplicate: bool = False
    """It came under an id another call of its round holds: refused."""
    arguments: dict[str, Any] | None = None
    """What it runs with, or what its gate had when it stopped it."""
    known: Any | None = None
    """Its report, once the model read its outcome: one cut before its
    observers heard it owes them that outcome."""
    reported: bool = False

    def as_ran(self) -> Any:
        """The call with the arguments it ran with, when its gate fixed them."""
        if self.arguments is None:
            return self.call
        return self.call.model_copy(update={"arguments": dict(self.arguments)})


_reporting: ContextVar[AnnouncedCall | None] = ContextVar("_reporting_call", default=None)


@contextmanager
def reporting(entry: AnnouncedCall | None) -> Iterator[None]:
    """Run the enclosed code for *entry*: a report claimed meanwhile under
    its id is its own, whichever other call holds the id."""
    token = _reporting.set(entry)
    try:
        yield
    finally:
        _reporting.reset(token)


class TurnCalls:
    """The turn's announced calls and the ids in flight in its round."""

    def __init__(self) -> None:
        self.entries: list[AnnouncedCall] = []
        self._holders: dict[str, AnnouncedCall] = {}
        # Reports claimed under an id no announced call carries (a call a
        # backend's provider served, reported through the turn).
        self._unannounced: set[str] = set()

    def announce(
        self, call: Any, *, marker: Any | None = None, external: bool = False
    ) -> AnnouncedCall:
        """Announce *call*, a duplicate when another call of the round holds
        its id, which then keeps it."""
        entry = AnnouncedCall(call, marker, external, duplicate=call.id in self._holders)
        if not entry.duplicate:
            self._holders[call.id] = entry
        self.entries.append(entry)
        return entry

    def next_round(self) -> None:
        """Free the ids of the round that ended: a call under one in the next
        round is a new call (RFC §12.4)."""
        self._holders.clear()

    def entry_of(self, call: Any) -> AnnouncedCall | None:
        """The entry of *call* itself, the object the round announced."""
        return next((e for e in reversed(self.entries) if e.call is call), None)

    def entry_for(self, call_id: str) -> AnnouncedCall | None:
        """The call a report under *call_id* is for: the one being reported
        now, else the one holding the id, else the latest under it."""
        current = _reporting.get()
        if current is not None and current.call.id == call_id:
            return current
        held = self._holders.get(call_id)
        if held is not None:
            return held
        return next((e for e in reversed(self.entries) if e.call.id == call_id), None)

    def claim(self, call_id: str) -> bool:
        """Claim the one report of the call *call_id* names now: ``False``
        when it was made."""
        entry = self.entry_for(call_id)
        if entry is None:
            if call_id in self._unannounced:
                return False
            self._unannounced.add(call_id)
            return True
        if entry.reported:
            return False
        entry.reported = True
        return True

    def was_reported(self, call_id: str) -> bool:
        """Whether the call *call_id* names now has had its one report."""
        entry = self.entry_for(call_id)
        return entry.reported if entry is not None else call_id in self._unannounced

    def unreported(self) -> list[AnnouncedCall]:
        """The announced calls no report claimed yet, in announcement order."""
        return [entry for entry in self.entries if not entry.reported]
