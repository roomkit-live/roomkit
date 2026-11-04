"""What the owner's pulls let the executor read of a turn's stream (RFC §10.6.4).

A pull grants a number of items. The executor reads text and reasoning deltas,
and what a generation used, ahead of the owner's reader within what was granted,
and never past any other item (a call's start, a segment break, the loop's end)
until a pull its reader sent once it read that item: each pull says how many
items the reader read, a joined burst counting its deltas, and one that says
fewer than were read through such an item grants nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.turns._codec import is_delta

if TYPE_CHECKING:
    from roomkit.turns.wire import Pull


class PullGate:
    """The reads one turn's pulls allow, and the read under way."""

    def __init__(self) -> None:
        self._credits = 0
        # The items read off the stream, and how many were read through the
        # last one that is no delta.
        self._read = 0
        self._barrier = 0
        self.closing = False
        """The reader closed the stream: nothing more is read."""
        self.reading = False
        """The stream is reading an item."""
        # Whether the read under way follows a delta: ahead of the reader.
        self._ahead = False

    @property
    def granted(self) -> bool:
        """Whether a pull lets the stream read its next item."""
        return self._credits > 0

    def grant(self, pull: Pull) -> None:
        """Take a pull: one sent before its reader read the last item that is
        no delta grants nothing past that item."""
        if pull.read >= self._barrier:
            self._credits += pull.count

    def start_read(self) -> None:
        """An item is being read, on a pull's credit."""
        self._credits -= 1
        self._read += 1
        self.reading = True

    def item_read(self, item: Any) -> None:
        """*item* came: past one that is no delta, only a later pull reads on."""
        self._ahead = is_delta(item)
        if not self._ahead:
            self._credits = 0
            self._barrier = self._read

    def end_read(self) -> None:
        self.reading = False

    def close(self) -> bool:
        """The reader closed: whether the read under way is one ahead of it,
        which the close cuts; one only a pull let start (a running call)
        finishes first."""
        self.closing = True
        return self.reading and self._ahead
