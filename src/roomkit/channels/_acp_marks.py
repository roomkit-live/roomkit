"""The fixed lines of the room context block an ACP prompt carries (RFC §6.4).

Apart from the block's builder so that the finder of a copy of a runtime mark
can name them without importing it: the builder reads each event's text
through that finder.
"""

from __future__ import annotations

ROOM_CONTEXT_OPENING = "[Room context"
"""How the room context block opens; what follows says how much it holds."""

ROOM_CONTEXT_CLOSING = " Context only; the request follows.]"
"""Ends the room context block's first line."""

ROOM_CONTEXT_END = "[End of room context]"
"""The room context block's last line."""
