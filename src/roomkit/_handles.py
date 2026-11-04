"""What can name an agent of a discussion (RFC §19.7.5 rule 4), apart from the
strategy: the channels read a stored handle through it, and importing
:mod:`roomkit.orchestration` from there runs into the AI channel it loads.
No dependency inside RoomKit but :mod:`roomkit._text`.
"""

from __future__ import annotations

import re
import unicodedata

from roomkit._text import identifier

EVERYONE = "all"

NAME_LIMIT = 32
"""The longest name a person is addressed by (a name joins every turn's notes):
a longer one is not addressable, never cut short, since a cut name could be
another person's whole one."""

_HANDLE = re.compile(r"[\w.\-]+")


def is_handle(value: object) -> bool:
    """Whether *value* can name an agent: a name's characters, as a message
    names it (NFC), at most :data:`NAME_LIMIT` of them, no final ``.`` (a
    name drops it) and not ``all``, which names every agent. A transcript
    labels the agent's turns with it unchanged (``@`` and its
    :func:`~roomkit._text.identifier`), so a handle that would change there
    (``dev-``, ``a--b``) is none."""
    return (
        isinstance(value, str)
        and len(value) <= NAME_LIMIT
        and _HANDLE.fullmatch(value) is not None
        and unicodedata.normalize("NFC", value) == value
        and identifier(value, "") == value
        and not value.endswith(".")
        and value.casefold() != EVERYONE
    )
