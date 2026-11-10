"""The channels an executor declares (RFC §10.6.7): names, and prefixes.

An entry ending in ``*`` declares every channel whose id starts with what
precedes it (``"org-*"``), and ``"*"`` alone every channel. Unlike a
predicate, such a declaration is data, so a distributed backend tells it to
the room owners of other processes as it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WILDCARD = "*"


@dataclass(frozen=True, slots=True)
class Hosted:
    """The channel ids an executor declares: names, and prefixes ending in
    ``*``.

    Raises:
        ValueError: An entry holds a ``*`` elsewhere than at its end.
    """

    entries: frozenset[str]
    _prefixes: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for entry in self.entries:
            if WILDCARD in entry[:-1]:
                raise ValueError(
                    f"hosts entry {entry!r}: a '*' ends an entry to declare a prefix "
                    "('org-*'), and stands nowhere else"
                )
        prefixes = sorted(entry[:-1] for entry in self.entries if entry.endswith(WILDCARD))
        object.__setattr__(self, "_prefixes", tuple(prefixes))

    def covers(self, channel_id: str) -> bool:
        """Whether this declaration names *channel_id* or a prefix of it."""
        return channel_id in self.entries or channel_id.startswith(self._prefixes)
