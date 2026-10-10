"""The names a message addresses, and an agent's answer that stays silent (RFC §19.7.5).

A name is ``@`` followed by an agent's channel id, or a person's name; ``@all``
names every agent. It starts the text or follows a character that is neither a
letter, a digit, ``_`` nor ``@`` (so ``ops@example.com`` names no one), it is
the longest run of identifier characters that follows (letters of any script,
digits, ``_ . -``), a final ``.`` excepted, and it matches ignoring case. Names
are read in what the speaker wrote: never in a fenced code block or a quoted
line.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

EVERYONE = "all"

NAME_LIMIT = 32
"""The longest name a person is addressed by: a name joins every turn's notes."""

_NAME = re.compile(r"(?<![\w@])@([\w.\-]+)")
_NOT_ID = re.compile(r"[^\w.\-]")
_FENCES = ("```", "~~~")
_BRACKETS = {"(": ")", "[": "]", "{": "}", "<": ">"}


@dataclass(frozen=True)
class Names:
    """What a message names: agents (by channel id) and people, in order."""

    agents: tuple[str, ...] = ()
    people: tuple[str, ...] = ()


def name_key(name: str) -> str:
    """A person's name kept to the characters a name holds (``Alice Martin`` →
    ``AliceMartin``, ``Hélène`` stays), the form a message names them by, at
    most :data:`NAME_LIMIT` characters."""
    return _NOT_ID.sub("", unicodedata.normalize("NFC", name))[:NAME_LIMIT]


def same_name(one: str, other: str) -> bool:
    """Whether two names read as one, ignoring case."""
    return one.casefold() == other.casefold()


def read_names(
    text: str, agents: Iterable[str], *, people: Iterable[str] = (), speaker: str | None = None
) -> Names:
    """The agents and people *text* names, in order, the speaker left out.

    Where a name reads as both an agent's channel id and a person's name, it
    is the agent's. A name that is neither addresses nobody.
    """
    agent_ids = list(agents)
    by_agent = {a.casefold(): a for a in agent_ids}
    by_person = {name_key(p).casefold(): p for p in people if name_key(p)}
    named_agents: list[str] = []
    named_people: list[str] = []
    for raw in _NAME.findall(_spoken(unicodedata.normalize("NFC", text))):
        key = (raw[:-1] if raw.endswith(".") else raw).casefold()
        if key == EVERYONE:
            found = [a for a in agent_ids]
        elif key in by_agent:
            found = [by_agent[key]]
        else:
            person = by_person.get(key[:NAME_LIMIT])
            if person is not None and person != speaker and person not in named_people:
                named_people.append(person)
            continue
        named_agents.extend(a for a in found if a != speaker and a not in named_agents)
    return Names(tuple(named_agents), tuple(named_people))


def _spoken(text: str) -> str:
    """*text* without its fenced code blocks and quoted lines: what the speaker
    says, not what it shows or repeats.

    One pass over the lines, never a regular expression: the text is a
    participant's, and a pattern that backtracks across lines (an unclosed
    fence, thousands of blank lines) would let one message stall the room.
    """
    kept: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        start = line.lstrip(" \t")
        if fence is not None:
            if start.startswith(fence):
                fence = None
            continue
        opening = next((f for f in _FENCES if start.startswith(f)), None)
        if opening is not None:
            fence = opening
        elif not start.startswith(">"):
            kept.append(line)
    return "\n".join(kept)


@dataclass(frozen=True)
class SilentToken:
    """The answer of an agent that stays silent, and the forms it is read in.

    The token is read ignoring case, surrounding whitespace, a final period and
    the brackets around it when it has some: ``(silent)``, ``Silent.``,
    ``SILENT``.
    """

    token: str = "(silent)"

    @property
    def _forms(self) -> tuple[str, ...]:
        token = self.token.strip().lower()
        if len(token) > 2 and _BRACKETS.get(token[0]) == token[-1]:
            return token, token[1:-1].strip()
        return (token,)

    def is_silent(self, text: str) -> bool:
        """Whether *text* is the token: says nothing. Empty text says nothing too."""
        said = text.strip().lower().removesuffix(".").rstrip()
        return not said or said in self._forms

    def may_become(self, text: str) -> bool:
        """Whether *text*, the start of a streamed answer, may still become the
        token: what is held back until it cannot."""
        said = text.strip().lower()
        if not said:
            return True
        return any(form.startswith(said) or said in (form, f"{form}.") for form in self._forms)
