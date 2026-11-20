"""What each turn of a discussion knows: the room and how it works (RFC §19.7.5 rule 14).

The notes join the turn's input as the runtime's own block. What they hold of
the room comes in a form nobody can turn into the runtime's words: agents'
handles, and the names people are addressed by, kept to a name's characters (no
space, no punctuation but ``_ . -``) and bounded. Everything else a
participant controls is quoted, on one line, between marks it cannot close
(``roomkit._text.quoted``): the label an author carries in the transcript
(``Alice (2, not Alice)`` for a second source that takes Alice's name) and what they
wrote. An Agent's name, role and description are the host's, and so are
the rules a discussion gives in place of the default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

HEADER = "[Discussion: the room and how it works]"

DEFAULT_RULES = (
    "How the room works: an agent takes a turn only when a message names it with @. "
    "Whenever another agent must act next, end your message with its handle, as in "
    '{handle}: a name without @ reaches nobody, nor does "your turn" or "over to you" '
    "alone, in any language, and a message that names nobody wakes nobody. Answer exactly "
    "{silent} when you have nothing to add."
)
"""The rules a turn's notes give when the host gives none (rule 14). An agent
told only that a name with ``@`` addresses another went on calling the others
by their bare names, and the discussion stopped: the rule asks for the
handle wherever someone must act next. One told that much still handed over
with "à toi" and no handle: the rule names that too."""


class TurnKind(StrEnum):
    ANSWER = "answer"
    INSTRUCTION = "instruction"
    REGENERATE = "regenerate"


@dataclass(frozen=True)
class AgentLine:
    """An agent of the room as the others are told about it."""

    handle: str
    name: str | None = None
    role: str | None = None
    description: str | None = None


def turn_notes(
    me: str,
    others: Sequence[AgentLine],
    people: Sequence[str],
    *,
    kind: TurnKind,
    answering: str,
    askers: Sequence[str],
    silent_token: str,
    rules: str | None = None,
) -> str:
    """The block a discussion's turn adds to the turn's input, for the agent
    whose handle is *me*.

    *answering* names the event the turn answers (its author's label and a
    quote of it); *askers* are who asked for the turn, agents as ``@handle``
    and people by their label. *rules* are the host's (:func:`room_rules`).
    """
    lines = [
        HEADER,
        f"You are @{me}, one of several agents in a conversation with people; "
        "one agent speaks at a time.",
    ]
    if others:
        lines.append("The other agents:")
        lines.extend(f"- @{a.handle}{_about(a)}" for a in others)
    if people:
        lines.append("People you may address: " + ", ".join(f"@{p}" for p in people) + ".")
    lines.append(_what_it_answers(kind, answering, askers))
    lines.append(room_rules(rules, f"@{others[0].handle if others else 'name'}", silent_token))
    return "\n".join(lines)


def room_rules(rules: str | None, handle: str, silent_token: str) -> str:
    """How the room works, as a turn's notes give it: the host's *rules*, else
    :data:`DEFAULT_RULES`, ``{handle}`` filled with *handle* and ``{silent}``
    with *silent_token*. Replaced rather than formatted: any other brace in a
    host's text stays as written."""
    text = rules if rules is not None else DEFAULT_RULES
    return text.replace("{handle}", handle).replace("{silent}", silent_token)


def _what_it_answers(kind: TurnKind, answering: str, askers: Sequence[str]) -> str:
    if kind == TurnKind.INSTRUCTION:
        return "This turn takes the application's instruction as its input."
    if kind == TurnKind.REGENERATE:
        return f"This turn answers again {answering}: your new answer replaces the earlier one."
    asked = f"; asked by {', '.join(askers)}" if askers else ""
    return f"This turn answers {answering}, read at its place in the conversation{asked}."


def _about(agent: AgentLine) -> str:
    said = " — ".join(part for part in (agent.name, agent.role, agent.description) if part)
    return f": {said}" if said else ""
