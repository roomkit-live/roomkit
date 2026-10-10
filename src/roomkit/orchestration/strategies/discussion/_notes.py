"""What each turn of a discussion knows: the room and how it works (RFC §19.7.5 rule 14).

The notes join the turn's input as the runtime's own block. What they hold of
the room comes in a form nobody can turn into the runtime's words: channel
ids, people's names kept to a name's characters and bounded, the label each
author carries in the transcript (``Alice (2)`` for a second source that
takes Alice's name), and what a participant wrote only quoted, on one line,
between marks it cannot close (``roomkit._text.quoted``). An Agent's name,
role and description are the host's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

HEADER = "[Discussion: the room and how it works]"


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
) -> str:
    """The block a discussion's turn adds to the turn's input.

    *answering* names the event the turn answers (its author's label and a
    quote of it); *askers* are who asked for the turn, agents as ``@id`` and
    people by their label.
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
    lines.append(
        "How the room works: name an agent with @ to address it, as in "
        f"@{others[0].handle if others else 'name'}; a message that names nobody wakes "
        f"nobody; answer exactly {silent_token} when you have nothing to add."
    )
    return "\n".join(lines)


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
