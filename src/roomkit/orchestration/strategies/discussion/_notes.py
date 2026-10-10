"""What each turn of a discussion knows: the room and how it works (RFC §19.7.5 rule 14).

The notes join the turn's input as the runtime's own block. Every name in them
is a channel id or a person's name kept to identifier characters, never text a
participant wrote: a display name cannot carry instructions into the runtime's
voice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

HEADER = "[Discussion: the room and how it works]"


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
    askers: Sequence[str],
    *,
    silent_token: str,
    instruction: bool = False,
) -> str:
    """The block a discussion's turn adds to the turn's input."""
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
    if instruction:
        lines.append("This turn takes the application's instruction as its input.")
    elif askers:
        lines.append(
            "This turn answers the message that asked for you, from "
            + ", ".join(f"@{a}" for a in askers)
            + ", read at its place in the conversation."
        )
    lines.append(
        "How the room works: name an agent with @ to address it, as in "
        f"@{others[0].handle if others else 'name'}; a message that names nobody wakes "
        f"nobody; answer exactly {silent_token} when you have nothing to add."
    )
    return "\n".join(lines)


def _about(agent: AgentLine) -> str:
    said = " — ".join(part for part in (agent.name, agent.role, agent.description) if part)
    return f": {said}" if said else ""
