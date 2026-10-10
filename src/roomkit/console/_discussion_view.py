"""What the discussion console shows: the room as it happens, the agents, the speak queue.

Pure state, no terminal: the console feeds it the room's messages, tool calls
and speak-queue changes, and the screen draws its lines and fragments, each a
``(style, text)`` pair (prompt_toolkit's formatted text). Styles are class
names the screen's palette defines.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from roomkit.orchestration.strategies.discussion import (
    SpeakQueue,
    SpeakQueueChange,
    SpeakQueueEvent,
)

Fragments = list[tuple[str, str]]

AGENT_STYLES = 5
"""How many agent colors the palette defines (``class:agent0`` and on)."""

_PREVIEW = 160


@dataclass(frozen=True)
class AgentCard:
    """An agent of the room as the panel presents it: its identity (RFC §19.1)."""

    handle: str
    name: str | None = None
    role: str | None = None
    description: str | None = None
    model: str | None = None
    tools: tuple[str, ...] = ()


class DiscussionView:
    """The room's lines, the agents' cards and the queue, kept as they change."""

    def __init__(
        self,
        cards: Sequence[AgentCard],
        *,
        you: str,
        queue: SpeakQueue | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.cards = tuple(cards)
        self.you = you
        self.queue = queue
        self.lines: list[tuple[str, str]] = []
        self.said: Counter[str] = Counter()
        self.used: Counter[str] = Counter()
        self._handles = tuple(c.handle for c in self.cards)
        self._on_change = on_change
        self._turn_said: dict[str, bool] = {}
        self._asked: set[tuple[str, str]] = set(queue.asked) if queue is not None else set()

    # -- What happens in the room --

    def person(
        self, label: str, addressed: Sequence[str] | None, text: str, *, mine: bool = False
    ) -> None:
        """A person's message, under the label the transcript gives its author
        (``Alice (2)`` for a second source taking Alice's name). Whether it is
        the person at the terminal is the console's to say, never a name's."""
        shown = f"@{self.you} (you)" if mine else label
        self._add(f"{shown} → {_to(addressed)}", "class:you" if mine else "class:person")
        self._body(text)

    def agent(self, author: str, addressed: Sequence[str] | None, text: str) -> None:
        self.said[author] += 1
        self._turn_said[author] = True
        card = self._card(author)
        about = f" · {card.model}" if card is not None and card.model else ""
        self._add(f"@{author}{about} → {_to(addressed)}", self.agent_style(author))
        self._body(text)

    def tool(self, agent: str, name: str, arguments: dict[str, Any], result: Any) -> None:
        self.used[agent] += 1
        self._add(f"  @{agent} ⚙ {tool_line(name, arguments, result)}", "class:dim")

    def note(self, text: str) -> None:
        self._add(f"· {text}", "class:note")

    def queue_changed(self, event: SpeakQueueEvent) -> None:
        """A change of the speak queue: the panel follows it, and what a
        person should notice joins the room as a note."""
        self.queue = event.queue
        for agent in event.channel_ids:
            self._turn_changed(event.change, agent)
        self._notice_asked(event.queue)
        notice = _NOTICES.get(event.change)
        if notice is not None:
            self.note(notice(", ".join(f"@{a}" for a in event.channel_ids)))
        self._changed()

    def _turn_changed(self, change: SpeakQueueChange, agent: str) -> None:
        if change == SpeakQueueChange.TURN_GIVEN:
            self._turn_said[agent] = False
        elif change == SpeakQueueChange.TURN_ENDED and self._turn_said.pop(agent, True) is False:
            self._add(f"@{agent} has nothing to add", "class:dim")
            self._add("")

    def _notice_asked(self, queue: SpeakQueue) -> None:
        asked = set(queue.asked)
        for agent, person in queue.asked:
            if (agent, person) in self._asked:
                continue
            whom = "you" if person.lower() == self.you.lower() else f"@{person}"
            self._add(f" >>> @{agent} is asking {whom} ", "class:ask")
            self._add("")
        self._asked = asked

    # -- What the screen draws --

    def agent_fragments(self) -> Fragments:
        fragments: Fragments = []
        for card in self.cards:
            fragments.extend(self._card_fragments(card))
        return fragments

    def status_text(self) -> str:
        queue = self.queue
        if queue is None:
            return "no discussion in this room"
        if queue.over:
            return "the discussion is over"
        parts = [f"speaking @{queue.speaking}" if queue.speaking else "nobody speaking"]
        if queue.queue:
            parts.append("next " + " ".join(f"@{a}" for a in queue.queue))
        if queue.waiting:
            parts.append("waiting for a person")
        return " · ".join(parts)

    def agent_style(self, handle: str) -> str:
        if handle not in self._handles:
            return "class:person"
        return f"class:agent{self._handles.index(handle) % AGENT_STYLES}"

    def _card_fragments(self, card: AgentCard) -> Fragments:
        marker, state, style = self._state(card.handle)
        title = " · ".join(part for part in (card.name, card.role) if part)
        fragments: Fragments = [
            (self.agent_style(card.handle), f"{marker} @{card.handle}"),
            (style, f"  {state}\n"),
        ]
        for line, line_style in (
            (title, ""),
            (card.model, ""),
            (card.description, "class:dim"),
            (f"tools: {', '.join(card.tools)}" if card.tools else None, "class:dim"),
        ):
            if line:
                fragments.append((line_style, f"  {line}\n"))
        counts = f"  {_count(self.said[card.handle], 'message')}"
        counts += f" · {_count(self.used[card.handle], 'tool call')}\n\n"
        fragments.append(("class:dim", counts))
        return fragments

    def _state(self, handle: str) -> tuple[str, str, str]:
        queue = self.queue
        if queue is None:
            return "·", "", "class:dim"
        if queue.speaking == handle:
            return "▶", "speaking", "class:speaking"
        asked = [person for agent, person in queue.asked if agent == handle]
        if handle in queue.listening:
            return "‖", "listening", "class:held"
        if asked:
            return "?", "asked " + ", ".join(f"@{p}" for p in asked), "class:you"
        if handle in queue.queue:
            place = queue.queue.index(handle) + 1
            return "•", "next" if place == 1 else f"#{place} in line", ""
        return "·", "idle", "class:dim"

    # -- Lines --

    def _card(self, handle: str) -> AgentCard | None:
        return next((c for c in self.cards if c.handle == handle), None)

    def _body(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            self._add(f"  {line}")
        self._add("")

    def _add(self, line: str, style: str = "") -> None:
        self.lines.append((style, line))
        self._changed()

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()


_NOTICES: dict[SpeakQueueChange, Callable[[str], str]] = {
    SpeakQueueChange.LISTENING: lambda who: f"listening only: {who}",
    SpeakQueueChange.TALKING_AGAIN: lambda who: f"talking again: {who}",
    SpeakQueueChange.WAITING: lambda _who: "the discussion waits for a person",
    SpeakQueueChange.OVER: lambda _who: "the discussion is over",
    SpeakQueueChange.INSTRUCTION_DROPPED: lambda who: f"an instruction for {who} was dropped",
}


def _to(addressed: Sequence[str] | None) -> str:
    if addressed is None:
        return "the room"
    return " ".join(f"@{a}" for a in addressed) or "nobody"


def tool_line(name: str, arguments: dict[str, Any], result: Any) -> str:
    """``name(args) → result``, both cut to stay on a line or two."""
    args = ", ".join(f"{k}={_cut(str(v), 60)!r}" for k, v in arguments.items())
    shown = "…" if result is None else _cut(str(result).replace("\n", " | "), _PREVIEW)
    return f"{name}({args}) → {shown}"


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"
