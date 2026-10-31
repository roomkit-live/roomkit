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

from roomkit._text import identifier
from roomkit.orchestration.strategies.discussion import (
    SpeakQueue,
    SpeakQueueChange,
    SpeakQueueEvent,
)

Fragments = list[tuple[str, str]]

AGENT_STYLES = 5
"""How many agent colors the palette defines (``class:agent0`` and on)."""

_PREVIEW = 160
_TASK_PREVIEW = 90
_CURSOR = "▍"


@dataclass
class _Live:
    """The answer an agent is writing, as it streams: shown until it is said."""

    agent: str
    stream: str = ""
    text: str = ""


@dataclass(frozen=True)
class AgentCard:
    """An agent of the room as the panel presents it: its identity (RFC §19.1)."""

    handle: str
    """The name the room addresses the agent by, shown as ``@handle``."""
    name: str | None = None
    role: str | None = None
    description: str | None = None
    model: str | None = None
    tools: tuple[str, ...] = ()
    channel_id: str | None = None
    """The agent's channel, when its handle is another name (a discussion's
    handle, RFC §19.7.5 rule 4); None: its handle is its channel id."""

    @property
    def channel(self) -> str:
        """The channel id the room's events, queue and tasks name the agent by."""
        return self.channel_id or self.handle


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
        self._channels = tuple(c.channel for c in self.cards)
        self._labels = {c.channel: f"@{c.handle}" for c in self.cards}
        self._on_change = on_change
        self._turn_said: dict[str, bool] = {}
        self._asked: set[tuple[str, str]] = set(queue.asked) if queue is not None else set()
        self.live: _Live | None = None
        self.working: dict[str, str] = {}
        """The room's running tasks: task id to the agent the task is for."""
        self._owners: dict[str, str] = {}
        """The agent a task is for, told before its start was observed: the
        hooks that report a start are not waited for (RFC §9.3)."""
        self.version = 0
        """Bumped at every change, the live text's included."""

    # -- What happens in the room --

    def person(
        self, label: str, addressed: Sequence[str] | None, text: str, *, mine: bool = False
    ) -> None:
        """A person's message, under the label the transcript gives its author
        (``Alice (2)`` for a second source taking Alice's name). Whether it is
        the person at the terminal is the console's to say, never a name's."""
        shown = f"@{self.you} (you)" if mine else label
        self._add(f"{shown} → {self._to(addressed)}", "class:you" if mine else "class:person")
        self._body(text)

    def agent(
        self, author: str, addressed: Sequence[str] | None, text: str, *, by: str | None = None
    ) -> None:
        """An agent's message; *by*, the worker that wrote it, for a task's
        result posted as the agent's message."""
        self.said[author] += 1
        self._turn_said[author] = True
        if by is None:
            # A posted result is no part of what the agent is writing.
            self._said_live(author, text)
        self._add(self._agent_header(author, addressed, by=by), self.agent_style(author))
        self._body(text)

    # -- What an agent is writing --

    def live_start(self, agent: str, stream: str = "") -> None:
        """*agent* starts streaming an answer, as *stream*."""
        self.live = _Live(agent, stream)
        self._changed()

    def live_text(self, delta: str, stream: str = "") -> None:
        """More of the answer being streamed; another stream's is ignored."""
        live = self.live
        if live is not None and delta and live.stream == stream:
            live.text += delta
            self._changed()

    def live_end(self, stream: str | None = None) -> None:
        """The stream ended (any stream, for None): what it said is in the
        room's lines."""
        if self.live is not None and stream in (None, self.live.stream):
            self.live = None
            self._changed()

    def _said_live(self, author: str, text: str) -> None:
        """*author*'s message is in: the live text keeps only what follows it."""
        live = self.live
        if live is None or live.agent != author:
            return
        said = text.strip()
        at = live.text.find(said) if said else -1
        live.text = live.text[at + len(said) :].lstrip() if at >= 0 else ""

    # -- The room's tasks --

    def task_started(self, task_id: str, agent: str, task: str) -> None:
        self.working[task_id] = self._owners.pop(task_id, agent)
        self.note(
            f"task {short_task(task_id)} started: {self.label(agent)} — "
            f"{_cut(task, _TASK_PREVIEW)}"
        )

    def task_for(self, task_id: str, agent: str) -> None:
        """The task *task_id* is *agent*'s, whatever channel does its work."""
        if task_id not in self.working:
            self._owners[task_id] = agent
            return
        self.working[task_id] = agent
        self._changed()

    def task_ended(self, task_id: str, status: str, seconds: float | None) -> None:
        self._owners.pop(task_id, None)
        agent = self.working.pop(task_id, None)
        took = f" after {seconds:.0f} s" if seconds is not None else ""
        whose = f" ({self.label(agent)})" if agent else ""
        self.note(f"task {short_task(task_id)} {status}{took}{whose}")

    def tool(self, agent: str, name: str, arguments: dict[str, Any], result: Any) -> None:
        self.used[agent] += 1
        self._add(f"  {self.label(agent)} ⚙ {tool_line(name, arguments, result)}", "class:dim")

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
            self.note(notice(", ".join(self.label(a) for a in event.channel_ids)))
        self._changed()

    def _turn_changed(self, change: SpeakQueueChange, agent: str) -> None:
        live = self.live
        if change == SpeakQueueChange.TURN_ENDED and live is not None and live.agent == agent:
            # A stream cut short may never say it ended.
            self.live_end()
        if change == SpeakQueueChange.TURN_GIVEN:
            self._turn_said[agent] = False
        elif change == SpeakQueueChange.TURN_ENDED and self._turn_said.pop(agent, True) is False:
            self._add(f"{self.label(agent)} has nothing to add", "class:dim")
            self._add("")

    def _notice_asked(self, queue: SpeakQueue) -> None:
        asked = set(queue.asked)
        for agent, person in queue.asked:
            if (agent, person) in self._asked:
                continue
            # A person's label may carry its @ already: one @ either way.
            name = person.removeprefix("@")
            whom = "you" if name.lower() == self.you.removeprefix("@").lower() else f"@{name}"
            self._add(f" >>> {self.label(agent)} is asking {whom} ", "class:ask")
            self._add("")
        self._asked = asked

    # -- What the screen draws --

    def shown_lines(self) -> list[tuple[str, str]]:
        """The room's lines, then the answer being streamed, if any."""
        live = self.live
        if live is None or not live.text:
            return list(self.lines)
        header = (self.agent_style(live.agent), self._agent_header(live.agent, None, writing=True))
        body = [("", f"  {line}") for line in live.text.splitlines() or [""]]
        style, last = body[-1]
        body[-1] = (style, last + _CURSOR)
        return [*self.lines, header, *body]

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
        parts = [f"speaking {self.label(queue.speaking)}" if queue.speaking else "nobody speaking"]
        if queue.queue:
            parts.append("next " + " ".join(self.label(a) for a in queue.queue))
        if queue.waiting:
            parts.append("waiting for a person")
        return " · ".join(parts)

    def label(self, agent: str) -> str:
        """The agent of channel *agent* as the panel names it: ``@`` and its
        handle, its channel id when no card has it."""
        return self._labels.get(agent, f"@{agent}")

    def agent_style(self, agent: str) -> str:
        if agent not in self._channels:
            return "class:person"
        return f"class:agent{self._channels.index(agent) % AGENT_STYLES}"

    def _card_fragments(self, card: AgentCard) -> Fragments:
        marker, state, style = self._state(card.channel)
        title = " · ".join(part for part in (card.name, card.role) if part)
        fragments: Fragments = [
            (self.agent_style(card.channel), f"{marker} @{card.handle}"),
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
        running = sum(1 for agent in self.working.values() if agent == card.channel)
        if running:
            fragments.append(("class:note", f"  ⧗ {_count(running, 'task')} running\n"))
        counts = f"  {_count(self.said[card.channel], 'message')}"
        counts += f" · {_count(self.used[card.channel], 'tool call')}\n\n"
        fragments.append(("class:dim", counts))
        return fragments

    def _state(self, channel: str) -> tuple[str, str, str]:
        queue = self.queue
        if queue is None:
            return "·", "", "class:dim"
        if queue.speaking == channel:
            return "▶", "speaking", "class:speaking"
        asked = [person for agent, person in queue.asked if agent == channel]
        if channel in queue.listening:
            return "‖", "listening", "class:held"
        if asked:
            whom = ", ".join(f"@{p.removeprefix('@')}" for p in asked)
            return "?", f"asked {whom}", "class:you"
        if channel in queue.queue:
            place = queue.queue.index(channel) + 1
            return "•", "next" if place == 1 else f"#{place} in line", ""
        return "·", "idle", "class:dim"

    # -- Lines --

    def _card(self, channel: str) -> AgentCard | None:
        return next((c for c in self.cards if c.channel == channel), None)

    def _agent_header(
        self,
        author: str,
        addressed: Sequence[str] | None,
        *,
        by: str | None = None,
        writing: bool = False,
    ) -> str:
        card = self._card(author)
        about = f" · {card.model}" if card is not None and card.model and by is None else ""
        # The worker may be another agent of the discussion: named by its handle.
        worker = (self._labels.get(by) or identifier(by, "worker")) if by else ""
        work = f" · a task's result, by {worker}" if by else ""
        where = "writing…" if writing else f"→ {self._to(addressed)}"
        return f"{self.label(author)}{about}{work} {where}"

    def _to(self, addressed: Sequence[str] | None) -> str:
        if addressed is None:
            return "the room"
        return " ".join(self.label(a) for a in addressed) or "nobody"

    def _body(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            self._add(f"  {line}")
        self._add("")

    def _add(self, line: str, style: str = "") -> None:
        self.lines.append((style, line))
        self._changed()

    def _changed(self) -> None:
        self.version += 1
        if self._on_change is not None:
            self._on_change()


_NOTICES: dict[SpeakQueueChange, Callable[[str], str]] = {
    SpeakQueueChange.LISTENING: lambda who: f"listening only: {who}",
    SpeakQueueChange.TALKING_AGAIN: lambda who: f"talking again: {who}",
    SpeakQueueChange.WAITING: lambda _who: "the discussion waits for a person",
    SpeakQueueChange.OVER: lambda _who: "the discussion is over",
    SpeakQueueChange.INSTRUCTION_DROPPED: lambda who: f"an instruction for {who} was dropped",
}


def tool_line(name: str, arguments: dict[str, Any], result: Any) -> str:
    """``name(args) → result``, both cut to stay on a line or two."""
    args = ", ".join(f"{k}={_cut(str(v), 60)!r}" for k, v in arguments.items())
    shown = "…" if result is None else _cut(str(result).replace("\n", " | "), _PREVIEW)
    return f"{name}({args}) → {shown}"


def short_task(task_id: str) -> str:
    """The id a person types for a task: six hex digits of ``task-<hex>``."""
    return task_id.removeprefix("task-")[:6]


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"
