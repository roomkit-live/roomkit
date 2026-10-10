"""DiscussionConsole — a full-screen terminal for a room a discussion holds (RFC §19.7.5).

The person at the terminal is one of the room's people: what they type goes
into the room through the transport channel they name, and the screen shows
the room as it happens (every message, every tool call), the agents with
their identity and live state, and the speak queue. It follows the room
through hooks it removes when it closes, so it can be opened on a running
room and closed without disturbing it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.console._discussion_screen import DiscussionScreen
from roomkit.console._discussion_view import AgentCard, DiscussionView
from roomkit.core.hooks import HookRegistration
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.models.tool_call import ToolCallEvent
    from roomkit.orchestration.strategies.discussion import SpeakQueueEvent

logger = logging.getLogger("roomkit.console.discussion")

Command = Callable[[str], Awaitable[None]]
"""A slash command of the host's: receives the rest of the line."""

EVERYONE = "all"

HELP = """You are @{you}: write to the room, name agents with @ to ask them ({agents}).
  @{first} what do you think?   asks @{first} (no name: the agents that asked you, else everyone)
  /listen @{first}              @{first} only listens: it answers only when you name it
  /talk @{first}                @{first} talks again (@all names every agent)
  /help · /quit{extra}"""


class DiscussionConsole:
    """A full-screen terminal on a room a discussion holds.

    Example::

        kit.register_channel(WebSocketChannel("oncall"))
        await kit.attach_channel(room_id, "oncall")
        console = DiscussionConsole(kit, room_id, channel_id="oncall")
        await console.run()  # until Ctrl-C or /quit

    Args:
        kit: The kit the room lives in.
        room_id: A room a :class:`~roomkit.orchestration.Discussion` holds.
        channel_id: The transport channel bound to the room the person
            writes through; their messages come from it.
        sender_id: Who the person is to that channel. Defaults to
            ``channel_id``, the name agents address them by when the room
            records no participant for them.
        cards: How the panel presents each agent. Defaults to the room's
            intelligence bindings and each agent's identity (name, role,
            description, model).
        title: What the header says about the room.
        welcome: The first lines of the room pane. Defaults to the help.
        commands: The host's own slash commands, by name without the ``/``.
        log_file: Where the kit's logs go while the screen is up. Without
            one, records of level WARNING and above are held and handed to
            the logging set up before, once the screen closes.
        log_level: The level written to ``log_file``.
    """

    def __init__(
        self,
        kit: RoomKit,
        room_id: str,
        *,
        channel_id: str,
        sender_id: str | None = None,
        cards: Sequence[AgentCard] | None = None,
        title: str | None = None,
        welcome: str | None = None,
        commands: Mapping[str, Command] | None = None,
        log_file: str | Path | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self._kit = kit
        self._room_id = room_id
        self._channel_id = channel_id
        self._sender_id = sender_id or channel_id
        self._cards = tuple(cards) if cards is not None else None
        self._title = title or f"room {room_id}"
        self._welcome = welcome
        self._commands = dict(commands or {})
        self._log_file = Path(log_file) if log_file is not None else None
        self._log_level = log_level
        self._hook_prefix = f"discussion_console_{uuid4().hex[:8]}"
        self._view: DiscussionView | None = None
        self._screen: DiscussionScreen | None = None

    @property
    def view(self) -> DiscussionView:
        """What the screen shows; available once :meth:`run` started."""
        if self._view is None:
            raise RuntimeError("The console is not running")
        return self._view

    async def run(self) -> None:
        """Show the room until the person quits (Ctrl-C, Ctrl-D or ``/quit``)."""
        cards = self._cards or await self._room_cards()
        view = DiscussionView(
            cards,
            you=self._sender_id,
            queue=self._kit.speak_queue(self._room_id),
            on_change=self._refresh,
        )
        self._view = view
        self._screen = DiscussionScreen(view, on_submit=self._submit, title=self._title)
        for line in (self._welcome or self._help()).splitlines():
            view.lines.append(("class:dim", line))
        view.lines.append(("", ""))
        hooks = self._hooks()
        for hook in hooks:
            self._kit.hook_engine.add_room_hook(self._room_id, hook)
        restore_logs = _capture_logs(self._log_file, self._log_level)
        try:
            await self._screen.run()
        finally:
            for hook in hooks:
                self._kit.hook_engine.remove_room_hook(self._room_id, hook.name)
            restore_logs()

    async def say(self, text: str) -> None:
        """Post *text* to the room as the person at the terminal."""
        await self._kit.process_inbound(
            InboundMessage(
                channel_id=self._channel_id,
                sender_id=self._sender_id,
                content=TextContent(body=text),
            )
        )

    def note(self, text: str) -> None:
        """Add a note of the host's to the room pane (not to the room)."""
        self.view.note(text)

    def exit(self) -> None:
        """Close the screen; :meth:`run` returns."""
        if self._screen is not None:
            self._screen.exit()

    # -- What the person types --

    async def _submit(self, text: str) -> None:
        try:
            if text.startswith("/"):
                await self._command(text[1:])
            else:
                await self.say(text)
        except Exception as exc:
            logger.exception("The console could not handle %r", text)
            self.view.note(f"failed: {type(exc).__name__}: {exc}")

    async def _command(self, line: str) -> None:
        name, _, rest = line.partition(" ")
        if name == "quit":
            self.exit()
        elif name == "help":
            for help_line in self._help().splitlines():
                self.view.note(help_line)
        elif name in ("listen", "talk"):
            await self._set_listening(name, rest)
        elif name in self._commands:
            await self._commands[name](rest)
        else:
            self.view.note(f"unknown command /{name}: /help lists them")

    async def _set_listening(self, name: str, rest: str) -> None:
        agents = self._named_agents(rest)
        if not agents:
            self.view.note(f"name the agents: /{name} @{self.view.cards[0].handle}")
            return
        if name == "listen":
            await self._kit.listen_only(self._room_id, agents)
        else:
            await self._kit.talk_again(self._room_id, agents)

    def _named_agents(self, text: str) -> list[str]:
        handles = [c.handle for c in self.view.cards]
        by_name = {h.lower(): h for h in handles}
        named: list[str] = []
        for word in text.split():
            key = word.removeprefix("@").rstrip(".,").lower()
            found = handles if key == EVERYONE else [by_name[key]] if key in by_name else []
            named.extend(h for h in found if h not in named)
        return named

    def _help(self) -> str:
        extra = "".join(f" · /{name}" for name in self._commands)
        handles = [card.handle for card in self.view.cards] or ["agent"]
        return HELP.format(
            you=self._sender_id,
            agents=" ".join(f"@{h}" for h in handles),
            first=handles[0],
            extra=extra,
        )

    # -- Following the room --

    def _hooks(self) -> list[HookRegistration]:
        return [
            HookRegistration(
                trigger=HookTrigger.AFTER_BROADCAST,
                execution=HookExecution.ASYNC,
                fn=self._on_message,
                name=f"{self._hook_prefix}_messages",
                event_types={EventType.MESSAGE},
            ),
            HookRegistration(
                trigger=HookTrigger.ON_TOOL_CALL,
                execution=HookExecution.ASYNC,
                fn=self._on_tool_call,  # ty: ignore[invalid-argument-type]
                name=f"{self._hook_prefix}_tools",
            ),
            HookRegistration(
                trigger=HookTrigger.ON_SPEAK_QUEUE,
                execution=HookExecution.ASYNC,
                fn=self._on_speak_queue,  # ty: ignore[invalid-argument-type]
                name=f"{self._hook_prefix}_queue",
            ),
        ]

    async def _on_message(self, event: RoomEvent, context: RoomContext) -> None:
        view = self.view
        source = event.source.channel_id
        text = event.content.body if isinstance(event.content, TextContent) else ""
        text = text or f"[{event.content.type}]"
        if any(card.handle == source for card in view.cards):
            view.agent(source, event.addressed_to, text)
        else:
            view.person(self._author(event, context), event.addressed_to, text)

    async def _on_tool_call(self, event: ToolCallEvent, context: RoomContext) -> None:
        self.view.tool(event.channel_id, event.name, event.arguments, event.result)

    async def _on_speak_queue(self, event: SpeakQueueEvent, context: RoomContext) -> None:
        self.view.queue_changed(event)

    def _author(self, event: RoomEvent, context: RoomContext) -> str:
        if event.source.channel_id == self._channel_id:
            return self._sender_id
        pid = event.source.participant_id
        participant = next((p for p in context.participants if p.id == pid), None)
        if participant is not None:
            return participant.display_name or participant.id
        sender = (event.metadata or {}).get("sender_name")
        return sender if isinstance(sender, str) and sender else event.source.channel_id

    def _refresh(self) -> None:
        if self._screen is not None:
            self._screen.refresh()

    async def _room_cards(self) -> list[AgentCard]:
        """The room's agents, from its intelligence bindings and their identity."""
        cards = []
        for binding in await self._kit.list_bindings(self._room_id):
            if binding.category != ChannelCategory.INTELLIGENCE:
                continue
            channel = self._kit.channels.get(binding.channel_id)
            cards.append(
                AgentCard(
                    handle=binding.channel_id,
                    name=getattr(channel, "name", None),
                    role=getattr(channel, "role", None),
                    description=getattr(channel, "description", None),
                    model=_model(channel),
                )
            )
        return cards


def _model(channel: Any) -> str | None:
    """The model an agent runs on, when its provider says (a provider may raise)."""
    try:
        model = getattr(getattr(channel, "provider", None), "model_name", None)
    except Exception:
        return None
    return model if isinstance(model, str) else None


class _HeldRecords(logging.Handler):
    """Keeps records while the screen owns the terminal."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _capture_logs(log_file: Path | None, level: int) -> Callable[[], None]:
    """Route the root logger away from the terminal; the function that restores it."""
    root = logging.getLogger()
    saved, saved_level = list(root.handlers), root.level
    for handler in saved:
        root.removeHandler(handler)
    held: _HeldRecords | None = None
    taking: logging.Handler
    if log_file is not None:
        taking = logging.FileHandler(log_file)
        taking.setLevel(level)
        taking.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root.setLevel(min(saved_level, level) if saved_level else level)
    else:
        taking = held = _HeldRecords()
    root.addHandler(taking)

    def restore() -> None:
        root.removeHandler(taking)
        taking.close()
        root.setLevel(saved_level)
        for handler in saved:
            root.addHandler(handler)
        for record in held.records if held is not None else ():
            root.handle(record)

    return restore
