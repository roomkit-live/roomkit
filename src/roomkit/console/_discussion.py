"""DiscussionConsole — a full-screen terminal for a room a discussion holds (RFC §19.7.5).

The person at the terminal is one of the room's people: what they type goes
into the room through the transport channel they name, and the screen shows
the room as it happens (every message, every tool call, the answer an agent
is writing), the agents with their identity, live state and running tasks,
and the speak queue. It follows the room through hooks and a subscription to
what the room publishes, removed when it closes, so it can be opened on a
running room and closed without disturbing it, whichever executors run the
room's agents.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.channels._discussion_turn import discussion_handles
from roomkit.channels._speaker import turn_labels
from roomkit.channels.acp import ACPChannel
from roomkit.console._discussion_screen import DiscussionScreen
from roomkit.console._discussion_view import AgentCard, DiscussionView, short_task
from roomkit.core.hooks import HookRegistration
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.orchestration.strategies.discussion._names import read_names
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.tasks._room_tasks import HOST_REACH, why_not_cancelled
from roomkit.tasks.background import BACKGROUND_TOOL
from roomkit.tasks.status import room_task_lines

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.orchestration.strategies.discussion import SpeakQueueEvent

logger = logging.getLogger("roomkit.console.discussion")

# The ephemeral events that tell an agent's presence (RFC §10.6.7).
_PRESENCE = {EphemeralEventType.PRESENCE_ONLINE, EphemeralEventType.PRESENCE_OFFLINE}

Command = Callable[[str], Awaitable[None]]
"""A slash command of the host's: receives the rest of the line."""

HELP = """You are @{you}: write to the room, name agents with @ to ask them ({agents}).
  @{first} what do you think?   asks @{first} (no name: the agents that asked you, else everyone)
  /listen @{first}              @{first} only listens: it answers only when you name it
  /talk @{first}                @{first} talks again (@all names every agent){tasks}
  /help · /quit{extra}"""

TASKS_HELP = """
  /task @{first} <what to do>      long work as a task, while the room goes on
  /tasks · /cancel <task>        the room's tasks, and stopping one"""

_TASK_WINDOW = 500
"""How many of the StatusBus's latest entries ``/tasks`` reads."""


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
            intelligence bindings, each agent's handle in the discussion
            (RFC §19.7.5 rule 4) and its identity (name, role, description,
            model).
        title: What the header says about the room.
        welcome: The first lines of the room pane. Defaults to the help.
        commands: The host's own slash commands, by name without the ``/``.
        tasks: Whether ``/task``, ``/tasks`` and ``/cancel`` start, list and
            stop the room's tasks (RFC §23.4). Defaults to on when an agent
            of the room has background work set up
            (:meth:`~roomkit.RoomKit.setup_background`).
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
        tasks: bool | None = None,
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
        self._tasks = tasks
        self._log_file = Path(log_file) if log_file is not None else None
        self._log_level = log_level
        self._hook_prefix = f"discussion_console_{uuid4().hex[:8]}"
        self._tool_arguments: dict[tuple[str, str], dict[str, Any]] = {}
        self._view: DiscussionView | None = None
        self._screen: DiscussionScreen | None = None

    @property
    def view(self) -> DiscussionView:
        """What the screen shows; available once :meth:`run` started."""
        if self._view is None:
            raise RuntimeError("The console is not running")
        return self._view

    async def run(self) -> None:
        """Show the room until the person quits (Ctrl-C, Ctrl-D or ``/quit``).

        Raises:
            RoomNotFoundError: the room does not exist (with no ``cards``,
                the panel reads the room's agents and their handles).
        """
        cards = self._cards or await self._room_cards()
        if self._tasks is None:
            setups = self._kit._background_setups
            self._tasks = any(card.channel in setups for card in cards)
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
        restore_logs = _capture_logs(self._log_file, self._log_level)
        hooks = self._hooks()
        live = None
        activity = None
        try:
            for hook in hooks:
                self._kit.hook_engine.add_room_hook(self._room_id, hook)
            activity = await self._kit.realtime.subscribe_to_room(self._room_id, self._on_activity)
            await self._read_presence()
            live = self._follow_streams()
            await self._screen.run()
        finally:
            for hook in hooks:
                self._kit.hook_engine.remove_room_hook(self._room_id, hook.name)
            if activity is not None:
                with contextlib.suppress(Exception):
                    await self._kit.realtime.unsubscribe(activity)
            if live is not None:
                live()
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
        elif self._tasks and name in ("task", "tasks", "cancel"):
            await self._task_command(name, rest)
        else:
            self.view.note(f"unknown command /{name}: /help lists them")

    # -- The room's tasks --

    async def _task_command(self, name: str, rest: str) -> None:
        if name == "task":
            await self._start_task(rest)
        elif name == "tasks":
            await self._list_tasks()
        else:
            await self._cancel_task(rest)

    async def _start_task(self, rest: str) -> None:
        first, _, task = rest.strip().partition(" ")
        agents = self._named_agents(first)
        if len(agents) != 1 or not task.strip():
            self.view.note(f"usage: /task @{self.view.cards[0].handle} <what to do>")
            return
        started = await self._kit.start_task(self._room_id, agents[0], task.strip())
        self.view.task_for(started.id, agents[0])

    async def _list_tasks(self) -> None:
        lines = await room_task_lines(self._kit.status_bus, self._room_id, _TASK_WINDOW)
        if not lines:
            self.view.note("no task in this room")
        now = datetime.now(UTC)
        for line in lines:
            since, ended = line.get("since"), line.get("ended")
            took = _seconds(since, ended or now)
            elapsed = f"{took:4.0f} s" if took is not None else "     "
            agent = self.view.label(self.view.working.get(line["task_id"], line["agent"]))
            self.view.note(
                f"{short_task(line['task_id'])}  {agent:<13} {line.get('status', '?'):<10}"
                f" {elapsed}  {str(line.get('task', ''))[:60]}"
            )

    async def _cancel_task(self, rest: str) -> None:
        wanted = rest.strip().removeprefix("task-")
        lines = await room_task_lines(self._kit.status_bus, self._room_id, _TASK_WINDOW)
        found = [
            t for t in lines if wanted and t["task_id"].removeprefix("task-").startswith(wanted)
        ]
        if len(found) != 1:
            self.view.note(
                f"no task {rest.strip()!r}: /tasks lists them"
                if not found
                else f"{rest.strip()!r} names several tasks: give more of its id"
            )
            return
        (line,) = found
        if line.get("status") != "running":
            self.view.note(f"task {short_task(line['task_id'])} had ended")
            return
        self.view.note(await self._cancelled(line["task_id"]))

    async def _cancelled(self, task_id: str) -> str:
        """What ``/cancel`` says of *task_id*, a task a line of the room lists
        running: cancelled with the host's reach when the kit runs it for this
        room, never from the line alone; else why not (RFC §23.4)."""
        short = short_task(task_id)
        record = self._kit._room_tasks
        reach = record.reach(task_id, self._room_id)
        # As an agent's cancel_task is decided, with the host's reach: one
        # asked as another's cancel ends joins it.
        cancelled = await self._kit._cancel_task_for(
            task_id, None, room_id=self._room_id, reaches=HOST_REACH
        )
        if cancelled:
            return f"task {short} cancelled"
        # Read again: the task may have ended while the cancel waited.
        why = why_not_cancelled(record.reach(task_id, self._room_id))
        if why is not None:
            return f"task {short} is still running: {why}"
        if reach is not None or record.ended_with(task_id, self._room_id) is not None:
            return f"task {short} had ended"
        return f"task {short} is not one this kit runs for this room"

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
        """The channel ids of the agents *text* names by handle, as the
        discussion reads a name (RFC §19.7.5 rule 4), ``@all`` every one; a
        command's bare word (``/listen dev``) names too."""
        words = " ".join(w if w.startswith("@") else f"@{w}" for w in text.split())
        handles = {c.handle: c.channel for c in self.view.cards}
        return list(read_names(words, handles).agents)

    def _help(self) -> str:
        extra = "".join(f" · /{name}" for name in self._commands)
        handles = [card.handle for card in self.view.cards] or ["agent"]
        return HELP.format(
            you=self._sender_id,
            agents=" ".join(f"@{h}" for h in handles),
            first=handles[0],
            tasks=TASKS_HELP.format(first=handles[0]) if self._tasks else "",
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
                trigger=HookTrigger.ON_ERROR,
                execution=HookExecution.ASYNC,
                fn=self._on_error,
                name=f"{self._hook_prefix}_errors",
            ),
            HookRegistration(
                trigger=HookTrigger.ON_SPEAK_QUEUE,
                execution=HookExecution.ASYNC,
                fn=self._on_speak_queue,  # ty: ignore[invalid-argument-type]
                name=f"{self._hook_prefix}_queue",
            ),
            HookRegistration(
                trigger=HookTrigger.ON_TASK_DELEGATED,
                execution=HookExecution.ASYNC,
                fn=self._on_task_started,
                name=f"{self._hook_prefix}_task_started",
            ),
            HookRegistration(
                trigger=HookTrigger.ON_TASK_COMPLETED,
                execution=HookExecution.ASYNC,
                fn=self._on_task_ended,
                name=f"{self._hook_prefix}_task_ended",
            ),
        ]

    async def _on_message(self, event: RoomEvent, context: RoomContext) -> None:
        view = self.view
        source = event.source.channel_id
        text = event.content.body if isinstance(event.content, TextContent) else ""
        text = text or f"[{event.content.type}]"
        if any(card.channel == source for card in view.cards):
            # A task's result posted as the agent's message names its worker.
            meta = event.metadata or {}
            worker = meta.get("agent_id") if meta.get("task_id") else None
            by = worker if isinstance(worker, str) and worker != source else None
            view.agent(source, event.addressed_to, text, by=by)
        else:
            view.person(
                self._author(event, context), event.addressed_to, text, mine=self._mine(event)
            )

    async def _on_error(self, event: RoomEvent, context: RoomContext) -> None:
        """A turn of one of the room's agents failed (RFC §9.2's ON_ERROR)."""
        source = event.source.channel_id
        if any(card.channel == source for card in self.view.cards):
            text = event.content.body if isinstance(event.content, TextContent) else ""
            self.view.failed(source, text)

    async def _read_presence(self) -> None:
        """Mark the agents no executor serves as the console opens; their
        presence events say what changes next (RFC §10.6.7)."""
        for presence in await self._kit.agent_presence(self._room_id):
            self.view.presence(presence.channel_id, presence.online, quiet=True)

    async def _on_activity(self, event: EphemeralEvent) -> None:
        """A tool call of the room's agents, or an agent's presence, as the
        room publishes it.

        Not a hook: ON_TOOL_CALL fires on the executor running the turn, which
        may be another process, and a room hook that executor did not register
        keeps it from taking the turn (RFC §10.6.7).
        """
        if event.type in _PRESENCE:
            if any(card.channel == event.user_id for card in self.view.cards):
                online = event.type == EphemeralEventType.PRESENCE_ONLINE
                self.view.presence(event.user_id, online)
            return
        calls = event.data.get("tool_calls")
        if not isinstance(calls, list):
            return
        agent = event.channel_id or str(event.data.get("channel_id") or "")
        if event.type == EphemeralEventType.TOOL_CALL_START:
            self._tool_calls_started(agent, calls)
        elif event.type == EphemeralEventType.TOOL_CALL_END:
            self._tool_calls_ended(agent, calls)

    def _tool_calls_started(self, agent: str, calls: list[dict[str, Any]]) -> None:
        """Keep each call's arguments: its end carries only its result."""
        for call in calls:
            arguments = call.get("arguments")
            key = (agent, str(call.get("id")))
            self._tool_arguments[key] = arguments if isinstance(arguments, dict) else {}

    def _tool_calls_ended(self, agent: str, calls: list[dict[str, Any]]) -> None:
        for call in calls:
            name, result = str(call.get("name")), call.get("result")
            arguments = self._tool_arguments.pop((agent, str(call.get("id"))), {})
            self.view.tool(agent, name, arguments, result)
            if name == BACKGROUND_TOOL:
                self._tool_started_task(agent, result)

    def _tool_started_task(self, agent: str, result: Any) -> None:
        """The task a ``work_in_background`` call started is the calling
        agent's, whatever channel does the work."""
        try:
            answer = json.loads(str(result))
        except (TypeError, ValueError):
            return
        task_id = answer.get("task_id") if isinstance(answer, dict) else None
        if isinstance(task_id, str):
            self.view.task_for(task_id, agent)

    async def _on_task_started(self, event: RoomEvent, context: RoomContext) -> None:
        meta = event.metadata or {}
        task_id = str(meta.get("task_id") or "")
        if task_id:
            self.view.task_started(
                task_id, str(meta.get("agent_id") or "?"), str(meta.get("task_input") or "")
            )

    async def _on_task_ended(self, event: RoomEvent, context: RoomContext) -> None:
        meta = event.metadata or {}
        task_id = str(meta.get("task_id") or "")
        if not task_id:
            return
        status = meta.get("task_status")
        duration = meta.get("duration_ms")
        seconds = duration / 1000 if isinstance(duration, int | float) else None
        self.view.task_ended(task_id, str(getattr(status, "value", status) or "ended"), seconds)

    # -- What an agent is writing --

    def _follow_streams(self) -> Callable[[], None] | None:
        """Follow the answers agents stream, through a connection of the
        person's channel when it takes one (a WebSocket channel); the
        function that stops it."""
        channel = self._kit.channels.get(self._channel_id)
        register = getattr(channel, "register_connection", None)
        unregister = getattr(channel, "unregister_connection", None)
        if register is None or unregister is None:
            return None
        connection = f"{self._hook_prefix}_live"

        async def ignore(_connection: str, _event: RoomEvent) -> None:
            return None

        register(connection, ignore, room_id=self._room_id, stream_send_fn=self._on_stream)
        return lambda: unregister(connection)

    async def _on_stream(self, _connection: str, message: Any) -> None:
        kind = getattr(message, "type", None)
        stream = str(getattr(message, "stream_id", ""))
        if kind == "stream_start":
            source = message.source.channel_id
            if any(card.channel == source for card in self.view.cards):
                self.view.live_start(source, stream)
        elif kind == "stream_chunk":
            self.view.live_text(message.delta, stream)
        elif kind in ("stream_end", "stream_error"):
            self.view.live_end(stream)

    async def _on_speak_queue(self, event: SpeakQueueEvent, context: RoomContext) -> None:
        self.view.queue_changed(event)

    def _mine(self, event: RoomEvent) -> bool:
        """Whether the person at the terminal sent *event*: their channel and
        sender, whatever name it carries."""
        sender = event.source.participant_id
        return event.source.channel_id == self._channel_id and sender in (None, self._sender_id)

    @staticmethod
    def _author(event: RoomEvent, context: RoomContext) -> str:
        """The label the transcript gives *event*'s author: their name kept to
        a name's characters, with the rank a look-alike of an earlier source
        carries, or their channel (``@sms1``, ``@sms1 (2)`` for a second
        nameless sender) when they have none, ranked among the room's recent
        turns."""
        labels = turn_labels([*context.recent_events, event], context)
        return labels.get(event.id) or f"@{event.source.channel_id}"

    def _refresh(self) -> None:
        if self._screen is not None:
            self._screen.refresh()

    async def _room_cards(self) -> list[AgentCard]:
        """The room's agents, from its intelligence bindings, their handles and
        their identity."""
        room = await self._kit.get_room(self._room_id)
        handles = discussion_handles(room.metadata)
        cards = []
        for binding in await self._kit.list_bindings(self._room_id):
            if binding.category != ChannelCategory.INTELLIGENCE:
                continue
            agent = self._kit.channels.get(binding.channel_id)
            cards.append(
                AgentCard(
                    handle=handles.get(binding.channel_id, binding.channel_id),
                    channel_id=binding.channel_id,
                    name=getattr(agent, "name", None),
                    role=getattr(agent, "role", None),
                    description=getattr(agent, "description", None),
                    model=_model(agent),
                )
            )
        return cards


def _model(channel: Any) -> str | None:
    """The model an agent runs on, when its provider says (a provider may raise)."""
    if isinstance(channel, ACPChannel):
        return "external agent (ACP)"
    try:
        model = getattr(getattr(channel, "provider", None), "model_name", None)
    except Exception:
        return None
    return model if isinstance(model, str) else None


def _seconds(since: Any, until: Any) -> float | None:
    if isinstance(since, datetime) and isinstance(until, datetime):
        return (until - since).total_seconds()
    return None


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
    held: _HeldRecords | None = None
    taking: logging.Handler
    if log_file is not None:
        # Opened first: a path that cannot be written leaves the logging as it was.
        taking = logging.FileHandler(log_file)
        taking.setLevel(level)
        taking.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root.setLevel(min(saved_level, level) if saved_level else level)
    else:
        taking = held = _HeldRecords()
    for handler in saved:
        root.removeHandler(handler)
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
