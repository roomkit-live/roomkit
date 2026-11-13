"""Agent Client Protocol intelligence channel.

``ACPChannel`` makes RoomKit an ACP client: every Room is mapped to a distinct
session owned by an external coding agent.  How that agent is reached is the
transport's business — spawned here over stdio by default, or wherever a custom
:class:`~roomkit.channels.acp_transport.ACPTransport` can carry the protocol.
The reverse integration (exposing a RoomKit agent as an ACP server) is
intentionally out of scope.

The optional ``agent-client-protocol`` dependency is imported lazily so that
``import roomkit`` continues to work when the ``acp`` extra is not installed.

This module holds the channel's construction and its public surface. The
mechanics live in mixins, one responsibility each:

- ``_acp_client.ACPConnectionMixin``: the connection to the agent and the SDK;
- ``_acp_sessions.ACPSessionsMixin``: room and standalone-turn sessions;
- ``_acp_turn.ACPTurnMixin``: one prompt, from sending it to the end of the turn;
- ``_acp_events.ACPEventsMixin``: the agent's updates, tools and permissions.

Public methods stay defined here: the API reference renders this class without
inherited members.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roomkit.channels._acp_client import (
    _SDK,
    _STABLE_PROTOCOL_VERSION,
    ACPConnectionMixin,
    _absolute_path,
    _ACPClient,
    _config_values,
    _TurnState,
)
from roomkit.channels._acp_context import (
    ACPContextContributor,
    acp_event_text,
    contributed_blocks,
)
from roomkit.channels._acp_events import ACPEventsMixin
from roomkit.channels._acp_sessions import (
    ACPSessionsMixin,
    MCPServersForRoom,
    _reported_options,
    _servers_source,
    _session_setup,
)
from roomkit.channels._acp_turn import ACPTurnMixin
from roomkit.channels._discussion_turn import discussion_notes
from roomkit.channels._instruction import is_standalone, mark_instruction
from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.channels._turn_notes import turn_notes
from roomkit.channels.acp_transport import ACPTransport, _agent_transport
from roomkit.channels.base import Channel
from roomkit.core._changes import opens_no_turn
from roomkit.core.task_utils import log_task_exception
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    EventType,
)
from roomkit.models.event import RoomEvent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.steering import Cancel, SteeringDirective
from roomkit.models.tool_call import AfterResponseCallback, ToolCallObserver

if TYPE_CHECKING:
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.tools.external import ExternalToolHandler

logger = logging.getLogger("roomkit.channels.acp")

_SHUTDOWN_TIMEOUT = 5.0
"""Seconds the agent gets to acknowledge cancellation and session closes."""

_DEFAULT_ROOM_HISTORY = 20
"""Room messages an agent catches up on when it is asked to act (RFC §19.3.2)."""

_CHANNEL_VERBS = frozenset(
    {"set_config_option", "close_session", "session_config", "config_options"}
)
"""The per-room verbs a host reaches through ``kit.channel_call`` (RFC §10.6.3)."""


class ACPChannel(ACPConnectionMixin, ACPSessionsMixin, ACPTurnMixin, ACPEventsMixin, Channel):
    """Connect a RoomKit Room to an external ACP coding agent.

    One connection to the agent is opened lazily for the channel and one ACP
    session is created per Room. Prompts are serialized inside each session,
    while different Rooms progress concurrently over the same connection.

    Pass ``command`` for the usual case — the agent is spawned here as a
    subprocess and spoken to over its stdio. Pass ``transport`` instead when
    the agent runs somewhere this process cannot spawn it (another machine,
    behind a relay); see :class:`~roomkit.channels.acp_transport.ACPTransport`.
    Exactly one of the two is required.

    Args:
        channel_id: RoomKit channel identifier.
        command: Executable and arguments used to start the ACP agent. No shell
            is involved. Mutually exclusive with ``transport``.
        transport: How to reach an agent this channel does not spawn. Mutually
            exclusive with ``command`` (and with ``env`` / ``inherit_env``,
            which configure the spawn).
        cwd: Absolute working directory declared to the ACP session — and, for
            a spawned agent, the process's own directory. With a custom
            transport it names a directory on the *agent's* machine.
        additional_directories: Additional absolute directories exposed in the
            ACP session declaration.
        env: Environment variables added to the SDK's restricted inherited
            environment. Spawned agents only.
        inherit_env: Names of parent-process environment variables to forward
            to the agent. The ACP SDK strips the environment down to
            ``HOME/LOGNAME/PATH/SHELL/TERM/USER`` (MCP practice), which
            silently breaks tooling a coding agent relies on — e.g. without
            ``SSH_AUTH_SOCK``, every git-over-SSH operation prompts for key
            passphrases on the controlling terminal. Values are read at each
            process spawn; unset names are skipped; explicit ``env`` entries
            win over inherited ones. Nothing is forwarded by default. Spawned
            agents only.
        mcp_servers: The MCP servers each session is given, as ACP MCP-server
            descriptors of the official SDK: a list, the same for every room,
            or a function of the room id (sync or async), called each time a
            session opens for that room, so a server can be told which room
            it acts for (a stdio server's ``env``, a URL).
        session_config: ACP session config values set on every session the
            channel opens for a room, ``{"mode": "default"}`` for instance:
            the room's first session, one opened again after a recovery or an
            agent restart, and a standalone turn's. A value the session
            already has is not set again. A value the host sets on a room
            with :meth:`set_config_option` is set again too, over this, on the
            room's next session, until :meth:`close_session` forgets it. A
            value the agent refuses fails the opening, and the turn with it:
            the session is released, never prompted (RFC §6.4, ACP agent
            channel, *Session setup*). The mode matters most, since it decides what
            the agent asks ``external_tool_handler`` before acting.
        authentication_method: Optional ACP authentication method identifier.
        external_tool_handler: Permission policy and tool observability bridge.
            Without a handler, every permission request is rejected.
        room_history: How many room messages the agent catches up on when it is
            asked to act, having been skipped while it was not (RFC §19.3.2).
            An ACP session holds its history in the agent's process, so a room
            where two agents are addressed in turn would otherwise leave each
            one with a private thread and no way to know it. ``0`` turns the
            catch-up off. Only what the agent missed is sent, and only what
            visibility would have delivered to it (RFC §7.5 rule 8). It is
            also the window this channel declares to the framework, which
            loads the largest window any bound channel declares (floored at
            50 events while a hook or an identity hook is registered, capped
            at 2000). When that tail stops short of what the agent missed, as
            it does on a room with no hook, the catch-up header says that
            earlier events were not loaded.
        context_contributor: What the host adds to a turn's prompt that the
            agent cannot go and fetch — member memories, a document corpus, an
            organisation's rules. Awaited once per solicited turn with the
            room's context and the triggering event; the blocks it returns
            open the prompt, ahead of the catch-up and the request. Non-empty
            blocks can supply the prompt for an event without text, such as a
            host-managed attachment. A turn with neither text nor blocks is
            skipped. One that
            raises is logged and the turn goes without it.

            Turn-scoped context only. An ACP session keeps what it was already
            told, so a block that never changes is paid for again every turn;
            what is stable belongs to ``instructions``, sent once per
            session, or to the agent's own configuration (``AGENTS.md``, MCP
            servers): this is conversation.

            Nothing here is bounded. RoomKit does not truncate the blocks — it
            knows neither their unit nor the agent's tokenizer, and the model
            can change mid-session — and does not bound how long the
            contributor takes. Both budgets are the host's, and a slow
            contributor delays the broadcast for the whole room, not just for
            this agent. Nor can RoomKit filter what the blocks carry: the
            catch-up is filtered per reader because it is made of room events
            (RFC §7.5 rule 8), and these are not.

            ``context.recent_events`` is the framework's tail, not a window
            this channel guarantees: sized by the largest window a bound
            channel declares, floored at 50 events while a hook or an identity
            hook is registered, capped at 2000.
            With ``room_history=0`` on a room with no hook and no other
            history reader, it holds the triggering event alone. A contributor
            that needs the room's history reads it from the store.
        name: The agent's name, as an Agent's (RFC §19.1): a discussion's
            notes, a dispatch policy and a console present it by its name,
            ``role`` and ``description``.
        role: The agent's role.
        description: What the agent does.
        instructions: The integrator's standing direction for the agent (a
            persona, how the room works), since its own system prompt is not
            RoomKit's to set: sent in the first prompt of each session the
            channel opens for a room, a session opened again included, as the
            application's block ahead of the catch-up, and not again while
            the session keeps it (RFC §6.4). What changes from turn to turn
            belongs to ``context_contributor``.
    """

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE
    direction = ChannelDirection.BIDIRECTIONAL

    def __init__(
        self,
        channel_id: str,
        command: Sequence[str] | None = None,
        *,
        transport: ACPTransport | None = None,
        cwd: str | Path,
        additional_directories: Sequence[str | Path] | None = None,
        env: Mapping[str, str] | None = None,
        inherit_env: Sequence[str] | None = None,
        mcp_servers: Sequence[Any] | MCPServersForRoom | None = None,
        session_config: Mapping[str, str | bool] | None = None,
        publish_thinking_text: bool = False,
        authentication_method: str | None = None,
        external_tool_handler: ExternalToolHandler | None = None,
        room_history: int = _DEFAULT_ROOM_HISTORY,
        context_contributor: ACPContextContributor | None = None,
        name: str | None = None,
        role: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
    ) -> None:
        super().__init__(channel_id)
        # Its identity, as an Agent's (RFC §19.1, §6.4).
        self.name = name
        self.role = role
        self.description = description
        self._instructions = instructions if instructions and instructions.strip() else None
        # Validated here, not in the transport: ``cwd`` is a session/new field
        # first — with a remote transport it names a directory on the agent's
        # machine, which this process may not have at all.
        self._cwd = _absolute_path(cwd, field_name="cwd")
        self._transport = _agent_transport(
            command, transport, cwd=self._cwd, env=env, inherit_env=inherit_env
        )

        self._additional_directories = [
            _absolute_path(path, field_name="additional_directories")
            for path in (additional_directories or ())
        ]
        self._mcp_servers = _servers_source(mcp_servers)
        self._session_setup = _session_setup(session_config)
        # Room -> config values the host set on its session, set again on the
        # room's next one until close_session forgets them (RFC §6.4, *Session setup*).
        self._room_config: dict[str, dict[str, str | bool]] = {}
        self._authentication_method = authentication_method
        self._external_tool_handler = external_tool_handler
        # ON_TOOL_CALL's report on a call the agent ran, wired by the kit.
        self._tool_report_hook: ToolCallObserver | None = None
        # ON_TOOL_CALL's observers only, for a call that never ran and that
        # the channel decided itself (RFC §9.3).
        self._tool_observer_hook: ToolCallObserver | None = None
        if room_history < 0:
            raise ValueError(
                "room_history is a count of messages to catch up on: pass 0 to turn "
                "the catch-up off"
            )
        self._room_history = room_history
        self._context_contributor = context_contributor

        self._loaded_sdk: _SDK | None = None
        self._client = _ACPClient(self)
        self._connection: Any = None
        self._connection_generation = 0
        self._message_queue: Any = None
        self._connect_lock = asyncio.Lock()
        self._room_locks: dict[str, asyncio.Lock] = {}
        # Room -> the lock its session is opened under, one opening at a time.
        self._opening_locks: dict[str, asyncio.Lock] = {}
        self._sessions: dict[str, str] = {}
        # Room -> the session a standalone turn is running in (RFC §10.1.1
        # step 7): opened for that turn, never prompted after it, never the room's.
        self._turn_sessions: dict[str, str] = {}
        self._session_rooms: dict[str, str] = {}
        # The agent's reasoning reaches a room's subscribers as its length; its
        # text only when the host asks: the channel has no visibility gate.
        self._publish_thinking_text = publish_thinking_text
        self._session_options: dict[str, list[Any]] = {}
        self._prompted_index: dict[str, int] = {}
        # Rooms whose session was sent a labelled request (RFC §6.4).
        self._labelled_rooms: set[str] = set()
        # Rooms whose session was sent the instructions (RFC §6.4).
        self._instructed_rooms: set[str] = set()
        self._turns: dict[str, _TurnState] = {}
        self._agent_info: dict[str, Any] | None = None
        self._agent_closes_sessions = True
        self._handler_started = False
        self._closed = False
        self._realtime: RealtimeBackend | None = None
        self._after_response_hook: AfterResponseCallback | None = None
        # Room -> how many times the channel was detached from it: a turn
        # started before a detach never prompts after it (RFC §10.6.3).
        self._detached: dict[str, int] = {}
        # The closes of detached rooms' sessions, run off the detach.
        self._closings: set[asyncio.Task[None]] = set()

    @property
    def info(self) -> dict[str, Any]:
        """Return ACP connection and agent metadata without exposing arguments."""
        return {
            "transport": self._transport.name,
            "protocol_version": _STABLE_PROTOCOL_VERSION,
            "sdk_version": self._loaded_sdk.version if self._loaded_sdk else None,
            "connected": self._connection is not None,
            "agent": self._agent_info,
            "session_count": len(self._sessions),
            "active_turns": self.active_turns,
        }

    @property
    def active_turns(self) -> int:
        """Turns in flight: registered by ``_prompt_stream`` once it holds the
        room, while its session is opened, dropped when its stream closes. The
        whole of the turn as the consumer sees it, not only while the agent is
        answering."""
        return len(self._turns)

    def session_config(self, room_id: str) -> dict[str, str | bool]:
        """Current ACP session config values for *room_id*, keyed by config id.

        Agents publish their tunables through this one list — ``model``,
        ``mode``, ``effort``, vendor switches. Empty until the room's session
        exists (sessions open on the first prompt).

        Tracks what the agent announces. A switch made *inside* the agent
        with its own slash command may not be announced at all — the ACP
        bridge for Claude Code relays ``/model`` output as plain text and
        sends no config update — so drive changes through
        :meth:`set_config_option` when the value must stay observable.
        """
        return _config_values(self._options_for(room_id))

    def config_options(self, room_id: str) -> list[dict[str, Any]]:
        """The agent's session tunables for *room_id*, as ACP describes them.

        Full descriptors — id, name, current value, available choices — for
        surfaces that let a user pick one (a model picker). Empty until the
        session exists. :meth:`session_config` is the values-only shortcut.
        """
        return [dict(option) for option in self._options_for(room_id)]

    async def set_config_option(
        self,
        room_id: str,
        config_id: str,
        value: str | bool,
    ) -> dict[str, str | bool]:
        """Set one session config option — ``set_config_option(room, "model", "opus")``.

        Returns the full config mapping the agent reports back, so the caller
        sees the value it landed on (agents resolve aliases) plus any option
        the change invalidated. Opens the room's session if the first prompt
        has not yet done so, which connects to the agent.

        The value belongs to the room, not only to this session: the room's
        next session (after a recovery or an agent restart) is set to it too,
        over ``session_config``, until :meth:`close_session`. A call that opens
        the room's session sets its value in place of the room's earlier one,
        so a value the agent has come to refuse is replaced, not replayed. A
        call the channel's detach from the room overtook keeps nothing: the
        detach forgot the room's values, and a session the call opened is
        closed (RFC §6.4, *Detaching*).
        """
        generation = self._detached.get(room_id, 0)
        connection = await self._ensure_connection()
        # Never on the room's turn lock: an in-flight turn holds it for its
        # whole duration, and waiting for that would deadlock a caller running
        # inside the turn (a tool handler). An opening has its own lock.
        session_id, opened = await self._room_session(room_id, connection, {config_id: value})
        if opened:
            # Opening the session set the value, in place of the room's.
            options = self._session_options.get(session_id, [])
        else:
            response = await connection.set_config_option(
                config_id=config_id,
                session_id=session_id,
                value=value,
            )
            options = _reported_options(response)
        values = _config_values(options)
        if await self._detached_during(room_id, generation, session_id, opened, connection):
            return values
        if self._sessions.get(room_id) != session_id:
            # Closed (or rebuilt) while the agent answered: what it reports,
            # and the value, are no longer the room's to keep.
            return values
        self._session_options[session_id] = options
        self._room_config.setdefault(room_id, {})[config_id] = values.get(config_id, value)
        await self._publish_config_options(session_id, options, values)
        return values

    async def channel_call(self, room_id: str, verb: str, arguments: dict[str, Any]) -> Any:
        """The room's session verbs, wherever its session is (RFC §10.6.3).

        What ``kit.channel_call`` reaches: ``set_config_option``
        (``config_id``, ``value``), ``close_session``, ``session_config``
        and ``config_options``, each answered as the method of that name
        answers it. Through the kit, the values the host sets belong to
        the room: the room's next executor sets them on the session it
        opens. A call made on this object directly changes this process's
        copy alone.

        ``await kit.channel_call("claude", room.id, "set_config_option",
        config_id="model", value="opus")``
        """
        if verb not in _CHANNEL_VERBS:
            raise ValueError(
                f"ACP channel {self.channel_id!r} has no per-room verb {verb!r}: "
                f"it has {', '.join(sorted(_CHANNEL_VERBS))}"
            )
        answer = getattr(self, verb)(room_id, **arguments)
        return await answer if inspect.isawaitable(answer) else answer

    def room_state(self, room_id: str) -> dict[str, Any]:
        """The config values the host set on the room's session, which belong
        to the room until :meth:`close_session` (RFC §6.4)."""
        config = self._room_config.get(room_id)
        return {"config": dict(config)} if config else {}

    def restore_room_state(self, room_id: str, state: dict[str, Any]) -> None:
        """Take back the config values the room keeps when this channel holds
        none for it: the room's next session is set to them (RFC §6.4)."""
        config = state.get("config")
        if room_id in self._room_config or not isinstance(config, dict):
            return
        values = {
            key: value
            for key, value in config.items()
            if isinstance(key, str) and isinstance(value, str | bool)
        }
        if values:
            self._room_config[room_id] = values

    async def steer(self, directive: SteeringDirective, *, turn_id: str) -> int | None:
        """Steer the turn *turn_id* (RFC §21.3, §10.6.8): an external agent
        takes a ``Cancel`` alone, as :meth:`cancel` sends it (the turn's
        session opening stopped, or its prompt cancelled), so the turn ends
        as a cancelled prompt ends (RFC §6.4): 1. ``None`` while the turn has
        not reached the agent yet, which its owner then cuts as the turn's
        cancel; 0 for another directive."""
        if not isinstance(directive, Cancel):
            return 0
        found = next(((k, t) for k, t in self._turns.items() if t.turn_id == turn_id), None)
        if found is None:
            return None
        key, turn = found
        await self._cancel_turn(key, turn)
        return 1

    async def on_room_attached(self, room_id: str, binding: ChannelBinding) -> None:
        """Compile the patterns that find a copy of a runtime mark as the
        channel joins a room, off the event loop: the room's first prompt
        then does not wait for them (RFC §6.4)."""
        await compile_mark_patterns()

    async def on_room_detached(self, room_id: str) -> None:
        """The room's session, and the values the host set on it, go with the
        binding (RFC §10.6.3, §6.4): a later attachment opens a new session,
        caught up on the room. A detach never waits for the agent: the
        session is closed off it, in the background once the room's running
        turn ended (which closes it itself when it prompted the room's
        session); a turn started before the detach that has not prompted yet
        ends without prompting, marked ``interrupted``."""
        self._room_config.pop(room_id, None)
        self._detached[room_id] = self._detached.get(room_id, 0) + 1
        session_id = self._sessions.get(room_id)
        if session_id is None:
            return
        closing = asyncio.get_running_loop().create_task(
            self._close_detached(room_id, session_id), name=f"roomkit-acp-close-{room_id}"
        )
        self._closings.add(closing)
        closing.add_done_callback(self._closings.discard)
        closing.add_done_callback(log_task_exception)

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.TEXT, ChannelMediaType.RICH],
            supports_rich_text=True,
        )

    @property
    def recent_events_window(self) -> int:
        """Room tail this channel reads — the catch-up window (RFC §19.3.2).

        The framework sizes ``RoomContext.recent_events`` to the largest window
        any bound channel declares, under a floor it keeps for hooks (50
        events, while one is registered). So a ``room_history`` under that
        floor reads a tail that was loaded anyway, one above it grows the tail
        to match, and on a room with no hook the declaration is what loads the
        tail at all: declaring the window is what keeps the two in step.
        """
        return self._room_history

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError("ACP intelligence channels do not accept inbound messages")

    async def deliver(
        self,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        return ChannelOutput.empty()

    async def on_event(
        self,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """Create a lazy ACP prompt stream for a Room event; none for an
        event that is no message nor instruction (RFC §6.4), nor for an edit
        or a delete, no message a turn may answer (RFC §10.3)."""
        if event.source.channel_id == self.channel_id or opens_no_turn(event):
            return ChannelOutput.empty()

        await compile_mark_patterns()
        text = acp_event_text(event)

        room_id = context.room.id if context.room is not None else event.room_id
        # Host-only blocks can be collected now. Catch-up is deliberately
        # computed inside the lazy stream, after connection recovery: a dead
        # transport discards its sessions and their prompted-index marks.
        blocks = await contributed_blocks(
            self._context_contributor, context, event, channel_id=self.channel_id
        )
        # A discussion's turn (RFC §19.7.5): who is in the room, which message
        # the turn answers and how the room works.
        notes = turn_notes(discussion_notes(event))
        if not text.strip() and not blocks and notes is None:
            return ChannelOutput.empty()
        instruction = text if event.type == EventType.INSTRUCTION else None
        if instruction is not None and instruction.strip():
            # The application's direction, never a participant's line (RFC
            # §10.1.1 step 6). The session keeps it once prompted, so the mark
            # is what makes later turns read it for what it was.
            text = mark_instruction(instruction)
        standalone = is_standalone(event)
        # One live record for the turn, handed to the stream and to the output
        # alike: the stop reason is only known when the prompt returns, and
        # every MESSAGE segment reads this mapping as it stands when it is
        # persisted. A dict literal here would be a snapshot taken now, before
        # the turn has an outcome to report.
        metadata = ResponseMetadata({"acp": {"protocol_version": _STABLE_PROTOCOL_VERSION}})
        if standalone:
            # The room's session did not produce this reply: its next catch-up
            # carries it (RFC §10.1.1 step 7), and this mark is how it knows.
            metadata["acp"]["standalone"] = True
        # The catch-up this turn sends covers the room up to its latest event,
        # and the cursor must say so. The trigger's own index is not that
        # bound: an instruction is never committed and carries index 0
        # (RFC §10.1.1), so marking it would replay the same catch-up next turn.
        seen_index = max((e.index for e in context.recent_events), default=event.index)
        return ChannelOutput(
            responded=True,
            response_stream=self._prompt_stream(
                room_id,
                event.id,
                blocks,
                context,
                event,
                text,
                max(seen_index, event.index),
                metadata,
                notes=notes,
                standalone=standalone,
                generation=self._detached.get(room_id, 0),
            ),
            response_metadata=metadata,
        )

    def _prompting(self, room_id: str) -> bool:
        """Whether a turn of *room_id* is running here, its prompt sent or its
        session being opened or rebuilt: what a cancellation reaches."""
        return any(turn.room_id == room_id for turn in self._turns.values())

    def session_id(self, room_id: str) -> str | None:
        """Return the process-local ACP session identifier for a Room."""
        return self._sessions.get(room_id)

    async def cancel(self, room_id: str) -> bool:
        """Request cancellation of the active ACP turn for a Room."""
        for key, turn in self._turns.items():
            if turn.room_id == room_id and turn.preparing:
                await self._cancel_turn(key, turn)
                return True
        # A standalone turn runs in its own session: that is the one to stop.
        session_id = self._turn_sessions.get(room_id) or self._sessions.get(room_id)
        connection = self._connection
        if session_id is None or connection is None:
            return False
        await connection.cancel(session_id)
        return True

    async def _cancel_turn(self, key: str, turn: _TurnState) -> None:
        """Stop *turn*, registered under *key*: a turn whose session is being
        opened or rebuilt has no prompt out, so its runner is stopped before
        one can go; a prompting turn's prompt is cancelled in its session,
        the key it is registered under."""
        if turn.preparing:
            turn.cancel_requested = True
            if turn.runner is not None:
                turn.runner.cancel()
            return
        connection = self._connection
        if connection is not None:
            await connection.cancel(key)

    async def _close_detached(self, room_id: str, session_id: str) -> None:
        """Close the session *room_id* had when it was detached, unless the
        room holds another by then (it was attached again and opened one)."""
        async with self._room_turn_lock(room_id), self._room_opening_lock(room_id):
            if self._sessions.get(room_id) != session_id:
                return
            await self._discard_room_session(room_id, self._connection)

    async def _detached_during(
        self, room_id: str, generation: int, session_id: str, opened: bool, connection: Any
    ) -> bool:
        """Whether the channel was detached from *room_id* since a host call
        started at *generation* (the number of its detaches then). A session
        the call opened is then the detached room's, discarded as a turn
        across a detach discards its own."""
        if self._detached.get(room_id, 0) == generation:
            return False
        if opened and self._sessions.get(room_id) == session_id:
            await self._discard_room_session(room_id, connection)
        return True

    async def close_session(self, room_id: str) -> bool:
        """Forget one Room's ACP session, and close it where the agent can.

        *Forget* is the whole of it: every map keyed by the session, and the
        room's turn lock once no session is left behind it, is dropped here.
        A long-lived channel cycling sessions (one per conversation, one per
        reconnect) would otherwise carry every dead session's config options
        until the channel itself closed. ``session/close`` goes only to an
        agent that announces it; one that does not keeps the session until the
        connection closes. Never raises for a close the agent refuses: returns
        ``True`` once the session is forgotten, ``False`` when there was none.

        The config values the host set on the room are forgotten too: its
        next session opens on ``session_config`` alone.
        """
        async with self._room_turn_lock(room_id), self._room_opening_lock(room_id):
            self._room_config.pop(room_id, None)
            try:
                return await self._discard_room_session(room_id, self._connection)
            finally:
                self._room_locks.pop(room_id, None)
                self._opening_locks.pop(room_id, None)

    async def close(self) -> None:
        """Cancel turns, close sessions where the agent can, and close the transport.

        Shutdown is bounded: the graceful ACP round trips share
        ``_SHUTDOWN_TIMEOUT``, and the transport teardown runs even when
        they time out, fail, or the caller is cancelled mid-close (a second
        Ctrl-C landing on ``close_session``). An agent that has stopped
        answering must not outlive — or hang — the process that started it.
        """
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.wait_for(self._say_goodbye(), _SHUTDOWN_TIMEOUT)
        except TimeoutError:
            logger.debug("ACP agent did not acknowledge shutdown in time; forcing teardown")
        except Exception:
            logger.debug("ACP graceful shutdown failed; forcing teardown", exc_info=True)
        finally:
            await self._teardown()

    async def _say_goodbye(self) -> None:
        """Best-effort graceful half: stop the turns, close the sessions."""
        preparing = [
            turn.runner
            for turn in self._turns.values()
            if turn.preparing and turn.runner is not None
        ]
        for runner in preparing:
            runner.cancel()
        connection = self._connection
        if connection is not None:
            await asyncio.gather(
                *(
                    connection.cancel(session_id)
                    for session_id, turn in self._turns.items()
                    if not turn.preparing
                ),
                return_exceptions=True,
            )
        runners = [turn.runner for turn in self._turns.values() if turn.runner is not None]
        for runner in runners:
            runner.cancel()
        if runners:
            await asyncio.gather(*runners, return_exceptions=True)

        if connection is not None:
            await asyncio.gather(
                *(
                    self._release_session(connection, session_id)
                    for session_id in self._sessions.values()
                )
            )

    async def _teardown(self) -> None:
        """Close the transport and drop the session state. Never raises."""
        try:
            async with self._connect_lock:
                await self._close_transport()
        except Exception:
            logger.debug("ACP transport teardown failed", exc_info=True)

        if self._external_tool_handler is not None and self._handler_started:
            self._handler_started = False
            with contextlib.suppress(Exception):
                await self._external_tool_handler.stop()

        for closing in list(self._closings):
            closing.cancel()
        self._detached.clear()
        self._turns.clear()
        self._sessions.clear()
        self._turn_sessions.clear()
        self._session_rooms.clear()
        self._session_options.clear()
        self._room_config.clear()
        self._prompted_index.clear()
        self._labelled_rooms.clear()
        self._instructed_rooms.clear()
