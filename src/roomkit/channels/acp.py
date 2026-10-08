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
    _model_dump,
    _TurnState,
)
from roomkit.channels._acp_context import (
    ACPContextContributor,
    acp_event_text,
    contributed_blocks,
)
from roomkit.channels._acp_events import ACPEventsMixin
from roomkit.channels._acp_sessions import ACPSessionsMixin
from roomkit.channels._acp_turn import ACPTurnMixin
from roomkit.channels._instruction import instruction_fingerprint, is_standalone, mark_instruction
from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.channels.acp_transport import ACPTransport, StdioACPTransport
from roomkit.channels.base import Channel
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
from roomkit.models.event import RoomEvent, is_tool_call_record
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.tool_call import AfterResponseCallback, ToolCallObserver

if TYPE_CHECKING:
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.tools.external import ExternalToolHandler

logger = logging.getLogger("roomkit.channels.acp")

_SHUTDOWN_TIMEOUT = 5.0
"""Seconds the agent gets to acknowledge cancellation and session closes."""

_DEFAULT_ROOM_HISTORY = 20
"""Room messages an agent catches up on when it is asked to act (RFC §19.3.2)."""


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
        mcp_servers: ACP MCP-server descriptors accepted by the official SDK.
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
            what is stable belongs to the agent's own configuration
            (``AGENTS.md``, MCP servers) — ACP has no instruction channel and
            this is not one, it is conversation.

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
        mcp_servers: Sequence[Any] | None = None,
        authentication_method: str | None = None,
        external_tool_handler: ExternalToolHandler | None = None,
        room_history: int = _DEFAULT_ROOM_HISTORY,
        context_contributor: ACPContextContributor | None = None,
    ) -> None:
        super().__init__(channel_id)
        if command is not None and transport is not None:
            raise ValueError(
                "command spawns the agent here and transport reaches one that is "
                "already running: pass one, not both"
            )
        # Validated here, not in the transport: ``cwd`` is a session/new field
        # first — with a remote transport it names a directory on the agent's
        # machine, which this process may not have at all.
        self._cwd = _absolute_path(cwd, field_name="cwd")
        if command is not None:
            self._transport: ACPTransport = StdioACPTransport(
                command, cwd=self._cwd, env=env, inherit_env=inherit_env
            )
        elif transport is not None:
            if env is not None or inherit_env is not None:
                raise ValueError(
                    "env and inherit_env configure the default subprocess spawn; "
                    "a custom transport carries its own environment"
                )
            self._transport = transport
        else:
            raise ValueError("pass command to spawn the agent, or transport to reach one")

        self._additional_directories = [
            _absolute_path(path, field_name="additional_directories")
            for path in (additional_directories or ())
        ]
        self._mcp_servers = list(mcp_servers or ())
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
        self._message_queue: Any = None
        self._connect_lock = asyncio.Lock()
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._sessions: dict[str, str] = {}
        # Room -> the session a standalone turn is running in (RFC §10.1.1
        # step 7): opened for that turn, never prompted after it, never the room's.
        self._turn_sessions: dict[str, str] = {}
        self._session_rooms: dict[str, str] = {}
        self._session_options: dict[str, list[Any]] = {}
        self._prompted_index: dict[str, int] = {}
        # Rooms whose session was sent a labelled request (RFC §6.4).
        self._labelled_rooms: set[str] = set()
        self._turns: dict[str, _TurnState] = {}
        self._agent_info: dict[str, Any] | None = None
        self._agent_closes_sessions = True
        self._handler_started = False
        self._closed = False
        self._realtime: RealtimeBackend | None = None
        self._after_response_hook: AfterResponseCallback | None = None

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
        """Turns in flight: registered by ``_prompt_stream`` when the prompt
        goes out, dropped when its stream closes. The whole of the turn as
        the consumer sees it, not only while the agent is answering."""
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
        """
        connection = await self._ensure_connection()
        session_id = self._sessions.get(room_id)
        if session_id is None and room_id in self._turn_sessions:
            # A standalone turn is in flight and holds the room's lock, so this
            # caller may be running inside it (a tool handler): waiting would
            # deadlock. That turn never opens the room's session, so nothing
            # races this one. The setting is the room's, not the turn's.
            session_id = await self._session_for(room_id, connection)
        elif session_id is None:
            # Session creation is serialized on the room's turn lock so a
            # concurrent first prompt cannot open a second session. An
            # existing session skips the lock deliberately: an in-flight turn
            # holds it for its whole duration, and waiting for that would
            # deadlock a caller running inside the turn (a tool handler).
            async with self._room_turn_lock(room_id):
                session_id = await self._session_for(room_id, connection)
        response = await connection.set_config_option(
            config_id=config_id,
            session_id=session_id,
            value=value,
        )
        options = _model_dump(getattr(response, "config_options", None))
        self._session_options[session_id] = options if isinstance(options, list) else []
        values = _config_values(self._session_options[session_id])
        await self._publish_config_options(session_id, self._session_options[session_id], values)
        return values

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
        """Create a lazy ACP prompt stream for a Room event."""
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()
        if is_tool_call_record(event):
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
        if not text.strip() and not blocks:
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
        if instruction is not None:
            metadata["instruction"] = instruction_fingerprint(instruction)
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
                standalone=standalone,
            ),
            response_metadata=metadata,
        )

    def session_id(self, room_id: str) -> str | None:
        """Return the process-local ACP session identifier for a Room."""
        return self._sessions.get(room_id)

    async def cancel(self, room_id: str) -> bool:
        """Request cancellation of the active ACP turn for a Room."""
        for turn in self._turns.values():
            if turn.room_id == room_id and turn.rebuilding:
                # The invalid session is already forgotten, but the logical
                # turn still exists. Stop it before a replacement can prompt.
                turn.cancel_requested = True
                if turn.runner is not None:
                    turn.runner.cancel()
                return True
        # A standalone turn runs in its own session: that is the one to stop.
        session_id = self._turn_sessions.get(room_id) or self._sessions.get(room_id)
        connection = self._connection
        if session_id is None or connection is None:
            return False
        await connection.cancel(session_id)
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
        """
        async with self._room_turn_lock(room_id):
            try:
                return await self._discard_room_session(room_id, self._connection)
            finally:
                self._room_locks.pop(room_id, None)

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
        rebuilding = [
            turn.runner
            for turn in self._turns.values()
            if turn.rebuilding and turn.runner is not None
        ]
        for runner in rebuilding:
            runner.cancel()
        connection = self._connection
        if connection is not None:
            await asyncio.gather(
                *(
                    connection.cancel(session_id)
                    for session_id, turn in self._turns.items()
                    if not turn.rebuilding
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

        self._turns.clear()
        self._sessions.clear()
        self._turn_sessions.clear()
        self._session_rooms.clear()
        self._session_options.clear()
        self._prompted_index.clear()
        self._labelled_rooms.clear()
