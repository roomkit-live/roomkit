"""Agent Client Protocol intelligence channel.

``ACPChannel`` makes RoomKit an ACP client: every Room is mapped to a distinct
session owned by an external coding agent.  How that agent is reached is the
transport's business — spawned here over stdio by default, or wherever a custom
:class:`~roomkit.channels.acp_transport.ACPTransport` can carry the protocol.
The reverse integration (exposing a RoomKit agent as an ACP server) is
intentionally out of scope.

The optional ``agent-client-protocol`` dependency is imported lazily so that
``import roomkit`` continues to work when the ``acp`` extra is not installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Mapping, Sequence
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
    _TurnDone,
    _TurnState,
    _usage_report,
    _usage_tokens,
)
from roomkit.channels._acp_context import (
    ACPContextContributor,
    compose_prompt,
    contributed_blocks,
    event_text,
    room_context_block,
)
from roomkit.channels._acp_events import ACPEventsMixin
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
from roomkit.models.event import RoomEvent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.streaming import StreamDelta
from roomkit.models.tool_call import AfterResponseCallback, AIResponseEvent, response_transcript
from roomkit.providers.ai.base import ProviderError
from roomkit.realtime.base import EphemeralEventType

if TYPE_CHECKING:
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.tools.external import ExternalToolHandler

logger = logging.getLogger("roomkit.channels.acp")

_CLEAN_STOP_REASON = "end_turn"
"""The one ACP stop reason that means the agent finished what it was asked."""

_SHUTDOWN_TIMEOUT = 5.0
"""Seconds the agent gets to acknowledge cancellation and session closes."""

_DEFAULT_ROOM_HISTORY = 20
"""Room messages an agent catches up on when it is asked to act (RFC §19.3.2)."""

_UNSEEN = -1
"""No prompt has left for this room yet — event indices start at 0."""


class ACPChannel(ACPConnectionMixin, ACPEventsMixin, Channel):
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
            visibility would have delivered to it (RFC §7.5 rule 8).
        context_contributor: What the host adds to a turn's prompt that the
            agent cannot go and fetch — member memories, a document corpus, an
            organisation's rules. Awaited once per solicited turn with the
            room's context and the triggering event; the blocks it returns
            open the prompt, ahead of the catch-up and the request. One that
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
        self._session_rooms: dict[str, str] = {}
        self._session_options: dict[str, list[Any]] = {}
        self._prompted_index: dict[str, int] = {}
        self._turns: dict[str, _TurnState] = {}
        self._agent_info: dict[str, Any] | None = None
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

    def _options_for(self, room_id: str) -> list[Any]:
        session_id = self._sessions.get(room_id)
        if session_id is None:
            return []
        return self._session_options.get(session_id, [])

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
        if session_id is None:
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
        if event.type in (EventType.TOOL_CALL_START, EventType.TOOL_CALL_END):
            return ChannelOutput.empty()

        text = event_text(event)
        if not text:
            return ChannelOutput.empty()

        room_id = context.room.id if context.room is not None else event.room_id
        # Host-only blocks can be collected now. Catch-up is deliberately
        # computed inside the lazy stream, after connection recovery: a dead
        # transport discards its sessions and their prompted-index marks.
        blocks = await contributed_blocks(
            self._context_contributor, context, event, channel_id=self.channel_id
        )
        # One live record for the turn, handed to the stream and to the output
        # alike: the stop reason is only known when the prompt returns, and
        # every MESSAGE segment reads this mapping as it stands when it is
        # persisted. A dict literal here would be a snapshot taken now, before
        # the turn has an outcome to report.
        metadata = ResponseMetadata({"acp": {"protocol_version": _STABLE_PROTOCOL_VERSION}})
        return ChannelOutput(
            responded=True,
            response_stream=self._prompt_stream(
                room_id,
                event.id,
                blocks,
                context,
                event,
                text,
                event.index,
                metadata,
            ),
            response_metadata=metadata,
        )

    def session_id(self, room_id: str) -> str | None:
        """Return the process-local ACP session identifier for a Room."""
        return self._sessions.get(room_id)

    async def cancel(self, room_id: str) -> bool:
        """Request cancellation of the active ACP turn for a Room."""
        session_id = self._sessions.get(room_id)
        connection = self._connection
        if session_id is None or connection is None:
            return False
        await connection.cancel(session_id)
        return True

    @contextlib.asynccontextmanager
    async def _room_turn_lock(self, room_id: str) -> AsyncIterator[None]:
        """Hold the room's turn lock, tolerating its retirement.

        ``close_session`` drops the entry while still holding the lock, so
        the map does not keep one lock per room the channel ever served. A
        coroutine that was queued on the retired lock therefore wakes owning
        an object nobody else can reach: it releases and retries on the
        current one. Without that re-check a fresh caller would take a
        brand-new lock and run the critical section alongside the waiter —
        which is how a room ends up with two sessions.
        """
        while True:
            lock = self._room_locks.setdefault(room_id, asyncio.Lock())
            await lock.acquire()
            if self._room_locks.get(room_id) is lock:
                break
            lock.release()
        try:
            yield
        finally:
            lock.release()

    async def close_session(self, room_id: str) -> bool:
        """Close and forget one Room's ACP session.

        *Forget* is the whole of it: every map keyed by the session, and the
        room's turn lock once no session is left behind it, is dropped here.
        A long-lived channel cycling sessions (one per conversation, one per
        reconnect) would otherwise carry every dead session's config options
        until the channel itself closed.
        """
        async with self._room_turn_lock(room_id):
            session_id = self._sessions.pop(room_id, None)
            if session_id is None:
                return False
            self._session_rooms.pop(session_id, None)
            self._session_options.pop(session_id, None)
            # The catch-up mark tracks what *this session* was told. A room
            # whose session is gone starts over: the next one opens empty and
            # has missed everything.
            self._prompted_index.pop(room_id, None)
            if self._connection is not None:
                await self._connection.close_session(session_id)
            self._room_locks.pop(room_id, None)
            return True

    async def close(self) -> None:
        """Cancel turns, close sessions, and close the transport.

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
        connection = self._connection
        if connection is not None:
            await asyncio.gather(
                *(connection.cancel(session_id) for session_id in self._turns),
                return_exceptions=True,
            )
        runners = [turn.runner for turn in self._turns.values() if turn.runner is not None]
        for runner in runners:
            runner.cancel()
        if runners:
            await asyncio.gather(*runners, return_exceptions=True)

        if connection is not None:
            await asyncio.gather(
                *(connection.close_session(session_id) for session_id in self._sessions.values()),
                return_exceptions=True,
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
        self._session_rooms.clear()
        self._session_options.clear()
        self._prompted_index.clear()

    async def _session_for(self, room_id: str, connection: Any) -> str:
        session_id = self._sessions.get(room_id)
        if session_id is not None:
            return session_id
        response = await connection.new_session(
            cwd=self._cwd,
            additional_directories=self._additional_directories or None,
            mcp_servers=self._mcp_servers,
            **{"roomkit.live/roomId": room_id},
        )
        session_id = response.session_id
        self._sessions[room_id] = session_id
        self._session_rooms[session_id] = room_id
        options = _model_dump(getattr(response, "config_options", None))
        self._session_options[session_id] = options if isinstance(options, list) else []
        await self._publish_config_options(
            session_id,
            self._session_options[session_id],
            _config_values(self._session_options[session_id]),
        )
        return session_id

    async def _prompt_stream(
        self,
        room_id: str,
        event_id: str,
        blocks: Sequence[str],
        context: RoomContext,
        trigger: RoomEvent,
        text: str,
        event_index: int,
        metadata: ResponseMetadata,
    ) -> AsyncIterator[StreamDelta]:
        async with self._room_turn_lock(room_id):
            connection = await self._ensure_connection()
            session_id = await self._session_for(room_id, connection)
            turn = _TurnState(room_id=room_id)
            self._turns[session_id] = turn
            catch_up = room_context_block(
                context,
                self.channel_id,
                after_index=self._prompted_index.get(room_id, _UNSEEN),
                trigger=trigger,
                limit=self._room_history,
            )
            prompt_text = compose_prompt(blocks, catch_up, text)
            prompt = [self._sdk().acp.text_block(prompt_text)]
            # The cursor commits only after the agent accepts the prompt. A
            # generator body that never runs, or a prompt rejected before
            # delivery, leaves the mark untouched so the next turn can replay
            # the missing room context instead of silently losing it.
            turn.runner = asyncio.create_task(
                self._run_prompt(
                    connection,
                    session_id,
                    event_id,
                    prompt,
                    turn,
                    room_id,
                    event_index,
                    metadata,
                )
            )

            try:
                while True:
                    item = await turn.queue.get()
                    if isinstance(item, _TurnDone):
                        # Whatever the turn left open closes here, in the
                        # stream, because the stored TOOL_CALL_END is
                        # persisted from the marker — and the finally below
                        # runs too late to yield one. The closing markers go
                        # through the same queue, so the terminal item is put
                        # back to be read after them; the second pass finds
                        # nothing open and falls through.
                        if await self._close_open_tools(turn, room_id, stream=True):
                            turn.queue.put_nowait(item)
                            continue
                        if item.error is not None:
                            if isinstance(item.error, asyncio.CancelledError):
                                return
                            if isinstance(item.error, ProviderError):
                                raise item.error
                            raise ProviderError(
                                f"ACP agent prompt failed: {item.error}",
                                provider="acp",
                            ) from item.error
                        turn.completed = True
                        return
                    yield item
            finally:
                if turn.runner is not None and not turn.runner.done():
                    with contextlib.suppress(Exception):
                        await connection.cancel(session_id)
                    turn.runner.cancel()
                    await asyncio.gather(turn.runner, return_exceptions=True)
                if turn.thinking_open:
                    await self._publish(
                        room_id,
                        EphemeralEventType.THINKING_END,
                        {"thinking": "", "round": 0},
                    )
                # A stream closed from the outside — the consumer was
                # cancelled, a muted binding dropped it — never reaches the
                # terminal item above. Its tools still have to stop spinning
                # for live surfaces; the stored row is beyond reach from here,
                # nothing can be yielded into a generator already closing.
                await self._close_open_tools(turn, room_id, stream=False)
                if self._turns.get(session_id) is turn:
                    self._turns.pop(session_id, None)
                if turn.completed:
                    await self._report_response(turn)

    async def _report_response(self, turn: _TurnState) -> None:
        """Announce a finished turn to whatever observes agent responses.

        Reached only from a turn that ran to its terminal item without an
        error, and once — an abandoned or failed turn produced no response to
        report. Observational, like the same report on an in-process AI
        channel: a hook that raises does not disturb the turn that is ending.
        """
        if self._after_response_hook is None:
            return
        segments, transcript = response_transcript("".join(chunks) for chunks in turn.segments)
        try:
            await self._after_response_hook(
                AIResponseEvent(
                    channel_id=self.channel_id,
                    response_content=transcript,
                    segments=segments,
                    room_id=turn.room_id,
                    tool_calls_count=len(turn.tools),
                    usage=_usage_report(turn.tokens, turn.context),
                    latency_ms=int((time.monotonic() - turn.started_at) * 1000),
                    streaming=True,
                )
            )
        except Exception:
            logger.debug("After-response hook failed (acp)", exc_info=True)

    async def _run_prompt(
        self,
        connection: Any,
        session_id: str,
        event_id: str,
        prompt: list[Any],
        turn: _TurnState,
        room_id: str,
        event_index: int,
        metadata: ResponseMetadata,
    ) -> None:
        """Run one prompt to its end, recording how that end came about.

        The turn's outcome rides ``metadata`` so that the MESSAGE segments
        persisted for this turn carry it: a caller with nobody watching (a
        scheduled run) has to tell an answer from a turn that stopped early,
        and the text alone cannot say which it is. Only an outcome that is
        *not* a clean end is written, the way an interrupted segment is
        marked ``cancelled`` and a finished one is marked by nothing.
        """
        acp_meta = metadata["acp"]
        try:
            response = await connection.prompt(
                session_id,
                prompt,
                **{"roomkit.live/eventId": event_id},
            )
            self._prompted_index[room_id] = max(
                event_index, self._prompted_index.get(room_id, _UNSEEN)
            )
            # The turn's own accounting, and the only place it is offered:
            # the usage notifications describe the context window, not what
            # answering cost.
            turn.tokens = _usage_tokens(getattr(response, "usage", None))
            # ``end_turn`` is the agent saying it finished; every other reason
            # (``refusal``, ``max_tokens``, ``cancelled``) means the work
            # stopped for a cause the caller must be able to act on. A reason
            # the agent did not name stays unwritten rather than being read as
            # either outcome.
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason and stop_reason != _CLEAN_STOP_REASON:
                acp_meta["stop_reason"] = stop_reason
            await self._drain_session_updates(session_id)
        except BaseException as exc:
            # The prompt never returned, so no stop reason exists to record:
            # the turn ended on the way, and that is the fact to carry.
            acp_meta["interrupted"] = True
            turn.queue.put_nowait(_TurnDone(error=exc))
        else:
            turn.queue.put_nowait(_TurnDone())
