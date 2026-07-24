"""Agent Client Protocol intelligence channel.

``ACPChannel`` makes RoomKit an ACP client: every Room is mapped to a distinct
session owned by an external coding-agent process.  The reverse integration
(exposing a RoomKit agent as an ACP server) is intentionally out of scope.

The optional ``agent-client-protocol`` dependency is imported lazily so that
``import roomkit`` continues to work when the ``acp`` extra is not installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roomkit.channels._acp_client import (
    _SDK,
    _STABLE_PROTOCOL_VERSION,
    ACPConnectionMixin,
    _absolute_path,
    _ACPClient,
    _TurnDone,
    _TurnState,
)
from roomkit.channels._acp_events import ACPEventsMixin
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
from roomkit.models.event import RichContent, RoomEvent, TextContent
from roomkit.models.streaming import StreamDelta
from roomkit.providers.ai.base import ProviderError
from roomkit.realtime.base import EphemeralEventType

if TYPE_CHECKING:
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.tools.external import ExternalToolHandler

logger = logging.getLogger("roomkit.channels.acp")


class ACPChannel(ACPConnectionMixin, ACPEventsMixin, Channel):
    """Connect a RoomKit Room to an external ACP coding agent over stdio.

    One agent process is created lazily for the channel and one ACP session is
    created per Room. Prompts are serialized inside each session, while
    different Rooms can use the same agent process concurrently.

    Args:
        channel_id: RoomKit channel identifier.
        command: Executable and arguments used to start the ACP agent. No shell
            is involved.
        cwd: Absolute working directory used for the process and ACP sessions.
        additional_directories: Additional absolute directories exposed in the
            ACP session declaration.
        env: Environment variables added to the SDK's restricted inherited
            environment.
        mcp_servers: ACP MCP-server descriptors accepted by the official SDK.
        authentication_method: Optional ACP authentication method identifier.
        external_tool_handler: Permission policy and tool observability bridge.
            Without a handler, every permission request is rejected.
    """

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE
    direction = ChannelDirection.BIDIRECTIONAL

    def __init__(
        self,
        channel_id: str,
        command: Sequence[str],
        *,
        cwd: str | Path,
        additional_directories: Sequence[str | Path] | None = None,
        env: Mapping[str, str] | None = None,
        mcp_servers: Sequence[Any] | None = None,
        authentication_method: str | None = None,
        external_tool_handler: ExternalToolHandler | None = None,
    ) -> None:
        super().__init__(channel_id)
        if isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty sequence of arguments")
        if any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError("every command argument must be a non-empty string")
        if env is not None and any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()
        ):
            raise ValueError("env keys and values must be strings")

        self._command = tuple(command)
        self._cwd = _absolute_path(cwd, field_name="cwd")
        self._additional_directories = [
            _absolute_path(path, field_name="additional_directories")
            for path in (additional_directories or ())
        ]
        self._env = dict(env) if env is not None else None
        self._mcp_servers = list(mcp_servers or ())
        self._authentication_method = authentication_method
        self._external_tool_handler = external_tool_handler

        self._loaded_sdk: _SDK | None = None
        self._client = _ACPClient(self)
        self._connection: Any = None
        self._process: Any = None
        self._process_context: Any = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._connect_lock = asyncio.Lock()
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._sessions: dict[str, str] = {}
        self._session_rooms: dict[str, str] = {}
        self._turns: dict[str, _TurnState] = {}
        self._agent_info: dict[str, Any] | None = None
        self._handler_started = False
        self._closed = False
        self._realtime: RealtimeBackend | None = None

    @property
    def info(self) -> dict[str, Any]:
        """Return ACP connection and agent metadata without exposing arguments."""
        return {
            "transport": "stdio",
            "protocol_version": _STABLE_PROTOCOL_VERSION,
            "sdk_version": self._loaded_sdk.version if self._loaded_sdk else None,
            "connected": self._connection is not None,
            "agent": self._agent_info,
            "session_count": len(self._sessions),
        }

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.TEXT, ChannelMediaType.RICH],
            supports_rich_text=True,
        )

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

        text = self._event_text(event)
        if not text:
            return ChannelOutput.empty()

        room_id = context.room.id if context.room is not None else event.room_id
        return ChannelOutput(
            responded=True,
            response_stream=self._prompt_stream(room_id, event.id, text),
            response_metadata={"acp": {"protocol_version": _STABLE_PROTOCOL_VERSION}},
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

    async def close_session(self, room_id: str) -> bool:
        """Close and forget one Room's ACP session."""
        lock = self._room_locks.setdefault(room_id, asyncio.Lock())
        async with lock:
            session_id = self._sessions.pop(room_id, None)
            if session_id is None:
                return False
            self._session_rooms.pop(session_id, None)
            if self._connection is not None:
                await self._connection.close_session(session_id)
            return True

    async def close(self) -> None:
        """Cancel turns, close sessions, and stop the ACP subprocess."""
        if self._closed:
            return
        self._closed = True

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

        async with self._connect_lock:
            await self._close_process()

        if self._external_tool_handler is not None and self._handler_started:
            await self._external_tool_handler.stop()
            self._handler_started = False

        self._turns.clear()
        self._sessions.clear()
        self._session_rooms.clear()

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
        return session_id

    async def _prompt_stream(
        self,
        room_id: str,
        event_id: str,
        text: str,
    ) -> AsyncIterator[StreamDelta]:
        lock = self._room_locks.setdefault(room_id, asyncio.Lock())
        async with lock:
            connection = await self._ensure_connection()
            session_id = await self._session_for(room_id, connection)
            turn = _TurnState(room_id=room_id)
            self._turns[session_id] = turn
            prompt = [self._sdk().acp.text_block(text)]
            turn.runner = asyncio.create_task(
                self._run_prompt(connection, session_id, event_id, prompt, turn)
            )

            try:
                while True:
                    item = await turn.queue.get()
                    if isinstance(item, _TurnDone):
                        if item.error is not None:
                            if isinstance(item.error, asyncio.CancelledError):
                                return
                            if isinstance(item.error, ProviderError):
                                raise item.error
                            raise ProviderError(
                                f"ACP agent prompt failed: {item.error}",
                                provider="acp",
                            ) from item.error
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
                if self._turns.get(session_id) is turn:
                    self._turns.pop(session_id, None)

    async def _run_prompt(
        self,
        connection: Any,
        session_id: str,
        event_id: str,
        prompt: list[Any],
        turn: _TurnState,
    ) -> None:
        try:
            await connection.prompt(
                session_id,
                prompt,
                **{"roomkit.live/eventId": event_id},
            )
        except BaseException as exc:
            turn.queue.put_nowait(_TurnDone(error=exc))
        else:
            turn.queue.put_nowait(_TurnDone())

    @staticmethod
    def _event_text(event: RoomEvent) -> str:
        content = event.content
        if isinstance(content, TextContent):
            return content.body
        if isinstance(content, RichContent):
            return content.plain_text or content.body
        return Channel.extract_text(event)
