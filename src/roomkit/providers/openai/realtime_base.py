"""Shared base for OpenAI-Realtime-wire-compatible providers.

The OpenAI Realtime WebSocket protocol is also spoken by xAI Grok. This base
owns the connection lifecycle and the outbound client API (connect, send,
disconnect) shared between them; the inbound server-event handling lives in
:class:`~roomkit.providers.openai.realtime_events.OpenAIRealtimeEventHandlersMixin`.
Subclasses supply only what genuinely differs: the session-config shape,
auth/URL, and a few provider-specific log lines.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from abc import abstractmethod
from typing import Any

from roomkit.providers.openai.realtime_events import OpenAIRealtimeEventHandlersMixin
from roomkit.voice.base import VoiceSession, VoiceSessionState

logger = logging.getLogger("roomkit.providers.openai.realtime_base")

_CONNECT_TIMEOUT = 30.0
_CLOSE_TIMEOUT = 2.0


class OpenAIRealtimeBase(OpenAIRealtimeEventHandlersMixin):
    """Connection lifecycle + outbound client API for OpenAI/xAI realtime.

    Subclasses must implement: :attr:`name`, :meth:`available_voices`,
    :attr:`_log_tag`, :attr:`_recv_task_prefix`, :attr:`_websockets_install_hint`,
    :meth:`_connect_url`, :meth:`_auth_headers`, and :meth:`_build_session_config`.
    """

    def __init__(self) -> None:
        super().__init__()
        # Active WebSocket connections: session_id -> ws
        self._connections: dict[str, Any] = {}
        self._receive_tasks: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, VoiceSession] = {}
        # Track active responses per session to avoid inject_text conflicts
        self._responding: set[str] = set()

    def is_responding(self, session_id: str) -> bool:
        return session_id in self._responding

    @staticmethod
    def _format_session_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Project tool dicts to the realtime ``session.tools`` shape.

        A function tool is reduced to the fields the API accepts
        (``type``/``name``/``description``/``parameters``), defaulting
        ``type`` to ``"function"``. Tool dicts may carry extra keys the
        caller uses elsewhere (e.g. ``tags`` for cross-lingual Tool
        Search); the API rejects those as unknown parameters, so they are
        dropped here. Native tools (xAI ``web_search``/``x_search``) carry
        a non-function ``type`` and pass through unchanged.
        """
        formatted: list[dict[str, Any]] = []
        for t in tools:
            if t.get("type", "function") != "function":
                formatted.append(dict(t))
                continue
            tool = {"type": "function"}
            for field in ("name", "description", "parameters"):
                if field in t:
                    tool[field] = t[field]
            formatted.append(tool)
        return formatted

    # -- Provider-specific extension points ---------------------------------

    @property
    @abstractmethod
    def _recv_task_prefix(self) -> str:
        """Prefix for the receive-loop task name (e.g. ``"openai_rt_recv"``)."""
        ...

    @property
    @abstractmethod
    def _websockets_install_hint(self) -> str:
        """Install command shown when the ``websockets`` dependency is missing."""
        ...

    @abstractmethod
    def _connect_url(self) -> str:
        """Full WebSocket URL to connect to."""
        ...

    @abstractmethod
    def _auth_headers(self) -> dict[str, str]:
        """Authorization headers for the WebSocket handshake."""
        ...

    @abstractmethod
    def _build_session_config(
        self,
        *,
        system_prompt: str | None,
        voice: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        input_sample_rate: int,
        output_sample_rate: int,
        server_vad: bool,
        pc: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the provider-specific ``session.update`` config payload.

        Implementations also perform any pre-connect validation (so it fails
        before a socket is opened) and emit the provider's "Sending
        session.update" info log.
        """
        ...

    # -- Connection lifecycle -----------------------------------------------

    def _import_websockets(self) -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                f"websockets is required for {self.name}. "
                f"Install with: {self._websockets_install_hint}"
            ) from exc
        return websockets

    async def connect(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        server_vad: bool = True,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        websockets = self._import_websockets()
        pc = provider_config or {}

        # Built before opening the socket so validation errors fail fast.
        session_config = self._build_session_config(
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            server_vad=server_vad,
            pc=pc,
        )

        ws = await asyncio.wait_for(
            websockets.connect(self._connect_url(), additional_headers=self._auth_headers()),
            timeout=_CONNECT_TIMEOUT,
        )

        self._connections[session.id] = ws
        self._sessions[session.id] = session

        await ws.send(json.dumps({"type": "session.update", "session": session_config}))

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        self._receive_tasks[session.id] = asyncio.create_task(
            self._receive_loop(session),
            name=f"{self._recv_task_prefix}:{session.id}",
        )

        logger.info("%s Realtime session connected: %s", self._log_tag, session.id)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio).decode("ascii"),
                }
            )
        )

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return

        logger.debug(
            "[%s →] conversation.item.create (input_text, role=%s, silent=%s)",
            self._log_tag,
            role,
            silent,
        )
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": role if role in ("user", "system") else "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )

        # Silent: add to context without requesting a response.
        if silent:
            logger.debug("[%s] Silent inject — no response.create", self._log_tag)
            return

        # Only request a new response if none is in progress.
        if session.id in self._responding:
            logger.debug(
                "[%s] Skipping response.create — response already active (session %s)",
                self._log_tag,
                session.id,
            )
            return

        logger.debug("[%s →] response.create", self._log_tag)
        await ws.send(json.dumps({"type": "response.create"}))

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return

        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    },
                }
            )
        )

        logger.debug("[%s →] response.create (after tool result)", self._log_tag)
        await ws.send(json.dumps({"type": "response.create"}))

    async def interrupt(self, session: VoiceSession) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        logger.debug("[%s →] response.cancel", self._log_tag)
        await ws.send(json.dumps({"type": "response.cancel"}))

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(json.dumps(event))

    async def send_activity_start(self, session: VoiceSession) -> None:
        """No-op — audio flows continuously via input_audio_buffer.append."""
        logger.debug("[%s] activity_start (no-op, session %s)", self._log_tag, session.id)

    async def send_activity_end(self, session: VoiceSession) -> None:
        """Commit audio buffer and request a response (manual VAD mode)."""
        ws = self._connections.get(session.id)
        if ws is None:
            return
        logger.debug("[%s →] input_audio_buffer.commit (session %s)", self._log_tag, session.id)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        if session.id in self._responding:
            logger.debug(
                "[%s] skip response.create — responding (session %s)", self._log_tag, session.id
            )
            return
        logger.debug("[%s →] response.create (session %s)", self._log_tag, session.id)
        await ws.send(json.dumps({"type": "response.create"}))

    async def disconnect(self, session: VoiceSession) -> None:
        # Cancel receive task
        task = self._receive_tasks.pop(session.id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        # Close WebSocket (short timeout to avoid blocking on close handshake)
        ws = self._connections.pop(session.id, None)
        self._sessions.pop(session.id, None)
        self._responding.discard(session.id)
        if ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.close(), timeout=_CLOSE_TIMEOUT)

        session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            session = self._sessions.get(session_id)
            if session:
                await self.disconnect(session)
