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

from roomkit.core.task_utils import cancel_and_wait
from roomkit.providers.ai.tool_declaration import ToolNameRule, declared_parameters
from roomkit.providers.openai.realtime_events import (
    OpenAIRealtimeEventHandlersMixin,
    _OutputAudioState,
)
from roomkit.providers.openai.response_calls import PendingResponse
from roomkit.voice._g711 import _G711Codec, _get_codec
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.injection import VoiceInjectionResult, say_line_instruction

logger = logging.getLogger("roomkit.providers.openai.realtime_base")

_CONNECT_TIMEOUT = 30.0
_CLOSE_TIMEOUT = 2.0


class OpenAIRealtimeBase(OpenAIRealtimeEventHandlersMixin):
    """Connection lifecycle + outbound client API for OpenAI/xAI realtime.

    Subclasses must implement: :attr:`name`, :meth:`available_voices`,
    :attr:`_log_tag`, :attr:`_recv_task_prefix`, :attr:`_websockets_install_hint`,
    :meth:`_connect_url`, :meth:`_auth_headers`, :meth:`_build_session_config`
    and :meth:`_reconfigure_patch`, and set ``_model`` to the realtime model id
    they connect to. :attr:`_reconfigurable_provider_config` names the
    provider config keys a live reconfigure applies.
    """

    _model: str = ""

    def __init__(self) -> None:
        super().__init__()
        # Active WebSocket connections: session_id -> ws
        self._connections: dict[str, Any] = {}
        self._receive_tasks: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, VoiceSession] = {}
        # Track active responses per session to avoid inject_text conflicts
        self._responding: set[str] = set()
        # Sessions whose caller is speaking: a continuation waits for the
        # floor to come back (RFC §12.4)
        self._floor_held: set[str] = set()
        # Sessions whose caller's turn met a response in progress and is
        # still owed a request (RFC §12.4)
        self._turns_owed: set[str] = set()
        # The current response's function calls: the model is asked to go on
        # once that response is done and every call has its output (RFC §12.4)
        self._pending_responses: dict[str, PendingResponse] = {}
        # provider_config as passed to connect, kept so mid-session calls
        # (image injection, for one) can read settings fixed at connect time
        self._provider_configs: dict[str, dict[str, Any]] = {}
        # WebSocket clients own playback. Keep the current assistant audio
        # item and its generated duration so a physical barge-in can truncate
        # the unheard tail in provider context.
        self._output_audio: dict[str, _OutputAudioState] = {}
        self._output_bytes_per_ms: dict[str, float] = {}
        self._audio_codecs: dict[str, tuple[_G711Codec | None, _G711Codec | None]] = {}

    @property
    def model_name(self) -> str:
        """The realtime model this provider connects to, end to end."""
        return self._model or super().model_name

    def is_responding(self, session_id: str) -> bool:
        return session_id in self._responding

    @property
    def _tool_name_rule(self) -> ToolNameRule | None:
        """The tool names this endpoint accepts, checked when the session's
        tools are declared; ``None`` where the server decides (RFC §6.7)."""
        return None

    def _format_session_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Project tool dicts to the realtime ``session.tools`` shape, their
        names checked against the endpoint's rule (RFC §6.7).

        A function tool is reduced to the fields the API accepts
        (``type``/``name``/``description``/``parameters``), defaulting
        ``type`` to ``"function"``. Tool dicts may carry extra keys the
        caller uses elsewhere (e.g. ``tags`` for cross-lingual Tool
        Search); the API rejects those as unknown parameters, so they are
        dropped here. Native tools (xAI ``web_search``/``x_search``) carry
        a non-function ``type`` and pass through unchanged.
        """
        rule = self._tool_name_rule
        formatted: list[dict[str, Any]] = []
        for t in tools:
            if t.get("type", "function") != "function":
                formatted.append(dict(t))
                continue
            if rule is not None:
                rule.check([t.get("name", "")])
            tool = {"type": "function"}
            for field in ("name", "description"):
                if field in t:
                    tool[field] = t[field]
            tool["parameters"] = declared_parameters(t.get("parameters"))
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

        formats = session_config.get("audio", {})
        codecs: list[_G711Codec | None] = []
        for direction in ("input", "output"):
            audio_type = formats.get(direction, {}).get("format", {}).get("type")
            if audio_type in {"audio/pcmu", "audio/pcma"}:
                # Build lookup tables off the event loop, before opening a
                # socket or receiving the first audio frame.
                law = "mulaw" if audio_type == "audio/pcmu" else "alaw"
                codecs.append(await asyncio.to_thread(_get_codec, law))
            else:
                codecs.append(None)

        ws = await asyncio.wait_for(
            websockets.connect(self._connect_url(), additional_headers=self._auth_headers()),
            timeout=_CONNECT_TIMEOUT,
        )

        self._connections[session.id] = ws
        self._sessions[session.id] = session
        self._provider_configs[session.id] = pc
        self._audio_codecs[session.id] = (codecs[0], codecs[1])
        # PCM is signed 16-bit mono; G.711 carries one byte per sample.
        # Read the format from the provider-built payload so the shared base
        # stays correct for OpenAI's 8 kHz G.711 and xAI's 8 kHz PCM.
        output_format = session_config.get("audio", {}).get("output", {}).get("format", {})
        audio_type = output_format.get("type", "audio/pcm")
        negotiated_rate = int(
            output_format.get(
                "rate",
                8000 if audio_type in {"audio/pcmu", "audio/pcma"} else output_sample_rate,
            )
        )
        bytes_per_sample = 1 if audio_type in {"audio/pcmu", "audio/pcma"} else 2
        self._output_bytes_per_ms[session.id] = negotiated_rate * bytes_per_sample / 1000

        try:
            await ws.send(json.dumps({"type": "session.update", "session": session_config}))
        except BaseException:
            await self._discard_connection(session, ws)
            raise

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        self._receive_tasks[session.id] = self._session_task(
            self._receive_loop(session),
            name=f"{self._recv_task_prefix}:{session.id}",
        )

        logger.info("%s Realtime session connected: %s", self._log_tag, session.id)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        codec = self._audio_codecs.get(session.id, (None, None))[0]
        if codec is not None:
            if len(audio) % 2:
                raise ValueError("PCM16 audio must contain complete two-byte samples")
            audio = codec.encode(audio)
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
    ) -> VoiceInjectionResult:
        """Add a message item, then request a response unless ``silent``.

        ``system`` and ``user`` are message roles here. The API has no item
        that makes the model speak a given text, so an ``assistant`` line
        becomes a ``system`` instruction to say it (RFC §12.4): as a user
        message, the model would answer its own line.
        """
        ws = self._connections.get(session.id)
        if ws is None:
            return VoiceInjectionResult(
                status="not_sent", reason="voice_not_connected", retryable=True
            )
        if role == "assistant":
            role, text = "system", say_line_instruction(text)

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

        await self._maybe_request_response(session, ws, silent=silent)
        return VoiceInjectionResult(status="sent")

    async def _maybe_request_response(
        self, session: VoiceSession, ws: Any, *, silent: bool
    ) -> None:
        """Ask the model to answer what was just added to the conversation.

        A silent injection is context only, and a response already in flight
        must not be doubled — the API answers a second ``response.create``
        with an error.
        """
        if silent:
            logger.debug("[%s] Silent inject — no response.create", self._log_tag)
            return
        await self._request_response(session, ws, "text injected")

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        held, _ = self._answerable_tool_call(session, call_id)
        if not held:
            return
        ws = self._connections.get(session.id)
        if ws is None:
            return
        # Off the response's books before the send yields, as off the open
        # calls: a call issued under the id meanwhile is a new call, which
        # holds its response (RFC §12.4).
        current = self._pending_responses.get(session.id)
        if current is not None:
            current.call_ids.discard(call_id)

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

        # A call of a response the conversation has left joins the one in
        # progress, or continues at once when none is (RFC §12.4)
        pending = self._pending_responses.setdefault(session.id, PendingResponse(finished=True))
        pending.had_calls = True
        await self._continue_after_tool_results(session)

    async def interrupt(self, session: VoiceSession) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        logger.debug("[%s →] response.cancel", self._log_tag)
        await ws.send(json.dumps({"type": "response.cancel"}))

    async def truncate_audio(self, session: VoiceSession, audio_end_ms: int) -> None:
        """Remove the unheard tail of the latest assistant audio item.

        The Realtime WebSocket server cannot observe client-side playback.
        ``RealtimeVoiceChannel`` therefore reports the physical duration it
        played when speech interrupts output. Cap that duration at the audio
        received so callback jitter or an underrun can never produce an API
        error for truncating beyond the generated item.
        """
        ws = self._connections.get(session.id)
        state = self._output_audio.get(session.id)
        bytes_per_ms = self._output_bytes_per_ms.get(session.id, 0.0)
        if ws is None or state is None or bytes_per_ms <= 0:
            return

        generated_ms = int(state.received_bytes / bytes_per_ms)
        played_ms = min(max(0, int(audio_end_ms)), generated_ms)
        if state.truncated_at_ms is not None:
            return

        logger.info(
            "[%s →] conversation.item.truncate item=%s played=%dms generated=%dms (session %s)",
            self._log_tag,
            state.item_id,
            played_ms,
            generated_ms,
            session.id,
        )
        # Reserve before awaiting the socket write so two simultaneous speech
        # callbacks cannot emit duplicate truncations. A failed write releases
        # the reservation so a reconnecting caller can retry.
        state.truncated_at_ms = played_ms
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": state.item_id,
                        "content_index": state.content_index,
                        "audio_end_ms": played_ms,
                    }
                )
            )
        except BaseException:
            if state.truncated_at_ms == played_ms:
                state.truncated_at_ms = None
            raise

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(json.dumps(event))

    async def send_activity_start(self, session: VoiceSession) -> None:
        """The caller takes the floor (manual VAD mode).

        Nothing goes on the wire, audio flows continuously via
        ``input_audio_buffer.append``; a continuation waits for the floor.
        """
        self._floor_held.add(session.id)
        logger.debug("[%s] activity_start (session %s)", self._log_tag, session.id)

    async def send_activity_end(self, session: VoiceSession) -> None:
        """Commit audio buffer and request a response (manual VAD mode).

        The request also covers a continuation held while the caller spoke.
        """
        self._floor_held.discard(session.id)
        ws = self._connections.get(session.id)
        if ws is None:
            return
        logger.debug("[%s →] input_audio_buffer.commit (session %s)", self._log_tag, session.id)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await self._request_response(session, ws, "activity end", owed=True)

    def _forget_session(self, session_id: str) -> None:
        """Drop every per-session record but the socket and the receive task."""
        self._provider_configs.pop(session_id, None)
        self._responding.discard(session_id)
        self._floor_held.discard(session_id)
        self._turns_owed.discard(session_id)
        self._pending_responses.pop(session_id, None)
        self._output_audio.pop(session_id, None)
        self._output_bytes_per_ms.pop(session_id, None)
        self._audio_codecs.pop(session_id, None)

    async def _abandon_open_calls(self, session: VoiceSession) -> None:
        """Report the calls this connection leaves unanswered: no other
        connection will read their results (RFC §12.4)."""
        await self._abandon_open_tool_calls(session)

    async def _discard_connection(
        self,
        session: VoiceSession,
        ws: Any,
        *,
        error_message: str | None = None,
    ) -> None:
        """Remove one exact socket from every ownership map and close it."""
        if self._connections.get(session.id) is not ws:
            return
        self._connections.pop(session.id, None)
        self._sessions.pop(session.id, None)
        self._receive_tasks.pop(session.id, None)
        await self._abandon_open_calls(session)
        self._forget_session(session.id)
        was_active = session.state == VoiceSessionState.ACTIVE
        session.state = VoiceSessionState.ENDED

        if error_message is not None and was_active:
            logger.warning("%s %s", self._log_tag, error_message)
            await self._fire(
                self._error_callbacks,
                session,
                "connection_closed",
                error_message,
                label="error",
            )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ws.close(), timeout=_CLOSE_TIMEOUT)

    async def _retire_lost_connection(self, session: VoiceSession, ws: Any, message: str) -> None:
        """Handle exceptional and clean peer closes with identical teardown."""
        await self._discard_connection(session, ws, error_message=message)

    async def reconfigure(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        """Apply a partial, in-band ``session.update``; never reconnect.

        Tool Search, skill activation and a handoff call ``reconfigure``
        mid-conversation. The base implementation disconnects and reconnects,
        which on this protocol throws away the conversation *and* the
        in-flight tool call (its ``call_id`` is connection-scoped). The
        protocol accepts a partial ``session.update`` at any time, so only the
        changed fields are sent, in each provider's shape
        (:meth:`_reconfigure_patch`), and the live session keeps its history.
        Fields left at ``None`` stay unchanged; an empty ``tools`` list clears
        them.
        """
        ws = self._connections.get(session.id)
        if ws is None:
            logger.debug("reconfigure skipped: no live connection (session %s)", session.id)
            return
        pc = provider_config or {}
        patch = self._reconfigure_patch(
            system_prompt=system_prompt, voice=voice, tools=tools, temperature=temperature, pc=pc
        )
        if patch:
            logger.info(
                "[%s →] session.update (reconfigure): fields=%s tools=%s (session %s)",
                self._log_tag,
                sorted(key for key in patch if key != "type"),
                len(tools) if tools is not None else "unchanged",
                session.id,
            )
            await ws.send(json.dumps({"type": "session.update", "session": patch}))
        self._remember_provider_config(session.id, pc)

    def _remember_provider_config(self, session_id: str, pc: dict[str, Any]) -> None:
        """Keep the provider config keys a reconfigure applied; name the others.

        A key the in-band update cannot change (turn detection, transcription,
        audio format) takes effect only when a session opens: recording it
        would claim a setting the live session does not have.
        """
        applied = self._reconfigurable_provider_config
        ignored = sorted(key for key in pc if key not in applied)
        if ignored:
            logger.warning(
                "[%s] reconfigure cannot change %s mid-session; they take effect on the "
                "next session (session %s)",
                self._log_tag,
                ignored,
                session_id,
            )
        kept = {key: value for key, value in pc.items() if key in applied}
        if not kept:
            return
        merged = dict(self._provider_configs.get(session_id, {}))
        for key, value in kept.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        self._provider_configs[session_id] = merged

    #: The ``provider_config`` keys a live reconfigure applies (sent, or local
    #: policy). Every other key takes effect only when a session opens.
    _reconfigurable_provider_config: frozenset[str] = frozenset()

    @abstractmethod
    def _reconfigure_patch(
        self,
        *,
        system_prompt: str | None,
        voice: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        pc: dict[str, Any],
    ) -> dict[str, Any] | None:
        """The ``session`` payload of an in-band update; ``None`` when nothing
        changes on the wire."""

    async def disconnect(self, session: VoiceSession) -> None:
        # Cancel receive task
        await cancel_and_wait(self._receive_tasks.pop(session.id, None), log_errors_to=logger)

        # Close WebSocket (short timeout to avoid blocking on close handshake)
        ws = self._connections.pop(session.id, None)
        self._sessions.pop(session.id, None)
        await self._abandon_open_calls(session)
        self._forget_session(session.id)
        if ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.close(), timeout=_CLOSE_TIMEOUT)

        session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            session = self._sessions.get(session_id)
            if session:
                await self.disconnect(session)
