"""FastRTC WebRTC-based realtime audio transport.

This module provides a RealtimeAudioTransport that uses FastRTC for WebRTC
audio transport in **passthrough mode** (no VAD). Raw audio flows
bidirectionally between the browser and the speech-to-speech AI provider,
which handles its own server-side VAD.

Unlike the FastRTCVoiceBackend (which uses ReplyOnPause for VAD), this
transport simply passes audio through — ideal for speech-to-speech providers
like OpenAI Realtime or Gemini Live.

Requires the ``fastrtc`` and ``numpy`` optional dependencies::

    pip install roomkit[fastrtc]

Usage::

    from roomkit.voice.realtime.fastrtc_transport import (
        FastRTCRealtimeTransport,
        mount_fastrtc_realtime,
    )

    transport = FastRTCRealtimeTransport()
    channel = RealtimeVoiceChannel("voice", provider=provider, transport=transport)

    # Mount WebRTC endpoints on FastAPI app
    mount_fastrtc_realtime(app, transport, path="/rtc-realtime")
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastrtc import AsyncStreamHandler

from roomkit.voice.realtime.base import RealtimeSession
from roomkit.voice.realtime.transport import (
    RealtimeAudioTransport,
    TransportAudioCallback,
    TransportDisconnectCallback,
)

if TYPE_CHECKING:
    import numpy as np
    from fastapi import FastAPI
    from fastrtc import Stream

logger = logging.getLogger("roomkit.voice.realtime.fastrtc_transport")

# Type alias for emit return: (sample_rate, ndarray) or None
EmitType = tuple[int, "np.ndarray"] | None


class _PassthroughHandler(AsyncStreamHandler):  # type: ignore[misc]
    """No-VAD passthrough handler for speech-to-speech.

    Inherits from FastRTC's AsyncStreamHandler to relay audio bidirectionally
    without performing voice activity detection. The provider's server-side
    VAD handles speech detection.

    Each WebRTC connection gets its own handler instance via ``copy()``.
    """

    def __init__(
        self,
        transport: FastRTCRealtimeTransport,
        *,
        input_sample_rate: int,
        output_sample_rate: int,
    ) -> None:
        import numpy as _np

        super().__init__(
            expected_layout="mono",
            output_sample_rate=output_sample_rate,
            input_sample_rate=input_sample_rate,
        )

        self._transport = transport
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._session: RealtimeSession | None = None
        self._webrtc_id: str | None = None

        self._np = _np

    def copy(self) -> _PassthroughHandler:
        """Create a per-connection handler instance (FastRTC requirement)."""
        return _PassthroughHandler(
            self._transport,
            input_sample_rate=self.input_sample_rate,
            output_sample_rate=self.output_sample_rate,
        )

    async def start_up(self) -> None:
        """Called by FastRTC when the WebRTC connection is established."""
        from fastrtc.utils import current_context

        ctx = current_context.get()
        if ctx:
            self._webrtc_id = ctx.webrtc_id
            self._transport._register_handler(self._webrtc_id, self)
            logger.info("WebRTC handler started: webrtc_id=%s", self._webrtc_id)

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Process incoming audio from the WebRTC client.

        Converts the numpy audio array to PCM16 LE bytes and fires
        the transport's audio callbacks.
        """
        if self._session is None:
            self._session = self._transport._get_session(self._webrtc_id)
            if self._session is None:
                return  # Not yet bound to a session

        _, audio_data = frame
        pcm_bytes = audio_data.astype(self._np.int16).tobytes()
        await self._transport._fire_audio_callbacks(self._session, pcm_bytes)

    async def emit(self) -> EmitType:
        """No-op: outgoing audio is sent directly via send_audio_direct()."""
        await asyncio.sleep(0.1)
        return None

    def shutdown(self) -> None:
        """Called by FastRTC when the WebRTC connection is closed."""
        if self._webrtc_id:
            self._transport._unregister_handler(self._webrtc_id)
            logger.info("WebRTC handler shutdown: webrtc_id=%s", self._webrtc_id)

    def send_message(self, message: str) -> None:
        """Send a message via the WebRTC DataChannel."""
        if self.channel:
            self.channel.send(message)

    def send_audio_direct(self, audio: bytes) -> None:
        """Send audio directly on the WebSocket, bypassing FastRTC's emit queue.

        Encodes PCM16 LE bytes as mu-law and sends immediately, avoiding
        the double-queue + 20ms sleep latency of the emit pipeline.
        """
        if not self.channel:
            return

        from roomkit.voice.backends.fastrtc import _pcm16_to_mulaw

        mulaw = _pcm16_to_mulaw(audio)
        payload = base64.b64encode(mulaw).decode("utf-8")
        self.channel.send(
            json.dumps(
                {
                    "event": "media",
                    "media": {"payload": payload},
                }
            )
        )


class FastRTCRealtimeTransport(RealtimeAudioTransport):
    """WebRTC-based realtime audio transport using FastRTC.

    Uses FastRTC in passthrough mode (no VAD) for speech-to-speech AI
    providers. Audio flows bidirectionally between the browser (via WebRTC)
    and the provider, which handles its own server-side VAD.

    Connection flow:
        1. ``mount_fastrtc_realtime(app, transport)`` creates a Stream
        2. Browser connects via WebRTC -> FastRTC calls handler.copy() -> start_up()
        3. start_up() reads webrtc_id, registers handler with transport
        4. Transport fires ``on_client_connected`` callback (if set)
        5. App calls ``channel.start_session(room_id, participant_id, connection=webrtc_id)``
        6. start_session() -> ``transport.accept(session, webrtc_id)`` maps session to handler
        7. receive()/emit() flow audio with session context
    """

    def __init__(
        self,
        *,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
    ) -> None:
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate

        # webrtc_id -> handler instance
        self._handlers: dict[str, _PassthroughHandler] = {}
        # session_id -> session
        self._sessions: dict[str, RealtimeSession] = {}
        # session_id -> webrtc_id
        self._session_handlers: dict[str, str] = {}
        # webrtc_id -> session
        self._webrtc_sessions: dict[str, RealtimeSession] = {}

        # Callbacks
        self._audio_callbacks: list[TransportAudioCallback] = []
        self._disconnect_callbacks: list[TransportDisconnectCallback] = []
        self._connected_callback: Callable[[str], Any] | None = None

        # FastRTC Stream (set by mount_fastrtc_realtime)
        self._stream: Stream | None = None

    @property
    def name(self) -> str:
        return "FastRTCRealtimeTransport"

    async def accept(self, session: RealtimeSession, connection: Any) -> None:
        """Accept a WebRTC connection for the given session.

        Args:
            session: The realtime session to bind.
            connection: The webrtc_id string identifying the WebRTC connection.
        """
        webrtc_id: str = connection
        self._sessions[session.id] = session
        self._session_handlers[session.id] = webrtc_id
        self._webrtc_sessions[webrtc_id] = session
        logger.info(
            "Session accepted: session=%s, webrtc_id=%s",
            session.id,
            webrtc_id,
        )

    async def send_audio(self, session: RealtimeSession, audio: bytes) -> None:
        """Send audio data to the connected WebRTC client.

        Sends directly on the WebSocket, bypassing FastRTC's emit queue
        for minimal latency.

        Args:
            session: The session to send audio to.
            audio: Raw PCM16 LE audio bytes.
        """
        webrtc_id = self._session_handlers.get(session.id)
        if webrtc_id is None:
            return
        handler = self._handlers.get(webrtc_id)
        if handler is not None:
            handler.send_audio_direct(audio)

    async def send_message(self, session: RealtimeSession, message: dict[str, Any]) -> None:
        """Send a JSON message via the WebRTC DataChannel.

        Args:
            session: The session to send the message to.
            message: JSON-serializable message dict.
        """
        webrtc_id = self._session_handlers.get(session.id)
        if webrtc_id is None:
            return
        handler = self._handlers.get(webrtc_id)
        if handler is not None and handler.channel:
            handler.send_message(json.dumps(message))

    async def disconnect(self, session: RealtimeSession) -> None:
        """Disconnect the client for the given session.

        Removes all mappings and sends a None sentinel to the handler's
        audio queue to signal the end of the stream.

        Args:
            session: The session to disconnect.
        """
        webrtc_id = self._session_handlers.pop(session.id, None)
        self._sessions.pop(session.id, None)
        if webrtc_id:
            self._webrtc_sessions.pop(webrtc_id, None)
            handler = self._handlers.get(webrtc_id)
            if handler:
                handler._audio_queue.put_nowait(None)  # Signal end
        logger.info("Session disconnected: session=%s", session.id)

    def on_audio_received(self, callback: TransportAudioCallback) -> None:
        """Register callback for audio received from the client.

        Args:
            callback: Called with (session, audio_bytes).
        """
        self._audio_callbacks.append(callback)

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        """Register callback for client disconnection.

        Args:
            callback: Called with (session) when the client disconnects.
        """
        self._disconnect_callbacks.append(callback)

    def on_client_connected(self, callback: Callable[[str], Any]) -> None:
        """Register callback fired when a new WebRTC client connects.

        Called with (webrtc_id: str). Use to auto-create sessions.

        Args:
            callback: Called with the webrtc_id when a client connects.
        """
        self._connected_callback = callback

    async def close(self) -> None:
        """Close all connections and release resources."""
        for session in list(self._sessions.values()):
            await self.disconnect(session)

    # ------------------------------------------------------------------
    # Internal methods called by _PassthroughHandler
    # ------------------------------------------------------------------

    def _register_handler(self, webrtc_id: str, handler: _PassthroughHandler) -> None:
        """Register a handler for a WebRTC connection."""
        self._handlers[webrtc_id] = handler
        if self._connected_callback is not None:
            try:
                result = self._connected_callback(webrtc_id)
                if hasattr(result, "__await__"):
                    asyncio.ensure_future(result)
            except Exception:
                logger.exception("Error in connected callback for webrtc_id=%s", webrtc_id)

    def _unregister_handler(self, webrtc_id: str) -> None:
        """Unregister a handler and fire disconnect callbacks."""
        self._handlers.pop(webrtc_id, None)
        session = self._webrtc_sessions.pop(webrtc_id, None)
        if session:
            self._session_handlers.pop(session.id, None)
            self._sessions.pop(session.id, None)
            asyncio.ensure_future(self._fire_disconnect_callbacks(session))

    def _get_session(self, webrtc_id: str | None) -> RealtimeSession | None:
        """Get the session bound to a WebRTC connection."""
        if webrtc_id is None:
            return None
        return self._webrtc_sessions.get(webrtc_id)

    async def _fire_audio_callbacks(self, session: RealtimeSession, audio: bytes) -> None:
        """Fire all registered audio callbacks."""
        for cb in self._audio_callbacks:
            try:
                result = cb(session, audio)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in audio callback for session %s", session.id)

    async def _fire_disconnect_callbacks(self, session: RealtimeSession) -> None:
        """Fire all registered disconnect callbacks."""
        for cb in self._disconnect_callbacks:
            try:
                result = cb(session)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in disconnect callback for session %s", session.id)


def mount_fastrtc_realtime(
    app: FastAPI,
    transport: FastRTCRealtimeTransport,
    *,
    path: str = "/rtc-realtime",
) -> None:
    """Mount FastRTC WebRTC endpoints for realtime voice transport.

    Creates a FastRTC Stream with a passthrough handler and mounts it
    on the given FastAPI app.

    Args:
        app: FastAPI application.
        transport: The FastRTCRealtimeTransport instance.
        path: Base path for WebRTC endpoints (default: /rtc-realtime).
    """
    from fastrtc import Stream

    handler = _PassthroughHandler(
        transport,
        input_sample_rate=transport._input_sample_rate,
        output_sample_rate=transport._output_sample_rate,
    )
    stream = Stream(handler=handler, modality="audio", mode="send-receive")
    transport._stream = stream
    stream.mount(app, path=path)
    logger.info("FastRTC realtime transport mounted at %s", path)
