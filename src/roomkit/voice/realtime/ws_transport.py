"""WebSocket-based realtime audio transport."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from roomkit.voice.backends.base import (
    TransportDisconnectCallback,
    VoiceBackend,
)
from roomkit.voice.base import AudioChunk, VoiceSession

logger = logging.getLogger("roomkit.voice.realtime.ws_transport")

# Transport audio callback: (session, audio_bytes) -> Any
TransportAudioCallback = Callable[["VoiceSession", bytes], Any]


class WebSocketRealtimeTransport(VoiceBackend):
    """Concrete WebSocket-based audio transport.

    Protocol:
    - Client sends: ``{"type": "audio", "data": "<base64 PCM>"}``
    - Server sends: ``{"type": "audio", "data": "<base64 PCM>"}``
    - Server sends: ``{"type": "transcription", "text": "...", "role": "...", "is_final": true}``
    - Server sends: ``{"type": "speaking", "speaking": true, "who": "user"|"assistant"}``

    Each accepted connection starts a background receive loop that
    decodes incoming audio and fires callbacks.

    Requires the ``websockets`` package.
    """

    def __init__(self) -> None:
        self._websockets: dict[str, Any] = {}  # session_id -> WebSocket
        self._receive_tasks: dict[str, asyncio.Task[None]] = {}
        self._audio_callbacks: list[TransportAudioCallback] = []
        self._disconnect_callbacks: list[TransportDisconnectCallback] = []
        self._sessions: dict[str, VoiceSession] = {}  # session_id -> session

    @property
    def name(self) -> str:
        return "WebSocketRealtimeTransport"

    async def accept(self, session: VoiceSession, connection: Any) -> None:
        """Accept a WebSocket connection for the given session.

        Starts a background receive loop to process incoming messages.

        Args:
            session: The realtime session.
            connection: A ``websockets`` WebSocket connection.
        """
        self._websockets[session.id] = connection
        self._sessions[session.id] = session
        self._receive_tasks[session.id] = asyncio.create_task(
            self._receive_loop(session),
            name=f"ws_rt_recv:{session.id}",
        )

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        if isinstance(audio, bytes):
            raw = audio
        else:
            return  # Streaming not supported in WS transport
        ws = self._websockets.get(session.id)
        if ws is None:
            return
        message = json.dumps(
            {
                "type": "audio",
                "data": base64.b64encode(raw).decode("ascii"),
            }
        )
        try:
            await ws.send(message)
        except Exception:
            logger.exception("Error sending audio to session %s", session.id)

    async def send_message(self, session: VoiceSession, message: dict[str, Any]) -> None:
        """Send a JSON message to the connected client.

        Not part of the VoiceBackend ABC — accessed via getattr by the channel.
        """
        ws = self._websockets.get(session.id)
        if ws is None:
            return
        try:
            await ws.send(json.dumps(message))
        except Exception:
            logger.exception("Error sending message to session %s", session.id)

    async def disconnect(self, session: VoiceSession) -> None:
        import contextlib

        # Cancel receive task
        task = self._receive_tasks.pop(session.id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        # Close WebSocket
        ws = self._websockets.pop(session.id, None)
        self._sessions.pop(session.id, None)
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    def on_audio_received(self, callback: TransportAudioCallback) -> None:  # type: ignore[override]
        self._audio_callbacks.append(callback)

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)

    async def close(self) -> None:
        """Close all connections."""
        for session_id in list(self._sessions.keys()):
            session = self._sessions.get(session_id)
            if session:
                await self.disconnect(session)

    async def _receive_loop(self, session: VoiceSession) -> None:
        """Background loop reading messages from the client WebSocket."""
        ws = self._websockets.get(session.id)
        if ws is None:
            return

        try:
            async for raw_message in ws:
                try:
                    if isinstance(raw_message, bytes):
                        # Raw binary audio
                        await self._fire_audio_callbacks(session, raw_message)
                        continue

                    data = json.loads(raw_message)
                    msg_type = data.get("type")

                    if msg_type == "audio":
                        audio_b64 = data.get("data", "")
                        audio_bytes = base64.b64decode(audio_b64)
                        await self._fire_audio_callbacks(session, audio_bytes)

                except (json.JSONDecodeError, KeyError, ValueError):
                    logger.warning("Invalid message from session %s", session.id)
                except Exception:
                    logger.exception("Error processing message from session %s", session.id)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("WebSocket closed for session %s", session.id)
        finally:
            # Fire disconnect callbacks
            for cb in self._disconnect_callbacks:
                try:
                    result = cb(session)
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    logger.exception("Error in disconnect callback for session %s", session.id)

    async def _fire_audio_callbacks(self, session: VoiceSession, audio: bytes) -> None:
        """Fire all registered audio callbacks."""
        for cb in self._audio_callbacks:
            try:
                result = cb(session, audio)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in audio callback for session %s", session.id)
