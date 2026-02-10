"""RealtimeAudioTransport abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from roomkit.voice.realtime.base import RealtimeSession

# Callback type aliases
TransportAudioCallback = Callable[[RealtimeSession, bytes], Any]
TransportDisconnectCallback = Callable[[RealtimeSession], Any]


class RealtimeAudioTransport(ABC):
    """Abstract base class for browser-to-server audio transport.

    Handles the WebSocket (or other protocol) connection between the
    user's browser and the server, carrying raw audio in both directions.

    Unlike VoiceBackend, this transport does NOT perform VAD —
    that's handled by the realtime provider's server-side VAD.

    Example:
        transport = WebSocketRealtimeTransport()

        transport.on_audio_received(handle_client_audio)
        transport.on_client_disconnected(handle_disconnect)

        await transport.accept(session, websocket_connection)
        await transport.send_audio(session, provider_audio_bytes)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Transport name (e.g. 'websocket', 'webrtc')."""
        ...

    @abstractmethod
    async def accept(self, session: RealtimeSession, connection: Any) -> None:
        """Accept a client connection for the given session.

        Args:
            session: The realtime session to bind to this connection.
            connection: The protocol-specific connection object
                (e.g. a WebSocket instance).
        """
        ...

    @abstractmethod
    async def send_audio(self, session: RealtimeSession, audio: bytes) -> None:
        """Send audio data to the connected client.

        Args:
            session: The session to send audio to.
            audio: Raw PCM audio bytes.
        """
        ...

    @abstractmethod
    async def send_message(self, session: RealtimeSession, message: dict[str, Any]) -> None:
        """Send a JSON message to the connected client.

        Used for transcriptions, speaking indicators, and other metadata.

        Args:
            session: The session to send the message to.
            message: JSON-serializable message dict.
        """
        ...

    @abstractmethod
    async def disconnect(self, session: RealtimeSession) -> None:
        """Disconnect the client for the given session.

        Args:
            session: The session to disconnect.
        """
        ...

    def on_audio_received(self, callback: TransportAudioCallback) -> None:
        """Register callback for audio received from the client.

        Args:
            callback: Called with (session, audio_bytes).
        """

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        """Register callback for client disconnection.

        Args:
            callback: Called with (session) when the client disconnects.
        """

    def end_of_response(self, session: RealtimeSession) -> None:  # noqa: B027
        """Signal end of AI response (for transports with pacing)."""

    def interrupt(self, session: RealtimeSession) -> None:  # noqa: B027
        """Signal interruption — flush queue, stop playback."""

    async def close(self) -> None:
        """Release all transport resources."""
