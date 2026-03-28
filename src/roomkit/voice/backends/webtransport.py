"""WebTransport voice backend using QUIC datagrams for low-latency audio.

Provides a voice backend that uses WebTransport (HTTP/3 over QUIC) for
real-time audio streaming.  Audio is sent/received as QUIC datagrams
(unreliable, like UDP) for minimum latency — no head-of-line blocking.

Requires the ``aioquic`` package::

    pip install aioquic

The backend runs a QUIC server on a separate UDP port alongside your
existing HTTP server.  Browsers connect via the WebTransport API::

    const transport = new WebTransport("https://localhost:4433/audio");

Usage::

    from roomkit.voice.backends.webtransport import (
        WebTransportBackend,
    )

    backend = WebTransportBackend(
        host="0.0.0.0",
        port=4433,
        certificate="cert.pem",
        private_key="key.pem",
    )

    # Register audio callback (called for each inbound audio datagram)
    backend.on_audio_received(voice_channel._on_audio_received)

    # Start the QUIC server
    await backend.start()

    # ...

    await backend.close()

Wire protocol (datagrams)::

    Client → Server:  [2 bytes sample_rate_le] [PCM-16 LE audio data]
    Server → Client:  [2 bytes sample_rate_le] [PCM-16 LE audio data]

The 2-byte header encodes the sample rate divided by 100 as a
little-endian uint16 (e.g. 16000 Hz → 160, 48000 Hz → 480).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from roomkit.core.task_utils import log_task_exception
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.base import (
    AudioReceivedCallback,
    SessionReadyCallback,
    TransportDisconnectCallback,
    VoiceBackend,
)
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession, VoiceSessionState

logger = logging.getLogger("roomkit.voice.webtransport")

# Datagram header: 2 bytes sample_rate/100 as little-endian uint16
_HEADER_STRUCT = struct.Struct("<H")
_HEADER_SIZE = _HEADER_STRUCT.size

# Session factory type: connection_id -> VoiceSession
SessionFactory = Callable[[str], Any]


class WebTransportBackend(VoiceBackend):
    """Voice backend using WebTransport (QUIC) datagrams for audio.

    Runs a QUIC/HTTP3 server that accepts WebTransport connections.
    Audio is exchanged as unreliable datagrams — low latency, no
    head-of-line blocking.

    Args:
        host: Bind address for the QUIC server.
        port: UDP port for the QUIC server.
        certificate: Path to TLS certificate file (PEM).
        private_key: Path to TLS private key file (PEM).
        input_sample_rate: Expected inbound audio sample rate.
        output_sample_rate: Outbound audio sample rate.
        path: URL path for WebTransport connections (default ``"/audio"``).
        max_datagram_size: Maximum datagram payload size in bytes.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",  # nosec B104
        port: int = 4433,
        certificate: str = "cert.pem",
        private_key: str = "key.pem",
        input_sample_rate: int = 16000,
        output_sample_rate: int = 16000,
        path: str = "/audio",
        max_datagram_size: int = 65536,
    ) -> None:
        self._host = host
        self._port = port
        self._certificate = certificate
        self._private_key = private_key
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._path = path
        self._max_datagram_size = max_datagram_size

        # Callbacks
        self._audio_received_callback: AudioReceivedCallback | None = None
        self._session_ready_callbacks: list[SessionReadyCallback] = []
        self._disconnect_callbacks: list[TransportDisconnectCallback] = []
        self._session_factory: SessionFactory | None = None

        # Session tracking
        self._sessions: dict[str, VoiceSession] = {}
        # WebTransport stream_id -> session_id
        self._stream_sessions: dict[int, str] = {}
        # session_id -> (protocol, stream_id) for sending datagrams back
        self._session_transports: dict[str, tuple[Any, int]] = {}

        self._server: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def name(self) -> str:
        return "webtransport"

    @property
    def capabilities(self) -> VoiceCapability:
        return VoiceCapability.INTERRUPTION

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        self._audio_received_callback = callback

    def on_session_ready(self, callback: SessionReadyCallback) -> None:
        self._session_ready_callbacks.append(callback)

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)

    def set_session_factory(self, factory: SessionFactory) -> None:
        """Set a factory called when a new WebTransport client connects.

        The factory receives a connection ID and should return a
        :class:`VoiceSession` (may be async).
        """
        self._session_factory = factory

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        session_id = str(uuid.uuid4())
        session_metadata = {
            **(metadata or {}),
            "transport": "webtransport",
            "input_sample_rate": self._input_sample_rate,
            "output_sample_rate": self._output_sample_rate,
        }
        session = VoiceSession(
            id=session_id,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VoiceSessionState.ACTIVE,
            metadata=session_metadata,
        )
        self._sessions[session_id] = session
        if self._loop is None:
            with contextlib.suppress(RuntimeError):
                self._loop = asyncio.get_running_loop()
        return session

    async def disconnect(self, session: VoiceSession) -> None:
        self._sessions.pop(session.id, None)
        transport_info = self._session_transports.pop(session.id, None)
        # Remove stream mapping
        if transport_info:
            _, stream_id = transport_info
            self._stream_sessions.pop(stream_id, None)
        session.state = VoiceSessionState.ENDED

    def get_session(self, session_id: str) -> VoiceSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    # ------------------------------------------------------------------
    # Audio send
    # ------------------------------------------------------------------

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        if isinstance(audio, bytes):
            self._send_datagram(session, audio, self._output_sample_rate)
        else:
            async for chunk in audio:
                self._send_datagram(session, chunk.data, chunk.sample_rate)

    def send_audio_sync(self, session: VoiceSession, chunk: AudioChunk) -> None:
        """Synchronously send audio — thread-safe for audio callback threads."""
        transport_info = self._session_transports.get(session.id)
        if transport_info is None:
            return
        protocol, stream_id = transport_info
        sample_rate = chunk.sample_rate or self._output_sample_rate
        header = _HEADER_STRUCT.pack(sample_rate // 100)
        payload = header + chunk.data

        # Schedule on event loop — aioquic protocol is not thread-safe
        loop = self._loop
        if loop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(protocol.send_datagram, stream_id, payload)

    def _send_datagram(self, session: VoiceSession, data: bytes, sample_rate: int) -> None:
        """Send audio data as a QUIC datagram to the client."""
        transport_info = self._session_transports.get(session.id)
        if transport_info is None:
            return
        protocol, stream_id = transport_info
        header = _HEADER_STRUCT.pack(sample_rate // 100)
        try:
            protocol.send_datagram(stream_id, header + data)
        except Exception:
            logger.debug("Failed to send datagram to %s", session.id[:8])

    def interrupt(self, session: VoiceSession) -> None:
        pass  # Datagrams are fire-and-forget, nothing to flush

    # ------------------------------------------------------------------
    # Inbound audio handling
    # ------------------------------------------------------------------

    def _handle_datagram(self, protocol: Any, stream_id: int, data: bytes) -> None:
        """Handle an inbound audio datagram from a client."""
        if len(data) <= _HEADER_SIZE:
            return

        sample_rate = _HEADER_STRUCT.unpack_from(data, 0)[0] * 100
        pcm_data = data[_HEADER_SIZE:]

        session_id = self._stream_sessions.get(stream_id)
        if session_id is None:
            return
        session = self._sessions.get(session_id)
        if session is None:
            return

        frame = AudioFrame(
            data=pcm_data,
            sample_rate=sample_rate or self._input_sample_rate,
            channels=1,
            sample_width=2,
        )

        if self._audio_received_callback:
            self._audio_received_callback(session, frame)

    # ------------------------------------------------------------------
    # WebTransport session lifecycle
    # ------------------------------------------------------------------

    async def _on_client_connect(self, protocol: Any, stream_id: int, path: str) -> bool:
        """Called when a WebTransport client connects.

        Returns True to accept, False to reject.
        """
        if path != self._path:
            logger.debug("Rejected WebTransport connection to %s", path)
            return False

        connection_id = f"wt-{stream_id}"

        if self._session_factory:
            result = self._session_factory(connection_id)
            if asyncio.iscoroutine(result):
                session = await result
            else:
                session = result
        else:
            session = await self.connect(
                room_id="default",
                participant_id=connection_id,
                channel_id="voice",
            )

        self._stream_sessions[stream_id] = session.id
        self._session_transports[session.id] = (protocol, stream_id)

        logger.info(
            "WebTransport client connected: session=%s stream=%d",
            session.id[:8],
            stream_id,
        )

        for cb in self._session_ready_callbacks:
            with contextlib.suppress(Exception):
                result = cb(session)
                if asyncio.iscoroutine(result):
                    await result

        return True

    async def _on_client_disconnect(self, stream_id: int) -> None:
        """Called when a WebTransport client disconnects."""
        session_id = self._stream_sessions.pop(stream_id, None)
        if session_id is None:
            return
        session = self._sessions.pop(session_id, None)
        if session is None:
            return

        logger.info("WebTransport client disconnected: session=%s", session_id[:8])
        self._session_transports.pop(session_id, None)
        session.state = VoiceSessionState.ENDED

        for cb in self._disconnect_callbacks:
            with contextlib.suppress(Exception):
                result = cb(session)
                if asyncio.iscoroutine(result):
                    await result

    # ------------------------------------------------------------------
    # QUIC server
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the QUIC/WebTransport server."""
        try:
            from aioquic.asyncio import serve
            from aioquic.quic.configuration import QuicConfiguration
        except ImportError as exc:
            raise ImportError(
                "WebTransportBackend requires aioquic. Install with: pip install aioquic"
            ) from exc

        self._loop = asyncio.get_running_loop()

        config = QuicConfiguration(
            alpn_protocols=["h3"],
            is_client=False,
            max_datagram_frame_size=self._max_datagram_size,
        )
        config.load_cert_chain(self._certificate, self._private_key)

        self._server = await serve(
            host=self._host,
            port=self._port,
            configuration=config,
            create_protocol=self._create_protocol,
        )

        logger.info(
            "WebTransport server started: %s:%d%s",
            self._host,
            self._port,
            self._path,
        )

    def _create_protocol(self, *args: Any, **kwargs: Any) -> Any:
        """Factory for QUIC connection protocols."""
        from aioquic.asyncio import QuicConnectionProtocol
        from aioquic.h3.connection import H3Connection

        backend = self

        class AudioTransportProtocol(QuicConnectionProtocol):
            def __init__(self, *a: Any, **kw: Any) -> None:
                super().__init__(*a, **kw)
                self._h3 = H3Connection(self._quic, enable_webtransport=True)
                self._wt_sessions: set[int] = set()

            def quic_event_received(self, event: Any) -> None:
                from aioquic.h3.events import (
                    DatagramReceived,
                    HeadersReceived,
                    WebTransportStreamDataReceived,
                )
                from aioquic.quic.events import StreamDataReceived

                for h3_event in self._h3.handle_event(event):
                    if isinstance(h3_event, HeadersReceived):
                        self._handle_headers(h3_event)
                    elif isinstance(h3_event, DatagramReceived):
                        backend._handle_datagram(self, h3_event.stream_id, h3_event.data)
                    elif isinstance(h3_event, WebTransportStreamDataReceived):
                        # Could handle reliable stream data here
                        pass

                # Detect closed CONNECT streams (session disconnect)
                if isinstance(event, StreamDataReceived) and event.end_stream:
                    stream_id = event.stream_id
                    if stream_id in self._wt_sessions:
                        self._wt_sessions.discard(stream_id)
                        task = asyncio.ensure_future(backend._on_client_disconnect(stream_id))
                        task.add_done_callback(log_task_exception)

            def _handle_headers(self, event: Any) -> None:
                headers = dict(event.headers)
                method = headers.get(b":method", b"").decode()
                path = headers.get(b":path", b"").decode()
                protocol = headers.get(b":protocol", b"").decode()

                if method == "CONNECT" and protocol == "webtransport":
                    # WebTransport session request
                    stream_id = event.stream_id
                    fut = asyncio.ensure_future(backend._on_client_connect(self, stream_id, path))

                    def _on_connect_done(f: Any, sid: int = stream_id) -> None:
                        exc = f.exception()
                        if exc:
                            logger.error("WebTransport connect error: %s", exc)
                            self._accept_or_reject(sid, False)
                        else:
                            self._accept_or_reject(sid, f.result())

                    fut.add_done_callback(_on_connect_done)

            def _accept_or_reject(self, stream_id: int, accept: bool) -> None:
                if accept:
                    self._wt_sessions.add(stream_id)
                    self._h3.send_headers(
                        stream_id=stream_id,
                        headers=[
                            (b":status", b"200"),
                            (
                                b"sec-webtransport-http3-draft",
                                b"draft02",
                            ),
                        ],
                    )
                else:
                    self._h3.send_headers(
                        stream_id=stream_id,
                        headers=[(b":status", b"403")],
                        end_stream=True,
                    )
                self.transmit()

            def send_datagram(self, stream_id: int, data: bytes) -> None:
                """Send a datagram on a WebTransport session."""
                self._h3.send_datagram(stream_id, data)
                self.transmit()

        return AudioTransportProtocol(*args, **kwargs)

    async def close(self) -> None:
        """Stop the QUIC server and disconnect all sessions."""
        if self._server is not None:
            self._server.close()
            self._server = None

        for session in list(self._sessions.values()):
            await self.disconnect(session)

        logger.info("WebTransport server stopped")
