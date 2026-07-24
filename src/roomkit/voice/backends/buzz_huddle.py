"""Buzz huddle voice transport — Opus audio in a Buzz ephemeral channel.

Bridges a `buzzkit.HuddleClient` (Opus over the relay's
``/huddle/{channel_id}/audio`` WebSocket) to RoomKit's realtime voice
pipeline. The huddle protocol is fixed at 48 kHz mono s16le PCM; the backend
resamples to/from the provider's rates itself (soxr, like
``TwilioWebSocketBackend``) — do **not** pass ``transport_sample_rate`` to
the channel, or audio will be resampled twice at the wrong rates::

    backend = BuzzHuddleBackend()  # 16 kHz in / 24 kHz out (Gemini Live)
    channel = RealtimeVoiceChannel(
        "voice",
        provider=GeminiLiveProvider(api_key=...),
        transport=backend,
    )
    huddle = HuddleClient(relay_url, nsec, huddle_id, parent_channel_id=parent)
    await huddle.connect()
    await channel.start_session(room_id, participant_id, huddle)

Resampling quality matters here: the channel's built-in fallback is linear
interpolation, whose imaging artifacts on a 24 kHz → 48 kHz upsample make
synthesized voices sound harsh/saturated. soxr keeps the spectrum clean.

Inbound audio from every remote peer is forwarded (unmixed) as raw PCM
bytes; overlapping speech from several peers arrives interleaved, which is
acceptable for the common one-human-plus-agent huddle. DTX comfort-noise
frames are dropped — the ``silence_fill`` ticker keeps the provider-facing
stream continuous instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from roomkit.voice.backends._resample import build_streaming_resampler
from roomkit.voice.backends.base import (
    AudioReceivedCallback,
    TransportDisconnectCallback,
    VoiceBackend,
)
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession

if TYPE_CHECKING:
    HuddleClient: Any = None
    HAS_BUZZKIT = True
else:
    try:
        from buzzkit import HuddleClient

        HAS_BUZZKIT = True
    except ImportError:
        HuddleClient = None
        HAS_BUZZKIT = False

logger = logging.getLogger("roomkit.voice.backends.buzz_huddle")


_FRAME_SECONDS = 0.02
HUDDLE_SAMPLE_RATE = 48_000

# Real speech arrives every 20 ms; only start filling silence after a gap
# clearly larger than network jitter, so fill never interleaves mid-speech.
_SILENCE_AFTER_SECONDS = 0.06


class BuzzHuddleBackend(VoiceBackend):
    """Realtime voice transport backed by a Buzz huddle.

    The ``connection`` given to :meth:`accept` must be a **connected**
    ``buzzkit.HuddleClient``. The backend owns it from that point on:
    it runs the client's event loop, and :meth:`disconnect` leaves the
    huddle and closes the socket.

    Outbound pacing (one Opus frame per 20 ms) and the wire protocol live
    in buzzkit; this class only moves PCM bytes and session state.

    ``silence_fill`` (default on) streams silence frames to the pipeline
    whenever no huddle audio is arriving — huddle senders go quiet between
    utterances (Opus DTX), but a realtime provider's server VAD needs to
    *hear* the post-speech silence to close the user's turn, exactly as it
    would from a continuously open microphone.

    ``provider_input_rate`` / ``provider_output_rate`` are the realtime
    provider's PCM rates (defaults match Gemini Live: 16 kHz in, 24 kHz
    out). The backend resamples huddle audio (48 kHz) to/from those rates
    internally — leave the channel's ``transport_sample_rate`` unset.
    """

    def __init__(
        self,
        *,
        silence_fill: bool = True,
        provider_input_rate: int = 16_000,
        provider_output_rate: int = 24_000,
        end_when_alone: bool = True,
        empty_huddle_grace: float = 90.0,
    ) -> None:
        """``end_when_alone`` (default on) ends the session when the last
        remote peer leaves the huddle. The relay keeps a huddle alive while
        ANY member is connected — this agent included — so without it the
        huddle and the provider session run forever. ``empty_huddle_grace``
        is how long to wait for a first peer in a huddle that is empty at
        join time (the announcement can precede the creator's audio socket).
        """
        if not HAS_BUZZKIT:
            raise ImportError(
                "BuzzHuddleBackend requires the buzzkit package. "
                "Install with: pip install roomkit[buzz]"
            )
        self._silence_fill = silence_fill
        self._input_rate = provider_input_rate
        self._output_rate = provider_output_rate
        self._end_when_alone = end_when_alone
        self._empty_grace = empty_huddle_grace
        # 20 ms of s16le silence at the provider's input rate.
        self._silence_frame = b"\x00\x00" * int(provider_input_rate * _FRAME_SECONDS)
        self._clients: dict[str, Any] = {}  # session_id -> HuddleClient
        self._pacers: dict[str, Any] = {}  # session_id -> OutboundAudioPacer
        self._receive_tasks: dict[str, asyncio.Task[None]] = {}
        self._silence_tasks: dict[str, asyncio.Task[None]] = {}
        self._alone_tasks: dict[str, asyncio.Task[None]] = {}
        # Sessions whose disconnect callbacks already fired (or must not fire
        # because teardown is deliberate). Guarded by the event loop.
        self._disconnect_fired: set[str] = set()
        self._last_audio_at: dict[str, float] = {}
        self._inbound_resample: dict[str, Callable[[bytes], bytes]] = {}
        self._outbound_resample: dict[str, Callable[[bytes], bytes]] = {}
        self._sessions: dict[str, VoiceSession] = {}
        self._audio_callbacks: list[AudioReceivedCallback] = []
        self._disconnect_callbacks: list[TransportDisconnectCallback] = []

    @property
    def name(self) -> str:
        return "BuzzHuddleBackend"

    @property
    def capabilities(self) -> VoiceCapability:
        return VoiceCapability.INTERRUPTION

    async def accept(self, session: VoiceSession, connection: Any) -> None:
        """Bind a connected ``buzzkit.HuddleClient`` to this session."""
        if not hasattr(connection, "events") or not hasattr(connection, "send_pcm"):
            raise TypeError(
                "BuzzHuddleBackend.accept expects a connected buzzkit.HuddleClient, "
                f"got {type(connection).__name__}"
            )
        self._clients[session.id] = connection
        self._sessions[session.id] = session
        session.metadata.setdefault("buzz_channel_id", getattr(connection, "channel_id", None))
        self._inbound_resample[session.id] = build_streaming_resampler(
            HUDDLE_SAMPLE_RATE, self._input_rate
        )
        self._outbound_resample[session.id] = build_streaming_resampler(
            self._output_rate, HUDDLE_SAMPLE_RATE
        )
        # Outbound pacing: prebuffer + jitter headroom so the receiver's jitter
        # buffer always leads wall-clock and absorbs this process's scheduling
        # jitter (GIL stalls from the provider/pipeline). Same pacer the SIP
        # backend uses. Runs at the huddle wire rate (48 kHz); its send_fn
        # encodes each frame to Opus and relays it via the (unpaced) client.
        from roomkit.voice.realtime.pacer import OutboundAudioPacer

        pacer = OutboundAudioPacer(
            send_fn=self._make_wire_sender(connection),
            sample_rate=HUDDLE_SAMPLE_RATE,
            channels=1,
            sample_width=2,
            prebuffer_ms=80,
            jitter_headroom_ms=60,
            fill_with_silence_when_idle=True,
        )
        await pacer.start()
        self._pacers[session.id] = pacer
        self._receive_tasks[session.id] = asyncio.create_task(
            self._receive_loop(session, connection),
            name=f"buzz_huddle_recv:{session.id}",
        )
        if self._silence_fill:
            self._silence_tasks[session.id] = asyncio.create_task(
                self._silence_loop(session),
                name=f"buzz_huddle_silence:{session.id}",
            )
        if self._end_when_alone:
            # Second events() subscriber — must start after the receive task:
            # the client hands its pre-connect event backlog to the first
            # subscriber, which belongs to the audio path.
            self._alone_tasks[session.id] = asyncio.create_task(
                self._alone_loop(session, connection),
                name=f"buzz_huddle_alone:{session.id}",
            )

    def _make_wire_sender(self, client: Any) -> Callable[[bytes], Any]:
        """Build the pacer's send_fn: encode 48 kHz PCM → relay Opus frames."""

        async def send(pcm48k: bytes) -> None:
            # send_pcm encodes to wire frames and hands them to the client's
            # unpaced relay loop; the pacer owns the timing.
            client.send_pcm(pcm48k)

        return send

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        """Resample provider PCM to 48 kHz and hand it to the outbound pacer."""
        pacer = self._pacers.get(session.id)
        resample = self._outbound_resample.get(session.id)
        if pacer is None or resample is None:
            return
        try:
            if isinstance(audio, bytes):
                pacer.push(resample(audio))
            else:
                async for chunk in audio:
                    pacer.push(resample(chunk.data))
        except Exception:
            logger.exception("Error sending audio to session %s", session.id)

    def interrupt(self, session: VoiceSession) -> None:
        """Drop queued + in-flight outbound audio (barge-in)."""
        pacer = self._pacers.get(session.id)
        if pacer is not None:
            pacer.interrupt()
        client = self._clients.get(session.id)
        if client is not None:
            client.clear_queue()

    async def cancel_audio(self, session: VoiceSession) -> bool:
        self.interrupt(session)
        return True

    def is_playing(self, session: VoiceSession) -> bool:
        client = self._clients.get(session.id)
        return client is not None and client.queued_frames > 0

    def end_of_response(self, session: VoiceSession) -> None:
        """Signal the pacer that the current response is fully delivered."""
        pacer = self._pacers.get(session.id)
        if pacer is not None:
            pacer.end_of_response()

    async def disconnect(self, session: VoiceSession) -> None:
        # Deliberate teardown: the dying receive loop must not report it as a
        # connection loss (the session owner already knows it's over).
        self._disconnect_fired.add(session.id)
        for tasks in (self._receive_tasks, self._silence_tasks, self._alone_tasks):
            task = tasks.pop(session.id, None)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        pacer = self._pacers.pop(session.id, None)
        if pacer is not None:
            with contextlib.suppress(Exception):
                await pacer.stop()
        client = self._clients.pop(session.id, None)
        self._sessions.pop(session.id, None)
        self._last_audio_at.pop(session.id, None)
        self._inbound_resample.pop(session.id, None)
        self._outbound_resample.pop(session.id, None)
        if client is not None:
            with contextlib.suppress(Exception):
                await client.leave()
        self._disconnect_fired.discard(session.id)

    def get_session(self, session_id: str) -> VoiceSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        self._audio_callbacks.append(callback)

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)

    async def close(self) -> None:
        for session in list(self._sessions.values()):
            await self.disconnect(session)

    async def _silence_loop(self, session: VoiceSession) -> None:
        """Stream silence to the pipeline while no huddle audio is arriving."""
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        while True:
            next_at += _FRAME_SECONDS
            delay = next_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_at = loop.time()  # fell behind — reset the cadence
            last = self._last_audio_at.get(session.id, 0.0)
            if loop.time() - last >= _SILENCE_AFTER_SECONDS:
                await self._fire_audio_callbacks(session, self._silence_frame)

    async def _receive_loop(self, session: VoiceSession, client: Any) -> None:
        """Forward decoded huddle audio to the pipeline until the huddle ends."""
        loop = asyncio.get_running_loop()
        resample = self._inbound_resample[session.id]
        try:
            async for event in client.events():
                pcm = getattr(event, "pcm", None)
                if pcm is None:
                    # Roster events: keep the speaker map in session metadata
                    # for observers; the pipeline itself is speaker-agnostic.
                    session.metadata["buzz_peers"] = dict(client.peers)
                    continue
                if event.is_dtx:
                    continue
                self._last_audio_at[session.id] = loop.time()
                await self._fire_audio_callbacks(session, resample(pcm))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Huddle receive loop failed for session %s", session.id)
        finally:
            await self._fire_disconnect(session, "connection_lost")

    async def _alone_loop(self, session: VoiceSession, client: Any) -> None:
        """End the session when the last remote peer leaves the huddle.

        The relay keeps a huddle alive while ANY member is connected — this
        agent included — so a session that never hangs up keeps the huddle
        (and the provider connection) open forever. A huddle that is empty
        at join time gets ``empty_huddle_grace`` seconds for a first peer to
        arrive before the session ends.
        """
        events = aiter(client.events())
        seen_peer = len(client.peers) > 1
        while True:
            try:
                if seen_peer:
                    event = await anext(events)
                else:
                    async with asyncio.timeout(self._empty_grace):
                        event = await anext(events)
            except TimeoutError:
                logger.info("Session %s: nobody joined the huddle — hanging up", session.id)
                break
            except StopAsyncIteration:
                return  # socket closed; the receive loop reports the disconnect
            if getattr(event, "pcm", None) is not None:
                continue  # audio frame; only roster changes matter here
            if len(client.peers) > 1:
                seen_peer = True
            elif seen_peer:
                logger.info("Session %s: last peer left the huddle — hanging up", session.id)
                break
        await self._fire_disconnect(session, "alone")

    async def _fire_disconnect(self, session: VoiceSession, reason: str) -> None:
        """Invoke the disconnect callbacks exactly once per session.

        ``reason`` lands in ``session.metadata["buzz_end_reason"]``: "alone"
        (last peer left, or nobody ever joined) or "connection_lost" (the
        relay dropped the audio socket). Callers deciding whether to rejoin
        the huddle read it from there.
        """
        if session.id in self._disconnect_fired:
            return
        self._disconnect_fired.add(session.id)
        session.metadata["buzz_end_reason"] = reason
        for cb in self._disconnect_callbacks:
            try:
                result = cb(session)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in disconnect callback for session %s", session.id)

    async def _fire_audio_callbacks(self, session: VoiceSession, audio: bytes) -> None:
        for cb in self._audio_callbacks:
            try:
                result = cb(session, audio)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in audio callback for session %s", session.id)
