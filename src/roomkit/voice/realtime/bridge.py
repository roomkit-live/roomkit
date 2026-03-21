"""Generic bridge between a RealtimeAudioVideoProvider and a voice/video backend.

Wires bidirectional audio and outbound avatar video between any
:class:`RealtimeAudioVideoProvider` and any :class:`VoiceBackend`,
with optional video encoding for backends that also implement
:class:`VideoBackend`.

Audio path:
    Backend audio in → ``provider.send_audio()``
    Provider audio out → resample → ``backend.send_audio()``

Video path (with encoder):
    Provider video out → ``VideoEncoderProvider.encode()`` → backend RTP

Video path (passthrough / no encoder):
    Provider video out → video taps only (display, recording)

Requirements for video encoding:
    pip install roomkit[video]   # PyAV + numpy
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeAudioVideoProvider, RealtimeVoiceProvider

if TYPE_CHECKING:
    from roomkit.video.avatar.base import AvatarProvider
    from roomkit.video.pipeline.config import VideoPipelineConfig
    from roomkit.video.pipeline.encoder.base import VideoEncoderProvider
    from roomkit.video.video_frame import VideoFrame
    from roomkit.voice.backends.base import VoiceBackend

logger = logging.getLogger("roomkit.voice.realtime.bridge")

# Lazy-loaded numpy
_np: Any = None


def _get_np() -> Any:
    global _np  # noqa: PLW0603
    if _np is None:
        import numpy as np

        _np = np
    return _np


# ---------------------------------------------------------------------------
# Audio resampler (provider rate → backend codec rate)
# ---------------------------------------------------------------------------


def resample_pcm(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample int16 PCM via linear interpolation.

    Args:
        pcm: Input PCM int16 LE bytes.
        from_rate: Source sample rate (Hz).
        to_rate: Target sample rate (Hz).

    Returns:
        Resampled PCM int16 LE bytes.
    """
    if from_rate == to_rate:
        return pcm
    np = _get_np()
    src = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_out = int(len(src) * to_rate / from_rate)
    if n_out == 0:
        return b""
    x_old = np.linspace(0, 1, len(src))
    x_new = np.linspace(0, 1, n_out)
    resampled = np.interp(x_new, x_old, src)
    result: bytes = resampled.astype(np.int16).tobytes()
    return result


# ---------------------------------------------------------------------------
# Per-call state
# ---------------------------------------------------------------------------


@dataclass
class _CallState:
    backend_session: VoiceSession
    provider_session: VoiceSession
    frame_count: int = 0
    audio_out_count: int = 0
    frame_seq: int = 0
    closed: bool = False
    encode_lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class RealtimeAVBridge:
    """Bridge a voice/video backend to a realtime voice or audio+video provider.

    Supports two video modes:

    **Provider video** (e.g. Anam full mode):
        Provider implements :class:`RealtimeAudioVideoProvider` and delivers
        both audio and video.  Video goes through the pipeline and encoder.

    **Avatar video** (e.g. OpenAI Realtime + Anam passthrough):
        Provider implements :class:`RealtimeVoiceProvider` (audio only).
        An :class:`AvatarProvider` receives the provider's TTS audio via
        ``feed_audio()`` and produces lip-synced video frames.

    In both cases, audio flows bidirectionally:

    - Backend audio → ``provider.send_audio()`` (user speech to AI)
    - Provider audio → resample → ``backend.send_audio()`` (AI speech to user)

    Example (Anam full — provider delivers audio+video)::

        bridge = RealtimeAVBridge(
            AnamRealtimeProvider(config), sip,
            encoder=PyAVVideoEncoder(fps=25),
        )

    Example (OpenAI + Anam avatar — provider delivers audio, avatar renders video)::

        bridge = RealtimeAVBridge(
            OpenAIRealtimeProvider(api_key="..."), sip,
            avatar=AnamAvatarProvider(anam_config),
            encoder=PyAVVideoEncoder(fps=25),
        )

    Args:
        provider: The realtime voice provider. Can be audio-only
            (:class:`RealtimeVoiceProvider`) or audio+video
            (:class:`RealtimeAudioVideoProvider`).
        backend: The voice backend (and optionally video backend).
        avatar: Optional :class:`AvatarProvider` for lip-synced video
            from the provider's audio output.  Used when the provider
            is audio-only (e.g. OpenAI Realtime + Anam passthrough).
        video_pipeline: Optional video processing pipeline applied to
            video frames before encoding (resize, filters, vision).
        encoder: Optional video encoder for outbound video.
        connecting_frame: Optional :class:`VideoFrame` shown while
            the provider negotiates (3-5s for WebRTC).
        connecting_fps: Frame rate for the connecting placeholder (default: 5).
        provider_sample_rate: Sample rate of provider audio output (Hz).
            Anam outputs 48000, OpenAI outputs 24000.  Default: 24000.
        video_fps: Frames per second for RTP timestamp calculation.
        on_call: Callback when a new call is connected.
        on_call_ended: Callback ``(session_id, video_frames, audio_chunks)``.
        on_transcription: Callback ``(role, text, is_final)``.
    """

    def __init__(
        self,
        provider: RealtimeVoiceProvider,
        backend: VoiceBackend,
        *,
        avatar: AvatarProvider | None = None,
        video_pipeline: VideoPipelineConfig | None = None,
        encoder: VideoEncoderProvider | None = None,
        connecting_frame: VideoFrame | None = None,
        connecting_fps: int = 5,
        provider_sample_rate: int = 24000,
        system_prompt: str | None = None,
        voice: str | None = None,
        video_fps: int = 25,
        on_call: Callable[[VoiceSession], Any] | None = None,
        on_call_ended: Callable[[str, int, int], Any] | None = None,
        on_transcription: Callable[[str, str, bool], Any] | None = None,
    ) -> None:
        self._provider = provider
        self._backend = backend
        self._avatar = avatar
        self._encoder = encoder
        self._connecting_frame = connecting_frame
        self._connecting_fps = connecting_fps
        self._provider_rate = provider_sample_rate
        self._system_prompt = system_prompt
        self._voice = voice
        self._video_fps = video_fps
        self._calls: dict[str, _CallState] = {}
        self._video_taps: list[Callable[..., Any]] = []
        self._on_call = on_call
        self._on_call_ended = on_call_ended
        self._on_transcription = on_transcription

        # Build video pipeline if configured
        if video_pipeline is not None and (
            video_pipeline.decoder
            or video_pipeline.resizer
            or video_pipeline.transforms
            or video_pipeline.filters
        ):
            from roomkit.video.pipeline.engine import VideoPipeline

            self._video_pipeline: Any = VideoPipeline(video_pipeline)
        else:
            self._video_pipeline = None
        self._video_pipeline_config = video_pipeline

        # Thread pool for H.264 encoding — avoids blocking the event loop
        # which would starve audio callbacks and cause dropped/delayed audio.
        self._encode_pool: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1) if encoder else None
        )

        # Wire provider audio → backend + avatar
        provider.on_audio(self._on_provider_audio)
        provider.on_transcription(self._on_provider_transcription)

        # Wire video source: either from provider (RealtimeAudioVideoProvider)
        # or from avatar (AvatarProvider in passthrough mode)
        if isinstance(provider, RealtimeAudioVideoProvider):
            provider.on_video(self._on_provider_video)
        if avatar is not None:
            avatar.on_video(self._on_avatar_video)
            provider.on_response_end(self._on_provider_response_end)

        # Wire backend → provider
        backend.on_audio_received(self._on_backend_audio)
        backend.on_client_disconnected(self._on_backend_disconnect)

        # Auto-register on_call if backend supports it (SIP, RTP)
        on_call_method = getattr(backend, "on_call", None)
        if on_call_method is not None:
            on_call_method(self._on_backend_call)

    # -- Public API ------------------------------------------------------------

    def add_video_tap(self, callback: Callable[..., Any]) -> None:
        """Register a tap that receives every provider video frame."""
        self._video_taps.append(callback)

    async def _send_connecting_loop(self, state: _CallState) -> None:
        """Send a placeholder frame while the provider negotiates."""
        if self._connecting_frame is None or self._encoder is None:
            return
        interval = 1.0 / self._connecting_fps
        try:
            while not state.closed:
                self._send_encoded_video(state, self._connecting_frame)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def connect(
        self,
        backend_session: VoiceSession,
        *,
        participant_id: str | None = None,
    ) -> VoiceSession:
        """Manually connect a backend session to the provider.

        Use this for backends that don't fire ``on_call`` (e.g. local mic).
        For SIP/RTP backends, connections are automatic via ``on_call``.

        Returns:
            The provider-side VoiceSession.
        """
        caller = participant_id or backend_session.participant_id or "unknown"
        provider_session = VoiceSession(
            id=backend_session.id,
            room_id=backend_session.room_id or backend_session.id,
            participant_id=caller,
            channel_id="realtime-av-bridge",
            state=VoiceSessionState.CONNECTING,
        )
        self._calls[backend_session.id] = _CallState(
            backend_session=backend_session,
            provider_session=provider_session,
        )
        # Start avatar if not yet started (first call triggers it)
        if self._avatar is not None and not self._avatar.is_started:
            await self._avatar.start(b"")

        # Pre-warm the SIP audio pacer so the first audio chunk isn't delayed
        ensure_pacer = getattr(self._backend, "_ensure_pacer", None)
        if ensure_pacer is not None:
            with contextlib.suppress(Exception):
                ensure_pacer(backend_session)

        # Send placeholder frame while provider negotiates (3-5s)
        placeholder_task: asyncio.Task[None] | None = None
        if self._connecting_frame is not None and self._encoder is not None:
            state = self._calls[backend_session.id]
            placeholder_task = asyncio.create_task(
                self._send_connecting_loop(state),
                name=f"connecting_placeholder:{backend_session.id}",
            )

        t0 = time.monotonic()
        try:
            await self._provider.connect(
                provider_session,
                system_prompt=self._system_prompt,
                voice=self._voice,
                output_sample_rate=self._provider_rate,
            )
        finally:
            if placeholder_task is not None:
                placeholder_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await placeholder_task

        # If the backend session died during the provider negotiation
        # (e.g. SIP BYE during the 3-4s WebRTC handshake), disconnect
        # the provider immediately to stop its consume loops.
        if backend_session.id not in self._calls:
            logger.info(
                "Backend session ended during provider negotiation, disconnecting",
            )
            await self._provider.disconnect(provider_session)
            return provider_session

        logger.info(
            "Bridge connected: %s (%s) in %.0fms",
            backend_session.id[:8],
            caller,
            (time.monotonic() - t0) * 1000,
        )
        return provider_session

    async def disconnect(self, session_id: str) -> None:
        """Disconnect a bridged session."""
        state = self._calls.pop(session_id, None)
        if state is None:
            return
        await self._provider.disconnect(state.provider_session)
        logger.info(
            "Bridge disconnected: %s (video=%d, audio=%d)",
            session_id[:8],
            state.frame_count,
            state.audio_out_count,
        )

    async def close(self) -> None:
        """Disconnect all active calls and clean up."""
        for sid in list(self._calls):
            await self.disconnect(sid)
        if self._encode_pool:
            self._encode_pool.shutdown(wait=False)
        if self._encoder:
            self._encoder.close()
        if self._avatar is not None:
            await self._avatar.close()
        await self._provider.close()

    # -- Backend → Provider (user audio in) ------------------------------------

    def _on_backend_audio(self, session: VoiceSession, audio: Any) -> None:
        state = self._calls.get(session.id)
        if state is None or state.closed:
            return
        # AudioFrame or raw bytes
        raw = audio.data if hasattr(audio, "data") else audio
        sample_rate = getattr(audio, "sample_rate", 16000)

        # Use the provider's public send_audio method.
        # Store rate for the provider's internal send_user_audio call.
        state.provider_session.metadata["_input_rate"] = sample_rate
        loop = asyncio.get_running_loop()
        if loop.is_running():
            loop.create_task(self._provider.send_audio(state.provider_session, raw))

    async def _on_backend_call(self, sip_session: VoiceSession) -> None:
        """Handle incoming call from SIP/RTP backend."""
        caller = sip_session.metadata.get("caller", "unknown")
        logger.info("Incoming call from %s", caller)

        await self.connect(sip_session, participant_id=caller)

        if self._on_call is not None:
            result = self._on_call(sip_session)
            if hasattr(result, "__await__"):
                await result

    def _on_backend_disconnect(self, session: object) -> None:
        sid = getattr(session, "id", None)
        if sid is None:
            return
        state = self._calls.pop(sid, None)
        if state is None:
            return
        state.closed = True
        logger.info(
            "Call ended: %s (video=%d, audio=%d)",
            sid[:8],
            state.frame_count,
            state.audio_out_count,
        )
        self._safe_create_task(self._provider.disconnect(state.provider_session))

        if self._on_call_ended is not None:
            result = self._on_call_ended(
                sid,
                state.frame_count,
                state.audio_out_count,
            )
            if hasattr(result, "__await__"):
                self._safe_create_task(result)

    @staticmethod
    def _safe_create_task(coro: Any) -> None:
        """Create an asyncio task, safely handling calls from non-async threads."""
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            # Called from a sync callback thread (e.g. SIP/WebRTC disconnect)
            loop = asyncio.get_event_loop_policy().get_event_loop()
            loop.call_soon_threadsafe(loop.create_task, coro)

    # -- Provider → Backend (AI audio/video out) -------------------------------

    def _on_provider_audio(self, session: VoiceSession, audio: bytes) -> None:
        state = self._calls.get(session.id)
        if state is None or state.closed:
            return
        # Check if backend session is still alive (SIP may have cleaned up
        # its internal state before firing the disconnect callback)
        backend_states = getattr(self._backend, "_session_states", None)
        if backend_states is not None and state.backend_session.id not in backend_states:
            state.closed = True
            logger.info("Backend session gone, stopping bridge: %s", session.id[:8])
            self._safe_create_task(self._provider.disconnect(state.provider_session))
            return
        state.audio_out_count += 1

        # Resample from provider rate to backend codec rate
        codec_rate = state.backend_session.metadata.get(
            "codec_sample_rate",
            16000,
        )
        resampled = resample_pcm(audio, self._provider_rate, codec_rate)

        self._safe_create_task(self._backend.send_audio(state.backend_session, resampled))

        # Feed audio to avatar for lip-sync (if configured)
        if self._avatar is not None and self._avatar.is_started:
            self._avatar.feed_audio(audio)

    def _on_provider_response_end(self, session: VoiceSession) -> None:
        """Signal end of response to avatar so it doesn't freeze."""
        if self._avatar is not None and self._avatar.is_started:
            self._avatar.end_turn()

    def _on_avatar_video(self, frame: Any) -> None:
        """Handle video from the avatar provider (passthrough mode)."""
        # Route to all active calls — avatar is shared across sessions
        for state in self._calls.values():
            if state.closed:
                continue
            state.frame_count += 1

            processed = frame
            if self._video_pipeline is not None:
                processed = self._video_pipeline.process_inbound("avatar", frame)
                if processed is None:
                    continue

            for tap in self._video_taps:
                tap(state.provider_session, processed)

            if self._encoder is not None and self._encode_pool is not None:
                self._encode_pool.submit(self._send_encoded_video, state, processed)

    def _on_provider_video(self, session: VoiceSession, frame: Any) -> None:
        state = self._calls.get(session.id)
        if state is None or state.closed:
            return
        state.frame_count += 1

        # Run through video pipeline (resize, filters, vision) if configured
        if self._video_pipeline is not None:
            processed = self._video_pipeline.process_inbound(session.id, frame)
            if processed is None:
                return
            frame = processed

        # Deliver to taps (display, recording)
        for tap in self._video_taps:
            tap(session, frame)

        # Encode in thread pool to avoid blocking audio callbacks
        if self._encoder is None or self._encode_pool is None:
            return
        self._encode_pool.submit(self._send_encoded_video, state, frame)

    def _send_encoded_video(self, state: _CallState, frame: Any) -> None:
        """Encode a raw frame and send via RTP (runs in thread pool).

        Uses the ``_video_call_sessions`` pattern established by
        :class:`AudioVideoChannel` for direct RTP frame delivery.
        """
        from roomkit.video.video_frame import VideoFrame

        if not isinstance(frame, VideoFrame):
            return

        with state.encode_lock:
            nals = self._encoder.encode(frame)  # type: ignore[union-attr]
            if not nals:
                return

            # Access internal video call session (same as AudioVideoChannel)
            vcs = getattr(self._backend, "_video_call_sessions", {}).get(
                state.backend_session.id,
            )
            if vcs is None:
                return

            is_key = any((nal[0] & 0x1F) == 5 for nal in nals if nal)
            rtp = getattr(vcs, "_session", None)
            if rtp is not None and hasattr(rtp, "send_frame_auto"):
                rtp.send_frame_auto(nals, is_key)
            else:
                ts = state.frame_seq * (90000 // self._video_fps)
                vcs.send_frame(nals, ts, is_key)
            state.frame_seq += 1

    def _on_provider_transcription(
        self,
        session: VoiceSession,
        text: str,
        role: str,
        is_final: bool,
    ) -> None:
        if self._on_transcription is not None and is_final:
            self._on_transcription(role, text, is_final)
