"""SIP-to-Anam bridge — wires a SIP video backend to an Anam avatar provider.

Handles all the plumbing:
- SIP caller audio → Anam ``send_user_audio()``
- Anam audio → resample (48kHz → SIP codec rate) → SIP ``send_audio()``
- Anam video → H.264 encode (PyAV) → SIP RTP ``send_frame()``
- Session lifecycle: connect on SIP INVITE, disconnect on BYE

Requirements:
    pip install roomkit[anam,sip,video]
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.providers.anam.realtime import AnamRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState

if TYPE_CHECKING:
    from collections.abc import Callable

    from roomkit.video.backends.sip import SIPVideoBackend

logger = logging.getLogger("roomkit.providers.anam.sip_bridge")

# Lazy-loaded optional deps
_av: Any = None
_np: Any = None


def _ensure_deps() -> None:
    global _av, _np  # noqa: PLW0603
    if _av is None:
        try:
            import av

            _av = av
        except ImportError as exc:
            msg = "PyAV is required for AnamSIPBridge. Install: pip install av"
            raise ImportError(msg) from exc
    if _np is None:
        try:
            import numpy as np

            _np = np
        except ImportError as exc:
            msg = "numpy is required for AnamSIPBridge. Install: pip install numpy"
            raise ImportError(msg) from exc


# ---------------------------------------------------------------------------
# H.264 encoder
# ---------------------------------------------------------------------------


class _H264Encoder:
    """Encode raw RGB frames to H.264 NAL units for SIP/RTP."""

    def __init__(self, width: int, height: int, fps: int = 25) -> None:
        _ensure_deps()
        self._ctx = _av.CodecContext.create("libx264", "w")
        self._ctx.width = width
        self._ctx.height = height
        self._ctx.pix_fmt = "yuv420p"
        self._ctx.time_base = fractions.Fraction(1, fps)
        self._ctx.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
        }
        self._ctx.open()
        self._pts = 0
        self._lock = threading.Lock()

    def encode(self, rgb_bytes: bytes) -> list[bytes]:
        """Encode one RGB frame → list of H.264 NAL units."""
        with self._lock:
            arr = _np.frombuffer(rgb_bytes, dtype=_np.uint8).reshape(
                self._ctx.height,
                self._ctx.width,
                3,
            )
            frame = _av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = self._pts
            self._pts += 1
            nals: list[bytes] = []
            for pkt in self._ctx.encode(frame):
                nals.extend(_split_nals(bytes(pkt)))
            return nals

    def close(self) -> None:
        with self._lock:
            for _pkt in self._ctx.encode(None):
                pass


def _split_nals(data: bytes) -> list[bytes]:
    """Split Annex-B byte stream into individual NAL units."""
    nals: list[bytes] = []
    i = 0
    while i < len(data):
        if data[i : i + 4] == b"\x00\x00\x00\x01":
            start = i + 4
        elif data[i : i + 3] == b"\x00\x00\x01":
            start = i + 3
        else:
            i += 1
            continue
        j = start
        while j < len(data):
            if data[j : j + 3] == b"\x00\x00\x01":
                break
            j += 1
        nals.append(data[start:j])
        i = j
    if not nals and data:
        nals.append(data)
    return nals


# ---------------------------------------------------------------------------
# Audio resampler (48kHz → SIP codec rate)
# ---------------------------------------------------------------------------

_ANAM_RATE = 48000


def _resample(pcm: bytes, target_rate: int) -> bytes:
    """Resample int16 PCM from 48kHz to target_rate."""
    if target_rate == _ANAM_RATE:
        return pcm
    _ensure_deps()
    src = _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float32)
    n_out = int(len(src) * target_rate / _ANAM_RATE)
    if n_out == 0:
        return b""
    x_old = _np.linspace(0, 1, len(src))
    x_new = _np.linspace(0, 1, n_out)
    resampled = _np.interp(x_new, x_old, src)
    return resampled.astype(_np.int16).tobytes()


# ---------------------------------------------------------------------------
# Per-call state
# ---------------------------------------------------------------------------


@dataclass
class _CallState:
    sip_session: VoiceSession
    voice_session: VoiceSession
    encoder: _H264Encoder | None = None
    frame_count: int = 0
    audio_out_count: int = 0
    frame_seq: int = 0


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class AnamSIPBridge:
    """Bridge SIP video calls to an Anam AI avatar.

    Wires bidirectional audio and outbound avatar video between a
    :class:`SIPVideoBackend` and an :class:`AnamRealtimeProvider`.

    Example::

        from roomkit.providers.anam import AnamConfig, AnamRealtimeProvider
        from roomkit.providers.anam.sip_bridge import AnamSIPBridge
        from roomkit.video.backends.sip import SIPVideoBackend

        sip = SIPVideoBackend(
            local_sip_addr=("0.0.0.0", 5060),
            local_rtp_ip="0.0.0.0",
            supported_video_codecs=["H264"],
        )
        provider = AnamRealtimeProvider(AnamConfig(...))

        bridge = AnamSIPBridge(provider, sip)
        await sip.start()
        # Incoming SIP calls are now connected to Anam automatically.
    """

    def __init__(
        self,
        provider: AnamRealtimeProvider,
        sip_backend: SIPVideoBackend,
        *,
        video_fps: int = 25,
        on_call: Callable[[VoiceSession], Any] | None = None,
        on_call_ended: Callable[[str, int, int], Any] | None = None,
        on_transcription: Callable[[str, str, bool], Any] | None = None,
    ) -> None:
        _ensure_deps()
        self._provider = provider
        self._sip = sip_backend
        self._video_fps = video_fps
        self._calls: dict[str, _CallState] = {}
        self._on_call = on_call
        self._on_call_ended = on_call_ended
        self._on_transcription = on_transcription

        # Wire provider callbacks
        provider.on_audio(self._on_anam_audio)
        provider.on_video(self._on_anam_video)
        provider.on_transcription(self._on_anam_transcription)

        # Wire SIP callbacks
        sip_backend.on_audio_received(self._on_sip_audio)
        sip_backend.on_call(self._on_sip_call)
        sip_backend.on_client_disconnected(self._on_sip_bye)

    # -- SIP → Anam ------------------------------------------------------------

    def _on_sip_audio(self, session: VoiceSession, audio: Any) -> None:
        state = self._calls.get(session.id)
        if state is None:
            return
        raw = audio.data if hasattr(audio, "data") else audio
        sample_rate = getattr(audio, "sample_rate", 16000)
        anam_state = self._provider._states.get(state.voice_session.id)
        if anam_state and anam_state.anam_session:
            anam_state.anam_session.send_user_audio(raw, sample_rate, 1)

    async def _on_sip_call(self, sip_session: VoiceSession) -> None:
        caller = sip_session.metadata.get("caller", "unknown")
        logger.info("SIP call from %s", caller)

        voice_session = VoiceSession(
            id=sip_session.id,
            room_id=sip_session.id,
            participant_id=caller,
            channel_id="anam-sip-bridge",
            state=VoiceSessionState.CONNECTING,
        )
        self._calls[sip_session.id] = _CallState(
            sip_session=sip_session,
            voice_session=voice_session,
        )

        await self._provider.connect(voice_session)
        logger.info("Anam avatar active for %s", caller)

        if self._on_call is not None:
            result = self._on_call(sip_session)
            if hasattr(result, "__await__"):
                await result

    def _on_sip_bye(self, session: object) -> None:
        sid = getattr(session, "id", "unknown")
        state = self._calls.pop(sid, None)
        if state is None:
            return
        logger.info(
            "Call ended: session=%s, video=%d, audio=%d",
            sid,
            state.frame_count,
            state.audio_out_count,
        )
        if state.encoder:
            state.encoder.close()
        asyncio.create_task(self._provider.disconnect(state.voice_session))

        if self._on_call_ended is not None:
            result = self._on_call_ended(sid, state.frame_count, state.audio_out_count)
            if hasattr(result, "__await__"):
                asyncio.create_task(result)

    # -- Anam → SIP ------------------------------------------------------------

    def _on_anam_audio(self, session: VoiceSession, audio: bytes) -> None:
        state = self._calls.get(session.id)
        if state is None:
            return
        state.audio_out_count += 1
        codec_rate = state.sip_session.metadata.get("codec_sample_rate", 16000)
        resampled = _resample(audio, codec_rate)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._sip.send_audio(state.sip_session, resampled))

    def _on_anam_video(self, session: VoiceSession, frame: Any) -> None:
        state = self._calls.get(session.id)
        if state is None:
            return
        state.frame_count += 1

        if state.encoder is None:
            state.encoder = _H264Encoder(frame.width, frame.height, self._video_fps)
            logger.info("H.264 encoder: %dx%d", frame.width, frame.height)

        nals = state.encoder.encode(frame.data)
        if not nals:
            return

        vcs = self._sip._video_call_sessions.get(state.sip_session.id)
        if vcs is None:
            return
        ts = state.frame_seq * (90000 // self._video_fps)
        is_key = any((nal[0] & 0x1F) == 5 for nal in nals if nal)
        vcs.send_frame(nals, ts, is_key)
        state.frame_seq += 1

    def _on_anam_transcription(
        self,
        session: VoiceSession,
        text: str,
        role: str,
        is_final: bool,
    ) -> None:
        if self._on_transcription is not None and is_final:
            self._on_transcription(role, text, is_final)

    # -- Cleanup ---------------------------------------------------------------

    async def close(self) -> None:
        """Disconnect all active calls and clean up encoders."""
        for state in list(self._calls.values()):
            if state.encoder:
                state.encoder.close()
            await self._provider.disconnect(state.voice_session)
        self._calls.clear()
        await self._provider.close()
