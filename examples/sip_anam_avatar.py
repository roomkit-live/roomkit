"""RoomKit — SIP-to-Anam AI Avatar bridge.

Accept incoming SIP video calls and connect them to an Anam AI avatar.
Caller audio is forwarded to Anam, and the avatar's audio+video is
streamed back to the caller over SIP/RTP.

Audio flow:
    SIP phone → RTP audio → Anam (send_user_audio)
    Anam TTS → PCM audio → SIP RTP → phone speaker

Video flow:
    Anam avatar → raw RGB → H.264 encode (PyAV) → SIP RTP → phone screen

Prerequisites:
    pip install roomkit[anam,sip,video]

Run with:
    export ANAM_API_KEY="your-api-key"
    export ANAM_AVATAR_ID="your-avatar-id"
    export ANAM_VOICE_ID="your-voice-id"
    export ANAM_LLM_ID="your-llm-id"
    uv run python examples/sip_anam_avatar.py

Environment variables:
    ANAM_API_KEY       Anam API key (required)
    ANAM_AVATAR_ID     Avatar ID from lab.anam.ai (required)
    ANAM_VOICE_ID      Voice ID from lab.anam.ai (required)
    ANAM_LLM_ID        LLM ID from lab.anam.ai (required)
    ANAM_PERSONA_ID    Pre-defined persona (alternative to above three)
    SIP_PORT           SIP listener port (default: 5060)
    RTP_IP             IP to advertise in SDP (default: 0.0.0.0)
    RTP_PORT_START     First RTP port to allocate (default: 10000)
    DEBUG              Set to 1 for verbose logging

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import os
import signal
import threading
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sip_anam_avatar")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)

import av
import numpy as np

from roomkit import AnamConfig, AnamRealtimeProvider
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.voice.base import VoiceSession, VoiceSessionState

# ---------------------------------------------------------------------------
# H.264 encoder (raw RGB → NAL units for SIP/RTP)
# ---------------------------------------------------------------------------


class H264Encoder:
    """Encode raw RGB frames to H.264 NAL units for SIP/RTP."""

    def __init__(self, width: int, height: int, fps: int = 25) -> None:
        self._ctx = av.CodecContext.create("libx264", "w")
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
            arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
                self._ctx.height,
                self._ctx.width,
                3,
            )
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
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
        # Find next start code
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
# Per-call bridge state
# ---------------------------------------------------------------------------


class _CallBridge:
    """Bridges one SIP call to one Anam session."""

    def __init__(
        self,
        sip_session: VoiceSession,
        sip_backend: SIPVideoBackend,
        provider: AnamRealtimeProvider,
        voice_session: VoiceSession,
    ) -> None:
        self.sip_session = sip_session
        self.sip_backend = sip_backend
        self.provider = provider
        self.voice_session = voice_session
        self.encoder: H264Encoder | None = None
        self.frame_count = 0
        self.audio_out_count = 0
        self._frame_seq = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # --- Validate environment -------------------------------------------------
    api_key = os.environ.get("ANAM_API_KEY", "")
    if not api_key:
        logger.error("Set ANAM_API_KEY environment variable")
        return

    persona_id = os.environ.get("ANAM_PERSONA_ID")
    avatar_id = os.environ.get("ANAM_AVATAR_ID")
    voice_id = os.environ.get("ANAM_VOICE_ID")
    llm_id = os.environ.get("ANAM_LLM_ID")

    if not persona_id and not (avatar_id and voice_id and llm_id):
        logger.error("Set either ANAM_PERSONA_ID or ANAM_AVATAR_ID + ANAM_VOICE_ID + ANAM_LLM_ID")
        return

    # --- SIP backend ----------------------------------------------------------
    sip_port = int(os.environ.get("SIP_PORT", "5060"))
    rtp_ip = os.environ.get("RTP_IP", "0.0.0.0")
    rtp_port_start = int(os.environ.get("RTP_PORT_START", "10000"))

    sip_backend = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", sip_port),
        local_rtp_ip=rtp_ip,
        rtp_port_start=rtp_port_start,
        supported_video_codecs=["H264"],
    )

    # --- Anam provider --------------------------------------------------------
    config = AnamConfig(
        api_key=api_key,
        persona_id=persona_id,
        avatar_id=avatar_id,
        voice_id=voice_id,
        llm_id=llm_id,
        system_prompt=(
            "You are a helpful AI avatar on a video call. "
            "Keep responses conversational and concise."
        ),
    )
    provider = AnamRealtimeProvider(config)

    # Active call bridges: sip_session_id → _CallBridge
    bridges: dict[str, _CallBridge] = {}

    # --- Anam audio output → resample → SIP speaker ---------------------------
    # Anam produces 48kHz mono int16 PCM (after _av_audio_to_pcm downmix).
    # SIP needs audio at codec_rate: 16kHz (G.722) or 8kHz (G.711).
    # Resample using linear interpolation for proper quality.
    _anam_rate = 48000

    def _resample(pcm: bytes, target_rate: int) -> bytes:
        """Resample int16 PCM from 48kHz to target_rate."""
        if target_rate == _anam_rate:
            return pcm
        src = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        ratio = target_rate / _anam_rate
        n_out = int(len(src) * ratio)
        if n_out == 0:
            return b""
        x_old = np.linspace(0, 1, len(src))
        x_new = np.linspace(0, 1, n_out)
        resampled = np.interp(x_new, x_old, src)
        return resampled.astype(np.int16).tobytes()

    def on_anam_audio(session: VoiceSession, audio: bytes) -> None:
        bridge = bridges.get(session.id)
        if bridge is None:
            return
        bridge.audio_out_count += 1
        codec_rate = bridge.sip_session.metadata.get("codec_sample_rate", 16000)

        # Debug: log first few audio chunks in detail
        if bridge.audio_out_count <= 3:
            samples = np.frombuffer(audio, dtype=np.int16)
            logger.info(
                "AUDIO DEBUG out #%d: %d bytes, %d samples (%.1fms@48k), "
                "min=%d max=%d rms=%.0f, sip_codec_rate=%d, metadata=%s",
                bridge.audio_out_count,
                len(audio),
                len(samples),
                len(samples) / 48.0,  # ms at 48kHz
                samples.min(),
                samples.max(),
                np.sqrt(np.mean(samples.astype(np.float64) ** 2)),
                codec_rate,
                {
                    k: v
                    for k, v in bridge.sip_session.metadata.items()
                    if "rate" in k or "codec" in k
                },
            )

        resampled = _resample(audio, codec_rate)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(sip_backend.send_audio(bridge.sip_session, resampled))

    provider.on_audio(on_anam_audio)

    # --- Anam video output → H.264 encode → SIP RTP --------------------------
    def on_anam_video(session: VoiceSession, frame: Any) -> None:
        bridge = bridges.get(session.id)
        if bridge is None:
            return
        bridge.frame_count += 1

        # Lazy-init encoder on first frame
        if bridge.encoder is None:
            bridge.encoder = H264Encoder(frame.width, frame.height, fps=25)
            logger.info(
                "H.264 encoder started: %dx%d (session %s)",
                frame.width,
                frame.height,
                session.id,
            )

        # Encode raw RGB → H.264 NALs
        nals = bridge.encoder.encode(frame.data)
        if not nals:
            return

        # Send to SIP video session
        vcs = sip_backend._video_call_sessions.get(bridge.sip_session.id)
        if vcs is None:
            return
        ts = bridge._frame_seq * (90000 // 25)
        is_key = any((nal[0] & 0x1F) == 5 for nal in nals if nal)
        vcs.send_frame(nals, ts, is_key)
        bridge._frame_seq += 1

        if bridge.frame_count % 30 == 1:
            logger.info("Avatar → SIP video frame #%d", bridge.frame_count)

    provider.on_video(on_anam_video)

    # --- Anam transcription → log ---------------------------------------------
    def on_transcription(
        session: VoiceSession,
        text: str,
        role: str,
        is_final: bool,
    ) -> None:
        if is_final:
            logger.info("[%s] %s", role.upper(), text)

    provider.on_transcription(on_transcription)

    # --- SIP audio → Anam microphone -----------------------------------------
    _sip_in_count = 0

    def on_sip_audio(session: VoiceSession, audio: Any) -> None:
        nonlocal _sip_in_count
        bridge = bridges.get(session.id)
        if bridge is None:
            return
        # SIP backend passes AudioFrame objects, extract raw PCM bytes
        raw = audio.data if hasattr(audio, "data") else audio
        sample_rate = getattr(audio, "sample_rate", 16000)

        _sip_in_count += 1
        if _sip_in_count <= 3:
            logger.info(
                "AUDIO DEBUG in #%d: type=%s, raw_type=%s, %d bytes, "
                "sample_rate=%d, audio_attrs=%s",
                _sip_in_count,
                type(audio).__name__,
                type(raw).__name__,
                len(raw) if isinstance(raw, (bytes, bytearray)) else -1,
                sample_rate,
                [a for a in dir(audio) if not a.startswith("_")],
            )

        anam_state = bridge.provider._states.get(bridge.voice_session.id)
        if anam_state and anam_state.anam_session:
            anam_state.anam_session.send_user_audio(raw, sample_rate, 1)

    sip_backend.on_audio_received(on_sip_audio)

    # --- On SIP INVITE: create Anam session -----------------------------------
    async def on_call(sip_session: VoiceSession) -> None:
        caller = sip_session.metadata.get("caller", "unknown")
        has_video = sip_session.metadata.get("has_video", False)
        logger.info("SIP call from %s (video=%s)", caller, has_video)

        # Create a VoiceSession for the Anam provider
        voice_session = VoiceSession(
            id=sip_session.id,
            room_id=sip_session.id,
            participant_id=caller,
            channel_id="anam-bridge",
            state=VoiceSessionState.CONNECTING,
        )

        bridge = _CallBridge(sip_session, sip_backend, provider, voice_session)
        bridges[sip_session.id] = bridge

        # Connect to Anam
        await provider.connect(voice_session)
        logger.info("Anam avatar active for caller %s", caller)

    sip_backend.on_call(on_call)

    # --- On SIP BYE: tear down Anam session -----------------------------------
    def on_call_ended(session: object) -> None:
        sid = getattr(session, "id", "unknown")
        bridge = bridges.pop(sid, None)
        if bridge is None:
            return
        logger.info(
            "Call ended: session=%s, video_frames=%d, audio_chunks=%d",
            sid,
            bridge.frame_count,
            bridge.audio_out_count,
        )
        if bridge.encoder:
            bridge.encoder.close()
        asyncio.create_task(provider.disconnect(bridge.voice_session))

    sip_backend.on_client_disconnected(on_call_ended)

    # --- Start ----------------------------------------------------------------
    await sip_backend.start()
    logger.info("SIP + Anam Avatar bridge on 0.0.0.0:%d", sip_port)
    logger.info("Call this SIP endpoint with a video phone to talk to the avatar.")
    logger.info("Press Ctrl+C to stop.\n")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # --- Cleanup --------------------------------------------------------------
    logger.info("\nStopping...")
    for bridge in bridges.values():
        if bridge.encoder:
            bridge.encoder.close()
        await provider.disconnect(bridge.voice_session)
    bridges.clear()
    await provider.close()
    await sip_backend.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
