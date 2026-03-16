"""RoomKit — SIP + OpenAI Realtime + Anam Avatar.

Accept SIP video calls. OpenAI Realtime handles the conversation
(speech-to-speech), Anam renders the avatar face (passthrough mode).

Architecture:
    SIP phone → audio → OpenAI Realtime (STT → LLM → TTS)
                                ↓
                         TTS audio out
                          ↓          ↓
                   SIP speaker    AnamAvatarProvider.feed_audio()
                   (caller)         ↓
                                 Anam Cloud (lip-sync)
                                    ↓
                              video frames
                                    ↓
                         H.264 encode → SIP video → phone screen

Prerequisites:
    pip install roomkit[anam,sip,video,realtime-openai]

Run with:
    export OPENAI_API_KEY="sk-..."
    export ANAM_API_KEY="your-api-key"
    export ANAM_AVATAR_ID="your-avatar-id"
    export ANAM_VOICE_ID="your-voice-id"
    export ANAM_LLM_ID="your-llm-id"
    uv run python examples/sip_openai_anam_avatar.py

Environment variables:
    OPENAI_API_KEY       OpenAI API key (required)
    OPENAI_MODEL         Realtime model (default: gpt-4o-mini-realtime-preview)
    OPENAI_VOICE         Voice preset (default: alloy)
    ANAM_API_KEY         Anam API key (required)
    ANAM_AVATAR_ID       Avatar from lab.anam.ai (required)
    ANAM_VOICE_ID        Voice from lab.anam.ai (required)
    ANAM_LLM_ID          LLM from lab.anam.ai (required)
    SIP_PORT             SIP listener port (default: 5060)
    RTP_IP               IP for SDP (default: 0.0.0.0)
    DEBUG                Set to 1 for verbose logging

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sip_openai_anam")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit").setLevel(logging.DEBUG)

import concurrent.futures

from roomkit import AnamConfig, VideoPipelineConfig
from roomkit.providers.anam.avatar import AnamAvatarProvider
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.video.backends.sip import SIPVideoBackend
from roomkit.video.pipeline.encoder.pyav import PyAVVideoEncoder
from roomkit.video.pipeline.engine import VideoPipeline
from roomkit.video.pipeline.filter.watermark import WatermarkFilter
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.bridge import resample_pcm


async def main() -> None:
    # --- Validate environment -------------------------------------------------
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anam_key = os.environ.get("ANAM_API_KEY", "")
    if not openai_key or not anam_key:
        logger.error("Set OPENAI_API_KEY and ANAM_API_KEY")
        return

    avatar_id = os.environ.get("ANAM_AVATAR_ID")
    voice_id = os.environ.get("ANAM_VOICE_ID")
    llm_id = os.environ.get("ANAM_LLM_ID")
    if not (avatar_id and voice_id and llm_id):
        logger.error("Set ANAM_AVATAR_ID + ANAM_VOICE_ID + ANAM_LLM_ID")
        return

    # --- SIP backend ----------------------------------------------------------
    sip = SIPVideoBackend(
        local_sip_addr=("0.0.0.0", int(os.environ.get("SIP_PORT", "5060"))),
        local_rtp_ip=os.environ.get("RTP_IP", "0.0.0.0"),
        rtp_port_start=int(os.environ.get("RTP_PORT_START", "10000")),
        supported_video_codecs=["H264"],
    )

    # --- OpenAI Realtime (speech-to-speech) -----------------------------------
    openai_provider = OpenAIRealtimeProvider(
        api_key=openai_key,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini-realtime-preview"),
    )

    # --- Anam avatar (passthrough — lip-sync only) ----------------------------
    avatar = AnamAvatarProvider(
        AnamConfig(
            api_key=anam_key,
            avatar_id=avatar_id,
            voice_id=voice_id,
            llm_id=llm_id,
            enable_audio_passthrough=True,
        ),
        audio_sample_rate=24000,
    )

    # --- H.264 encoder for SIP video ------------------------------------------
    encoder = PyAVVideoEncoder(fps=25, bitrate=3_000_000, preset="medium")
    encode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # --- Per-call state -------------------------------------------------------
    calls: dict[str, dict[str, Any]] = {}

    # --- Wire OpenAI audio → SIP speaker + Anam avatar ------------------------
    def on_openai_audio(session: VoiceSession, audio: bytes) -> None:
        state = calls.get(session.id)
        if state is None:
            return

        # Send audio to SIP caller (resample 24kHz → SIP codec rate)
        sip_session = state["sip_session"]
        codec_rate = sip_session.metadata.get("codec_sample_rate", 16000)
        resampled = resample_pcm(audio, 24000, codec_rate)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(sip.send_audio(sip_session, resampled))

        # Feed audio to Anam for lip-sync (already 24kHz from OpenAI)
        avatar.feed_audio(audio)

    openai_provider.on_audio(on_openai_audio)

    # --- Wire OpenAI response end → Anam end_turn ----------------------------
    def on_openai_response_end(session: VoiceSession) -> None:
        avatar.end_turn()

    openai_provider.on_response_end(on_openai_response_end)

    # --- Wire OpenAI transcription → log --------------------------------------
    def on_transcription(
        session: VoiceSession,
        text: str,
        role: str,
        is_final: bool,
    ) -> None:
        if is_final:
            logger.info("[%s] %s", role.upper(), text)

    openai_provider.on_transcription(on_transcription)

    # --- Wire Anam avatar video → encode → SIP --------------------------------
    video_pipeline_cfg = VideoPipelineConfig(
        filters=[
            WatermarkFilter(
                "RoomKit | OpenAI + Anam | {timestamp}",
                position="bottom-left",
                font_scale=0.4,
            ),
        ],
    )
    video_pipeline = VideoPipeline(video_pipeline_cfg)

    def on_avatar_video(frame: Any) -> None:
        # Run through pipeline (watermark)
        processed = video_pipeline.process_inbound("avatar", frame)
        if processed is None:
            return

        # Encode in thread pool and send to all active SIP sessions
        for state in calls.values():
            sip_session = state["sip_session"]
            vcs = getattr(sip, "_video_call_sessions", {}).get(sip_session.id)
            if vcs is None:
                continue

            nals = encoder.encode(processed)
            if not nals:
                continue
            is_key = any((nal[0] & 0x1F) == 5 for nal in nals if nal)
            rtp = getattr(vcs, "_session", None)
            if rtp is not None and hasattr(rtp, "send_frame_auto"):
                rtp.send_frame_auto(nals, is_key)
            else:
                state["frame_seq"] = state.get("frame_seq", 0) + 1
                vcs.send_frame(nals, state["frame_seq"] * 3600, is_key)

    avatar.on_video(on_avatar_video)

    # --- Wire SIP audio → OpenAI ----------------------------------------------
    def on_sip_audio(session: VoiceSession, audio: Any) -> None:
        state = calls.get(session.id)
        if state is None:
            return
        raw = audio.data if hasattr(audio, "data") else audio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(openai_provider.send_audio(state["openai_session"], raw))

    sip.on_audio_received(on_sip_audio)

    # --- Start Anam avatar (one shared instance for all calls) ----------------
    await avatar.start(b"")
    logger.info("Anam avatar ready (passthrough mode)")

    # --- On SIP INVITE: connect OpenAI ----------------------------------------
    async def on_call(sip_session: VoiceSession) -> None:
        caller = sip_session.metadata.get("caller", "unknown")
        logger.info("SIP call from %s", caller)

        # Create a session for OpenAI
        openai_session = VoiceSession(
            id=sip_session.id,
            room_id=sip_session.id,
            participant_id=caller,
            channel_id="openai-realtime",
            state=VoiceSessionState.CONNECTING,
        )
        calls[sip_session.id] = {
            "sip_session": sip_session,
            "openai_session": openai_session,
        }

        voice = os.environ.get("OPENAI_VOICE", "alloy")
        await openai_provider.connect(
            openai_session,
            system_prompt=(
                "You are a helpful AI assistant on a video call. "
                "Keep responses conversational and concise."
            ),
            voice=voice,
            output_sample_rate=24000,
        )
        logger.info("OpenAI Realtime + Anam avatar active for %s", caller)

    sip.on_call(on_call)

    # --- On SIP BYE: disconnect OpenAI ----------------------------------------
    def on_bye(session: object) -> None:
        sid = getattr(session, "id", None)
        if sid is None:
            return
        state = calls.pop(sid, None)
        if state is None:
            return
        logger.info("Call ended: %s", sid[:8])
        asyncio.create_task(openai_provider.disconnect(state["openai_session"]))

    sip.on_client_disconnected(on_bye)

    # --- Start ----------------------------------------------------------------
    await sip.start()
    logger.info(
        "SIP + OpenAI Realtime + Anam Avatar on 0.0.0.0:%s",
        os.environ.get("SIP_PORT", "5060"),
    )
    logger.info("Call this SIP endpoint with a video phone.")
    logger.info("Press Ctrl+C to stop.\n")

    stop = asyncio.Event()

    def _signal() -> None:
        if stop.is_set():
            raise SystemExit(1)
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal)
    await stop.wait()

    logger.info("Shutting down...")
    await avatar.stop()
    encode_pool.shutdown(wait=False)
    await openai_provider.close()
    await sip.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
