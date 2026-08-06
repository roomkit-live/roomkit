#!/usr/bin/env python3
"""Play a pre-recorded WAV prompt to a caller while the AI stays silent.

``VoiceChannel.play()`` pushes a WAV straight to the transport -- no TTS,
no LLM round-trip.  Combined with ``kit.mute()`` it gives the classic
telephony move: an announcement the AI neither talks over nor answers.

Two mutes, two different effects (both demonstrated below):

    kit.mute(room, "ai")      the AI's brain still runs (memory, context),
                              only its voice is dropped
    kit.mute(room, "voice")   inbound audio is dropped before the VAD, so
                              the caller cannot barge in and the AI hears
                              nothing that was said during the prompt

``play()`` itself is never blocked by either mute: it writes to the voice
backend directly instead of going through the broadcast pipeline.

Modes:
    default       in-memory demo with MockVoiceBackend -- runs anywhere
    SIP_MODE=1    accept real SIP INVITEs (requires ``pip install roomkit[sip]``)

Run with:
    uv run python examples/voice_sip_play_prompt.py
    SIP_MODE=1 uv run python examples/voice_sip_play_prompt.py

Environment variables (all optional):
    PROMPT_WAV      16-bit mono WAV to play (default: a generated 440 Hz tone)
    SIP_CODEC_RATE  transport rate -- 8000 for G.711, 16000 for G.722 (default 8000)
    SIP_HOST        SIP listen address, SIP_MODE=1 only (default 0.0.0.0)
    SIP_PORT        SIP listen port, SIP_MODE=1 only    (default 5060)
    RTP_IP          RTP bind address, SIP_MODE=1 only   (default 0.0.0.0)
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import struct
import sys
import tempfile
import wave
from collections.abc import AsyncIterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import env_bool, run_until_stopped, setup_console, setup_logging

logger = setup_logging("voice_sip_play_prompt")

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.session_event import SessionStartedEvent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceCapability, VoiceSession
from roomkit.voice.pipeline import (
    AudioFormat,
    AudioPipelineConfig,
    AudioPipelineContract,
    MockVADProvider,
    VADEvent,
    VADEventType,
)
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOM_ID = "prompt-demo"
VOICE_CHANNEL = "voice"
AI_CHANNEL = "ai"

# SIP negotiates G.711 (8 kHz) or G.722 (16 kHz).  The pipeline contract
# tells RoomKit to resample outbound audio to that rate -- which is what
# lets play() accept a WAV recorded at any sample rate.
CODEC_RATE = int(os.environ.get("SIP_CODEC_RATE", "8000"))
INTERNAL_RATE = 16000

# The prompt is deliberately NOT at the codec rate: the pipeline resamples it.
PROMPT_RATE = 22050


# ---------------------------------------------------------------------------
# Prompt asset
# ---------------------------------------------------------------------------


def prompt_wav_path() -> Path:
    """Return the WAV to play -- ``PROMPT_WAV`` or a generated 440 Hz tone.

    play() requires 16-bit PCM mono, uncompressed.  Convert anything else
    up front::

        ffmpeg -i prompt.mp3 -ac 1 -ar 8000 -acodec pcm_s16le prompt.wav
    """
    configured = os.environ.get("PROMPT_WAV")
    if configured:
        return Path(configured)

    path = Path(tempfile.gettempdir()) / f"roomkit_prompt_{PROMPT_RATE}.wav"
    if not path.exists():
        frames = bytearray()
        for i in range(PROMPT_RATE):  # 1 second
            value = int(12000 * math.sin(2 * math.pi * 440 * i / PROMPT_RATE))
            frames += struct.pack("<h", value)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(PROMPT_RATE)
            wf.writeframes(bytes(frames))
        logger.info("Generated prompt: %s (%d Hz tone)", path, PROMPT_RATE)
    return path


# ---------------------------------------------------------------------------
# The pattern: silence the AI around a prompt
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def ai_silenced(
    kit: RoomKit,
    room_id: str,
    *,
    deafen: bool = True,
) -> AsyncIterator[None]:
    """Hold the AI quiet for the duration of the block.

    Args:
        kit: The framework instance.
        room_id: Room whose channels are muted.
        deafen: Also mute the voice channel, so inbound audio is dropped
            before the VAD -- the caller cannot barge in and nothing said
            during the prompt reaches STT or the AI.  Set ``False`` for
            hold music, where the caller should still be heard.
    """
    await kit.mute(room_id, AI_CHANNEL)
    if deafen:
        await kit.mute(room_id, VOICE_CHANNEL)
    try:
        yield
    finally:
        if deafen:
            await kit.unmute(room_id, VOICE_CHANNEL)
        await kit.unmute(room_id, AI_CHANNEL)


async def play_prompt(
    kit: RoomKit,
    voice: VoiceChannel,
    session: VoiceSession,
    wav: Path,
    *,
    deafen: bool = True,
) -> None:
    """Play *wav* to *session* with the AI silenced.

    ``await play()`` returns when the audio has drained: on SIP the pacer
    clocks it out at real time, so the await lasts the length of the file.
    """
    async with ai_silenced(kit, ROOM_ID, deafen=deafen):
        # Cut any TTS already in flight -- otherwise the two overlap.
        await voice.interrupt(session)
        await voice.play(session, wav, text="[prompt]")


# ---------------------------------------------------------------------------
# Shared wiring
# ---------------------------------------------------------------------------


def build_pipeline(vad: MockVADProvider | None) -> AudioPipelineConfig:
    """Pipeline whose contract pins the transport rate to the SIP codec."""
    return AudioPipelineConfig(
        vad=vad,
        contract=AudioPipelineContract(
            transport_inbound_format=AudioFormat(sample_rate=CODEC_RATE),
            transport_outbound_format=AudioFormat(sample_rate=CODEC_RATE),
            internal_format=AudioFormat(sample_rate=INTERNAL_RATE),
        ),
    )


def register_prompt_hook(kit: RoomKit, voice: VoiceChannel, wav: Path) -> None:
    """Play the welcome prompt as soon as the audio path is live."""

    @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_session_started(event: SessionStartedEvent, ctx: object) -> None:
        if event.session is None:
            return
        logger.info("Session live -- playing welcome prompt")
        await play_prompt(kit, voice, event.session, wav)
        logger.info("Welcome prompt done -- the AI is listening again")


# ---------------------------------------------------------------------------
# Mock mode -- the four states, verifiable without any infrastructure
# ---------------------------------------------------------------------------


async def run_mock(wav: Path) -> None:
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    # Two speech turns: SPEECH_START, silence, SPEECH_END -- twice.
    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"turn-1", duration_ms=900.0),
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"turn-2", duration_ms=900.0),
        ]
    )
    stt = MockSTTProvider(transcripts=["I want to talk to a human.", "Never mind, thanks."])
    tts = MockTTSProvider()

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    voice = VoiceChannel(
        VOICE_CHANNEL,
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=build_pipeline(vad),
    )
    ai = AIChannel(AI_CHANNEL, provider=MockAIProvider(responses=["Sure.", "Of course."]))
    kit.register_channel(voice)
    kit.register_channel(ai)

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, VOICE_CHANNEL)
    await kit.attach_channel(ROOM_ID, AI_CHANNEL, category=ChannelCategory.INTELLIGENCE)

    register_prompt_hook(kit, voice, wav)

    async def caller_speaks(session: VoiceSession) -> None:
        """Feed three frames -- the mock VAD turns them into one utterance."""
        for _ in range(3):
            await backend.simulate_audio_received(
                session,
                AudioFrame(data=b"\x00\x00" * (CODEC_RATE // 50), sample_rate=CODEC_RATE),
            )
        await asyncio.sleep(0.3)

    # --- 1. Welcome prompt, played from ON_SESSION_STARTED -------------------
    session = await kit.join(ROOM_ID, VOICE_CHANNEL, participant_id="caller-1")
    await asyncio.sleep(0.3)  # let the async hook finish

    with wave.open(str(wav), "rb") as wf:
        source_bytes = wf.getnframes() * 2
        source_rate = wf.getframerate()
    sent_bytes = sum(len(a) for _, a in backend.sent_audio)
    print("\n--- 1. Welcome prompt (ON_SESSION_STARTED) ---")
    print(f"  WAV read:        {source_bytes} bytes @ {source_rate} Hz")
    print(f"  Sent to transport: {sent_bytes} bytes @ {CODEC_RATE} Hz (pipeline resampled)")
    print(f"  Transcript shown to the UI: {backend.sent_transcriptions[-1][1]!r}")

    # --- 2. Caller talks over a prompt, AI muted + voice deafened ------------
    print("\n--- 2. Caller talks during the prompt (AI muted, voice deafened) ---")
    streams_before = len(backend.sent_audio)
    async with ai_silenced(kit, ROOM_ID, deafen=True):
        await voice.play(session, wav)
        await caller_speaks(session)
        # Playback stays "live" for ~2 s after the audio drains, so echo is
        # not transcribed as caller speech.  interrupt() ends that window
        # immediately -- here so the next demo turn starts from a clean slate.
        await voice.interrupt(session)
    played = len(backend.sent_audio) - streams_before
    print(f"  Audio streams sent: {played} (mute never blocks play() itself)")
    print(f"  STT calls: {len(stt.calls)} (frames dropped before the VAD)")
    print(f"  TTS calls: {len(tts.calls)} (the AI never spoke)")

    # --- 3. Only the AI is muted: the caller is still heard -------------------
    print("\n--- 3. Caller speaks with only the AI muted ---")
    await kit.mute(ROOM_ID, AI_CHANNEL)
    await caller_speaks(session)
    await kit.unmute(ROOM_ID, AI_CHANNEL)
    events = await kit.store.list_events(ROOM_ID)
    transcribed = [e for e in events if getattr(e.content, "body", None) == stt.transcripts[0]]
    print(f"  STT calls: {len(stt.calls)} (the caller was transcribed)")
    print(f"  Stored in the timeline: {len(transcribed)} event(s)")
    print(f"  TTS calls: {len(tts.calls)} (brain ran, voice suppressed)")

    # --- 4. Nothing muted: the AI answers out loud ---------------------------
    print("\n--- 4. Normal turn, nothing muted ---")
    await caller_speaks(session)
    print(f"  STT calls: {len(stt.calls)}")
    print(f"  TTS calls: {len(tts.calls)} (the AI answered)")

    if console_cleanup:
        await console_cleanup()
    await kit.close()
    print("\nDone.")


# ---------------------------------------------------------------------------
# SIP mode -- the same wiring against a real trunk
# ---------------------------------------------------------------------------


async def run_sip(wav: Path) -> None:
    # Local import: SIP is an optional extra (roomkit[sip]) and mock mode
    # must run without it installed.
    from roomkit.voice.backends.sip import SIPVoiceBackend

    sip_host = os.environ.get("SIP_HOST", "0.0.0.0")  # nosec B104
    sip_port = int(os.environ.get("SIP_PORT", "5060"))
    rtp_ip = os.environ.get("RTP_IP", "0.0.0.0")  # nosec B104

    backend = SIPVoiceBackend(
        local_sip_addr=(sip_host, sip_port),
        local_rtp_ip=rtp_ip,
    )

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # A real deployment swaps the mocks for Deepgram/ElevenLabs/an LLM --
    # the prompt playback path is identical either way.
    voice = VoiceChannel(
        VOICE_CHANNEL,
        stt=MockSTTProvider(),
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=build_pipeline(vad=None),
    )
    ai = AIChannel(AI_CHANNEL, provider=MockAIProvider(responses=["Understood."]))
    kit.register_channel(voice)
    kit.register_channel(ai)

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, VOICE_CHANNEL)
    await kit.attach_channel(ROOM_ID, AI_CHANNEL, category=ChannelCategory.INTELLIGENCE)

    register_prompt_hook(kit, voice, wav)

    @backend.on_call
    async def handle_call(session: VoiceSession) -> None:
        logger.info("Incoming call from %s", session.metadata.get("caller"))
        await kit.join(ROOM_ID, VOICE_CHANNEL, session=session)

    @backend.on_call_disconnected
    async def handle_disconnect(session: VoiceSession) -> None:
        logger.info("Call ended -- session=%s", session.id)
        await kit.leave(session)

    await backend.start()
    logger.info(
        "Listening for INVITEs on %s:%d -- prompt=%s, codec rate=%d Hz",
        sip_host,
        sip_port,
        wav,
        CODEC_RATE,
    )

    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await backend.close()

    await run_until_stopped(kit, cleanup=cleanup)


async def main() -> None:
    wav = prompt_wav_path()
    if not wav.exists():
        print(f"Error: prompt WAV not found: {wav}")
        sys.exit(1)
    if env_bool("SIP_MODE", default=False):
        await run_sip(wav)
    else:
        await run_mock(wav)


if __name__ == "__main__":
    asyncio.run(main())
