"""Bridge a multi-party call with an AI moderator that can speak.

Demonstrates AudioBridge + TTS: human participants are bridged while an
AI moderator can interject with spoken announcements.  Bridge audio and
TTS audio coexist — the AI speaks to participants without disrupting the
bridge between them.

Architecture::

    Caller A ──mic──► Pipeline ──► STT ──► ON_TRANSCRIPTION hook
                          │                       │
                          └──bridge──► Caller B    └──► transcript log
                                                          │
                   AI moderator ──TTS──► Both callers    │
                                                          │
                     (all callers leave) ──────► final transcript

This example uses mock providers so it runs locally without any
external services.

Run with:
    uv run python examples/voice_bridge_with_ai.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import (
    ChannelBinding,
    ChannelType,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    MockVADProvider,
    VADEvent,
    VADEventType,
)
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_bridge_with_ai")

ROOM_ID = "conference-room"


async def main() -> None:
    kit = RoomKit()

    # --- Transcript accumulator -----------------------------------------------
    transcript: list[tuple[str, str]] = []

    # --- Mock backend ---------------------------------------------------------
    backend = MockVoiceBackend()

    # --- Pipeline: VAD for speech detection -----------------------------------
    # 4 VAD events per speaker (SPEECH_START, 2 silent, SPEECH_END) x 2 speakers
    vad = MockVADProvider(
        events=[
            # Alice speaks
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"speech-alice",
                duration_ms=1500.0,
            ),
            # Bob speaks
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"speech-bob",
                duration_ms=1200.0,
            ),
        ]
    )

    # --- STT: mock transcriptions ---
    stt = MockSTTProvider(
        transcripts=[
            "I think we should launch the product next quarter.",
            "Agreed, but we need more user testing first.",
        ]
    )

    # --- TTS: for AI moderator to speak ---
    tts = MockTTSProvider()

    # --- Voice channel with bridge + STT + TTS ---
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad),
        bridge=True,
    )
    kit.register_channel(voice)

    # --- Room ---
    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "voice")

    # --- Hooks: capture live transcriptions ---
    @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_session_started(event, ctx):
        logger.info("Session joined: %s", event.session.participant_id)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        speaker = event.session.participant_id or event.session.id
        transcript.append((speaker, event.text))
        logger.info("[TRANSCRIPT %s] %s", speaker, event.text)
        return HookResult.allow()

    # --- Simulate two callers joining ---
    print("\n=== Participants join the conference ===\n")

    binding = ChannelBinding(room_id=ROOM_ID, channel_id="voice", channel_type=ChannelType.VOICE)

    session_a = await backend.connect(ROOM_ID, "alice", "voice")
    voice.bind_session(session_a, ROOM_ID, binding)
    logger.info("Alice joined (participants: %d)", voice._bridge.get_participant_count(ROOM_ID))

    session_b = await backend.connect(ROOM_ID, "bob", "voice")
    voice.bind_session(session_b, ROOM_ID, binding)
    logger.info("Bob joined (participants: %d)", voice._bridge.get_participant_count(ROOM_ID))

    # --- Conversation: both speak (bridge + STT active) ---
    print("\n=== Conversation in progress ===\n")

    for session, name in [(session_a, "Alice"), (session_b, "Bob")]:
        logger.info("%s is speaking...", name)
        for _ in range(4):
            frame = AudioFrame(data=b"\x00\x80" * 160, sample_rate=16000)
            await backend.simulate_audio_received(session, frame)
        await asyncio.sleep(0.05)

    # Let async hooks and STT complete
    await asyncio.sleep(0.3)

    # --- AI moderator interjects via TTS ---
    # This demonstrates that TTS and bridge coexist: the AI can speak
    # to participants while the bridge continues forwarding audio.
    print("\n=== AI moderator speaks ===\n")

    await voice.say(session_a, "Great discussion! Let me summarize the key points.")
    await voice.say(session_b, "Great discussion! Let me summarize the key points.")
    await asyncio.sleep(0.05)
    logger.info("AI moderator spoke to both participants")

    # --- Verify results ---
    bridge_sends = len(backend.sent_audio)
    alice_received = [sid for sid, _ in backend.sent_audio if sid == session_a.id]
    bob_received = [sid for sid, _ in backend.sent_audio if sid == session_b.id]

    # --- Participants leave ---
    print("\n=== Participants leaving ===\n")

    for session, name in [(session_b, "Bob"), (session_a, "Alice")]:
        voice.unbind_session(session)
        logger.info("%s left", name)

    # --- Display results ---
    print("\n=== Conversation Transcript ===\n")
    for speaker, text in transcript:
        print(f"  {speaker}: {text}")

    print("\n=== Stats ===")
    print("  Participants: 2 (Alice, Bob)")
    print(f"  Transcript entries: {len(transcript)}")
    print(f"  Bridge + TTS audio sends: {bridge_sends}")
    print(f"  Alice received {len(alice_received)} audio frames (bridge + TTS)")
    print(f"  Bob received {len(bob_received)} audio frames (bridge + TTS)")

    await kit.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
