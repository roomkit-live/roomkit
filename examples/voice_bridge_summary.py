"""Bridge a multi-party call and summarize the conversation with AI.

Demonstrates the AudioBridge in conference mode with live transcription.
All participants' speech is transcribed in real time and collected into
a transcript log.  When all participants leave, the accumulated
transcript is sent to an AI channel which generates a meeting summary.

Architecture::

    Caller A ──mic──► Pipeline ──► STT ──► ON_TRANSCRIPTION hook
                          │                       │
                          └──bridge──► Caller B    └──► transcript log
                                                          │
                        (all callers leave) ───────────► AI summary

This example uses mock providers so it runs locally without any
external services.

Run with:
    uv run python examples/voice_bridge_summary.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import (
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    MockAIProvider,
    RoomKit,
    TextContent,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
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
logger = logging.getLogger("voice_bridge_summary")

ROOM_ID = "conference-room"


async def main() -> None:
    kit = RoomKit()

    # --- Transcript accumulator -----------------------------------------------
    # Collects (speaker, text) tuples from ON_TRANSCRIPTION hooks.
    transcript: list[tuple[str, str]] = []

    # --- Mock backend ---------------------------------------------------------
    backend = MockVoiceBackend()

    # --- Pipeline: VAD for speech detection -----------------------------------
    # The VoiceChannel shares one pipeline across all sessions.  We need
    # 4 VAD events per speaker (SPEECH_START, 2 silent, SPEECH_END) × 3.
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
            # Charlie speaks
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"speech-charlie",
                duration_ms=1800.0,
            ),
        ]
    )
    pipeline = AudioPipelineConfig(vad=vad)

    # --- STT: each call to transcribe returns the next line -------------------
    stt = MockSTTProvider(
        transcripts=[
            "Hi everyone, let's discuss the Q3 roadmap.",
            "I think we should prioritize the mobile app redesign.",
            "Agreed, and we need to finalize the API migration timeline.",
        ]
    )

    # --- TTS: not used during bridge-only mode, but VoiceChannel needs it -----
    tts = MockTTSProvider()

    # --- Voice channel with bridge (no AI during the call) --------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline,
        bridge=True,
    )
    kit.register_channel(voice)

    # --- Room (voice only — AI added after the call) --------------------------
    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "voice")

    # --- Hooks: capture live transcriptions -----------------------------------
    @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_session_started(event, ctx):
        logger.info("Session joined: %s", event.session.participant_id)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        speaker = event.session.participant_id or event.session.id
        transcript.append((speaker, event.text))
        logger.info("[TRANSCRIPT %s] %s", speaker, event.text)
        return HookResult.allow()

    # --- Simulate three callers joining the bridge ----------------------------
    print("\n=== Three participants join the conference ===\n")

    binding = ChannelBinding(room_id=ROOM_ID, channel_id="voice", channel_type=ChannelType.VOICE)

    session_a = await backend.connect(ROOM_ID, "alice", "voice")
    voice.bind_session(session_a, ROOM_ID, binding)
    count = voice._bridge.get_participant_count(ROOM_ID)
    logger.info("Alice joined (participants: %d)", count)

    session_b = await backend.connect(ROOM_ID, "bob", "voice")
    voice.bind_session(session_b, ROOM_ID, binding)
    count = voice._bridge.get_participant_count(ROOM_ID)
    logger.info("Bob joined (participants: %d)", count)

    session_c = await backend.connect(ROOM_ID, "charlie", "voice")
    voice.bind_session(session_c, ROOM_ID, binding)
    count = voice._bridge.get_participant_count(ROOM_ID)
    logger.info("Charlie joined (participants: %d)", count)

    # --- Simulate each participant speaking -----------------------------------
    # Each sends 4 audio frames: the mock VAD fires SPEECH_START on the first,
    # nothing on the next two, then SPEECH_END on the fourth — triggering STT.
    print("\n=== Conversation in progress ===\n")

    for session, name in [
        (session_a, "Alice"),
        (session_b, "Bob"),
        (session_c, "Charlie"),
    ]:
        logger.info("%s is speaking...", name)
        for _ in range(4):
            frame = AudioFrame(data=b"\x00\x80" * 160, sample_rate=16000)
            await backend.simulate_audio_received(session, frame)
        # Small pause between speakers
        await asyncio.sleep(0.05)

    # Let async hooks complete
    await asyncio.sleep(0.1)

    # --- Participants leave ---------------------------------------------------
    print("\n=== Participants leaving ===\n")

    for session, name in [
        (session_c, "Charlie"),
        (session_b, "Bob"),
        (session_a, "Alice"),
    ]:
        voice.unbind_session(session)
        logger.info("%s left", name)

    # --- Generate AI summary from accumulated transcript ----------------------
    print("\n=== Generating meeting summary ===\n")

    transcript_text = "\n".join(f"{speaker}: {text}" for speaker, text in transcript)
    logger.info("Full transcript:\n%s", transcript_text)

    # Now attach the AI channel and send the transcript for summarization
    ai = AIChannel(
        "ai-summarizer",
        provider=MockAIProvider(
            responses=[
                (
                    "Meeting Summary\n\n"
                    "Participants discussed the Q3 roadmap.\n"
                    "Key decisions:\n"
                    "1. Prioritize the mobile app redesign\n"
                    "2. Finalize the API migration timeline\n\n"
                    "Action items: schedule follow-up to set deadlines."
                ),
            ]
        ),
        system_prompt=(
            "You are a meeting assistant. Given a conversation transcript, "
            "produce a concise summary with key decisions and action items."
        ),
    )
    kit.register_channel(ai)
    await kit.attach_channel(ROOM_ID, "ai-summarizer", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(
            channel_id="voice",
            sender_id="system",
            content=TextContent(
                body=("Please summarize this meeting transcript:\n\n" + transcript_text)
            ),
            room_id=ROOM_ID,
        )
    )

    # Let AI respond
    await asyncio.sleep(0.1)

    # --- Display results ------------------------------------------------------
    events = await kit.store.list_events(ROOM_ID)
    ai_events = [
        e
        for e in events
        if isinstance(e.content, TextContent) and e.source.channel_id == "ai-summarizer"
    ]

    if ai_events:
        print("=== AI Meeting Summary ===\n")
        print(ai_events[-1].content.body)
    else:
        print("(No AI summary generated)")

    print("\n=== Conversation stats ===")
    print("  Participants: 3 (Alice, Bob, Charlie)")
    print(f"  Transcript entries: {len(transcript)}")
    print(f"  Total room events: {len(events)}")

    await kit.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
