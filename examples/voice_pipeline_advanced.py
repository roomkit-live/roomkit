"""RoomKit -- Advanced audio pipeline features.

Demonstrates features beyond the basic pipeline example:
- Audio recording with ON_RECORDING_STARTED / ON_RECORDING_STOPPED hooks
- Turn detection with ON_TURN_COMPLETE / ON_TURN_INCOMPLETE hooks
- Semantic interruption with BackchannelDetector and ON_BACKCHANNEL hook
- Capability-aware pipeline (NATIVE_AEC skips AEC stage)

All mock providers — runs without external dependencies.

Run with:
    uv run python examples/voice_pipeline_advanced.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import (
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookTrigger,
    MockAIProvider,
    RoomKit,
    TextContent,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceCapability
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    BackchannelDecision,
    MockAECProvider,
    MockAGCProvider,
    MockAudioRecorder,
    MockBackchannelDetector,
    MockDenoiserProvider,
    MockTurnDetector,
    MockVADProvider,
    RecordingConfig,
    TurnDecision,
    VADConfig,
    VADEvent,
    VADEventType,
)
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    kit = RoomKit()

    # --- Pipeline providers ---------------------------------------------------

    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.95),
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"\x00\x00" * 160,
                duration_ms=800.0,
            ),
            # Second utterance
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.90),
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"\x00\x00" * 160,
                duration_ms=600.0,
            ),
        ]
    )

    denoiser = MockDenoiserProvider()

    # AEC is configured but the backend declares NATIVE_AEC — so the
    # pipeline will skip the AEC stage automatically.
    aec = MockAECProvider()
    agc = MockAGCProvider()

    # --- Recording (auto-starts when session becomes active) ------------------
    recorder = MockAudioRecorder()
    recording_config = RecordingConfig(storage="/tmp/recordings")

    # --- Turn detection -------------------------------------------------------
    # First utterance: incomplete (user hasn't finished).
    # Second utterance: complete (ready to route to AI).
    turn_detector = MockTurnDetector(
        decisions=[
            TurnDecision(
                is_complete=False,
                confidence=0.4,
                reason="waiting for more",
                suggested_wait_ms=500.0,
            ),
            TurnDecision(is_complete=True, confidence=0.95, reason="sentence complete"),
        ]
    )

    # --- Semantic interruption ------------------------------------------------
    # BackchannelDetector distinguishes "uh-huh" from real interruptions.
    backchannel_detector = MockBackchannelDetector(
        decisions=[
            BackchannelDecision(is_backchannel=True, confidence=0.9),
        ]
    )

    # --- Pipeline config ------------------------------------------------------
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        denoiser=denoiser,
        aec=aec,
        agc=agc,
        recorder=recorder,
        recording_config=recording_config,
        turn_detector=turn_detector,
        backchannel_detector=backchannel_detector,
        vad_config=VADConfig(silence_threshold_ms=500),
    )

    # --- Backend with NATIVE_AEC capability -----------------------------------
    # This causes the pipeline to skip the AEC stage.
    backend = MockVoiceBackend()
    backend._capabilities = VoiceCapability.INTERRUPTION | VoiceCapability.NATIVE_AEC

    stt = MockSTTProvider(transcripts=["I need to", "reschedule my appointment"])
    tts = MockTTSProvider()

    # --- Voice channel with SEMANTIC interruption -----------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline_config,
        interruption=InterruptionConfig(strategy=InterruptionStrategy.SEMANTIC),
    )
    kit.register_channel(voice)

    ai = AIChannel("ai", provider=MockAIProvider(responses=["Sure, let me help you reschedule."]))
    kit.register_channel(ai)

    await kit.create_room(room_id="demo")
    await kit.attach_channel("demo", "voice")
    await kit.attach_channel("demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks ----------------------------------------------------------------

    @kit.hook(
        HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC, name="log_rec_start"
    )
    async def on_recording_started(event, ctx):
        print(f"[hook] Recording started: {event.id}")

    @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC, name="log_rec_stop")
    async def on_recording_stopped(event, ctx):
        print(f"[hook] Recording stopped: {event.id} ({event.duration_seconds}s)")

    @kit.hook(
        HookTrigger.ON_TURN_COMPLETE, execution=HookExecution.ASYNC, name="log_turn_complete"
    )
    async def on_turn_complete(event, ctx):
        print(f"[hook] Turn complete: '{event.text}' (confidence={event.confidence})")

    @kit.hook(
        HookTrigger.ON_TURN_INCOMPLETE, execution=HookExecution.ASYNC, name="log_turn_incomplete"
    )
    async def on_turn_incomplete(event, ctx):
        print(f"[hook] Turn incomplete: '{event.text}' (confidence={event.confidence})")

    @kit.hook(HookTrigger.ON_BACKCHANNEL, execution=HookExecution.ASYNC, name="log_backchannel")
    async def on_backchannel(event, ctx):
        print(f"[hook] Backchannel detected: '{event.text}'")

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, name="log_transcription")
    async def on_transcription(text, ctx):
        from roomkit import HookResult

        print(f"[hook] Transcription: {text}")
        return HookResult.allow()

    # --- Simulate session -----------------------------------------------------
    session = await backend.connect("demo", "user-1", "voice")
    binding = ChannelBinding(room_id="demo", channel_id="voice", channel_type=ChannelType.VOICE)
    voice.bind_session(session, "demo", binding)

    # Send 6 frames: two 3-frame utterances
    for _i in range(6):
        frame = AudioFrame(data=b"\x00\x00" * 160, sample_rate=16000)
        await backend.simulate_audio_received(session, frame)

    await asyncio.sleep(0.2)

    # --- Inspect results ------------------------------------------------------
    # AEC should have 0 frames processed (skipped due to NATIVE_AEC)
    print(f"\nAEC processed {len(aec.frames)} frames (expected 0 — NATIVE_AEC)")
    print(f"AGC processed {len(agc.frames)} frames")
    print(f"Denoiser processed {len(denoiser.frames)} frames")
    print(f"Recorder started {len(recorder.started)} recording(s)")
    print(f"Turn detector evaluated {len(turn_detector.evaluations)} time(s)")
    print(f"STT transcribed {len(stt.calls)} utterance(s)")

    events = await kit.store.list_events("demo")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        body = ev.content.body if isinstance(ev.content, TextContent) else str(ev.content)
        print(f"  [{ev.source.channel_id}] {body}")

    # Clean up
    voice.unbind_session(session)
    await asyncio.sleep(0.05)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
