"""RoomKit -- Voice channel with audio processing pipeline.

Demonstrates how to set up a VoiceChannel with the AudioPipeline
for voice activity detection (VAD), denoising, speaker diarization,
echo cancellation (AEC), gain control (AGC), and DTMF detection.

Audio flow (inbound):

    Backend → [Resampler] → [Recorder] → [AEC] → [AGC] → [Denoiser]
    → VAD → [Diarization] + [DTMF] → STT → Room

This example uses mock providers so it runs without any external
dependencies.

Run with:
    uv run python examples/voice_pipeline.py
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
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    DiarizationResult,
    DTMFEvent,
    MockAECProvider,
    MockAGCProvider,
    MockDenoiserProvider,
    MockDiarizationProvider,
    MockDTMFDetector,
    MockVADProvider,
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
    # The VAD provider detects speech start/end in the audio stream.
    # Here we pre-configure a mock sequence: SPEECH_START, then SPEECH_END
    # with accumulated audio bytes.
    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.95),
            None,  # intermediate frame, no event
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"accumulated-speech-audio",
                duration_ms=1200.0,
            ),
        ]
    )

    # Optional: denoiser cleans up audio before VAD processes it.
    denoiser = MockDenoiserProvider()

    # Optional: diarization identifies which speaker is talking.
    diarizer = MockDiarizationProvider(
        results=[
            DiarizationResult(speaker_id="speaker_0", confidence=0.92, is_new_speaker=True),
            DiarizationResult(speaker_id="speaker_0", confidence=0.95, is_new_speaker=False),
            DiarizationResult(speaker_id="speaker_1", confidence=0.88, is_new_speaker=True),
        ]
    )

    # Optional: AEC removes echo from speaker playback in the mic signal.
    aec = MockAECProvider()

    # Optional: AGC normalises audio levels.
    agc = MockAGCProvider()

    # Optional: DTMF detector recognises telephone keypad tones.
    dtmf = MockDTMFDetector(
        events=[
            None,
            DTMFEvent(digit="5", duration_ms=120.0, confidence=0.99),
            None,
        ]
    )

    # --- Pipeline config ------------------------------------------------------
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        denoiser=denoiser,
        diarization=diarizer,
        aec=aec,
        agc=agc,
        dtmf=dtmf,
        vad_config=VADConfig(silence_threshold_ms=500, min_speech_duration_ms=250),
    )

    # --- Backend + STT + TTS --------------------------------------------------
    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["Hello, how can I help you?"])
    tts = MockTTSProvider()

    # --- Voice channel --------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline_config,
        # Use CONFIRMED interruption: only interrupt after 300 ms of speech
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.CONFIRMED,
            min_speech_ms=300,
        ),
    )
    kit.register_channel(voice)

    # --- AI channel (responds to transcribed speech) --------------------------
    ai = AIChannel("ai", provider=MockAIProvider(responses=["I can help with that!"]))
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="demo")
    await kit.attach_channel("demo", "voice")
    await kit.attach_channel("demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks ----------------------------------------------------------------
    # React to pipeline events via the hook system.
    # Voice pipeline hooks fire asynchronously (execution=ASYNC).
    # ON_TRANSCRIPTION is a sync hook that can modify/block the text.

    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC, name="log_speech_start")
    async def on_speech_start(session, ctx):
        print(f"[hook] Speech started: session={session.id}")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC, name="log_speech_end")
    async def on_speech_end(session, ctx):
        print(f"[hook] Speech ended: session={session.id}")

    @kit.hook(HookTrigger.ON_SPEAKER_CHANGE, execution=HookExecution.ASYNC, name="log_speaker")
    async def on_speaker_change(event, ctx):
        print(f"[hook] Speaker changed to {event.speaker_id} (confidence={event.confidence})")

    @kit.hook(HookTrigger.ON_DTMF, execution=HookExecution.ASYNC, name="log_dtmf")
    async def on_dtmf(event, ctx):
        print(f"[hook] DTMF detected: digit={event.digit}")

    @kit.hook(HookTrigger.ON_BARGE_IN, execution=HookExecution.ASYNC, name="log_barge_in")
    async def on_barge_in(event, ctx):
        print(
            f"[hook] Barge-in: interrupted '{event.interrupted_text}'"
            f" at {event.audio_position_ms}ms"
        )

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, name="log_transcription")
    async def on_transcription(text, ctx):
        from roomkit import HookResult

        print(f"[hook] Transcription: {text}")
        return HookResult.allow()

    # --- Simulate a voice session ---------------------------------------------
    session = await backend.connect("demo", "user-1", "voice")

    # Bind session to room (required for routing)
    binding = ChannelBinding(room_id="demo", channel_id="voice", channel_type=ChannelType.VOICE)
    voice.bind_session(session, "demo", binding)

    # Simulate 3 audio frames arriving from the client.
    # The pipeline processes each frame through the full inbound chain:
    #   [AEC] -> [AGC] -> Denoiser -> VAD -> Diarization + DTMF
    # The mock VAD returns SPEECH_START on frame 1, nothing on frame 2,
    # and SPEECH_END on frame 3 -- triggering STT transcription.
    for i in range(3):
        frame = AudioFrame(data=f"audio-chunk-{i}".encode(), sample_rate=16000)
        await backend.simulate_audio_received(session, frame)

    # Give async hooks time to fire
    await asyncio.sleep(0.1)

    # --- Inspect results ------------------------------------------------------
    print(f"\nAEC processed {len(aec.frames)} frames")
    print(f"AGC processed {len(agc.frames)} frames")
    print(f"Denoiser processed {len(denoiser.frames)} frames")
    print(f"VAD processed {len(vad.frames)} frames")
    print(f"Diarizer processed {len(diarizer.frames)} frames")
    print(f"DTMF processed {len(dtmf.frames)} frames")
    print(f"STT transcribed {len(stt.calls)} utterance(s)")
    print(f"TTS synthesized {len(tts.calls)} response(s)")

    events = await kit.store.list_events("demo")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        body = ev.content.body if isinstance(ev.content, TextContent) else str(ev.content)
        print(f"  [{ev.source.channel_id}] {body}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
