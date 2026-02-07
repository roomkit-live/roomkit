"""RoomKit -- Debug audio recording with WavFileRecorder.

Demonstrates how to capture pipeline audio to WAV files for debugging.
The WavFileRecorder writes raw PCM audio to disk so you can listen to
exactly what the pipeline processes — useful for verifying AEC, denoiser,
and AGC behaviour.

Three channel modes are shown:
  - MIXED:    single mono WAV (inbound + outbound averaged)
  - SEPARATE: two WAV files (inbound + outbound)
  - STEREO:   single stereo WAV (inbound=left, outbound=right)

All mock providers — runs without external dependencies.

Run with:
    uv run python examples/wav_recorder.py
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from roomkit import (
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookTrigger,
    MockAIProvider,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    MockVADProvider,
    RecordingChannelMode,
    RecordingConfig,
    VADEvent,
    VADEventType,
    WavFileRecorder,
)
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("wav_recorder")


def make_audio_frame(value: int = 100, num_samples: int = 160) -> AudioFrame:
    """Create a 16-bit PCM audio frame with a constant sample value."""
    import struct

    data = b"".join(struct.pack("<h", value) for _ in range(num_samples))
    return AudioFrame(data=data, sample_rate=16000, channels=1, sample_width=2)


async def demo_mixed(kit: RoomKit, output_dir: Path) -> None:
    """Record inbound + outbound mixed into a single mono WAV."""
    logger.info("=== MIXED mode ===")

    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"speech", duration_ms=600.0),
        ]
    )
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        recorder=WavFileRecorder(),
        recording_config=RecordingConfig(
            storage=str(output_dir / "mixed"),
            channels=RecordingChannelMode.MIXED,
        ),
    )

    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["Hello"])
    tts = MockTTSProvider()

    voice = VoiceChannel(
        "voice-mixed", stt=stt, tts=tts, backend=backend, pipeline=pipeline_config
    )
    kit.register_channel(voice)

    ai = AIChannel("ai-mixed", provider=MockAIProvider(responses=["Hi there!"]))
    kit.register_channel(ai)

    await kit.create_room(room_id="mixed-demo")
    await kit.attach_channel("mixed-demo", "voice-mixed")
    await kit.attach_channel("mixed-demo", "ai-mixed", category=ChannelCategory.INTELLIGENCE)

    session = await backend.connect("mixed-demo", "user-1", "voice-mixed")
    binding = ChannelBinding(
        room_id="mixed-demo", channel_id="voice-mixed", channel_type=ChannelType.VOICE
    )
    voice.bind_session(session, "mixed-demo", binding)

    # Simulate 3 inbound audio frames
    for _ in range(3):
        await backend.simulate_audio_received(session, make_audio_frame(value=500))

    await asyncio.sleep(0.1)
    voice.unbind_session(session)
    await asyncio.sleep(0.05)

    # Show output files
    for f in sorted((output_dir / "mixed").glob("*.wav")):
        logger.info("  Created: %s (%d bytes)", f.name, f.stat().st_size)


async def demo_separate(kit: RoomKit, output_dir: Path) -> None:
    """Record inbound and outbound into separate WAV files."""
    logger.info("=== SEPARATE mode ===")

    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"speech", duration_ms=600.0),
        ]
    )
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        recorder=WavFileRecorder(),
        recording_config=RecordingConfig(
            storage=str(output_dir / "separate"),
            channels=RecordingChannelMode.SEPARATE,
        ),
    )

    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["Hello"])
    tts = MockTTSProvider()

    voice = VoiceChannel(
        "voice-separate", stt=stt, tts=tts, backend=backend, pipeline=pipeline_config
    )
    kit.register_channel(voice)

    ai = AIChannel("ai-separate", provider=MockAIProvider(responses=["Hi!"]))
    kit.register_channel(ai)

    await kit.create_room(room_id="separate-demo")
    await kit.attach_channel("separate-demo", "voice-separate")
    await kit.attach_channel("separate-demo", "ai-separate", category=ChannelCategory.INTELLIGENCE)

    session = await backend.connect("separate-demo", "user-1", "voice-separate")
    binding = ChannelBinding(
        room_id="separate-demo", channel_id="voice-separate", channel_type=ChannelType.VOICE
    )
    voice.bind_session(session, "separate-demo", binding)

    for _ in range(3):
        await backend.simulate_audio_received(session, make_audio_frame(value=300))

    await asyncio.sleep(0.1)
    voice.unbind_session(session)
    await asyncio.sleep(0.05)

    for f in sorted((output_dir / "separate").glob("*.wav")):
        logger.info("  Created: %s (%d bytes)", f.name, f.stat().st_size)


async def demo_stereo(kit: RoomKit, output_dir: Path) -> None:
    """Record inbound (left) + outbound (right) as a stereo WAV."""
    logger.info("=== STEREO mode ===")

    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"speech", duration_ms=600.0),
        ]
    )
    pipeline_config = AudioPipelineConfig(
        vad=vad,
        recorder=WavFileRecorder(),
        recording_config=RecordingConfig(
            storage=str(output_dir / "stereo"),
            channels=RecordingChannelMode.STEREO,
        ),
    )

    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["Hello"])
    tts = MockTTSProvider()

    voice = VoiceChannel(
        "voice-stereo", stt=stt, tts=tts, backend=backend, pipeline=pipeline_config
    )
    kit.register_channel(voice)

    ai = AIChannel("ai-stereo", provider=MockAIProvider(responses=["Hi!"]))
    kit.register_channel(ai)

    await kit.create_room(room_id="stereo-demo")
    await kit.attach_channel("stereo-demo", "voice-stereo")
    await kit.attach_channel("stereo-demo", "ai-stereo", category=ChannelCategory.INTELLIGENCE)

    session = await backend.connect("stereo-demo", "user-1", "voice-stereo")
    binding = ChannelBinding(
        room_id="stereo-demo", channel_id="voice-stereo", channel_type=ChannelType.VOICE
    )
    voice.bind_session(session, "stereo-demo", binding)

    for _ in range(3):
        await backend.simulate_audio_received(session, make_audio_frame(value=800))

    await asyncio.sleep(0.1)
    voice.unbind_session(session)
    await asyncio.sleep(0.05)

    for f in sorted((output_dir / "stereo").glob("*.wav")):
        logger.info("  Created: %s (%d bytes)", f.name, f.stat().st_size)


async def main() -> None:
    output_dir = Path(tempfile.mkdtemp(prefix="roomkit_wav_"))
    logger.info("Output directory: %s\n", output_dir)

    kit = RoomKit()

    # Hook to log recording lifecycle events
    @kit.hook(
        HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC, name="log_rec_start"
    )
    async def on_recording_started(event, ctx):
        logger.info("[hook] Recording started: %s", event.id)

    @kit.hook(
        HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC, name="log_rec_stop"
    )
    async def on_recording_stopped(event, ctx):
        logger.info(
            "[hook] Recording stopped: %s (%.2fs, %d bytes)",
            event.id,
            event.duration_seconds,
            event.size_bytes,
        )

    await demo_mixed(kit, output_dir)
    print()
    await demo_separate(kit, output_dir)
    print()
    await demo_stereo(kit, output_dir)

    await kit.close()

    print(f"\nAll recordings saved to: {output_dir}")
    print("Open them in Audacity or any audio player to inspect.")


if __name__ == "__main__":
    asyncio.run(main())
