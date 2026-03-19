"""Continuous CPU profiling with Pyroscope.

Demonstrates how to use RoomKit's PyroscopeProfiler to profile a voice
application and attribute CPU samples to individual rooms and sessions.

Requirements:
    pip install 'roomkit[pyroscope]'

Start a local Pyroscope server first:
    docker run -p 4040:4040 grafana/pyroscope

Run with:
    uv run python examples/telemetry_pyroscope.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.telemetry.pyroscope import PyroscopeProfiler
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline.vad.mock import MockVADProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
logger = logging.getLogger(__name__)

ROOM_ID = "demo-room"

# ---------------------------------------------------------------------------
# Pyroscope profiler
# ---------------------------------------------------------------------------

profiler = PyroscopeProfiler(
    application_name="roomkit-demo",
    server_address="http://localhost:4040",
    tags={"env": "development"},
)

# For Grafana Cloud, use:
# profiler = PyroscopeProfiler(
#     application_name="roomkit-demo",
#     server_address="https://profiles-prod-001.grafana.net",
#     basic_auth_username="<instance-id>",
#     basic_auth_password="<api-token>",
#     tenant_id="<tenant-id>",
# )


async def main() -> None:
    profiler.start()

    # ---------------------------------------------------------------------------
    # Framework
    # ---------------------------------------------------------------------------

    kit = RoomKit()

    backend = MockVoiceBackend()
    vad = MockVADProvider(events=[])
    stt = MockSTTProvider(responses=["hello world"])
    tts = MockTTSProvider()

    voice = VoiceChannel("voice", backend=backend, stt=stt, tts=tts, vad=vad)
    kit.register_channel(voice)

    # ---------------------------------------------------------------------------
    # Hook: profile each transcription with session tags
    # ---------------------------------------------------------------------------

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def on_transcription(event, ctx) -> HookResult:
        # Tag this code block so Pyroscope attributes its CPU to this session
        with profiler.tag_session(
            room_id=event.session.room_id,
            session_id=event.session.id,
            backend="mock",
        ):
            logger.info("[STT] %s", event.text)
        return HookResult.allow()

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "voice")

    session = await backend.create_session(ROOM_ID, "user-1", "voice")
    logger.info("Session started: %s", session.id)

    # Simulate some audio frames
    for _ in range(50):
        backend.simulate_audio_received(session)
        await asyncio.sleep(0.02)

    logger.info("Done — check Pyroscope UI at http://localhost:4040")
    profiler.stop()


if __name__ == "__main__":
    asyncio.run(main())
