"""RoomKit — Capture that outlives a session: never lose the start of a phrase.

A ``VoiceBackend`` takes the microphone when a session starts and gives it back
when the session ends.  Anything that must listen *before* a session exists —
a wake word, a level meter — therefore has to hand the device over at exactly
the wrong moment: while the person is still talking.

``LocalMicSource`` owns the device instead.  The detector subscribes to it, the
session subscribes to it too, and a mark taken when speech began lets the
session replay what was said before it existed.  See RFC Section 12.12.

**This example is deliberately not a wake word.** The trigger here is "the
first speech segment opens a session", which is enough to show the primitive
without dragging in a detection stack.  A real detector — acoustic
fingerprinting rather than transcription, which mangles trigger phrases — lives
in the application, not in roomkit.

What to look for: say a whole sentence in one breath, without pausing after the
first few words.  The assistant answers the *question*, instead of asking you
to repeat it, because the words you spoke before the session opened were
replayed into it.

Requirements:
    pip install roomkit[realtime-gemini,local-audio]

Run with:
    GOOGLE_API_KEY=... uv run python examples/shared_mic_capture.py

Environment variables:
    GOOGLE_API_KEY      (required) Google API key
    GEMINI_MODEL        Model name (default: gemini-3.1-flash-live-preview)
    GEMINI_VOICE        Voice preset (default: Aoede)
    BACKLOG_SECONDS     Ring size retained for replay (default: 10)
    ENERGY_THRESHOLD    RMS speech threshold, raise it in a noisy room
                        (default: 300)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import run_until_stopped, setup_logging

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.capture import CaptureMark, LocalMicSource
from roomkit.voice.pipeline.vad.base import VADEventType
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider

logger = setup_logging("shared_mic_capture")

SAMPLE_RATE = 24000  # Gemini speaks at 24 kHz; one rate for the whole chain
BLOCK_MS = 20


class SpeechTrigger:
    """Opens a session on the first speech segment, replaying it from its start.

    Stands in for a wake-word detector.  The shape is the one a real detector
    uses: mark at SPEECH_START, decide at SPEECH_END, hand the mark to the
    session.
    """

    def __init__(self, mic: LocalMicSource, channel: RealtimeVoiceChannel) -> None:
        self._mic = mic
        self._channel = channel
        self._vad = EnergyVADProvider(
            energy_threshold=float(os.environ.get("ENERGY_THRESHOLD", "300")),
            silence_threshold_ms=600,
        )
        self._frames: asyncio.Queue[AudioFrame] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._mark: CaptureMark | None = None
        self._session = None
        self._armed = True

    # -- capture thread -----------------------------------------------------

    def enqueue(self, frame: AudioFrame) -> None:
        """Subscriber callback.  Runs on the capture thread — enqueue only.

        Anything heavier here (a VAD, an ONNX encoder) starves the device and
        degrades capture for every other subscriber.  See RFC Section 12.12.
        """
        self._loop.call_soon_threadsafe(self._frames.put_nowait, frame)

    # -- event loop ---------------------------------------------------------

    async def run(self) -> None:
        while True:
            frame = await self._frames.get()
            if not self._armed:
                continue
            event = self._vad.process(frame, "trigger")
            if event is None:
                continue
            if event.type is VADEventType.SPEECH_START:
                self._mark = self._mic.mark()
                logger.info("Speech started — marked the backlog here")
            elif event.type is VADEventType.SPEECH_END:
                await self._open_session(event.duration_ms or 0.0)

    async def _open_session(self, duration_ms: float) -> None:
        self._armed = False  # one shot: this example opens a single session
        logger.info("Speech ended after %.0fms — opening the session", duration_ms)

        # A real detector detaches here: source frames are pre-AEC, so a
        # subscriber left attached would hear the assistant's own voice.
        self._session = await self._channel.start_session(
            "capture-demo",
            "local-user",
            connection=None,
            metadata={"capture_since": self._mark},
        )

        expected = int(duration_ms / 1000 * SAMPLE_RATE) * 2
        logger.info(
            "Session live. Backlog replayed into it covers roughly %d bytes of the "
            "%.0fms you just spoke — that is the part a per-session device would "
            "have lost.",
            expected,
            duration_ms,
        )

    async def close(self) -> None:
        if self._session is not None:
            await self._channel.end_session(self._session)


async def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY to run this example.")
        print("  GOOGLE_API_KEY=... uv run python examples/shared_mic_capture.py")
        return

    # --- The microphone, owned by nobody in particular ---------------------
    mic = LocalMicSource(
        sample_rate=SAMPLE_RATE,
        block_duration_ms=BLOCK_MS,
        backlog_seconds=float(os.environ.get("BACKLOG_SECONDS", "10")),
    )
    mic.start()
    logger.info("Microphone open, no session in sight. Say a full sentence.")

    kit = RoomKit()
    provider = GeminiLiveProvider(
        api_key=api_key,
        model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview"),
    )

    # The backend subscribes to the shared source instead of opening the
    # device.  It still owns everything per-session: mute, gating, AEC.
    transport = LocalAudioBackend(
        source=mic,
        output_sample_rate=SAMPLE_RATE,
        mute_mic_during_playback=True,
    )

    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
        system_prompt=(
            "You are a voice assistant. Answer what the user asked. If their "
            "question arrived incomplete, say so plainly."
        ),
        voice=os.environ.get("GEMINI_VOICE", "Aoede"),
        input_sample_rate=SAMPLE_RATE,
    )
    kit.register_channel(channel)

    await kit.create_room(room_id="capture-demo")
    await kit.attach_channel("capture-demo", "voice")

    trigger = SpeechTrigger(mic, channel)
    detector = mic.subscribe(trigger.enqueue, name="speech-trigger")
    logger.info("Listening (subscriber %s). Press Ctrl+C to stop.\n", detector.name)

    task = asyncio.create_task(trigger.run())

    async def cleanup() -> None:
        task.cancel()
        detector.unsubscribe()
        await trigger.close()
        mic.stop()  # the source's lifecycle is ours, not the backend's

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())
