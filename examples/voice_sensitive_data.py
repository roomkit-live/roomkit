"""RoomKit -- DTMF redaction and recording encryption (RFC Section 17.6).

A voice turn carries two kinds of sensitive payload the framework can expose
without anyone meaning it to: the digits a caller types on their keypad, and
the audio it writes to disk. This example shows the two surfaces that cover
them.

    DTMFRedaction        masks digits wherever the framework exposes one --
                         frame metadata (which reaches recorders, debug taps
                         and logs) and DTMFDetectedEvent.redacted_digit
    RecordingEncryption  encrypts a finished recording at rest, and deletes
                         the plaintext

The ON_DTMF hook still receives the raw digit, deliberately: that hook is how
an IVR reads the card number it exists to collect. Redaction protects the
places the digit reaches *incidentally*.

The cipher below is a stand-in so the example runs with no dependencies and no
key management. It is NOT encryption -- see the note in `run()`.

Run with:
    uv run python examples/voice_sensitive_data.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import ChannelCategory, HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    DTMFEvent,
    DTMFRedaction,
    MockDTMFDetector,
    MockVADProvider,
    RecordingConfig,
    RecordingEncryption,
    WavFileRecorder,
)
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logger = setup_logging("voice_sensitive_data")

# A caller entering a card number, one tone at a time.
CARD_DIGITS = "4111111111111111"


class ReversingEncryption(RecordingEncryption):
    """Stand-in cipher: reverses the bytes. Do not ship this.

    It stands in for the real thing to keep the example dependency-free. What
    it demonstrates is the *contract* a real implementation must honour: leave
    no plaintext behind, and return the path of the encrypted artifact.
    """

    @property
    def name(self) -> str:
        return "reversing-stand-in"

    def encrypt_file(self, path: str) -> str:
        source = Path(path)
        target = source.with_suffix(source.suffix + ".enc")
        target.write_bytes(source.read_bytes()[::-1])
        source.unlink()
        return str(target)


async def run() -> None:
    output_dir = Path("./recordings/sensitive-data")

    backend = MockVoiceBackend()
    pipeline = AudioPipelineConfig(
        vad=MockVADProvider(),
        dtmf=MockDTMFDetector(events=[DTMFEvent(digit=d, duration_ms=80.0) for d in CARD_DIGITS]),
        # Keeping the first and last four is the card-number shape the RFC
        # gives as its example. The defaults mask everything, which is the
        # right starting point for a PIN.
        dtmf_redaction=DTMFRedaction(keep_first=4, keep_last=4),
        recorder=WavFileRecorder(),
        recording_config=RecordingConfig(
            storage=str(output_dir),
            encryption=ReversingEncryption(),
        ),
    )

    voice = VoiceChannel(
        "voice",
        stt=MockSTTProvider(transcripts=["card entered"]),
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=pipeline,
    )

    kit = RoomKit(voice=backend)
    kit.register_channel(voice)
    await kit.create_room(room_id="payment-line")
    await kit.attach_channel("payment-line", "voice", category=ChannelCategory.TRANSPORT)

    collected: list[str] = []

    @kit.hook(HookTrigger.ON_DTMF, execution=HookExecution.ASYNC)
    async def on_dtmf(event, ctx) -> None:
        # The IVR reads the real digit here -- that is the point of the hook.
        collected.append(event.digit)
        # Anything that logs or stores uses the redacted form instead. It
        # equals `digit` when no redaction is configured, so this line is safe
        # unconditionally.
        logger.info("DTMF received: %s", event.redacted_digit)

    session = await kit.join("payment-line", "voice", participant_id="caller-1")

    logger.info("Caller enters %d digits...", len(CARD_DIGITS))
    for _ in CARD_DIGITS:
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x00" * 160))
        await asyncio.sleep(0.01)

    logger.info("")
    logger.info("What the IVR collected  : %s", "".join(collected))
    logger.info(
        "What a log would show   : %s",
        DTMFRedaction(keep_first=4, keep_last=4).mask("".join(collected)),
    )
    logger.info("")

    await kit.leave(session)
    await asyncio.sleep(0.1)
    await kit.close()

    written = sorted(p.name for p in output_dir.glob("*")) if output_dir.exists() else []
    logger.info("Files on disk           : %s", written or "(none)")
    logger.info(
        "Plaintext .wav left     : %s", [n for n in written if n.endswith(".wav")] or "none"
    )
    logger.info("")
    logger.info(
        "A recording the cipher cannot encrypt is deleted rather than left in "
        "the clear -- a caller who asked for encryption at rest is never handed "
        "an unencrypted file."
    )


if __name__ == "__main__":
    asyncio.run(run())
