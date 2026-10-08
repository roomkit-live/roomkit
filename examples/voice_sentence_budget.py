"""A voice agent that says at most four sentences per reply.

Asked to "explain", a language model may talk for a minute however short the
prompt asks it to be. ``VoiceChannel(max_sentences=4)`` caps what is spoken:
once four sentences are said, a reply that goes on stops at its first word past
them, as if the person had spoken over the agent. The model generates nothing
more, no tool call it would make next starts, and the room keeps the text
produced up to there (marked ``cancelled``) rather than the whole answer nobody
heard.

The user's speech, the model and the TTS are scripted stand-ins, so the example
runs without keys: the model streams a twelve-sentence explanation word by word,
and the TTS prints each sentence it is given.

Run with:
    uv run python examples/voice_sentence_budget.py
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSProvider

logger = setup_logging("example.voice_sentence_budget")

EXPLANATION = " ".join(
    f"Point {n} of the invoice explanation is here, with its details." for n in range(1, 13)
)


class Explainer(AIProvider):
    """Streams a long explanation word by word, as a model would."""

    def __init__(self) -> None:
        self.words_generated = 0

    @property
    def model_name(self) -> str:
        return "explainer"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content=EXPLANATION)

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        for word in EXPLANATION.split(" "):
            self.words_generated += 1
            yield word + " "
            await asyncio.sleep(0.005)


class PrintingTTS(TTSProvider):
    """Prints each sentence it is asked to say, and returns a little silence."""

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        async for sentence in text_stream:
            logger.info("  spoken: %s", sentence)
            yield AudioChunk(data=b"\x00\x00" * 160, sample_rate=16000)


async def main() -> None:
    stt = MockSTTProvider(transcripts=["Can you explain my invoice?"])
    backend = MockVoiceBackend()
    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"speech"),
        ]
    )
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=PrintingTTS(),
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad),
        max_sentences=4,
    )
    explainer = Explainer()
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(voice)
    kit.register_channel(AIChannel("agent", provider=explainer))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    await kit.attach_channel(room.id, "agent")
    session = await kit.join(room.id, "voice", participant_id="caller")

    logger.info('Caller: "Can you explain my invoice?"')
    for data in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        await backend.simulate_audio_received(session, AudioFrame(data=data))
    await asyncio.sleep(1.0)

    [reply] = [e for e in await kit.store.list_events(room.id) if e.source.channel_id == "agent"]
    total = len(EXPLANATION.split(" "))
    logger.info("Words generated: %d of %d", explainer.words_generated, total)
    logger.info(
        "The room keeps (cancelled=%s): %s", reply.metadata.get("cancelled"), reply.content.body
    )
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
