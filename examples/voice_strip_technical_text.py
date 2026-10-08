"""Never read technical text aloud: a tool call, a note to self, a separator.

A language model sometimes writes into a spoken reply what was never meant to
be heard: its next tool call as JSON instead of calling the tool, a
"(Note: ...)" to itself, a "---" between two parts. ``StripTechnicalText``,
given to a voice channel as its ``tts_filter``, keeps them out of speech; the
text stored in the room stays as the model wrote it, and each removal is
logged as a warning.

1. A whole reply, through ``VoiceChannel.say()``: the TTS is asked to say the
   words around the tool call, not the tool call.
2. A streamed reply, as the AI channel hands it to the voice channel token by
   token: the filter works before the reply is cut into sentences, so a JSON
   object holding a full stop is removed whole.

The TTS and the transport are mocks, so the example runs without keys.

Run with:
    uv run python examples/voice_strip_technical_text.py
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import VoiceChannel
from roomkit.voice import StripEmoji, StripTechnicalText, TTSFilterChain
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.tts.filters import filtered_stream
from roomkit.voice.tts.mock import MockTTSProvider
from roomkit.voice.tts.sentence_splitter import split_sentences

logger = setup_logging("example.voice_strip_technical_text")

REPLY = (
    "I'll ask the weather service. "
    '{"name": "delegate", "arguments": {"task": "Weather in Montreal. Today"}} '
    "(Note: the user prefers Celsius.) It will rain this afternoon. 🌧️"
)

TOKENS = [
    "Let me look. ",
    '{"name": "lookup", ',
    '"arguments": {"q": "Opening hours. Sunday"}}',
    " ---",
    " The shop opens at ten.",
]


async def tokens() -> AsyncIterator[str]:
    for token in TOKENS:
        yield token
        await asyncio.sleep(0)


async def main() -> None:
    tts = MockTTSProvider()
    backend = MockVoiceBackend()
    voice = VoiceChannel(
        "voice",
        tts=tts,
        backend=backend,
        tts_filter=TTSFilterChain(StripTechnicalText(), StripEmoji()),
    )
    session = await backend.connect("room", "user", "voice")

    logger.info("The model wrote:  %s", REPLY)
    await voice.say(session, REPLY)
    logger.info("The voice says:   %s", tts.calls[-1]["text"])

    logger.info("Streamed tokens:  %s", "".join(TOKENS))
    sentences = split_sentences(filtered_stream(tokens(), StripTechnicalText()))
    async for sentence in sentences:
        logger.info("  sentence spoken: %s", sentence)


if __name__ == "__main__":
    asyncio.run(main())
