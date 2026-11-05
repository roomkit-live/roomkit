"""Synthesize speech with Rime Coda or Mist v3, then transcribe it with Rime STT.

Install: pip install 'roomkit[rime]'
Run: RIME_API_KEY=... uv run python examples/voice_rime.py

Optional: RIME_MODEL=coda|mistv3, RIME_LANGUAGE=en|es, RIME_VOICE=astra,
RIME_TEXT='What is two plus two?'. Choose a voice matching the language.
This example uses no microphone, audio device, or separate LLM.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, setup_logging

from roomkit.voice.stt import RimeSTTConfig, RimeSTTProvider
from roomkit.voice.tts import RimeTTSConfig, RimeTTSProvider

logger = setup_logging("voice_rime")


async def main() -> None:
    key = require_env("RIME_API_KEY")["RIME_API_KEY"]
    language = os.environ.get("RIME_LANGUAGE", "en")
    tts = RimeTTSProvider(
        RimeTTSConfig(
            api_key=key,
            model=os.environ.get("RIME_MODEL", "coda"),  # type: ignore[arg-type]
            language=language,
            voice_id=os.environ.get("RIME_VOICE", "astra"),
        )
    )
    stt = RimeSTTProvider(RimeSTTConfig(api_key=key, language=language))
    try:
        audio = await tts.synthesize(os.environ.get("RIME_TEXT", "What is two plus two?"))
        transcript = await stt.transcribe(audio)
        logger.info(
            "Synthesized %.2f seconds; transcript: %s", audio.duration_seconds, transcript.text
        )
    finally:
        await asyncio.gather(tts.close(), stt.close())


if __name__ == "__main__":
    asyncio.run(main())
