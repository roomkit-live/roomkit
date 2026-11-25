"""Scripted synthesis for the voice scenarios (RFC §12.2, §12.2.2, §12.3.13).

``voice.open``'s ``tts`` scripts the channel's TTS: the context level it
declares, the format and length of its chunks, a synthesis that fails, a gate
a sentence's synthesis holds at, and a curated voice catalog; ``tts_context``
sets the channel's TTSContextConfig. ``voice.read`` returns the context each
synthesis call received and the contexts released; ``voice.play``,
``voice.list_voices`` and ``voice.synthesize_dialogue`` drive the rest.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

from roomkit.voice import (
    AudioChunk,
    DialogueTurn,
    TTSContext,
    TTSContextConfig,
    TTSContextLevel,
    VoiceInfo,
)
from roomkit.voice.tts import MockTTSProvider

if TYPE_CHECKING:
    from roomkit import RoomKit
    from roomkit.voice import ConversationTurn
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates


class ScriptedSynthesisError(RuntimeError):
    """A synthesis the script fails, as a provider's error ends one."""


class UnsupportedError(Exception):
    """A method the transport or the provider does not support (README, Errors)."""


_SAMPLE = b"\x01\x00"
"""One 16-bit sample of value 1: a scripted word's audio, sound and not silence."""


class ScriptedTTS(MockTTSProvider):
    """The mock TTS, streaming its input sentence by sentence, as ``tts`` scripts it.

    Each sentence it synthesizes is one entry of ``calls`` (``synthesized``);
    each call the channel makes, a sentence's or a whole streamed reply's,
    appends the context it was given to ``contexts``.
    """

    def __init__(self, spec: Mapping[str, Any], gates: Gates) -> None:
        super().__init__(context_level=TTSContextLevel(spec.get("context_level", "none")))
        self._spec = spec
        self._gates = gates

    @property
    def supports_streaming_input(self) -> bool:
        return True

    @property
    def max_dialogue_speakers(self) -> int:
        return int(self._spec.get("max_dialogue_speakers", 0))

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None, context: TTSContext | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.contexts.append(context)
        async for chunk in self._sentence(text, voice):
            yield chunk

    async def synthesize_stream_input(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        context: TTSContext | None = None,
    ) -> AsyncIterator[AudioChunk]:
        self.contexts.append(context)
        async for text in text_stream:
            async for chunk in self._sentence(text, voice):
                yield chunk

    async def _sentence(self, text: str, voice: str | None) -> AsyncIterator[AudioChunk]:
        """One sentence's audio: a chunk per word, after its gate, unless it fails."""
        self.calls.append({"text": text, "voice": voice or self.default_voice})
        gates = self._spec.get("gates", [])
        index = len(self.calls) - 1
        if index < len(gates) and gates[index] is not None:
            await self._gates.hold(gates[index])
        if "fails" in self._spec:
            raise ScriptedSynthesisError(self._spec["fails"])
        words = text.split()
        for i, word in enumerate(words):
            yield AudioChunk(
                data=self._audio(word),
                sample_rate=16000,
                format=self._spec.get("format", "pcm_s16le"),
                is_final=i == len(words) - 1,
            )

    def _audio(self, word: str) -> bytes:
        if "word_ms" in self._spec:
            return _SAMPLE * (16 * int(self._spec["word_ms"]))
        raw = f"mock-audio-{word}".encode()
        return raw + b"\x00" if len(raw) % 2 else raw


def scripted_tts(spec: Mapping[str, Any], gates: Gates) -> ScriptedTTS:
    """The channel's TTS; with ``voices``, the curated catalog its class offers."""
    if "voices" not in spec:
        return ScriptedTTS(spec, gates)
    catalog = [VoiceInfo(**voice) for voice in spec["voices"]]

    class CatalogTTS(ScriptedTTS):
        @classmethod
        def available_voices(cls) -> list[VoiceInfo]:
            return list(catalog)

    return CatalogTTS(spec, gates)


def tts_context(spec: Mapping[str, Any]) -> TTSContextConfig | None:
    return TTSContextConfig(**spec["tts_context"]) if "tts_context" in spec else None


def _audio_record(turn: ConversationTurn) -> str | None:
    if turn.audio is None:
        return None
    return base64.b64encode(turn.audio.data).decode("ascii")


def _context_record(context: TTSContext | None) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        "context_id": context.context_id,
        "next_turn_id": context.next_turn_id,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "role": turn.role,
                "participant_id": turn.participant_id,
                "text": turn.text,
                "audio": _audio_record(turn),
                "played_ms": turn.played_ms,
                "interrupted": turn.interrupted,
            }
            for turn in context.turns
        ],
    }


def read_synthesis(tts: ScriptedTTS | None) -> dict[str, Any]:
    """``voice.read``'s view of the TTS: the contexts received, the ones released."""
    if tts is None:
        return {"tts_contexts": [], "released_contexts": []}
    return {
        "tts_contexts": [_context_record(context) for context in tts.contexts],
        "released_contexts": list(tts.released),
    }


def _voice_record(voice: VoiceInfo) -> dict[str, Any]:
    return {
        "id": voice.id,
        "name": voice.name,
        "language": voice.language,
        "gender": voice.gender,
        "description": voice.description,
    }


async def _play(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    audio = base64.b64decode(spec["audio"], validate=True)
    await fixture.channel.play(
        fixture.sessions[spec["participant_id"]], audio, text=spec.get("text")
    )


async def _list_voices(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    tts = harness.voice_fixtures[spec["channel_id"]].tts
    voices = await tts.list_voices(
        language=spec.get("language"), gender=spec.get("gender"), query=spec.get("query")
    )
    return {"voices": [_voice_record(voice) for voice in voices]}


async def _synthesize_dialogue(
    kit: RoomKit, harness: Harness, spec: Mapping[str, Any]
) -> dict[str, Any]:
    tts = harness.voice_fixtures[spec["channel_id"]].tts
    turns = [DialogueTurn(speaker=t["speaker"], text=t["text"]) for t in spec["turns"]]
    try:
        audio = await tts.synthesize_dialogue(turns, spec["voices"])
    except NotImplementedError as error:
        raise UnsupportedError(str(error)) from error
    return {"transcript": audio.transcript}


SYNTHESIS_CALLS = {
    "voice.play": _play,
    "voice.list_voices": _list_voices,
    "voice.synthesize_dialogue": _synthesize_dialogue,
}
SYNTHESIS_ARGUMENTS = {
    "voice.play": {"channel_id", "participant_id", "audio", "text"},
    "voice.list_voices": {"channel_id", "language", "gender", "query"},
    "voice.synthesize_dialogue": {"channel_id", "turns", "voices"},
}
SYNTHESIS_OPEN_ARGUMENTS = {"tts", "tts_context"}
