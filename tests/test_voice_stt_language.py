"""VoiceChannel chooses the STT language per session, from the next stream on.

Covers the three STT modes (VAD, continuous, batch), the events that carry
the reported language, and the ``STTLanguageLock`` policy end to end.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, STTLanguageLock, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.stt.mock import MockSTTProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utterances(n: int) -> list[VADEvent | None]:
    events: list[VADEvent | None] = []
    for _ in range(n):
        events += [
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"fake-audio"),
        ]
    return events


class _ScriptedSTT(STTProvider):
    """Streaming STT that plays one scripted final per stream and records the language asked.

    A scripted empty text is what a pinned stream produces when it hears a
    language it cannot transcribe: the channel falls back to ``transcribe``,
    which answers empty as well — a final with nothing in it.
    """

    def __init__(self, script: list[tuple[str, str | None, float | None]]) -> None:
        self._script = list(script)
        self.languages_requested: list[str | None] = []
        self.batch_languages_requested: list[str | None] = []

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_language_override(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        self.batch_languages_requested.append(language)
        return TranscriptionResult(text="", is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        self.languages_requested.append(language)
        async for _ in audio_stream:
            pass
        if not self._script:
            return
        text, reported, confidence = self._script.pop(0)
        yield TranscriptionResult(
            text=text, is_final=True, language=reported, confidence=confidence
        )


class _ContinuousSTT(_ScriptedSTT):
    """Answers right after the first chunk, the way a server-endpointing STT does."""

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        self.languages_requested.append(language)
        async for _ in audio_stream:
            break
        if not self._script:
            return
        text, reported, confidence = self._script.pop(0)
        yield TranscriptionResult(
            text=text, is_final=True, language=reported, confidence=confidence
        )


class _PartialSTT(STTProvider):
    """Yields a tagged partial, then a final."""

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_language_override(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="batch")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            pass
        yield TranscriptionResult(text="bon", is_final=False, confidence=0.4, language="fr")
        yield TranscriptionResult(text="bonjour", is_final=True, language="fr")


class _NoOverrideSTT(STTProvider):
    """A provider on the older contract: no per-call language."""

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="hello")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            pass
        yield TranscriptionResult(text="hello", is_final=True)


def _vad_kit(
    stt: STTProvider,
    *,
    utterances: int = 1,
    lock: STTLanguageLock | None = None,
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend]:
    backend = MockVoiceBackend()
    pipeline = AudioPipelineConfig(vad=MockVADProvider(events=_utterances(utterances)))
    channel = VoiceChannel(
        "voice-1", stt=stt, backend=backend, pipeline=pipeline, stt_language_lock=lock
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)
    return kit, channel, backend


async def _connect(kit: RoomKit) -> Any:
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    return await kit.connect_voice(room.id, "user-1", "voice-1")


async def _speak(backend: MockVoiceBackend, session: Any) -> None:
    """One utterance: SPEECH_START, a mid frame, SPEECH_END."""
    for byte in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        await backend.simulate_audio_received(session, AudioFrame(data=byte))
    await asyncio.sleep(0.15)


# ---------------------------------------------------------------------------
# set_stt_language / get_stt_language
# ---------------------------------------------------------------------------


class TestPerSessionLanguage:
    async def test_default_stream_carries_no_language(self) -> None:
        stt = MockSTTProvider(streaming=True)
        kit, channel, backend = _vad_kit(stt)
        session = await _connect(kit)

        await _speak(backend, session)

        assert stt.languages_requested == [None]
        assert channel.get_stt_language(session) is None

    async def test_choice_applies_to_the_next_utterance(self) -> None:
        stt = MockSTTProvider(streaming=True)
        kit, channel, backend = _vad_kit(stt, utterances=3)
        session = await _connect(kit)

        await _speak(backend, session)
        channel.set_stt_language(session, "fr-CA")
        assert channel.get_stt_language(session) == "fr-CA"
        await _speak(backend, session)
        channel.set_stt_language(session, None)
        await _speak(backend, session)

        assert stt.languages_requested == [None, "fr-CA", None]

    async def test_batch_fallback_gets_the_language(self) -> None:
        stt = MockSTTProvider()  # not streaming: VAD mode transcribes in batch
        kit, channel, backend = _vad_kit(stt)
        session = await _connect(kit)

        channel.set_stt_language(session, "fr-CA")
        await _speak(backend, session)

        assert stt.languages_requested == ["fr-CA"]

    async def test_batch_mode_flush_gets_the_language(self) -> None:
        stt = MockSTTProvider(streaming=True)
        backend = MockVoiceBackend()
        channel = VoiceChannel(
            "voice-1", stt=stt, backend=backend, pipeline=AudioPipelineConfig(), batch_mode=True
        )
        kit = RoomKit(stt=stt, voice=backend)
        kit.register_channel(channel)
        session = await _connect(kit)

        channel.set_stt_language(session, "fr-CA")
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00" * 100))
        result = await channel.flush_stt(session)

        assert result.text == "Hello"
        assert stt.languages_requested == ["fr-CA"]

    async def test_cleared_when_the_session_goes_away(self) -> None:
        stt = MockSTTProvider(streaming=True)
        kit, channel, backend = _vad_kit(stt)
        session = await _connect(kit)
        channel.set_stt_language(session, "fr-CA")

        await kit.disconnect_voice(session)

        assert channel.get_stt_language(session) is None
        assert session.id not in channel._stt_languages

    async def test_requires_a_provider_that_honours_it(self) -> None:
        kit, channel, backend = _vad_kit(_NoOverrideSTT())
        session = await _connect(kit)

        with pytest.raises(RuntimeError, match="per-session language"):
            channel.set_stt_language(session, "fr-CA")

    async def test_requires_a_provider(self) -> None:
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend, pipeline=AudioPipelineConfig())
        kit = RoomKit(voice=backend)
        kit.register_channel(channel)
        session = await _connect(kit)

        with pytest.raises(RuntimeError, match="STT provider"):
            channel.set_stt_language(session, "fr-CA")

    async def test_a_provider_without_override_is_never_handed_one(self) -> None:
        """Even with a language stored, the call stays audio-only."""
        stt = _NoOverrideSTT()
        kit, channel, backend = _vad_kit(stt)
        session = await _connect(kit)
        channel._stt_languages[session.id] = "fr-CA"

        assert channel._stt_call_kwargs(session.id) == {}


# ---------------------------------------------------------------------------
# Events carry what the provider reported
# ---------------------------------------------------------------------------


class TestEventsCarryLanguage:
    async def test_transcription_event_language(self) -> None:
        stt = MockSTTProvider(streaming=True, languages=["fr"])
        kit, channel, backend = _vad_kit(stt)
        seen: list[str | None] = []

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event, ctx):
            seen.append(event.language)
            return HookResult.allow()

        session = await _connect(kit)
        await _speak(backend, session)

        assert seen == ["fr"]

    async def test_partial_event_language(self) -> None:
        kit, channel, backend = _vad_kit(_PartialSTT())
        partials: list[str | None] = []
        finals: list[str | None] = []

        @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, execution=HookExecution.ASYNC)
        async def on_partial(event, ctx):
            partials.append(event.language)

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event, ctx):
            finals.append(event.language)
            return HookResult.allow()

        session = await _connect(kit)
        await _speak(backend, session)

        assert partials == ["fr"]
        assert finals == ["fr"]

    async def test_batch_result_language(self) -> None:
        stt = MockSTTProvider(languages=["es"])
        kit, channel, backend = _vad_kit(stt)
        seen: list[str | None] = []

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event, ctx):
            seen.append(event.language)
            return HookResult.allow()

        session = await _connect(kit)
        await _speak(backend, session)

        assert seen == ["es"]


# ---------------------------------------------------------------------------
# STTLanguageLock through the channel
# ---------------------------------------------------------------------------


class TestLanguageLockInVadMode:
    def test_requires_a_provider_that_honours_it(self) -> None:
        with pytest.raises(ValueError, match="per-session language"):
            VoiceChannel(
                "voice-1",
                stt=_NoOverrideSTT(),
                backend=MockVoiceBackend(),
                stt_language_lock=STTLanguageLock(),
            )
        with pytest.raises(ValueError, match="STT provider"):
            VoiceChannel(
                "voice-1", backend=MockVoiceBackend(), stt_language_lock=STTLanguageLock()
            )

    async def test_sessions_start_detecting(self) -> None:
        stt = _ScriptedSTT([("hello", None, 0.9)])
        kit, channel, backend = _vad_kit(stt, lock=STTLanguageLock())
        session = await _connect(kit)

        assert channel.get_stt_language(session) == "multi"
        await _speak(backend, session)
        assert stt.languages_requested == ["multi"]

    async def test_the_reported_language_pins_the_next_utterance(self) -> None:
        stt = _ScriptedSTT([("bonjour", "fr", 0.9), ("encore", None, 0.9)])
        lock = STTLanguageLock(prefer={"fr": "fr-CA"})
        kit, channel, backend = _vad_kit(stt, utterances=2, lock=lock)
        session = await _connect(kit)

        await _speak(backend, session)
        await _speak(backend, session)

        assert stt.languages_requested == ["multi", "fr-CA"]
        assert channel.get_stt_language(session) == "fr-CA"

    async def test_misses_release_back_to_detecting(self) -> None:
        stt = _ScriptedSTT(
            [
                ("bonjour", "fr", 0.9),  # multi -> lock fr-CA
                ("", None, None),  # fr-CA hears nothing: miss 1
                ("", None, None),  # miss 2: release
                ("hello", "en", 0.9),  # multi again -> lock en
            ]
        )
        lock = STTLanguageLock(prefer={"fr": "fr-CA"})
        kit, channel, backend = _vad_kit(stt, utterances=4, lock=lock)
        session = await _connect(kit)

        for _ in range(4):
            await _speak(backend, session)

        assert stt.languages_requested == ["multi", "fr-CA", "fr-CA", "multi"]
        # an empty stream falls back to batch, which is asked in the same language
        assert stt.batch_languages_requested == ["fr-CA", "fr-CA"]
        assert channel.get_stt_language(session) == "en"

    async def test_low_confidence_releases(self) -> None:
        stt = _ScriptedSTT([("bonjour", "fr", 0.9), ("krzt", None, 0.1), ("krzt", None, 0.1)])
        kit, channel, backend = _vad_kit(stt, utterances=3, lock=STTLanguageLock())
        session = await _connect(kit)

        for _ in range(3):
            await _speak(backend, session)

        assert stt.languages_requested == ["multi", "fr", "fr"]
        assert channel.get_stt_language(session) == "multi"

    async def test_unbind_forgets_the_session(self) -> None:
        stt = _ScriptedSTT([("bonjour", "fr", 0.9)])
        lock = STTLanguageLock()
        kit, channel, backend = _vad_kit(stt, lock=lock)
        session = await _connect(kit)
        await _speak(backend, session)
        assert lock.language_for(session.id) == "fr"

        await kit.disconnect_voice(session)

        assert lock.language_for(session.id) == "multi"


# ---------------------------------------------------------------------------
# Continuous mode: the language lands on the next cycle
# ---------------------------------------------------------------------------


def _continuous_kit(
    stt: STTProvider, lock: STTLanguageLock | None = None
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend]:
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(),  # no VAD + streaming STT = continuous
        stt_language_lock=lock,
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)
    return kit, channel, backend


_CHUNK = AudioFrame(data=b"\x01\x00" * 1600)  # 3200 bytes: one STT buffer flush


class TestContinuousMode:
    async def test_set_language_reopens_the_cycle(self) -> None:
        stt = _ContinuousSTT([("hello", None, 0.9), ("bonjour", None, 0.9)])
        kit, channel, backend = _continuous_kit(stt)
        session = await _connect(kit)
        assert channel._continuous_stt

        await backend.simulate_audio_received(session, _CHUNK)
        await asyncio.sleep(0.3)
        assert stt.languages_requested == [None]

        channel.set_stt_language(session, "fr-CA")
        await asyncio.sleep(0.3)
        await backend.simulate_audio_received(session, _CHUNK)
        await asyncio.sleep(0.3)

        assert stt.languages_requested == [None, "fr-CA"]
        await channel.close()

    async def test_lock_pins_the_next_cycle(self) -> None:
        stt = _ContinuousSTT([("bonjour", "fr", 0.9), ("encore", None, 0.9)])
        lock = STTLanguageLock(prefer={"fr": "fr-CA"})
        kit, channel, backend = _continuous_kit(stt, lock=lock)
        seen: list[str | None] = []

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event, ctx):
            seen.append(event.language)
            return HookResult.allow()

        session = await _connect(kit)

        await backend.simulate_audio_received(session, _CHUNK)
        await asyncio.sleep(0.3)
        await backend.simulate_audio_received(session, _CHUNK)
        await asyncio.sleep(0.3)

        assert stt.languages_requested == ["multi", "fr-CA"]
        assert seen[:1] == ["fr"]
        await channel.close()
