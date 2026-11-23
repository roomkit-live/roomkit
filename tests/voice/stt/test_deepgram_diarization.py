"""Deepgram speaker segments with ``diarize_model`` (RFC §12.2.3).

Deepgram labels each word, so one final can hold two voices: the words become
one segment per run of the same label. ``diarize=True`` alone keeps its old
behaviour (labels in ``words`` only), so existing configurations are unchanged.

Only the streaming tests need the SDK (the provider imports its event types
there); the rest run without the ``deepgram`` extra, as CI installs it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import VoiceChannel
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, SpeakerSegment
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider, _speaker_segments


def _word(word: str, speaker: int | str | None, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(
        word=word.lower().strip(".,?"), punctuated_word=word, speaker=speaker, start=start, end=end
    )


# One Deepgram final that runs across a change of voice, as observed on the
# live service (nova-3, French, 2026-09-27).
_MIXED = [
    _word("explosent.", 0, 8.98, 9.46),
    _word("C'est", 1, 9.66, 9.82),
    _word("ce", 1, 9.82, 9.9),
    _word("que", 1, 9.9, 10.02),
    _word("je", 1, 10.02, 10.1),
    _word("craignais.", 1, 10.1, 10.62),
]
_MIXED_TEXT = "explosent. C'est ce que je craignais."


def _message(words: list[Any], text: str, *, is_final: bool = True) -> SimpleNamespace:
    alt = SimpleNamespace(transcript=text, confidence=0.9, words=words)
    return SimpleNamespace(channel=SimpleNamespace(alternatives=[alt]), is_final=is_final)


class _Connection:
    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages
        self._handlers: dict[Any, Any] = {}

    def on(self, event: Any, handler: Any) -> None:
        self._handlers[event] = handler

    async def start_listening(self) -> None:
        events = pytest.importorskip("deepgram.core.events")
        for message in self._messages:
            self._handlers[events.EventType.MESSAGE](message)
        self._handlers[events.EventType.CLOSE](None)

    async def send_media(self, _data: bytes) -> None:
        return None

    async def send_close_stream(self) -> None:
        return None

    async def __aenter__(self) -> _Connection:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _FakeDeepgram:
    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages
        self.connect_opts: list[dict[str, Any]] = []

    def AsyncDeepgramClient(self, **_kwargs: Any) -> Any:  # noqa: N802 — SDK name
        fake = self

        class _V1:
            @staticmethod
            def connect(**opts: Any) -> _Connection:
                fake.connect_opts.append(opts)
                return _Connection(fake._messages)

        return SimpleNamespace(listen=SimpleNamespace(v1=_V1()))


def _provider(messages: list[Any] | None = None, **config: Any) -> tuple[Any, _FakeDeepgram]:
    fake = _FakeDeepgram(messages or [])
    provider = DeepgramSTTProvider.__new__(DeepgramSTTProvider)
    provider._config = DeepgramConfig(api_key="k", model="nova-3", language="fr", **config)
    provider._dg = fake
    provider._client = fake.AsyncDeepgramClient()
    return provider, fake


async def _stream(provider: Any) -> list[Any]:
    async def audio() -> Any:
        yield AudioChunk(data=b"\x00\x00" * 160, sample_rate=16000)

    async def collect() -> list[Any]:
        return [r async for r in provider.transcribe_stream(audio())]

    return await asyncio.wait_for(collect(), timeout=5)


class TestConfig:
    def test_diarize_model_claims_speaker_segments(self) -> None:
        provider, _ = _provider(diarize_model="latest")
        assert provider.supports_diarization is True

    @pytest.mark.parametrize("config", [{}, {"diarize": True}], ids=["default", "legacy"])
    def test_without_diarize_model_nothing_is_claimed(self, config: dict[str, Any]) -> None:
        provider, _ = _provider(**config)
        assert provider.supports_diarization is False

    def test_diarize_and_diarize_model_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="exclusive"):
            DeepgramConfig(api_key="k", diarize=True, diarize_model="latest")


class TestConnectOptions:
    def test_diarize_model_travels_as_a_query_parameter_without_diarize(self) -> None:
        provider, _ = _provider(diarize_model="latest")
        opts = provider._build_connect_options(16000)

        # The service refuses diarize next to diarize_model, even "false";
        # and SDK 6 has no diarize_model keyword, hence request_options.
        assert "diarize" not in opts
        assert opts["request_options"] == {
            "additional_query_parameters": {"diarize_model": "latest"}
        }

    def test_legacy_diarize_is_unchanged(self) -> None:
        provider, _ = _provider(diarize=True)
        opts = provider._build_connect_options(16000)
        assert opts["diarize"] == "true"
        assert "request_options" not in opts


class TestSegments:
    def test_runs_of_one_label_become_one_segment(self) -> None:
        assert _speaker_segments(_MIXED, _MIXED_TEXT) == [
            SpeakerSegment("0", "explosent.", 8980, 9460),
            SpeakerSegment("1", "C'est ce que je craignais.", 9660, 10620),
        ]

    def test_dict_words_work_too(self) -> None:
        words = [
            {"word": "oui", "punctuated_word": "Oui.", "speaker": 2, "start": 1.0, "end": 1.2}
        ]
        assert _speaker_segments(words, "Oui.") == [SpeakerSegment("2", "Oui.", 1000, 1200)]

    @pytest.mark.parametrize("spelling", ["unknown", "UU", "PENDING"])
    def test_a_word_labelled_nobody_is_unattributed(self, spelling: str) -> None:
        """The spellings of nobody are Deepgram's adapter's to name (RFC §12.2.3)."""
        words = [_word("Bonjour.", spelling, 0.1, 0.5)]
        assert _speaker_segments(words, "Bonjour.") == [SpeakerSegment(None, "Bonjour.", 100, 500)]

    def test_text_without_words_is_unattributed(self) -> None:
        assert _speaker_segments([], "Bonjour.") == [SpeakerSegment(None, "Bonjour.")]
        assert _speaker_segments(None, "") == []


@pytest.fixture
def _sdk() -> None:
    pytest.importorskip("deepgram")


@pytest.mark.usefixtures("_sdk")
class TestStreaming:
    async def test_a_final_across_two_voices_becomes_two_segments(self) -> None:
        provider, fake = _provider([_message(_MIXED, _MIXED_TEXT)], diarize_model="latest")

        results = await _stream(provider)

        assert results[0].segments == [
            SpeakerSegment("0", "explosent.", 8980, 9460),
            SpeakerSegment("1", "C'est ce que je craignais.", 9660, 10620),
        ]
        assert results[0].speaker is None  # two voices, no single speaker
        assert fake.connect_opts[0]["request_options"]["additional_query_parameters"] == {
            "diarize_model": "latest"
        }

    async def test_partials_carry_no_segments(self) -> None:
        provider, _ = _provider(
            [_message(_MIXED[:2], "explosent. C'est", is_final=False)], diarize_model="latest"
        )
        results = await _stream(provider)
        assert results[0].is_final is False
        assert results[0].segments == []

    async def test_legacy_diarize_keeps_labels_in_words_only(self) -> None:
        provider, _ = _provider([_message(_MIXED, _MIXED_TEXT)], diarize=True)
        results = await _stream(provider)
        assert results[0].segments == []
        assert [w.speaker for w in results[0].words] == [0, 1, 1, 1, 1, 1]

    async def test_words_without_labels_are_unattributed(self) -> None:
        words = [_word("Bonjour.", None, 0.1, 0.5)]
        provider, _ = _provider([_message(words, "Bonjour.")], diarize_model="latest")
        results = await _stream(provider)
        assert results[0].segments == [SpeakerSegment(None, "Bonjour.", 100, 500)]


class TestBatch:
    async def test_the_clip_comes_back_as_segments(self) -> None:
        provider, _ = _provider(diarize_model="latest")
        alt = SimpleNamespace(transcript=_MIXED_TEXT, confidence=0.9, words=_MIXED)
        response = SimpleNamespace(
            results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[alt])])
        )
        provider._client.listen.v1.media = SimpleNamespace(
            transcribe_file=AsyncMock(return_value=response)
        )

        result = await provider.transcribe(AudioChunk(data=b"\x00\x00" * 160, sample_rate=16000))

        assert [s.speaker for s in result.segments] == ["0", "1"]
        kwargs = provider._client.listen.v1.media.transcribe_file.call_args.kwargs
        assert kwargs["request_options"]["additional_query_parameters"] == {
            "sample_rate": "16000",
            "channels": "1",
            "diarize_model": "latest",
        }


class TestVoiceChannel:
    def test_continuous_mode_carries_deepgram_speakers(self) -> None:
        provider, _ = _provider(diarize_model="latest")
        VoiceChannel(
            "voice-1", stt=provider, backend=MockVoiceBackend(), pipeline=AudioPipelineConfig()
        )

    def test_behind_a_vad_diarize_model_is_refused(self) -> None:
        provider, _ = _provider(diarize_model="latest")
        with pytest.raises(ValueError, match="continuous mode"):
            VoiceChannel(
                "voice-1",
                stt=provider,
                backend=MockVoiceBackend(),
                pipeline=AudioPipelineConfig(vad=MockVADProvider(events=[])),
            )

    def test_behind_a_vad_legacy_diarize_still_works(self) -> None:
        # Backward compatibility: diarize=True never claimed speaker segments.
        provider, _ = _provider(diarize=True)
        VoiceChannel(
            "voice-1",
            stt=provider,
            backend=MockVoiceBackend(),
            pipeline=AudioPipelineConfig(vad=MockVADProvider(events=[])),
        )
