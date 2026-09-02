"""Tests for the Deepgram STT provider."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from roomkit.voice.base import AudioChunk


def _mock_deepgram_module() -> MagicMock:
    """Create a MagicMock that stands in for the deepgram module."""
    mod = MagicMock()
    # AsyncDeepgramClient must return a mock with async transcribe_file
    client = MagicMock()
    mod.AsyncDeepgramClient.return_value = client
    return mod


def _make_provider(dg_mock: MagicMock, **config_kwargs: Any) -> Any:
    """Build a DeepgramSTTProvider with mocked deepgram dependency."""
    with patch.dict("sys.modules", {"deepgram": dg_mock}):
        import roomkit.voice.stt.deepgram as stt_mod

        importlib.reload(stt_mod)
        from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider

        cfg = DeepgramConfig(**config_kwargs)
        return DeepgramSTTProvider(cfg)


class TestDeepgramSTTProvider:
    def test_constructor_stores_config(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="test-key")
        assert provider._config.api_key == "test-key"
        assert provider._config.model == "nova-2"
        dg.AsyncDeepgramClient.assert_called_once_with(api_key="test-key")

    def test_name(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="k")
        assert provider.name == "DeepgramSTT"

    def test_supports_streaming(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="k")
        assert provider.supports_streaming is True

    async def test_transcribe_batch(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="k")

        # Build a fake Deepgram response
        alt = SimpleNamespace(transcript="hello world", confidence=0.95)
        channel = SimpleNamespace(alternatives=[alt])
        response = SimpleNamespace(results=SimpleNamespace(channels=[channel]))
        provider._client.listen.v1.media.transcribe_file = AsyncMock(return_value=response)

        audio = AudioChunk(data=b"\x00\x01" * 100, sample_rate=16000)
        result = await provider.transcribe(audio)

        assert result.text == "hello world"
        assert result.confidence == 0.95
        provider._client.listen.v1.media.transcribe_file.assert_awaited_once()

    async def test_transcribe_batch_empty_response(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="k")

        # Response with no channels/alternatives
        response = SimpleNamespace(results=SimpleNamespace(channels=[]))
        provider._client.listen.v1.media.transcribe_file = AsyncMock(return_value=response)

        audio = AudioChunk(data=b"\x00\x01" * 100, sample_rate=16000)
        result = await provider.transcribe(audio)

        assert result.text == ""

    async def test_close(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="k")
        # close() should not raise
        await provider.close()

    def test_build_connect_options(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="k", model="nova-3", language="fr")
        opts = provider._build_connect_options(sample_rate=24000)
        assert opts["model"] == "nova-3"
        assert opts["language"] == "fr"
        assert opts["sample_rate"] == "24000"
        assert opts["encoding"] == "linear16"


# ---------------------------------------------------------------------------
# Language: what Deepgram reports, and the per-call override
# ---------------------------------------------------------------------------


def _alt(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(transcript="x", confidence=0.9, **fields)


class TestReportedLanguage:
    """``_reported_language`` reads every shape Deepgram uses, and only reports."""

    def test_word_majority_wins(self) -> None:
        from roomkit.voice.stt.deepgram import _reported_language

        words = [
            SimpleNamespace(word="bonjour", language="fr"),
            SimpleNamespace(word="hello", language="en"),
            SimpleNamespace(word="monde", language="fr"),
        ]
        assert _reported_language(_alt(words=words, languages=["en", "fr"])) == "fr"

    def test_tie_keeps_the_first_heard(self) -> None:
        from roomkit.voice.stt.deepgram import _reported_language

        words = [
            SimpleNamespace(word="hello", language="en"),
            SimpleNamespace(word="bonjour", language="fr"),
        ]
        assert _reported_language(_alt(words=words)) == "en"

    def test_languages_list_when_words_are_untagged(self) -> None:
        from roomkit.voice.stt.deepgram import _reported_language

        words = [SimpleNamespace(word="hola")]
        assert _reported_language(_alt(words=words, languages=["es"])) == "es"
        # v2 shape: objects with a score
        scored = [{"language": "de", "score": 0.97}]
        assert _reported_language(_alt(words=words, languages=scored)) == "de"
        objs = [SimpleNamespace(language="it", score=0.9)]
        assert _reported_language(_alt(words=words, languages=objs)) == "it"

    def test_dict_words_are_read_too(self) -> None:
        from roomkit.voice.stt.deepgram import _reported_language

        words = [{"word": "oui", "language": "fr"}, {"word": "oui", "language": "fr"}]
        assert _reported_language(_alt(words=words)) == "fr"

    def test_pinned_stream_reports_nothing(self) -> None:
        """No tags, no list: None — never an echo of the requested language."""
        from roomkit.voice.stt.deepgram import _reported_language

        words = [SimpleNamespace(word="bonjour"), SimpleNamespace(word="monde")]
        assert _reported_language(_alt(words=words)) is None
        assert _reported_language(_alt()) is None

    def test_prerecorded_detection_on_the_channel(self) -> None:
        from roomkit.voice.stt.deepgram import _reported_language

        channel = SimpleNamespace(detected_language="pt")
        assert _reported_language(_alt(words=[]), channel) == "pt"


class TestLanguageOverride:
    def test_supports_language_override(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="k")
        assert provider.supports_language_override is True

    def test_connect_options_take_the_override(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="k", model="nova-3", language="multi")
        assert provider._build_connect_options(16000)["language"] == "multi"
        assert provider._build_connect_options(16000, None)["language"] == "multi"
        assert provider._build_connect_options(16000, "fr-CA")["language"] == "fr-CA"

    async def test_batch_transcribe_takes_the_override_and_reports(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="k", language="multi")

        alt = SimpleNamespace(
            transcript="bonjour le monde",
            confidence=0.95,
            words=[SimpleNamespace(word="bonjour", language="fr")],
        )
        channel = SimpleNamespace(alternatives=[alt])
        response = SimpleNamespace(results=SimpleNamespace(channels=[channel]))
        provider._client.listen.v1.media.transcribe_file = AsyncMock(return_value=response)

        audio = AudioChunk(data=b"\x00\x01" * 100, sample_rate=16000)
        result = await provider.transcribe(audio, language="fr-CA")

        kwargs = provider._client.listen.v1.media.transcribe_file.await_args.kwargs
        assert kwargs["language"] == "fr-CA"
        assert result.language == "fr"

    async def test_batch_transcribe_default_is_the_config(self) -> None:
        dg = _mock_deepgram_module()
        provider = _make_provider(dg, api_key="k", language="multi")
        alt = SimpleNamespace(transcript="hi", confidence=0.9)
        response = SimpleNamespace(
            results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[alt])])
        )
        provider._client.listen.v1.media.transcribe_file = AsyncMock(return_value=response)

        result = await provider.transcribe(AudioChunk(data=b"\x00\x01" * 10, sample_rate=16000))

        kwargs = provider._client.listen.v1.media.transcribe_file.await_args.kwargs
        assert kwargs["language"] == "multi"
        assert result.language is None
