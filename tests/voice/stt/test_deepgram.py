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
