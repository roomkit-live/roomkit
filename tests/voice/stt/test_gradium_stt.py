"""Tests for the Gradium STT provider."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.voice.base import AudioChunk


def _mock_gradium_module() -> MagicMock:
    """Create a MagicMock that stands in for the gradium module."""
    mod = MagicMock()
    client = MagicMock()
    mod.GradiumClient.return_value = client
    return mod


def _make_provider(grad_mock: MagicMock, **config_kwargs: Any) -> Any:
    """Build a GradiumSTTProvider with mocked gradium dependency."""
    with patch.dict("sys.modules", {"gradium": grad_mock}):
        import roomkit.voice.stt.gradium as stt_mod

        importlib.reload(stt_mod)
        from roomkit.voice.stt.gradium import GradiumSTTConfig, GradiumSTTProvider

        cfg = GradiumSTTConfig(**config_kwargs)
        return GradiumSTTProvider(cfg)


class TestGradiumSTTProvider:
    def test_constructor_stores_config(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="test-key")
        assert provider._config.api_key == "test-key"
        assert provider._config.region == "us"
        assert provider._client is None  # lazy init

    def test_name(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k")
        assert provider.name == "GradiumSTT"

    def test_supports_streaming(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k")
        assert provider.supports_streaming is True

    def test_get_client_lazy_init(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", region="eu")
        assert provider._client is None

        with patch.dict("sys.modules", {"gradium": grad}):
            client = provider._get_client()

        grad.GradiumClient.assert_called_once_with(
            base_url="https://eu.api.gradium.ai/api/",
            api_key="k",
        )
        assert provider._client is client

    async def test_transcribe_batch(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k")

        fake_result = SimpleNamespace(text="transcribed text")
        fake_client = MagicMock()
        fake_client.stt = AsyncMock(return_value=fake_result)
        provider._client = fake_client

        audio = AudioChunk(data=b"\x00\x01" * 100, sample_rate=16000)
        result = await provider.transcribe(audio)

        assert result.text == "transcribed text"
        fake_client.stt.assert_awaited_once()

    async def test_transcribe_rejects_url_audio(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k")

        audio_with_url = SimpleNamespace(url="https://example.com/audio.wav")
        with pytest.raises(ValueError, match="URL-based AudioContent"):
            await provider.transcribe(audio_with_url)

    async def test_close_clears_client(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k")
        provider._client = MagicMock()
        await provider.close()
        assert provider._client is None

    def test_build_setup(self) -> None:
        grad = _mock_gradium_module()
        provider = _make_provider(grad, api_key="k", model_name="fast", language="fr")
        setup = provider._build_setup()
        assert setup["model_name"] == "fast"
        assert setup["json_config"]["language"] == "fr"
        assert setup["json_config"]["delay_in_frames"] == 7
