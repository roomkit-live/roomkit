"""Tests for the ElevenLabs TTS provider."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


def _mock_elevenlabs_module() -> MagicMock:
    """Create a MagicMock that stands in for the elevenlabs module."""
    mod = MagicMock()
    # VoiceSettings used in _make_voice_settings()
    mod.VoiceSettings = MagicMock()
    # AsyncElevenLabs client constructor
    client = MagicMock()
    mod.client.AsyncElevenLabs.return_value = client
    return mod


def _make_provider(el_mock: MagicMock, **config_kwargs: Any) -> Any:
    """Build an ElevenLabsTTSProvider with mocked elevenlabs dependency."""
    with patch.dict(
        "sys.modules",
        {
            "elevenlabs": el_mock,
            "elevenlabs.client": el_mock.client,
        },
    ):
        import roomkit.voice.tts.elevenlabs as tts_mod

        importlib.reload(tts_mod)
        from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

        cfg = ElevenLabsConfig(**config_kwargs)
        return ElevenLabsTTSProvider(cfg)


class TestElevenLabsTTSProvider:
    def test_constructor_stores_config(self) -> None:
        el = _mock_elevenlabs_module()
        provider = _make_provider(el, api_key="test-key")
        assert provider._config.api_key == "test-key"
        assert provider._config.voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert provider._client is None  # lazy

    def test_name(self) -> None:
        el = _mock_elevenlabs_module()
        provider = _make_provider(el, api_key="k")
        assert provider.name == "ElevenLabsTTS"

    def test_default_voice(self) -> None:
        el = _mock_elevenlabs_module()
        provider = _make_provider(el, api_key="k", voice_id="custom-voice")
        assert provider.default_voice == "custom-voice"

    def test_supports_streaming_input_false(self) -> None:
        el = _mock_elevenlabs_module()
        provider = _make_provider(el, api_key="k")
        assert provider.supports_streaming_input is False

    def test_voice_settings_normal_mode(self) -> None:
        el = _mock_elevenlabs_module()
        provider = _make_provider(
            el,
            api_key="k",
            stability=0.7,
            similarity_boost=0.8,
            style=0.3,
            use_speaker_boost=True,
        )
        settings = provider._build_voice_settings()
        assert settings["stability"] == 0.7
        assert settings["similarity_boost"] == 0.8
        assert settings["style"] == 0.3
        assert settings["use_speaker_boost"] is True

    def test_voice_settings_expressive_mode(self) -> None:
        el = _mock_elevenlabs_module()
        provider = _make_provider(
            el,
            api_key="k",
            expressive=True,
            stability=0.5,
            similarity_boost=0.75,
            style=0.3,
        )
        # Expressive mode forces v3 model and omits style/speaker_boost
        assert provider._config.model_id == "eleven_v3"
        settings = provider._build_voice_settings()
        assert "style" not in settings
        assert "use_speaker_boost" not in settings
        assert settings["stability"] == 0.5
        assert settings["similarity_boost"] == 0.75

    async def test_synthesize(self) -> None:
        el = _mock_elevenlabs_module()
        provider = _make_provider(el, api_key="k", output_format="pcm_16000")

        fake_client = MagicMock()
        fake_client.text_to_speech.convert = AsyncMock(return_value=b"\x00\x01" * 50)
        provider._client = fake_client

        with patch.dict("sys.modules", {"elevenlabs": el, "elevenlabs.client": el.client}):
            result = await provider.synthesize("hello world")

        assert result.transcript == "hello world"
        assert result.url.startswith("data:audio/pcm;base64,")
        fake_client.text_to_speech.convert.assert_awaited_once()

    async def test_close(self) -> None:
        el = _mock_elevenlabs_module()
        provider = _make_provider(el, api_key="k")
        provider._client = MagicMock()
        await provider.close()
        assert provider._client is None

    def test_sample_rate_detection(self) -> None:
        el = _mock_elevenlabs_module()
        p1 = _make_provider(el, api_key="k", output_format="mp3_44100_128")
        assert p1._get_sample_rate() == 44100

        p2 = _make_provider(el, api_key="k", output_format="pcm_16000")
        assert p2._get_sample_rate() == 16000

        p3 = _make_provider(el, api_key="k", output_format="pcm_24000")
        assert p3._get_sample_rate() == 24000

    def test_mime_type_detection(self) -> None:
        el = _mock_elevenlabs_module()
        p1 = _make_provider(el, api_key="k", output_format="mp3_44100_128")
        assert p1._get_mime_type() == "audio/mpeg"

        p2 = _make_provider(el, api_key="k", output_format="pcm_16000")
        assert p2._get_mime_type() == "audio/pcm"

        p3 = _make_provider(el, api_key="k", output_format="ulaw_8000")
        assert p3._get_mime_type() == "audio/basic"
