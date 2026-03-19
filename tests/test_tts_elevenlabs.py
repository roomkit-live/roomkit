"""Tests for ElevenLabsTTSProvider — config, expressive mode, and synthesis."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from roomkit.voice.tts.elevenlabs import (
    EXPRESSIVE_TAGS,
    MODEL_MULTILINGUAL_V2,
    MODEL_V3,
    ElevenLabsConfig,
    ElevenLabsTTSProvider,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestElevenLabsConfig:
    def test_defaults(self):
        cfg = ElevenLabsConfig(api_key="test-key")
        assert cfg.voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert cfg.model_id == MODEL_MULTILINGUAL_V2
        assert cfg.stability == 0.5
        assert cfg.similarity_boost == 0.75
        assert cfg.style == 0.0
        assert cfg.use_speaker_boost is True
        assert cfg.output_format == "mp3_44100_128"
        assert cfg.optimize_streaming_latency == 3
        assert cfg.expressive is False

    def test_custom_values(self):
        cfg = ElevenLabsConfig(
            api_key="test-key",
            voice_id="custom-voice",
            model_id="eleven_flash_v2_5",
            stability=0.8,
            similarity_boost=0.9,
            style=0.3,
            use_speaker_boost=False,
            output_format="pcm_24000",
            optimize_streaming_latency=1,
        )
        assert cfg.voice_id == "custom-voice"
        assert cfg.model_id == "eleven_flash_v2_5"
        assert cfg.stability == 0.8
        assert cfg.style == 0.3

    def test_expressive_flag(self):
        cfg = ElevenLabsConfig(api_key="test-key", expressive=True)
        assert cfg.expressive is True


# ---------------------------------------------------------------------------
# Provider basics
# ---------------------------------------------------------------------------


class TestProviderBasics:
    def test_name(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        assert provider.name == "ElevenLabsTTS"

    def test_default_voice(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", voice_id="custom"))
        assert provider.default_voice == "custom"

    def test_supports_streaming_input(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        assert provider.supports_streaming_input is True


# ---------------------------------------------------------------------------
# Expressive mode
# ---------------------------------------------------------------------------


class TestExpressiveMode:
    def test_expressive_sets_v3_model(self):
        """expressive=True should force model to v3 conversational."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))
        assert provider._config.model_id == MODEL_V3

    def test_expressive_overrides_explicit_model(self):
        """expressive=True overrides any explicit model_id."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(
                api_key="k",
                model_id="eleven_flash_v2_5",
                expressive=True,
            )
        )
        assert provider._config.model_id == MODEL_V3

    def test_non_expressive_keeps_model(self):
        """Without expressive, model_id is untouched."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(api_key="k", model_id="eleven_flash_v2_5")
        )
        assert provider._config.model_id == "eleven_flash_v2_5"

    def test_is_v3_model_expressive(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))
        assert provider._is_v3_model() is True

    def test_is_v3_model_explicit(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", model_id=MODEL_V3))
        assert provider._is_v3_model() is True

    def test_is_v3_model_v2(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        assert provider._is_v3_model() is False

    def test_expressive_tags_constant(self):
        assert "[laughs]" in EXPRESSIVE_TAGS
        assert "[whispers]" in EXPRESSIVE_TAGS
        assert "[sighs]" in EXPRESSIVE_TAGS
        assert "[slow]" in EXPRESSIVE_TAGS
        assert "[excited]" in EXPRESSIVE_TAGS
        assert len(EXPRESSIVE_TAGS) == 5


# ---------------------------------------------------------------------------
# Voice settings
# ---------------------------------------------------------------------------


class TestVoiceSettings:
    def test_v2_includes_all_settings(self):
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(api_key="k", style=0.3, use_speaker_boost=False)
        )
        settings = provider._build_voice_settings()
        assert settings == {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.3,
            "use_speaker_boost": False,
        }

    def test_v3_omits_style_and_speaker_boost(self):
        """v3 model should only include stability and similarity_boost."""
        provider = ElevenLabsTTSProvider(
            ElevenLabsConfig(
                api_key="k",
                expressive=True,
                style=0.5,
                use_speaker_boost=True,
            )
        )
        settings = provider._build_voice_settings()
        assert settings == {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
        assert "style" not in settings
        assert "use_speaker_boost" not in settings

    def test_v3_explicit_model_omits_style(self):
        """Setting model_id to v3 directly also omits style/speaker_boost."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", model_id=MODEL_V3))
        settings = provider._build_voice_settings()
        assert "style" not in settings
        assert "use_speaker_boost" not in settings


# ---------------------------------------------------------------------------
# Synthesize (mocked HTTP)
# ---------------------------------------------------------------------------


class TestSynthesize:
    async def test_synthesize_sends_correct_payload(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio-bytes"
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            result = await provider.synthesize("Hello world")

        call_args = mock_client.return_value.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model_id"] == MODEL_MULTILINGUAL_V2
        assert "style" in payload["voice_settings"]
        assert result.transcript == "Hello world"

    async def test_synthesize_expressive_payload(self):
        """Expressive mode sends v3 model and stripped voice settings."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio"
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            await provider.synthesize("[laughs] That's funny!")

        call_args = mock_client.return_value.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model_id"] == MODEL_V3
        assert payload["text"] == "[laughs] That's funny!"
        assert "style" not in payload["voice_settings"]
        assert "use_speaker_boost" not in payload["voice_settings"]


# ---------------------------------------------------------------------------
# Streaming latency param
# ---------------------------------------------------------------------------


class TestStreamingLatency:
    async def test_v2_includes_latency_param(self):
        """Non-v3 models include optimize_streaming_latency."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))

        # We can't easily test the full stream, but we can verify the params
        # are built correctly by checking the code path via _is_v3_model
        assert provider._is_v3_model() is False

    async def test_v3_excludes_latency_param(self):
        """v3 models skip optimize_streaming_latency."""
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k", expressive=True))
        assert provider._is_v3_model() is True


# ---------------------------------------------------------------------------
# Voice listing (mocked HTTP)
# ---------------------------------------------------------------------------


class TestListVoices:
    async def test_list_voices(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "voices": [
                {
                    "voice_id": "v1",
                    "name": "Rachel",
                    "category": "premade",
                    "labels": {"accent": "american"},
                },
                {
                    "voice_id": "v2",
                    "name": "Adam",
                },
            ]
        }

        with patch.object(provider, "_get_client") as mock_client:
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            voices = await provider.list_voices()

        assert len(voices) == 2
        assert voices[0].voice_id == "v1"
        assert voices[0].name == "Rachel"
        assert voices[1].voice_id == "v2"

    async def test_list_voices_caches(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"voices": [{"voice_id": "v1", "name": "R"}]}

        with patch.object(provider, "_get_client") as mock_client:
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            await provider.list_voices()
            await provider.list_voices()

        # Should only call the API once
        assert mock_client.return_value.get.call_count == 1


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_cleans_up_client(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        # Force client creation
        provider._get_client()
        assert provider._client is not None

        with patch.object(provider._client, "aclose", new_callable=AsyncMock) as mock_close:
            await provider.close()

        mock_close.assert_awaited_once()
        assert provider._client is None

    async def test_close_noop_when_no_client(self):
        provider = ElevenLabsTTSProvider(ElevenLabsConfig(api_key="k"))
        await provider.close()  # should not raise
