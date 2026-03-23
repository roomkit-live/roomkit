"""Tests for the Qwen3-TTS provider."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _mock_deps() -> tuple[MagicMock, MagicMock]:
    """Create mocks for qwen_tts and torch modules."""
    qwen_tts = MagicMock()
    torch = MagicMock()
    torch.bfloat16 = "bf16_sentinel"
    torch.float16 = "f16_sentinel"
    torch.float32 = "f32_sentinel"
    torch.cuda.is_available.return_value = False
    return qwen_tts, torch


def _make_provider(qwen_tts: MagicMock, torch: MagicMock, **config_kwargs: Any) -> Any:
    """Build a Qwen3TTSProvider with mocked dependencies."""
    with patch.dict("sys.modules", {"qwen_tts": qwen_tts, "torch": torch}):
        import roomkit.voice.tts.qwen3 as tts_mod

        importlib.reload(tts_mod)
        from roomkit.voice.tts.qwen3 import Qwen3TTSConfig, Qwen3TTSProvider, VoiceCloneConfig

        voices = config_kwargs.pop("voices", None)
        cfg = Qwen3TTSConfig(**config_kwargs)
        if voices is not None:
            cfg.voices = {name: VoiceCloneConfig(**v) for name, v in voices.items()}
        return Qwen3TTSProvider(cfg)


class TestQwen3TTSProvider:
    def test_constructor_stores_config(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(qwen_tts, torch)
        assert provider._config.model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        assert provider._model is None

    def test_name(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(qwen_tts, torch)
        assert provider.name == "Qwen3TTS"

    def test_default_voice_none_when_no_voices(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(qwen_tts, torch)
        assert provider.default_voice is None

    def test_default_voice_returns_first(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(
            qwen_tts,
            torch,
            voices={
                "alice": {"ref_audio": "alice.wav", "ref_text": "Hello"},
                "bob": {"ref_audio": "bob.wav", "ref_text": "Hi"},
            },
        )
        assert provider.default_voice == "alice"

    def test_resolve_voice_by_name(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(
            qwen_tts,
            torch,
            voices={"alice": {"ref_audio": "a.wav", "ref_text": "text"}},
        )
        voice = provider._resolve_voice("alice")
        assert voice.ref_audio == "a.wav"
        assert voice.ref_text == "text"

    def test_resolve_voice_none_uses_first(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(
            qwen_tts,
            torch,
            voices={"alice": {"ref_audio": "a.wav", "ref_text": "text"}},
        )
        voice = provider._resolve_voice(None)
        assert voice.ref_audio == "a.wav"

    def test_resolve_voice_missing_raises(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(
            qwen_tts,
            torch,
            voices={"alice": {"ref_audio": "a.wav", "ref_text": "text"}},
        )
        with pytest.raises(ValueError, match="Voice 'bob' not found"):
            provider._resolve_voice("bob")

    def test_resolve_voice_no_voices_raises(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(qwen_tts, torch)
        with pytest.raises(ValueError, match="No voices configured"):
            provider._resolve_voice(None)

    def test_load_model_from_pretrained(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(qwen_tts, torch, dtype="bfloat16")

        with patch.dict("sys.modules", {"qwen_tts": qwen_tts, "torch": torch}):
            model = provider._load_model()

        qwen_tts.Qwen3TTSModel.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="auto",
            dtype="bf16_sentinel",
        )
        assert provider._model is model

    def test_load_model_unsupported_dtype(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(qwen_tts, torch, dtype="int8")

        with (
            patch.dict("sys.modules", {"qwen_tts": qwen_tts, "torch": torch}),
            pytest.raises(ValueError, match="Unsupported dtype"),
        ):
            provider._load_model()

    def test_load_model_with_attn_implementation(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(
            qwen_tts,
            torch,
            dtype="float16",
            attn_implementation="flash_attention_2",
        )

        with patch.dict("sys.modules", {"qwen_tts": qwen_tts, "torch": torch}):
            provider._load_model()

        call_kwargs = qwen_tts.Qwen3TTSModel.from_pretrained.call_args[1]
        assert call_kwargs["attn_implementation"] == "flash_attention_2"

    def test_import_error_without_qwen_tts(self) -> None:
        """Constructor should raise ImportError when qwen_tts is missing."""
        with patch.dict("sys.modules", {"qwen_tts": None}):
            import roomkit.voice.tts.qwen3 as tts_mod

            importlib.reload(tts_mod)
            from roomkit.voice.tts.qwen3 import Qwen3TTSConfig, Qwen3TTSProvider

            with pytest.raises(ImportError, match="qwen-tts is required"):
                Qwen3TTSProvider(Qwen3TTSConfig())

    async def test_close_releases_model(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(qwen_tts, torch)
        provider._model = MagicMock()
        provider._cached_prompts["test"] = "value"

        await provider.close()

        assert provider._model is None
        assert len(provider._cached_prompts) == 0

    async def test_close_noop_when_no_model(self) -> None:
        qwen_tts, torch = _mock_deps()
        provider = _make_provider(qwen_tts, torch)
        # Should not raise when no model is loaded
        await provider.close()
        assert provider._model is None
