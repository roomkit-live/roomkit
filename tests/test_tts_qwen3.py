"""Tests for Qwen3-TTS provider."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from roomkit.voice.base import AudioChunk

# ---------------------------------------------------------------------------
# Helpers: mock qwen_tts so tests run without GPU / real weights
# ---------------------------------------------------------------------------


class _FakeQwen3TTSModel:
    """Fake model for testing without real qwen_tts."""

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: Any
    ) -> _FakeQwen3TTSModel:
        return cls()

    def create_voice_clone_prompt(
        self, ref_audio: str, ref_text: str | None = None, **kwargs: Any
    ) -> list[str]:
        return [f"prompt:{ref_audio}:{ref_text}"]

    def generate_voice_clone(self, **kwargs: Any) -> tuple[list[Any], int]:
        # Return list of arrays (one per text input): 0.5s of silence at 16kHz
        samples = np.zeros(8000, dtype=np.float32)
        return [samples], 16000


def _make_fake_torch() -> ModuleType:
    """Create a minimal fake torch module with dtype constants."""
    mod = ModuleType("torch")
    mod.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    mod.float16 = "float16"  # type: ignore[attr-defined]
    mod.float32 = "float32"  # type: ignore[attr-defined]
    mod.dtype = type("dtype", (), {})  # type: ignore[attr-defined]
    return mod


def _install_fake_qwen_tts() -> ModuleType:
    """Install fake qwen_tts and torch modules into sys.modules."""
    mod = ModuleType("qwen_tts")
    mod.Qwen3TTSModel = _FakeQwen3TTSModel  # type: ignore[attr-defined]
    sys.modules["qwen_tts"] = mod
    # _load_model() imports torch for dtype mapping; provide a fake
    if "torch" not in sys.modules:
        sys.modules["torch"] = _make_fake_torch()
    return mod


def _uninstall_fake_qwen_tts() -> None:
    """Remove the fake qwen_tts module."""
    sys.modules.pop("qwen_tts", None)


@pytest.fixture(autouse=True)
def _fake_qwen_tts():
    """Install fake qwen_tts for all tests, clean up after."""
    _install_fake_qwen_tts()
    yield
    _uninstall_fake_qwen_tts()


# Imports after fixture ensures qwen_tts is available at import time.
# The autouse fixture runs before each test, but we need the module
# available at import time for the top-level import below.
_install_fake_qwen_tts()

from roomkit.voice.tts.qwen3 import (  # noqa: E402
    Qwen3TTSConfig,
    Qwen3TTSProvider,
    VoiceCloneConfig,
    _numpy_to_pcm_s16le,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestQwen3TTSConfig:
    def test_defaults(self):
        cfg = Qwen3TTSConfig()
        assert cfg.model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        assert cfg.device_map == "auto"
        assert cfg.dtype == "bfloat16"
        assert cfg.attn_implementation is None
        assert cfg.language == "English"
        assert cfg.voices == {}
        assert cfg.x_vector_only_mode is False
        assert cfg.max_new_tokens == 4096

    def test_custom_values(self):
        voice = VoiceCloneConfig(ref_audio="/path/to/audio.wav", ref_text="Hello world")
        cfg = Qwen3TTSConfig(
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map="cuda:0",
            dtype="float16",
            language="Chinese",
            voices={"alice": voice},
        )
        assert cfg.model_id == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        assert cfg.device_map == "cuda:0"
        assert "alice" in cfg.voices
        assert cfg.voices["alice"].ref_audio == "/path/to/audio.wav"


class TestVoiceCloneConfig:
    def test_creation(self):
        vc = VoiceCloneConfig(ref_audio="ref.wav", ref_text="test text")
        assert vc.ref_audio == "ref.wav"
        assert vc.ref_text == "test text"


# ---------------------------------------------------------------------------
# Import check
# ---------------------------------------------------------------------------


class TestImportCheck:
    def test_raises_without_qwen_tts(self):
        """Provider raises ImportError with helpful message when qwen_tts missing."""
        import builtins

        real_import = builtins.__import__

        def _block_qwen_tts(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "qwen_tts":
                raise ImportError("No module named 'qwen_tts'")
            return real_import(name, *args, **kwargs)

        # Clear cached provider module so the import check re-runs
        sys.modules.pop("roomkit.voice.tts.qwen3", None)
        _uninstall_fake_qwen_tts()

        try:
            builtins.__import__ = _block_qwen_tts
            # Re-import the module with qwen_tts blocked
            from roomkit.voice.tts.qwen3 import Qwen3TTSConfig as Cfg  # noqa: N813
            from roomkit.voice.tts.qwen3 import VoiceCloneConfig as VCC  # noqa: N813

            cfg = Cfg(voices={"v": VCC(ref_audio="a.wav", ref_text="hi")})

            with pytest.raises(ImportError, match="qwen-tts is required"):
                from roomkit.voice.tts.qwen3 import (  # noqa: N813
                    Qwen3TTSProvider as Prov,
                )

                Prov(cfg)
        finally:
            builtins.__import__ = real_import
            _install_fake_qwen_tts()


# ---------------------------------------------------------------------------
# PCM conversion
# ---------------------------------------------------------------------------


class TestNumpyToPcmS16le:
    def test_silence(self):
        samples = np.zeros(100, dtype=np.float32)
        pcm = _numpy_to_pcm_s16le(samples)
        assert len(pcm) == 200  # 100 samples * 2 bytes

    def test_max_amplitude(self):
        samples = np.ones(1, dtype=np.float32)
        pcm = _numpy_to_pcm_s16le(samples)
        value = int.from_bytes(pcm, byteorder="little", signed=True)
        assert value == 32767

    def test_clipping(self):
        samples = np.array([2.0, -2.0], dtype=np.float32)
        pcm = _numpy_to_pcm_s16le(samples)
        assert len(pcm) == 4
        val1 = int.from_bytes(pcm[0:2], byteorder="little", signed=True)
        val2 = int.from_bytes(pcm[2:4], byteorder="little", signed=True)
        assert val1 == 32767  # clipped to max
        assert val2 == -32767  # clipped to min (note: -1.0 * 32767)


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


class TestVoiceResolution:
    def test_default_voice_is_first(self):
        cfg = Qwen3TTSConfig(
            voices={
                "alice": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi"),
                "bob": VoiceCloneConfig(ref_audio="b.wav", ref_text="hey"),
            }
        )
        provider = Qwen3TTSProvider(cfg)
        assert provider.default_voice == "alice"

    def test_default_voice_none_when_empty(self):
        cfg = Qwen3TTSConfig()
        provider = Qwen3TTSProvider(cfg)
        assert provider.default_voice is None

    def test_resolve_named_voice(self):
        cfg = Qwen3TTSConfig(voices={"alice": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi")})
        provider = Qwen3TTSProvider(cfg)
        resolved = provider._resolve_voice("alice")
        assert resolved.ref_audio == "a.wav"

    def test_resolve_none_gets_first(self):
        cfg = Qwen3TTSConfig(voices={"alice": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi")})
        provider = Qwen3TTSProvider(cfg)
        resolved = provider._resolve_voice(None)
        assert resolved.ref_audio == "a.wav"

    def test_resolve_missing_voice_raises(self):
        cfg = Qwen3TTSConfig(voices={"alice": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi")})
        provider = Qwen3TTSProvider(cfg)
        with pytest.raises(ValueError, match="Voice 'bob' not found"):
            provider._resolve_voice("bob")

    def test_resolve_empty_voices_raises(self):
        cfg = Qwen3TTSConfig()
        provider = Qwen3TTSProvider(cfg)
        with pytest.raises(ValueError, match="No voices configured"):
            provider._resolve_voice(None)


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


class TestProviderName:
    def test_name(self):
        cfg = Qwen3TTSConfig(voices={"v": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi")})
        provider = Qwen3TTSProvider(cfg)
        assert provider.name == "Qwen3TTS"


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    async def test_warmup_loads_model_and_prompts(self):
        cfg = Qwen3TTSConfig(voices={"alice": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi")})
        provider = Qwen3TTSProvider(cfg)
        assert provider._model is None

        await provider.warmup()

        assert provider._model is not None
        assert "alice" in provider._cached_prompts


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    async def test_synthesize_returns_audio_content(self):
        cfg = Qwen3TTSConfig(voices={"alice": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi")})
        provider = Qwen3TTSProvider(cfg)
        await provider.warmup()

        result = await provider.synthesize("Hello world")

        assert result.type == "audio"
        assert result.url.startswith("data:audio/wav;base64,")
        assert result.mime_type == "audio/wav"
        assert result.transcript == "Hello world"
        assert result.duration_seconds is not None
        assert result.duration_seconds == pytest.approx(0.5, abs=0.01)

    async def test_synthesize_with_named_voice(self):
        cfg = Qwen3TTSConfig(
            voices={
                "alice": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi"),
                "bob": VoiceCloneConfig(ref_audio="b.wav", ref_text="hey"),
            }
        )
        provider = Qwen3TTSProvider(cfg)
        await provider.warmup()

        result = await provider.synthesize("Test", voice="bob")
        assert result.transcript == "Test"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestSynthesizeStream:
    async def test_stream_yields_chunks_and_final(self):
        cfg = Qwen3TTSConfig(voices={"alice": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi")})
        provider = Qwen3TTSProvider(cfg)
        await provider.warmup()

        chunks: list[AudioChunk] = []
        async for chunk in provider.synthesize_stream("Hello"):
            chunks.append(chunk)

        # Should have at least 2 chunks (data + final marker)
        assert len(chunks) >= 2

        # All non-final chunks should have data
        for c in chunks[:-1]:
            assert len(c.data) > 0
            assert c.is_final is False
            assert c.sample_rate == 16000
            assert c.format == "pcm_s16le"

        # Last chunk is the final marker
        assert chunks[-1].is_final is True
        assert chunks[-1].data == b""

    async def test_stream_total_pcm_matches_full_audio(self):
        """Total PCM from stream chunks should match full synthesize output size."""
        cfg = Qwen3TTSConfig(voices={"alice": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi")})
        provider = Qwen3TTSProvider(cfg)
        await provider.warmup()

        # Collect stream chunks (excluding final marker)
        stream_pcm = b""
        async for chunk in provider.synthesize_stream("Hello"):
            if not chunk.is_final:
                stream_pcm += chunk.data

        # 8000 samples * 2 bytes = 16000 bytes (0.5s at 16kHz)
        assert len(stream_pcm) == 16000


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_releases_model(self):
        cfg = Qwen3TTSConfig(voices={"alice": VoiceCloneConfig(ref_audio="a.wav", ref_text="hi")})
        provider = Qwen3TTSProvider(cfg)
        await provider.warmup()
        assert provider._model is not None

        await provider.close()
        assert provider._model is None
        assert provider._cached_prompts == {}


# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------


class TestLazyLoaders:
    def test_get_qwen3_tts_provider(self):
        from roomkit.voice import get_qwen3_tts_provider

        cls = get_qwen3_tts_provider()
        assert cls.__name__ == "Qwen3TTSProvider"

    def test_get_qwen3_tts_config(self):
        from roomkit.voice import get_qwen3_tts_config

        cls = get_qwen3_tts_config()
        assert cls.__name__ == "Qwen3TTSConfig"

    def test_get_qwen3_voice_clone_config(self):
        from roomkit.voice import get_qwen3_voice_clone_config

        cls = get_qwen3_voice_clone_config()
        assert cls.__name__ == "VoiceCloneConfig"
