"""Tests for NeuTTS provider."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from roomkit.voice.base import AudioChunk

# ---------------------------------------------------------------------------
# Helpers: mock neutts so tests run without GPU / real weights
# ---------------------------------------------------------------------------

_FAKE_SAMPLE_RATE = 24000


class _FakeNeuTTS:
    """Fake model for testing without real neutts."""

    def __init__(
        self,
        backbone_repo: str = "neuphonic/neutts-nano-french-q8-gguf",
        backbone_device: str = "cpu",
        codec_repo: str = "neuphonic/neucodec",
        codec_device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.backbone_repo = backbone_repo
        self.codec_repo = codec_repo

    def encode_reference(self, ref_audio_path: str) -> np.ndarray:
        return np.zeros(100, dtype=np.float32)

    def infer(self, text: str, ref_codes: Any, ref_text: str, **kwargs: Any) -> np.ndarray:
        # Return 0.5s of silence at 24kHz
        return np.zeros(12000, dtype=np.float32)

    def infer_stream(self, text: str, ref_codes: Any, ref_text: str, **kwargs: Any):
        # Yield 3 chunks of 4000 samples each (total 12000 = 0.5s at 24kHz)
        for _ in range(3):
            yield np.zeros(4000, dtype=np.float32)


def _install_fake_neutts() -> ModuleType:
    """Install fake neutts module into sys.modules."""
    mod = ModuleType("neutts")
    mod.NeuTTS = _FakeNeuTTS  # type: ignore[attr-defined]
    sys.modules["neutts"] = mod
    return mod


def _uninstall_fake_neutts() -> None:
    """Remove the fake neutts module."""
    sys.modules.pop("neutts", None)


@pytest.fixture(autouse=True)
def _fake_neutts():
    """Install fake neutts for all tests, clean up after."""
    _install_fake_neutts()
    yield
    _uninstall_fake_neutts()


# Install before top-level import so the module is available at import time.
_install_fake_neutts()

from roomkit.voice.tts.neutts import (  # noqa: E402
    NeuTTSConfig,
    NeuTTSProvider,
    NeuTTSVoiceConfig,
    _numpy_to_pcm_s16le,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestNeuTTSConfig:
    def test_defaults(self):
        cfg = NeuTTSConfig()
        assert cfg.backbone_repo == "neuphonic/neutts-nano-french-q8-gguf"
        assert cfg.codec_repo == "neuphonic/neucodec"
        assert cfg.device == "cpu"
        assert cfg.voices == {}
        assert cfg.streaming_pre_buffer == 2

    def test_custom_values(self):
        voice = NeuTTSVoiceConfig(ref_audio="/path/to/audio.wav", ref_text="Bonjour")
        cfg = NeuTTSConfig(
            backbone_repo="neuphonic/neutts-nano-q8-gguf",
            codec_repo="/local/neucodec",
            device="cuda",
            voices={"marie": voice},
        )
        assert cfg.backbone_repo == "neuphonic/neutts-nano-q8-gguf"
        assert cfg.codec_repo == "/local/neucodec"
        assert cfg.device == "cuda"
        assert "marie" in cfg.voices
        assert cfg.voices["marie"].ref_audio == "/path/to/audio.wav"


class TestNeuTTSVoiceConfig:
    def test_creation(self):
        vc = NeuTTSVoiceConfig(ref_audio="ref.wav", ref_text="test text")
        assert vc.ref_audio == "ref.wav"
        assert vc.ref_text == "test text"


# ---------------------------------------------------------------------------
# Import check
# ---------------------------------------------------------------------------


class TestImportCheck:
    def test_raises_without_neutts(self):
        """Provider raises ImportError with helpful message when neutts missing."""
        import builtins

        real_import = builtins.__import__

        def _block_neutts(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "neutts":
                raise ImportError("No module named 'neutts'")
            return real_import(name, *args, **kwargs)

        # Clear cached provider module so the import check re-runs
        sys.modules.pop("roomkit.voice.tts.neutts", None)
        _uninstall_fake_neutts()

        try:
            builtins.__import__ = _block_neutts
            from roomkit.voice.tts.neutts import NeuTTSConfig as Cfg  # noqa: N813
            from roomkit.voice.tts.neutts import NeuTTSVoiceConfig as VCC  # noqa: N814

            cfg = Cfg(voices={"v": VCC(ref_audio="a.wav", ref_text="hi")})

            with pytest.raises(ImportError, match="neutts is required"):
                from roomkit.voice.tts.neutts import (  # noqa: N813
                    NeuTTSProvider as Prov,
                )

                Prov(cfg)
        finally:
            builtins.__import__ = real_import
            _install_fake_neutts()


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
        assert val1 == 32767
        assert val2 == -32767


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


class TestVoiceResolution:
    def test_default_voice_is_first(self):
        cfg = NeuTTSConfig(
            voices={
                "marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi"),
                "jean": NeuTTSVoiceConfig(ref_audio="b.wav", ref_text="hey"),
            }
        )
        provider = NeuTTSProvider(cfg)
        assert provider.default_voice == "marie"

    def test_default_voice_none_when_empty(self):
        cfg = NeuTTSConfig()
        provider = NeuTTSProvider(cfg)
        assert provider.default_voice is None

    def test_resolve_named_voice(self):
        cfg = NeuTTSConfig(voices={"marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")})
        provider = NeuTTSProvider(cfg)
        resolved = provider._resolve_voice("marie")
        assert resolved.ref_audio == "a.wav"

    def test_resolve_none_gets_first(self):
        cfg = NeuTTSConfig(voices={"marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")})
        provider = NeuTTSProvider(cfg)
        resolved = provider._resolve_voice(None)
        assert resolved.ref_audio == "a.wav"

    def test_resolve_missing_voice_raises(self):
        cfg = NeuTTSConfig(voices={"marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")})
        provider = NeuTTSProvider(cfg)
        with pytest.raises(ValueError, match="Voice 'bob' not found"):
            provider._resolve_voice("bob")

    def test_resolve_empty_voices_raises(self):
        cfg = NeuTTSConfig()
        provider = NeuTTSProvider(cfg)
        with pytest.raises(ValueError, match="No voices configured"):
            provider._resolve_voice(None)


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


class TestProviderName:
    def test_name(self):
        cfg = NeuTTSConfig(voices={"v": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")})
        provider = NeuTTSProvider(cfg)
        assert provider.name == "NeuTTS"


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    async def test_warmup_loads_model_and_refs(self):
        cfg = NeuTTSConfig(voices={"marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")})
        provider = NeuTTSProvider(cfg)
        assert provider._model is None

        await provider.warmup()

        assert provider._model is not None
        assert "marie" in provider._cached_refs


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    async def test_synthesize_returns_audio_content(self):
        cfg = NeuTTSConfig(voices={"marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")})
        provider = NeuTTSProvider(cfg)
        await provider.warmup()

        result = await provider.synthesize("Bonjour le monde")

        assert result.type == "audio"
        assert result.url.startswith("data:audio/wav;base64,")
        assert result.mime_type == "audio/wav"
        assert result.transcript == "Bonjour le monde"
        assert result.duration_seconds is not None
        assert result.duration_seconds == pytest.approx(0.5, abs=0.01)

    async def test_synthesize_with_named_voice(self):
        cfg = NeuTTSConfig(
            voices={
                "marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi"),
                "jean": NeuTTSVoiceConfig(ref_audio="b.wav", ref_text="hey"),
            }
        )
        provider = NeuTTSProvider(cfg)
        await provider.warmup()

        result = await provider.synthesize("Test", voice="jean")
        assert result.transcript == "Test"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestSynthesizeStream:
    async def test_stream_yields_chunks_and_final(self):
        cfg = NeuTTSConfig(voices={"marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")})
        provider = NeuTTSProvider(cfg)
        await provider.warmup()

        chunks: list[AudioChunk] = []
        async for chunk in provider.synthesize_stream("Bonjour"):
            chunks.append(chunk)

        # Should have at least 2 chunks (data + final marker)
        assert len(chunks) >= 2

        # All non-final chunks should have data
        for c in chunks[:-1]:
            assert len(c.data) > 0
            assert c.is_final is False
            assert c.sample_rate == 24000
            assert c.format == "pcm_s16le"

        # Last chunk is the final marker
        assert chunks[-1].is_final is True
        assert chunks[-1].data == b""

    async def test_stream_total_pcm_matches_expected(self):
        """Total PCM from stream should match the fake model output."""
        cfg = NeuTTSConfig(voices={"marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")})
        provider = NeuTTSProvider(cfg)
        await provider.warmup()

        stream_pcm = b""
        async for chunk in provider.synthesize_stream("Bonjour"):
            if not chunk.is_final:
                stream_pcm += chunk.data

        # 3 chunks * 4000 samples * 2 bytes = 24000 bytes
        assert len(stream_pcm) == 24000

    async def test_pre_buffer_disabled(self):
        """streaming_pre_buffer=0 yields chunks immediately."""
        cfg = NeuTTSConfig(
            voices={"marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")},
            streaming_pre_buffer=0,
        )
        provider = NeuTTSProvider(cfg)
        await provider.warmup()

        chunks: list[AudioChunk] = []
        async for chunk in provider.synthesize_stream("Bonjour"):
            chunks.append(chunk)

        # 3 data chunks + final marker
        assert len(chunks) == 4
        for c in chunks[:-1]:
            assert len(c.data) > 0

    async def test_stream_disables_watermarker(self):
        """Streaming should disable Perth watermarker to avoid per-chunk artifacts."""
        cfg = NeuTTSConfig(
            voices={"marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")},
            streaming_pre_buffer=0,
        )
        provider = NeuTTSProvider(cfg)
        await provider.warmup()

        # Simulate a watermarker being present
        sentinel = object()
        provider._model.watermarker = sentinel

        async for _ in provider.synthesize_stream("Bonjour"):
            pass

        # Watermarker should be restored after streaming completes
        assert provider._model.watermarker is sentinel


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_releases_model(self):
        cfg = NeuTTSConfig(voices={"marie": NeuTTSVoiceConfig(ref_audio="a.wav", ref_text="hi")})
        provider = NeuTTSProvider(cfg)
        await provider.warmup()
        assert provider._model is not None

        await provider.close()
        assert provider._model is None
        assert provider._cached_refs == {}


# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------


class TestLazyLoaders:
    def test_get_neutts_provider(self):
        from roomkit.voice import get_neutts_provider

        cls = get_neutts_provider()
        assert cls.__name__ == "NeuTTSProvider"

    def test_get_neutts_config(self):
        from roomkit.voice import get_neutts_config

        cls = get_neutts_config()
        assert cls.__name__ == "NeuTTSConfig"

    def test_get_neutts_voice_config(self):
        from roomkit.voice import get_neutts_voice_config

        cls = get_neutts_voice_config()
        assert cls.__name__ == "NeuTTSVoiceConfig"
