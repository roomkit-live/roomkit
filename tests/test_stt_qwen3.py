"""Tests for Qwen3-ASR STT provider."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from roomkit.voice.base import AudioChunk

# ---------------------------------------------------------------------------
# Helpers: mock qwen_asr so tests run without GPU / real weights
# ---------------------------------------------------------------------------


class _FakeQwen3ASRModel:
    """Fake model for testing without real qwen_asr."""

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: Any
    ) -> _FakeQwen3ASRModel:
        return cls()

    @classmethod
    def LLM(cls, model_id: str, **kwargs: Any) -> _FakeQwen3ASRModel:  # noqa: N802
        return cls()

    def transcribe(self, **kwargs: Any) -> list[str]:
        return ["Hello world"]

    def init_streaming_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"chunks_fed": 0}

    def streaming_transcribe(self, **kwargs: Any) -> str:
        state = kwargs.get("state", {})
        n = state.get("chunks_fed", 0)
        state["chunks_fed"] = n + 1
        return f"partial {n + 1}"

    def finish_streaming_transcribe(self, **kwargs: Any) -> str:
        return "Hello world final"


def _make_fake_torch() -> ModuleType:
    """Create a minimal fake torch module with dtype constants."""
    mod = ModuleType("torch")
    mod.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    mod.float16 = "float16"  # type: ignore[attr-defined]
    mod.float32 = "float32"  # type: ignore[attr-defined]
    mod.dtype = type("dtype", (), {})  # type: ignore[attr-defined]

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            pass

    mod.cuda = _FakeCuda  # type: ignore[attr-defined]
    return mod


def _install_fake_qwen_asr() -> ModuleType:
    """Install fake qwen_asr and torch modules into sys.modules."""
    mod = ModuleType("qwen_asr")
    mod.Qwen3ASRModel = _FakeQwen3ASRModel  # type: ignore[attr-defined]
    sys.modules["qwen_asr"] = mod
    if "torch" not in sys.modules:
        sys.modules["torch"] = _make_fake_torch()
    return mod


def _uninstall_fake_qwen_asr() -> None:
    """Remove the fake qwen_asr module."""
    sys.modules.pop("qwen_asr", None)


@pytest.fixture(autouse=True)
def _fake_qwen_asr():
    """Install fake qwen_asr for all tests, clean up after."""
    _install_fake_qwen_asr()
    yield
    _uninstall_fake_qwen_asr()


# Install once at import time so the top-level import below works.
_install_fake_qwen_asr()

from roomkit.voice.stt.qwen3 import (  # noqa: E402
    Qwen3ASRConfig,
    Qwen3ASRProvider,
    _pcm_s16le_to_float32_np,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestQwen3ASRConfig:
    def test_defaults(self):
        cfg = Qwen3ASRConfig()
        assert cfg.model_id == "Qwen/Qwen3-ASR-0.6B"
        assert cfg.backend == "transformers"
        assert cfg.device_map == "auto"
        assert cfg.dtype == "bfloat16"
        assert cfg.language is None
        assert cfg.chunk_size_sec == 2.0
        assert cfg.unfixed_chunk_num == 2
        assert cfg.unfixed_token_num == 5
        assert cfg.gpu_memory_utilization == 0.3
        assert cfg.max_new_tokens == 2048

    def test_custom_values(self):
        cfg = Qwen3ASRConfig(
            model_id="Qwen/Qwen3-ASR-4B",
            backend="vllm",
            device_map="cuda:0",
            dtype="float16",
            language="en",
            chunk_size_sec=1.0,
        )
        assert cfg.model_id == "Qwen/Qwen3-ASR-4B"
        assert cfg.backend == "vllm"
        assert cfg.device_map == "cuda:0"
        assert cfg.dtype == "float16"
        assert cfg.language == "en"
        assert cfg.chunk_size_sec == 1.0


# ---------------------------------------------------------------------------
# Import check
# ---------------------------------------------------------------------------


class TestImportCheck:
    def test_raises_without_qwen_asr(self):
        """Provider raises ImportError with helpful message when qwen_asr missing."""
        import builtins

        real_import = builtins.__import__

        def _block_qwen_asr(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "qwen_asr":
                raise ImportError("No module named 'qwen_asr'")
            return real_import(name, *args, **kwargs)

        sys.modules.pop("roomkit.voice.stt.qwen3", None)
        _uninstall_fake_qwen_asr()

        try:
            builtins.__import__ = _block_qwen_asr
            from roomkit.voice.stt.qwen3 import Qwen3ASRConfig as Cfg  # noqa: N813

            cfg = Cfg()

            with pytest.raises(ImportError, match="qwen-asr is required"):
                from roomkit.voice.stt.qwen3 import (  # noqa: N813
                    Qwen3ASRProvider as Prov,
                )

                Prov(cfg)
        finally:
            builtins.__import__ = real_import
            _install_fake_qwen_asr()


# ---------------------------------------------------------------------------
# PCM conversion
# ---------------------------------------------------------------------------


class TestPcmToFloat32Np:
    def test_silence(self):
        data = b"\x00\x00" * 100
        samples = _pcm_s16le_to_float32_np(data)
        assert samples.shape == (100,)
        assert samples.dtype == np.float32
        assert np.allclose(samples, 0.0)

    def test_max_amplitude(self):
        import struct

        data = struct.pack("<h", 32767)
        samples = _pcm_s16le_to_float32_np(data)
        assert samples.shape == (1,)
        assert samples[0] == pytest.approx(32767 / 32768.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


class TestProviderName:
    def test_name(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig())
        assert provider.name == "Qwen3ASR"


# ---------------------------------------------------------------------------
# Streaming support
# ---------------------------------------------------------------------------


class TestSupportsStreaming:
    def test_vllm_supports_streaming(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig(backend="vllm"))
        assert provider.supports_streaming is True

    def test_transformers_no_streaming(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig(backend="transformers"))
        assert provider.supports_streaming is False


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    async def test_warmup_loads_model(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig())
        assert provider._model is None

        await provider.warmup()

        assert provider._model is not None


# ---------------------------------------------------------------------------
# Transcribe (batch)
# ---------------------------------------------------------------------------


class TestTranscribe:
    async def test_transcribe_returns_result(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig())
        await provider.warmup()

        audio = AudioChunk(data=b"\x00\x00" * 1600, sample_rate=16000)
        result = await provider.transcribe(audio)

        assert result.text == "Hello world"
        assert isinstance(result, type(result))

    async def test_transcribe_rejects_url_audio(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig())
        await provider.warmup()

        class _FakeUrlAudio:
            url = "https://example.com/audio.wav"
            data = b""

        with pytest.raises(ValueError, match="does not support URL-based"):
            await provider.transcribe(_FakeUrlAudio())  # type: ignore[arg-type]

    async def test_transcribe_vllm_backend(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig(backend="vllm"))
        await provider.warmup()

        audio = AudioChunk(data=b"\x00\x00" * 1600, sample_rate=16000)
        result = await provider.transcribe(audio)

        assert result.text == "Hello world"


# ---------------------------------------------------------------------------
# Streaming transcription
# ---------------------------------------------------------------------------


class TestTranscribeStream:
    async def test_stream_yields_partials_and_final(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig(backend="vllm"))
        await provider.warmup()

        async def _audio_gen() -> AsyncIterator[AudioChunk]:
            for _ in range(3):
                yield AudioChunk(data=b"\x00\x00" * 1600, sample_rate=16000)
            yield AudioChunk(data=b"", sample_rate=16000, is_final=True)

        from collections.abc import AsyncIterator

        results: list[Any] = []
        async for r in provider.transcribe_stream(_audio_gen()):
            results.append(r)

        # Should have partial results + final
        assert len(results) >= 2
        # Last result should be final
        assert results[-1].is_final is True
        assert results[-1].text == "Hello world final"
        # Earlier results should be partial
        for r in results[:-1]:
            assert r.is_final is False

    async def test_stream_transformers_falls_back_to_batch(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig(backend="transformers"))
        await provider.warmup()

        async def _audio_gen() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=b"\x00\x00" * 1600, sample_rate=16000)

        from collections.abc import AsyncIterator

        results: list[Any] = []
        async for r in provider.transcribe_stream(_audio_gen()):
            results.append(r)

        # Batch fallback returns single final result
        assert len(results) == 1
        assert results[0].is_final is True
        assert results[0].text == "Hello world"


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_releases_model(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig())
        await provider.warmup()
        assert provider._model is not None

        await provider.close()
        assert provider._model is None

    async def test_close_noop_when_no_model(self):
        provider = Qwen3ASRProvider(Qwen3ASRConfig())
        await provider.close()  # Should not raise
        assert provider._model is None


# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------


class TestLazyLoaders:
    def test_get_qwen3_asr_provider(self):
        from roomkit.voice import get_qwen3_asr_provider

        cls = get_qwen3_asr_provider()
        assert cls.__name__ == "Qwen3ASRProvider"

    def test_get_qwen3_asr_config(self):
        from roomkit.voice import get_qwen3_asr_config

        cls = get_qwen3_asr_config()
        assert cls.__name__ == "Qwen3ASRConfig"
