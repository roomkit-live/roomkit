"""Tests for the Qwen3-ASR STT provider."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _mock_deps() -> tuple[MagicMock, MagicMock]:
    """Create mocks for qwen_asr and torch modules."""
    qwen_asr = MagicMock()
    torch = MagicMock()
    torch.bfloat16 = "bf16_sentinel"
    torch.float16 = "f16_sentinel"
    torch.float32 = "f32_sentinel"
    torch.cuda.is_available.return_value = False
    return qwen_asr, torch


def _make_provider(qwen_asr: MagicMock, torch: MagicMock, **config_kwargs: Any) -> Any:
    """Build a Qwen3ASRProvider with mocked dependencies."""
    with patch.dict("sys.modules", {"qwen_asr": qwen_asr, "torch": torch}):
        import roomkit.voice.stt.qwen3 as stt_mod

        importlib.reload(stt_mod)
        from roomkit.voice.stt.qwen3 import Qwen3ASRConfig, Qwen3ASRProvider

        cfg = Qwen3ASRConfig(**config_kwargs)
        return Qwen3ASRProvider(cfg)


class TestQwen3ASRProvider:
    def test_constructor_stores_config(self) -> None:
        qwen_asr, torch = _mock_deps()
        provider = _make_provider(qwen_asr, torch)
        assert provider._config.model_id == "Qwen/Qwen3-ASR-0.6B"
        assert provider._config.backend == "transformers"
        assert provider._model is None

    def test_name(self) -> None:
        qwen_asr, torch = _mock_deps()
        provider = _make_provider(qwen_asr, torch)
        assert provider.name == "Qwen3ASR"

    def test_supports_streaming_transformers(self) -> None:
        qwen_asr, torch = _mock_deps()
        provider = _make_provider(qwen_asr, torch, backend="transformers")
        assert provider.supports_streaming is False

    def test_supports_streaming_vllm(self) -> None:
        qwen_asr, torch = _mock_deps()
        provider = _make_provider(qwen_asr, torch, backend="vllm")
        assert provider.supports_streaming is True

    def test_load_model_transformers(self) -> None:
        qwen_asr, torch = _mock_deps()
        provider = _make_provider(qwen_asr, torch, backend="transformers", dtype="bfloat16")

        with patch.dict("sys.modules", {"qwen_asr": qwen_asr, "torch": torch}):
            model = provider._load_model()

        qwen_asr.Qwen3ASRModel.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-ASR-0.6B",
            device_map="auto",
            dtype="bf16_sentinel",
        )
        assert provider._model is model

    def test_load_model_vllm(self) -> None:
        qwen_asr, torch = _mock_deps()
        provider = _make_provider(
            qwen_asr,
            torch,
            backend="vllm",
            dtype="float16",
            gpu_memory_utilization=0.5,
            max_new_tokens=1024,
        )

        with patch.dict("sys.modules", {"qwen_asr": qwen_asr, "torch": torch}):
            model = provider._load_model()

        qwen_asr.Qwen3ASRModel.LLM.assert_called_once_with(
            "Qwen/Qwen3-ASR-0.6B",
            dtype="f16_sentinel",
            gpu_memory_utilization=0.5,
            max_new_tokens=1024,
        )
        assert provider._model is model

    def test_load_model_unsupported_dtype(self) -> None:
        qwen_asr, torch = _mock_deps()
        provider = _make_provider(qwen_asr, torch, dtype="int8")

        with (
            patch.dict("sys.modules", {"qwen_asr": qwen_asr, "torch": torch}),
            pytest.raises(ValueError, match="Unsupported dtype"),
        ):
            provider._load_model()

    def test_import_error_without_qwen_asr(self) -> None:
        """Constructor should raise ImportError when qwen_asr is missing."""
        with patch.dict("sys.modules", {"qwen_asr": None}):
            import roomkit.voice.stt.qwen3 as stt_mod

            importlib.reload(stt_mod)
            from roomkit.voice.stt.qwen3 import Qwen3ASRConfig, Qwen3ASRProvider

            with pytest.raises(ImportError, match="qwen-asr is required"):
                Qwen3ASRProvider(Qwen3ASRConfig())

    async def test_close_releases_model(self) -> None:
        qwen_asr, torch = _mock_deps()
        provider = _make_provider(qwen_asr, torch)
        provider._model = MagicMock()

        with patch.dict("sys.modules", {"torch": torch}):
            await provider.close()

        assert provider._model is None
