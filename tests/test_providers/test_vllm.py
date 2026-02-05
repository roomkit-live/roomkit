"""Tests for the vLLM provider factory."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from roomkit.providers.vllm.config import VLLMConfig


class _FakeAPIStatusError(Exception):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _mock_openai_module() -> MagicMock:
    mod = MagicMock()
    mod.APIStatusError = _FakeAPIStatusError
    return mod


class TestVLLMConfig:
    def test_required_model(self) -> None:
        with pytest.raises(TypeError):
            VLLMConfig()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        cfg = VLLMConfig(model="meta-llama/Llama-3-8B")
        assert cfg.model == "meta-llama/Llama-3-8B"
        assert cfg.base_url == "http://localhost:8000/v1"
        assert cfg.api_key == "none"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.7

    def test_custom_values(self) -> None:
        cfg = VLLMConfig(
            model="my-model",
            base_url="http://gpu-server:9000/v1",
            api_key="secret",
            max_tokens=2048,
            temperature=0.3,
        )
        assert cfg.base_url == "http://gpu-server:9000/v1"
        assert cfg.api_key == "secret"
        assert cfg.max_tokens == 2048
        assert cfg.temperature == 0.3


class TestCreateVLLMProvider:
    def test_returns_openai_provider(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            # Reload to pick up mocked openai
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(model="meta-llama/Llama-3-8B")
            provider = create_vllm_provider(cfg)

            assert type(provider).__name__ == "OpenAIAIProvider"
            assert provider._config.model == "meta-llama/Llama-3-8B"
            assert provider._config.base_url == "http://localhost:8000/v1"
            assert provider._config.api_key.get_secret_value() == "none"
            assert provider._config.max_tokens == 1024
            assert provider._config.temperature == 0.7

    def test_custom_config_propagated(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(
                model="my-model",
                base_url="http://gpu:9000/v1",
                api_key="tok",
                max_tokens=512,
                temperature=0.1,
            )
            provider = create_vllm_provider(cfg)

            assert provider._config.model == "my-model"
            assert provider._config.base_url == "http://gpu:9000/v1"
            assert provider._config.api_key.get_secret_value() == "tok"
            assert provider._config.max_tokens == 512
            assert provider._config.temperature == 0.1

    def test_import_error_when_openai_missing(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(model="test")
            with pytest.raises(ImportError, match="openai is required"):
                create_vllm_provider(cfg)
