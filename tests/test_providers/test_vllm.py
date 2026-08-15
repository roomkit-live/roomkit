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


class _FakeAPIConnectionError(Exception):
    pass


def _mock_openai_module() -> MagicMock:
    mod = MagicMock()
    mod.APIStatusError = _FakeAPIStatusError
    mod.APIConnectionError = _FakeAPIConnectionError
    return mod


class TestVLLMConfig:
    def test_required_model(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VLLMConfig()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        cfg = VLLMConfig(model="meta-llama/Llama-3-8B")
        assert cfg.model == "meta-llama/Llama-3-8B"
        assert cfg.base_url == "http://localhost:8000/v1"
        assert cfg.api_key.get_secret_value() == "none"
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
        assert cfg.api_key.get_secret_value() == "secret"
        assert cfg.max_tokens == 2048
        assert cfg.temperature == 0.3

    def test_headers_and_extra_body_default_none(self) -> None:
        cfg = VLLMConfig(model="m")
        assert cfg.headers is None
        assert cfg.extra_body is None

    def test_reasoning_knobs_default_none(self) -> None:
        cfg = VLLMConfig(model="m")
        assert cfg.enable_thinking is None
        assert cfg.reasoning_effort is None
        # Unset knobs add nothing: a vanilla body stays vanilla.
        assert cfg.chat_template_kwargs() == {}

    def test_chat_template_kwargs_from_reasoning_knobs(self) -> None:
        cfg = VLLMConfig(model="m", enable_thinking=False, reasoning_effort="low")
        assert cfg.chat_template_kwargs() == {
            "enable_thinking": False,
            "reasoning_effort": "low",
        }

    def test_chat_template_kwargs_keeps_thinking_false(self) -> None:
        # False is a real value, not "unset": it must survive into the kwargs.
        cfg = VLLMConfig(model="m", enable_thinking=False)
        assert cfg.chat_template_kwargs() == {"enable_thinking": False}


class TestCreateVLLMProvider:
    def test_returns_openai_provider(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            # Reload to pick up mocked openai
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(model="meta-llama/Llama-3-8B")
            provider = create_vllm_provider(cfg)

            # An OpenAIAIProvider on the wire — asserted through inherited
            # behaviour rather than class identity, which importlib.reload
            # above makes meaningless.
            assert provider.supports_streaming is True
            assert provider.name == "vllm"
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

    def test_headers_and_extra_body_propagated(self) -> None:
        # vLLM-tier headers/extra_body map onto the underlying OpenAIConfig
        # as default_headers/extra_body — auth proxying and guided decoding.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(
                model="m",
                headers={"X-Proxy": "v1"},
                extra_body={"guided_choice": ["yes", "no"]},
            )
            provider = create_vllm_provider(cfg)

            assert provider._config.default_headers == {"X-Proxy": "v1"}
            assert provider._config.extra_body == {"guided_choice": ["yes", "no"]}

    def test_reasoning_knobs_reach_extra_body(self) -> None:
        # vLLM renders the chat template server-side, so reasoning is steered
        # through chat_template_kwargs in the request body.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(model="m", enable_thinking=False)
            provider = create_vllm_provider(cfg)

            assert provider._config.extra_body == {
                "chat_template_kwargs": {"enable_thinking": False}
            }

    def test_reasoning_knobs_merge_with_extra_body(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(
                model="m",
                reasoning_effort="low",
                extra_body={"guided_choice": ["yes", "no"]},
            )
            provider = create_vllm_provider(cfg)

            assert provider._config.extra_body == {
                "guided_choice": ["yes", "no"],
                "chat_template_kwargs": {"reasoning_effort": "low"},
            }

    def test_explicit_extra_body_template_kwargs_win(self) -> None:
        # extra_body stays the escape hatch for templates this config does
        # not model, so an explicit entry overrides the derived one.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(
                model="m",
                enable_thinking=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            provider = create_vllm_provider(cfg)

            assert provider._config.extra_body == {
                "chat_template_kwargs": {"enable_thinking": True}
            }

    def test_config_extra_body_not_mutated(self) -> None:
        # The caller's dict must not gain the derived key.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            original = {"guided_choice": ["yes", "no"]}
            cfg = VLLMConfig(model="m", enable_thinking=False, extra_body=original)
            create_vllm_provider(cfg)

            assert original == {"guided_choice": ["yes", "no"]}

    def test_model_metadata_is_the_servers_not_openais(self) -> None:
        # A local server runs whatever weights someone loaded onto it. The
        # provider used to inherit OpenAI's hosted catalog, which described
        # someone else's models — and would hand out one of their context
        # windows for any id that happened to collide.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            provider = create_vllm_provider(VLLMConfig(model="Qwen/Qwen3-VL-8B-Instruct"))

            assert provider.available_models() == []
            # Unknown model → unknown window, so callers degrade instead of
            # trimming history against a number that came from OpenAI.
            assert provider.context_window is None
            # And a collision with a real OpenAI id borrows nothing from it.
            assert create_vllm_provider(VLLMConfig(model="gpt-4o")).context_window is None

    def test_images_reach_a_local_multimodal_server(self) -> None:
        # The parent's fallback prefixes are OpenAI's own model names, which no
        # local id matches — inheriting them reported every vLLM deployment as
        # text-only and dropped images before the wire.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            provider = create_vllm_provider(VLLMConfig(model="Qwen/Qwen3-VL-8B-Instruct"))

            assert provider.supports_vision is True

    def test_import_error_when_openai_missing(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(model="test")
            with pytest.raises(ImportError, match=r"openai is required.*roomkit\[vllm\]"):
                create_vllm_provider(cfg)
