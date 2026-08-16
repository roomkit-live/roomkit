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

    def test_sampling_knobs_default_none(self) -> None:
        cfg = VLLMConfig(model="m")
        assert cfg.top_p is None
        assert cfg.top_k is None
        assert cfg.min_p is None
        assert cfg.presence_penalty is None
        assert cfg.repetition_penalty is None
        # "The server decides" is not the same as sending its documented
        # default, so an unset knob adds nothing to the body.
        assert cfg.sampling_body() == {}

    def test_sampling_body_from_knobs(self) -> None:
        cfg = VLLMConfig(
            model="m",
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
        )
        assert cfg.sampling_body() == {
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }

    def test_sampling_body_keeps_explicit_zero(self) -> None:
        # Qwen's own guidance sets min_p=0.0 and presence_penalty=0.0 in
        # thinking mode. A truthiness test would drop both as "unset".
        cfg = VLLMConfig(model="m", min_p=0.0, presence_penalty=0.0)
        assert cfg.sampling_body() == {"min_p": 0.0, "presence_penalty": 0.0}

    def test_sampling_body_emits_only_what_was_set(self) -> None:
        cfg = VLLMConfig(model="m", top_k=20)
        assert cfg.sampling_body() == {"top_k": 20}


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

    def test_sampling_knobs_reach_extra_body(self) -> None:
        # top_k / min_p / repetition_penalty are vLLM extensions the OpenAI SDK
        # has no argument for, and top_p / presence_penalty are read from the
        # same body — so all five ride extra_body rather than named params.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(model="m", top_p=0.8, top_k=20, presence_penalty=1.5)
            provider = create_vllm_provider(cfg)

            assert provider._config.extra_body == {
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 1.5,
            }

    def test_sampling_and_reasoning_coexist(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(model="m", top_p=0.8, enable_thinking=False)
            provider = create_vllm_provider(cfg)

            assert provider._config.extra_body == {
                "top_p": 0.8,
                "chat_template_kwargs": {"enable_thinking": False},
            }

    def test_explicit_extra_body_sampling_wins(self) -> None:
        # Same rule as the template kwargs: extra_body is the escape hatch for
        # a server this config does not model, so it overrides the typed field.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            cfg = VLLMConfig(model="m", top_p=0.8, extra_body={"top_p": 0.95})
            provider = create_vllm_provider(cfg)

            assert provider._config.extra_body == {"top_p": 0.95}

    def test_no_sampling_anywhere_sends_nothing(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.vllm import create_vllm_provider

            provider = create_vllm_provider(VLLMConfig(model="m"))

            assert provider._config.extra_body is None

    def test_turn_reasoning_overrides_config(self) -> None:
        # The per-turn value is the more specific one and must reach the wire.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.ai.base import AIContext
            from roomkit.providers.vllm import create_vllm_provider

            provider = create_vllm_provider(
                VLLMConfig(model="m", enable_thinking=False, reasoning_effort="low")
            )
            kwargs: dict[str, object] = {}
            provider._apply_sampling_kwargs(kwargs, AIContext(enable_thinking=True))

            assert kwargs["extra_body"] == {
                # The turn flipped the switch; the configured effort it says
                # nothing about survives rather than being dropped.
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}
            }

    def test_turn_without_reasoning_keeps_config(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.ai.base import AIContext
            from roomkit.providers.vllm import create_vllm_provider

            provider = create_vllm_provider(VLLMConfig(model="m", enable_thinking=False))
            kwargs: dict[str, object] = {}
            provider._apply_sampling_kwargs(kwargs, AIContext())

            assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_no_reasoning_anywhere_sends_nothing(self) -> None:
        # Neither layer set anything: the body stays vanilla and the model's
        # own default applies.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.ai.base import AIContext
            from roomkit.providers.vllm import create_vllm_provider

            provider = create_vllm_provider(VLLMConfig(model="m"))
            kwargs: dict[str, object] = {}
            provider._apply_sampling_kwargs(kwargs, AIContext())

            assert "extra_body" not in kwargs

    def test_reasoning_sent_on_tool_turns(self) -> None:
        # The OpenAI parent omits a configured effort on tool turns; a local
        # server couples nothing to tools, and an agentic turn is exactly where
        # steering the cost matters.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            import roomkit.providers.openai.ai as ai_mod

            importlib.reload(ai_mod)

            from roomkit.providers.ai.base import AIContext, AITool
            from roomkit.providers.vllm import create_vllm_provider

            provider = create_vllm_provider(VLLMConfig(model="m", enable_thinking=False))
            tool = AITool(name="t", description="d", parameters={"type": "object"})
            kwargs: dict[str, object] = {}
            provider._apply_sampling_kwargs(kwargs, AIContext(tools=[tool]))

            assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
            # The parent's top-level field would not be read by a local
            # template, so it must not be sent.
            assert "reasoning_effort" not in kwargs

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
