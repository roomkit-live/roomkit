"""Tests for the DeepSeek chat provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ModelInfo, ProviderError
from roomkit.providers.deepseek.config import DeepSeekConfig


class _FakeAPIStatusError(Exception):
    """Stub for openai.APIStatusError used in tests."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeAPIConnectionError(Exception):
    """Stub for openai.APIConnectionError used in tests."""


def _mock_openai_module() -> MagicMock:
    """Return a MagicMock that behaves like the openai module."""
    mod = MagicMock()
    mod.APIStatusError = _FakeAPIStatusError
    mod.APIConnectionError = _FakeAPIConnectionError
    return mod


def _config(**overrides: Any) -> DeepSeekConfig:
    defaults: dict[str, Any] = {"api_key": "sk-test-key", "model": "deepseek-v4-pro"}
    defaults.update(overrides)
    return DeepSeekConfig(**defaults)


def _mock_response(text: str = "Hello!", finish_reason: str = "stop") -> SimpleNamespace:
    """Build a fake chat completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish_reason,
            ),
        ],
        model="deepseek-v4-pro",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=25),
    )


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {"messages": [AIMessage(role="user", content="Hi")]}
    defaults.update(overrides)
    return AIContext(**defaults)


def _provider(**cfg_overrides: Any) -> Any:
    """Build a provider without touching the SDK — for pure request-shaping tests."""
    from roomkit.providers.deepseek.ai import DeepSeekAIProvider

    provider = DeepSeekAIProvider.__new__(DeepSeekAIProvider)
    provider._config = _config(**cfg_overrides)
    return provider


class TestDeepSeekConfig:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.base_url == "https://api.deepseek.com/v1"
        # DeepSeek never adopted max_completion_tokens, so the parent's
        # conservative default is the correct one here.
        assert cfg.use_max_completion_tokens is False
        assert cfg.enable_thinking is None
        assert cfg.reasoning_effort is None

    def test_model_is_required(self) -> None:
        with pytest.raises(ValueError, match="model"):
            DeepSeekConfig(api_key="sk-test-key")  # type: ignore[call-arg]

    def test_inherits_openai_request_fields(self) -> None:
        # Subclassing OpenAIConfig means every request field the inherited
        # provider reads exists here — guards against config drift.
        cfg = _config(extra_body={"top_k": 20})
        assert cfg.extra_body == {"top_k": 20}
        assert cfg.supports_custom_temperature is True
        assert cfg.temperature == 0.7
        assert cfg.timeout == 30.0

    def test_custom_base_url(self) -> None:
        assert (
            _config(base_url="https://proxy.internal/v1").base_url == "https://proxy.internal/v1"
        )


class TestDeepSeekAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.deepseek.ai import DeepSeekAIProvider

            provider = DeepSeekAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="Hi there!")
            )

            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.finish_reason == "stop"

    def test_provider_name_is_deepseek(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.deepseek.ai import DeepSeekAIProvider

            assert DeepSeekAIProvider(_config())._provider_name == "deepseek"

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped_with_deepseek_provider(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.deepseek.ai import DeepSeekAIProvider

            provider = DeepSeekAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                side_effect=_FakeAPIStatusError("rate limited", status_code=429)
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.provider == "deepseek"
            assert exc_info.value.status_code == 429
            assert exc_info.value.retryable is True

    def test_client_created_against_deepseek_endpoint(self) -> None:
        mock_mod = _mock_openai_module()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            from roomkit.providers.deepseek.ai import DeepSeekAIProvider

            DeepSeekAIProvider(_config())

            mock_mod.AsyncOpenAI.assert_called_once_with(
                api_key="sk-test-key",
                base_url="https://api.deepseek.com/v1",
                timeout=30.0,
                max_retries=0,
                default_headers=None,
            )

    @pytest.mark.asyncio
    async def test_generate_with_tools_uses_inherited_path(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.deepseek.ai import DeepSeekAIProvider

            provider = DeepSeekAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            await provider.generate(
                _context(
                    tools=[
                        AITool(
                            name="search",
                            description="Search for info",
                            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
                        )
                    ]
                )
            )

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["tools"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_output_cap_sent_as_max_tokens(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.deepseek.ai import DeepSeekAIProvider

            provider = DeepSeekAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            await provider.generate(_context(max_tokens=512))

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["max_tokens"] == 512
            assert "max_completion_tokens" not in call_kwargs

    def test_lazy_import_error(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import roomkit.providers.deepseek.ai as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="openai is required"):
                mod.DeepSeekAIProvider(_config())


class TestDeepSeekCatalog:
    def test_available_models_nonempty_and_unique(self) -> None:
        from roomkit.providers.deepseek.ai import DeepSeekAIProvider

        models = DeepSeekAIProvider.available_models()
        assert models
        assert all(isinstance(m, ModelInfo) for m in models)
        ids = [m.id for m in models]
        assert len(ids) == len(set(ids))
        assert all(m.id.startswith("deepseek-") for m in models)

    def test_every_model_priced_and_sized(self) -> None:
        from roomkit.providers.deepseek.ai import DeepSeekAIProvider

        for model in DeepSeekAIProvider.available_models():
            assert model.context_window, f"{model.id} has no context window"
            assert model.pricing is not None, f"{model.id} has no price"
            assert model.pricing.cache_read_per_million is not None

    def test_no_model_claims_vision(self) -> None:
        # DeepSeek's API is text-only; a True here would let images through to
        # an endpoint that rejects them.
        from roomkit.providers.deepseek.ai import DeepSeekAIProvider

        assert all(m.supports_vision is False for m in DeepSeekAIProvider.available_models())

    def test_configured_model_reports_no_vision(self) -> None:
        assert _provider().supports_vision is False
        assert _provider(model="deepseek-v5-unknown").supports_vision is False


class TestDeepSeekThinking:
    """Thinking rides a nested object, not OpenAI's top-level effort string."""

    def _thinking(self, ctx: AIContext | None = None, **cfg: Any) -> Any:
        kwargs: dict[str, Any] = {}
        _provider(**cfg)._apply_sampling_kwargs(kwargs, ctx or _context())
        return kwargs.get("extra_body", {}).get("thinking")

    def test_omitted_when_nothing_configured(self) -> None:
        # Silence leaves DeepSeek's own default (thinking on) in charge.
        assert self._thinking() is None

    def test_effort_nested_inside_thinking_object(self) -> None:
        # The parent's top-level reasoning_effort is silently ignored by this
        # API; the nested form is the only one that takes effect.
        assert self._thinking(reasoning_effort="high") == {
            "type": "enabled",
            "reasoning_effort": "high",
        }

    def test_top_level_reasoning_effort_never_sent(self) -> None:
        kwargs: dict[str, Any] = {}
        _provider(reasoning_effort="high")._apply_sampling_kwargs(kwargs, _context())
        assert "reasoning_effort" not in kwargs

    def test_config_switch_disables(self) -> None:
        assert self._thinking(enable_thinking=False) == {"type": "disabled"}

    def test_config_switch_enables(self) -> None:
        assert self._thinking(enable_thinking=True) == {"type": "enabled"}

    def test_zero_budget_disables(self) -> None:
        assert self._thinking(_context(thinking_budget=0)) == {"type": "disabled"}

    def test_positive_budget_enables_without_a_size(self) -> None:
        # DeepSeek ignores token budgets, so the number is deliberately dropped
        # rather than translated into an effort tier the vendor never published.
        assert self._thinking(_context(thinking_budget=4096)) == {"type": "enabled"}

    def test_budget_outranks_config(self) -> None:
        assert self._thinking(_context(thinking_budget=0), enable_thinking=True) == {
            "type": "disabled"
        }

    def test_sent_on_tool_turns(self) -> None:
        # Diverges from the OpenAI parent on purpose: DeepSeek documents
        # thinking and tool calls as compatible.
        ctx = _context(tools=[AITool(name="x", description="d", parameters={})])
        assert self._thinking(ctx, reasoning_effort="low") == {
            "type": "enabled",
            "reasoning_effort": "low",
        }

    def test_temperature_still_applied(self) -> None:
        kwargs: dict[str, Any] = {}
        _provider()._apply_sampling_kwargs(kwargs, _context(temperature=0.3))
        assert kwargs["temperature"] == 0.3

    def test_temperature_dropped_when_unsupported(self) -> None:
        kwargs: dict[str, Any] = {}
        _provider(supports_custom_temperature=False)._apply_sampling_kwargs(
            kwargs, _context(temperature=0.3)
        )
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_config_extra_body_survives_thinking(self) -> None:
        # The inherited _apply_extra_body merges rather than replaces, so a
        # caller's own passthrough fields and the thinking object coexist.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.deepseek.ai import DeepSeekAIProvider

            provider = DeepSeekAIProvider(_config(extra_body={"top_k": 20}, enable_thinking=True))
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            await provider.generate(_context())

            extra_body = provider._client.chat.completions.create.call_args[1]["extra_body"]
            assert extra_body == {"top_k": 20, "thinking": {"type": "enabled"}}


class TestDeepSeekUsage:
    """DeepSeek reports cache hits under its own counter names."""

    def _usage(self, **fields: Any) -> dict[str, int]:
        from roomkit.providers.deepseek.ai import DeepSeekAIProvider

        return DeepSeekAIProvider._usage_from(SimpleNamespace(**fields))

    def test_cache_hits_are_not_billed_as_input(self) -> None:
        # Regression: the inherited implementation reads only OpenAI's
        # prompt_tokens_details.cached_tokens, so all 1000 tokens would land in
        # input_tokens and be priced at 50x the cache rate.
        usage = self._usage(
            prompt_tokens=1000,
            completion_tokens=50,
            prompt_cache_hit_tokens=960,
            prompt_cache_miss_tokens=40,
        )
        assert usage == {
            "input_tokens": 40,
            "output_tokens": 50,
            "cache_read_input_tokens": 960,
        }

    def test_miss_counter_derived_when_absent(self) -> None:
        usage = self._usage(prompt_tokens=1000, completion_tokens=50, prompt_cache_hit_tokens=600)
        assert usage["input_tokens"] == 400
        assert usage["cache_read_input_tokens"] == 600

    def test_zero_hit_reports_no_cache_counter(self) -> None:
        usage = self._usage(
            prompt_tokens=100,
            completion_tokens=5,
            prompt_cache_hit_tokens=0,
            prompt_cache_miss_tokens=100,
        )
        assert usage == {"input_tokens": 100, "output_tokens": 5}

    def test_falls_back_to_openai_shape(self) -> None:
        # A proxy in front of DeepSeek may normalise usage to OpenAI's shape.
        usage = self._usage(
            prompt_tokens=100,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=30, cache_write_tokens=None),
        )
        assert usage == {
            "input_tokens": 70,
            "output_tokens": 5,
            "cache_read_input_tokens": 30,
        }
