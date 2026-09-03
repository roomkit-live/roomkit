"""Tests for the Qwen (Alibaba Model Studio) chat provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ModelInfo, ProviderError
from roomkit.providers.qwen.config import QwenConfig


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


def _config(**overrides: Any) -> QwenConfig:
    defaults: dict[str, Any] = {"api_key": "sk-test-key", "model": "qwen3.7-max"}
    defaults.update(overrides)
    return QwenConfig(**defaults)


def _mock_response(text: str = "Hello!", finish_reason: str = "stop") -> SimpleNamespace:
    """Build a fake chat completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish_reason,
            ),
        ],
        model="qwen3.7-max",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=25),
    )


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {"messages": [AIMessage(role="user", content="Hi")]}
    defaults.update(overrides)
    return AIContext(**defaults)


def _provider(**cfg_overrides: Any) -> Any:
    """Build a provider without touching the SDK — for pure request-shaping tests."""
    from roomkit.providers.qwen.ai import QwenAIProvider

    provider = QwenAIProvider.__new__(QwenAIProvider)
    provider._config = _config(**cfg_overrides)
    return provider


class TestQwenConfig:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        assert cfg.enable_thinking is None
        assert cfg.reasoning_effort is None

    def test_model_is_required(self) -> None:
        with pytest.raises(ValueError, match="model"):
            QwenConfig(api_key="sk-test-key")  # type: ignore[call-arg]

    def test_inherits_openai_request_fields(self) -> None:
        # Subclassing OpenAIConfig means every request field the inherited
        # provider reads exists here — guards against config drift.
        cfg = _config(extra_body={"top_k": 20})
        assert cfg.extra_body == {"top_k": 20}
        assert cfg.supports_custom_temperature is True
        assert cfg.temperature == 0.7
        assert cfg.timeout == 30.0

    def test_regional_base_url(self) -> None:
        # Beijing, US and the workspace-scoped MaaS forms are all plain
        # overrides — there is no correct default for a workspace id.
        cfg = _config(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        assert cfg.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


class TestQwenAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.qwen.ai import QwenAIProvider

            provider = QwenAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="Hi there!")
            )

            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.finish_reason == "stop"

    def test_provider_name_is_qwen(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.qwen.ai import QwenAIProvider

            assert QwenAIProvider(_config())._provider_name == "qwen"

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped_with_qwen_provider(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.qwen.ai import QwenAIProvider

            provider = QwenAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                side_effect=_FakeAPIStatusError("rate limited", status_code=429)
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.provider == "qwen"
            assert exc_info.value.status_code == 429
            assert exc_info.value.retryable is True

    def test_client_created_against_model_studio_endpoint(self) -> None:
        mock_mod = _mock_openai_module()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            from roomkit.providers.qwen.ai import QwenAIProvider

            QwenAIProvider(_config())

            mock_mod.AsyncOpenAI.assert_called_once_with(
                api_key="sk-test-key",
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                timeout=httpx.Timeout(30.0, connect=5.0),
                max_retries=0,
                default_headers=None,
            )

    @pytest.mark.asyncio
    async def test_generate_with_tools_uses_inherited_path(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.qwen.ai import QwenAIProvider

            provider = QwenAIProvider(_config())
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

    def test_lazy_import_error(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import roomkit.providers.qwen.ai as mod

            importlib.reload(mod)

            # `qwen-ai`, not the inherited `openai` and not the voice extras.
            with pytest.raises(ImportError, match=r"openai is required.*roomkit\[qwen-ai\]"):
                mod.QwenAIProvider(_config())


class TestQwenModelDiscovery:
    """Model Studio publishes no models endpoint, so the catalog is the answer."""

    @pytest.mark.asyncio
    async def test_list_models_returns_catalog_without_calling_the_api(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.qwen.ai import QwenAIProvider

            provider = QwenAIProvider(_config())
            provider._client = MagicMock()
            provider._client.models.list = AsyncMock(
                side_effect=AssertionError("must not query /v1/models")
            )

            models = await provider.list_models()

            assert [m.id for m in models] == [m.id for m in QwenAIProvider.available_models()]
            provider._client.models.list.assert_not_called()


class TestQwenCatalog:
    def test_available_models_nonempty_and_unique(self) -> None:
        from roomkit.providers.qwen.ai import QwenAIProvider

        models = QwenAIProvider.available_models()
        assert models
        assert all(isinstance(m, ModelInfo) for m in models)
        ids = [m.id for m in models]
        assert len(ids) == len(set(ids))
        assert all(m.id.startswith("qwen") for m in models)

    def test_every_model_declares_a_context_window(self) -> None:
        from roomkit.providers.qwen.ai import QwenAIProvider

        for model in QwenAIProvider.available_models():
            assert model.context_window, f"{model.id} has no context window"

    def test_unpriced_models_are_the_multi_tier_ones(self) -> None:
        # Alibaba tiers these across three and four input-length bands, which
        # ModelPricing (one threshold) cannot represent; no rate beats a rate
        # that understates a long-context bill by up to 12x.
        from roomkit.providers.qwen.ai import QwenAIProvider

        unpriced = {m.id for m in QwenAIProvider.available_models() if m.pricing is None}
        assert unpriced == {"qwen3-coder-plus", "qwen3-vl-plus"}

    def test_priced_models_carry_the_cache_read_rate(self) -> None:
        from roomkit.providers.qwen.ai import QwenAIProvider

        priced = [m for m in QwenAIProvider.available_models() if m.pricing is not None]
        assert priced
        for model in priced:
            assert model.pricing is not None
            assert model.pricing.cache_read_per_million is not None
            # Alibaba never charges this provider for a cache write: the 125%
            # rate applies to explicit cache creation, which it never requests.
            assert model.pricing.cache_write_per_million is None


class TestQwenVision:
    """The lineup is mixed, and the OpenAI prefix fallback knows none of it."""

    def test_catalog_decides_for_known_models(self) -> None:
        assert _provider(model="qwen3.7-plus").supports_vision is True
        assert _provider(model="qwen3-vl-plus").supports_vision is True
        # The flagship is text-only, and saying so is what lets a caller route
        # an image to the plus line instead.
        assert _provider(model="qwen3.7-max").supports_vision is False

    def test_unknown_model_defaults_to_vision(self) -> None:
        # Regression: the inherited implementation prefix-matches gpt-4o / o1 /
        # o3 and would report False for every qwen id, silently dropping images.
        assert _provider(model="qwen3.8-plus").supports_vision is True


class TestQwenThinking:
    """Thinking rides a boolean switch plus a token budget, not an effort tier."""

    def _extra(self, ctx: AIContext | None = None, **cfg: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        _provider(**cfg)._apply_sampling_kwargs(kwargs, ctx or _context())
        return kwargs.get("extra_body", {})

    def test_omitted_when_nothing_configured(self) -> None:
        assert self._extra() == {}

    def test_config_switch_enables(self) -> None:
        assert self._extra(enable_thinking=True) == {"enable_thinking": True}

    def test_config_switch_disables(self) -> None:
        assert self._extra(enable_thinking=False) == {"enable_thinking": False}

    def test_budget_maps_straight_onto_the_vendor_parameter(self) -> None:
        # The one provider where roomkit's per-turn budget is not an
        # approximation: Qwen takes a token cap by that name.
        assert self._extra(_context(thinking_budget=2048)) == {
            "enable_thinking": True,
            "thinking_budget": 2048,
        }

    def test_zero_budget_disables(self) -> None:
        assert self._extra(_context(thinking_budget=0)) == {"enable_thinking": False}

    def test_budget_outranks_config(self) -> None:
        assert self._extra(_context(thinking_budget=0), enable_thinking=True) == {
            "enable_thinking": False
        }

    def test_reasoning_effort_never_sent(self) -> None:
        # Model Studio accepts it only for the third-party DeepSeek models it
        # hosts; sending it to a Qwen model risks a rejected request for a
        # setting that would not apply anyway.
        kwargs: dict[str, Any] = {}
        _provider(reasoning_effort="high")._apply_sampling_kwargs(kwargs, _context())
        assert "reasoning_effort" not in kwargs
        assert kwargs.get("extra_body", {}) == {}

    def test_sent_on_tool_turns(self) -> None:
        ctx = _context(tools=[AITool(name="x", description="d", parameters={})])
        assert self._extra(ctx, enable_thinking=True) == {"enable_thinking": True}

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
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.qwen.ai import QwenAIProvider

            provider = QwenAIProvider(_config(extra_body={"top_k": 20}, enable_thinking=True))
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            await provider.generate(_context())

            extra_body = provider._client.chat.completions.create.call_args[1]["extra_body"]
            assert extra_body == {"top_k": 20, "enable_thinking": True}
