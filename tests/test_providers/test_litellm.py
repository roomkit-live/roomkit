"""Tests for the LiteLLM proxy provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ProviderError
from roomkit.providers.litellm.config import LiteLLMConfig


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


def _config(**overrides: Any) -> LiteLLMConfig:
    defaults: dict[str, Any] = {
        "api_key": "sk-virtual-key",
        "model": "claude-sonnet",
    }
    defaults.update(overrides)
    return LiteLLMConfig(**defaults)


def _mock_response(
    text: str = "Hello!",
    finish_reason: str = "stop",
    model: str = "claude-sonnet",
) -> SimpleNamespace:
    """Build a fake chat completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish_reason,
            ),
        ],
        model=model,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=25),
    )


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {
        "messages": [AIMessage(role="user", content="Hi")],
    }
    defaults.update(overrides)
    return AIContext(**defaults)


class TestLiteLLMConfig:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.base_url == "http://localhost:4000"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.7
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 0

    def test_model_is_required(self) -> None:
        # Which alias routes where is the deployment's decision — no default.
        with pytest.raises(ValidationError):
            LiteLLMConfig(api_key="sk-key")  # type: ignore[call-arg]

    def test_api_key_is_required(self) -> None:
        # A gateway's whole point is per-key auth and budgets.
        with pytest.raises(ValidationError):
            LiteLLMConfig(model="claude-sonnet")  # type: ignore[call-arg]

    def test_inherits_openai_request_fields(self) -> None:
        # Subclassing OpenAIConfig means every request field the provider reads
        # exists here — guards against the Azure-style config-drift bug.
        cfg = _config(reasoning_effort="high", use_max_completion_tokens=True)
        assert cfg.reasoning_effort == "high"
        assert cfg.use_max_completion_tokens is True
        assert cfg.supports_custom_temperature is True


class TestLiteLLMAIProvider:
    async def test_generate_success(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.litellm.ai import LiteLLMAIProvider

            provider = LiteLLMAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="Hi there!")
            )

            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.finish_reason == "stop"

    def test_provider_name_is_litellm(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.litellm.ai import LiteLLMAIProvider

            provider = LiteLLMAIProvider(_config())
            assert provider._provider_name == "litellm"

    async def test_sdk_error_wrapped_with_litellm_provider(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.litellm.ai import LiteLLMAIProvider

            provider = LiteLLMAIProvider(_config())
            provider._client = MagicMock()
            exc = _FakeAPIStatusError("rate limited", status_code=429)
            provider._client.chat.completions.create = AsyncMock(side_effect=exc)

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is True
            assert exc_info.value.provider == "litellm"
            assert exc_info.value.status_code == 429

    def test_client_created_against_proxy(self) -> None:
        mock_mod = _mock_openai_module()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            from roomkit.providers.litellm.ai import LiteLLMAIProvider

            LiteLLMAIProvider(_config())

            mock_mod.AsyncOpenAI.assert_called_once_with(
                api_key="sk-virtual-key",
                base_url="http://localhost:4000",
                timeout=30.0,
                max_retries=0,
                default_headers=None,
            )

    def test_lazy_import_error_names_litellm_extra(self) -> None:
        # The SDK import happens at instantiation, so the hint must name this
        # provider's own extra, not the openai one.
        with patch.dict("sys.modules", {"openai": None}):
            from roomkit.providers.litellm.ai import LiteLLMAIProvider

            with pytest.raises(ImportError, match=r"openai is required.*roomkit\[litellm\]"):
                LiteLLMAIProvider(_config())


class TestLiteLLMMetadata:
    def test_model_metadata_is_the_deployments_not_openais(self) -> None:
        # A gateway serves whatever aliases its operator configured. Inheriting
        # OpenAI's catalog would hand out one of their context windows for any
        # alias that happened to collide.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.litellm.ai import LiteLLMAIProvider

            provider = LiteLLMAIProvider(_config())

            assert provider.available_models() == []
            assert provider.context_window is None
            # A collision with a real OpenAI id borrows nothing from it.
            assert LiteLLMAIProvider(_config(model="gpt-4o")).context_window is None

    def test_supports_vision_true(self) -> None:
        # Whether a routed model reads images is the gateway's call; the
        # parent's OpenAI-name prefixes say nothing about deployment aliases.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.litellm.ai import LiteLLMAIProvider

            assert LiteLLMAIProvider(_config()).supports_vision is True

    def test_parse_model_maps_metadata(self) -> None:
        from roomkit.providers.litellm.ai import LiteLLMAIProvider

        info = LiteLLMAIProvider._parse_model(
            "claude-sonnet",
            {
                "max_input_tokens": 200_000,
                "supports_vision": True,
                "input_cost_per_token": 3e-06,
                "output_cost_per_token": 1.5e-05,
                "cache_read_input_token_cost": 3e-07,
                "cache_creation_input_token_cost": 3.75e-06,
            },
        )
        assert info.id == "claude-sonnet"
        assert info.context_window == 200_000
        assert info.supports_vision is True
        assert info.pricing is not None
        assert info.pricing.input_per_million == pytest.approx(3.0)
        assert info.pricing.output_per_million == pytest.approx(15.0)
        assert info.pricing.cache_read_per_million == pytest.approx(0.3)
        assert info.pricing.cache_write_per_million == pytest.approx(3.75)

    def test_parse_model_unknown_alias_reports_nothing(self) -> None:
        # An operator-defined alias absent from the cost map stays "unknown"
        # rather than gaining an invented window or price.
        from roomkit.providers.litellm.ai import LiteLLMAIProvider

        info = LiteLLMAIProvider._parse_model("my-custom-route", {})
        assert info.context_window is None
        assert info.supports_vision is None
        assert info.pricing is None

    def test_parse_model_keeps_vision_false(self) -> None:
        # False is the cost map's real answer, not an absence.
        from roomkit.providers.litellm.ai import LiteLLMAIProvider

        info = LiteLLMAIProvider._parse_model("text-route", {"supports_vision": False})
        assert info.supports_vision is False

    def test_partial_pricing_is_no_pricing(self) -> None:
        # A rate for only one side of the meter would bill half a conversation.
        from roomkit.providers.litellm.ai import LiteLLMAIProvider

        info = LiteLLMAIProvider._parse_model("half-priced", {"input_cost_per_token": 3e-06})
        assert info.pricing is None

    def test_zero_zero_pricing_is_no_pricing(self) -> None:
        # LiteLLM defaults *unknown* costs to 0 rather than null (seen live on
        # 1.79.0), so 0/0 is indistinguishable from unmapped — a $0 price
        # would tell a budget dashboard the route is free while the gateway
        # may well be billing it.
        from roomkit.providers.litellm.ai import LiteLLMAIProvider

        info = LiteLLMAIProvider._parse_model(
            "unmapped-alias", {"input_cost_per_token": 0, "output_cost_per_token": 0}
        )
        assert info.pricing is None

    async def test_list_models_collapses_load_balanced_deployments(self) -> None:
        # /model/info lists one entry per deployment; a load-balanced group
        # repeats the public name and must surface as one model.
        from roomkit.providers.litellm.ai import LiteLLMAIProvider

        provider = LiteLLMAIProvider.__new__(LiteLLMAIProvider)
        provider._fetch_model_info = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {
                    "model_name": "claude-sonnet",
                    "model_info": {"max_input_tokens": 200_000, "supports_vision": True},
                },
                {"model_name": "claude-sonnet", "model_info": {"max_input_tokens": 200_000}},
                {"model_name": "gpt-5.5", "model_info": {"max_input_tokens": 400_000}},
                {"litellm_params": {"model": "entry-without-a-name"}},
            ]
        )

        models = {m.id: m for m in await provider.list_models()}
        assert set(models) == {"claude-sonnet", "gpt-5.5"}
        assert models["claude-sonnet"].context_window == 200_000
        assert models["claude-sonnet"].supports_vision is True
        assert models["gpt-5.5"].context_window == 400_000


class TestLiteLLMReasoning:
    """Thinking rides LiteLLM's normalised ``reasoning_effort`` / ``thinking``."""

    def _provider(self, **cfg_overrides: Any) -> Any:
        from roomkit.providers.litellm.ai import LiteLLMAIProvider

        provider = LiteLLMAIProvider.__new__(LiteLLMAIProvider)
        provider._config = _config(**cfg_overrides)
        return provider

    def test_effort_from_config_when_no_budget(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider(reasoning_effort="high")._apply_sampling_kwargs(kwargs, _context())
        assert kwargs["reasoning_effort"] == "high"

    def test_turn_effort_outranks_config(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider(reasoning_effort="high")._apply_sampling_kwargs(
            kwargs, _context(reasoning_effort="low")
        )
        assert kwargs["reasoning_effort"] == "low"

    def test_budget_maps_to_thinking_object(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider()._apply_sampling_kwargs(kwargs, _context(thinking_budget=4096))
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled", "budget_tokens": 4096}
        assert "reasoning_effort" not in kwargs

    def test_zero_budget_sends_no_reasoning_params(self) -> None:
        # LiteLLM has no disable token every translator accepts (Gemini
        # rejects "none", Anthropic rejects "none" and "disable" — seen live
        # on 1.79.0 as 500s), so 0 must send nothing rather than a spelling
        # that breaks on some routes. The configured effort must not leak
        # through either.
        kwargs: dict[str, Any] = {}
        self._provider(reasoning_effort="high")._apply_sampling_kwargs(
            kwargs, _context(thinking_budget=0)
        )
        assert "reasoning_effort" not in kwargs
        assert "extra_body" not in kwargs

    def test_omitted_when_no_effort_and_no_budget(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider()._apply_sampling_kwargs(kwargs, _context())
        assert "reasoning_effort" not in kwargs
        assert "extra_body" not in kwargs

    def test_reasoning_skipped_on_tool_turns(self) -> None:
        kwargs: dict[str, Any] = {}
        ctx = _context(
            tools=[AITool(name="x", description="d", parameters={})], thinking_budget=4096
        )
        self._provider(reasoning_effort="high")._apply_sampling_kwargs(kwargs, ctx)
        assert "reasoning_effort" not in kwargs
        assert "extra_body" not in kwargs

    def test_temperature_still_applied(self) -> None:
        kwargs: dict[str, Any] = {}
        self._provider()._apply_sampling_kwargs(kwargs, _context(temperature=0.3))
        assert kwargs["temperature"] == 0.3

    async def test_config_extra_body_merges_with_thinking(self) -> None:
        # A configured extra_body must not clobber the thinking object set in
        # _apply_sampling_kwargs — both ride the request together.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.litellm.ai import LiteLLMAIProvider

            provider = LiteLLMAIProvider(_config(extra_body={"metadata": {"tags": ["roomkit"]}}))
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
            await provider.generate(_context(thinking_budget=2048))
            extra_body = provider._client.chat.completions.create.call_args[1]["extra_body"]
            assert extra_body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
            assert extra_body["metadata"] == {"tags": ["roomkit"]}
