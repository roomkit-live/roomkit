"""Tests for the OpenAI AI provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.providers.ai.openai_dialect import ThinkTagParser, extract_think_tags
from roomkit.providers.ai.response_schema import ResponseSchemaError
from roomkit.providers.openai.config import OpenAIConfig


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


def _config(**overrides: Any) -> OpenAIConfig:
    defaults: dict[str, Any] = {"api_key": "sk-test-key", "model": "gpt-5.6-sol"}
    defaults.update(overrides)
    return OpenAIConfig(**defaults)


def _mock_response(
    text: str = "Hello!",
    finish_reason: str = "stop",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 25,
    tool_calls: list[dict[str, Any]] | None = None,
    cached_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    reasoning_content: str | None = None,
    reasoning: str | None = None,
) -> SimpleNamespace:
    """Build a fake OpenAI chat completion response.

    ``reasoning_content`` / ``reasoning`` are the dedicated trace fields
    OpenAI-compatible servers use instead of inline ``<think>`` tags; they are
    attached only when set, because OpenAI's own responses carry neither.
    """
    mock_tool_calls = None
    if tool_calls:
        mock_tool_calls = [
            SimpleNamespace(
                id=tc.get("id", "call_123"),
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc.get("arguments", "{}"),
                ),
            )
            for tc in tool_calls
        ]
    message = SimpleNamespace(content=text, tool_calls=mock_tool_calls)
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    if reasoning is not None:
        message.reasoning = reasoning
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            ),
        ],
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=(
                None
                if cached_tokens is None and cache_write_tokens is None
                else SimpleNamespace(
                    cached_tokens=cached_tokens,
                    cache_write_tokens=cache_write_tokens,
                )
            ),
        ),
    )


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {
        "messages": [AIMessage(role="user", content="Hi")],
    }
    defaults.update(overrides)
    return AIContext(**defaults)


class TestOpenAIAIProvider:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="Hi there!")
            )

            result = await provider.generate(_context())

            assert result.content == "Hi there!"
            assert result.finish_reason == "stop"
            assert result.metadata["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            ctx = _context(system_prompt="You are helpful.")
            await provider.generate(ctx)

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            messages = call_kwargs["messages"]
            assert messages[0] == {"role": "system", "content": "You are helpful."}
            assert messages[1] == {"role": "user", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_token_limit_kwarg_name_follows_config(self) -> None:
        # An explicit compatibility flag selects the exact argument and never
        # sends both forms.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            for flag, expected, forbidden in (
                (False, "max_tokens", "max_completion_tokens"),
                (True, "max_completion_tokens", "max_tokens"),
            ):
                provider = OpenAIAIProvider(_config(use_max_completion_tokens=flag))
                provider._client = MagicMock()
                provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
                await provider.generate(_context(max_tokens=321))
                call_kwargs = provider._client.chat.completions.create.call_args[1]
                assert call_kwargs[expected] == 321
                assert forbidden not in call_kwargs

    @pytest.mark.asyncio
    async def test_config_max_tokens_used_when_turn_sets_none(self) -> None:
        # A turn that sets no cap falls back to the configured one. A non-None
        # default on AIContext would shadow the config and make
        # OpenAIConfig.max_tokens — and every config built from it, such as
        # vLLM's — unreachable.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config(max_tokens=4096, use_max_completion_tokens=False))
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            await provider.generate(_context())

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_streaming_also_falls_back_to_config_max_tokens(self) -> None:
        # The streaming path must apply the same fallback as generate(): a cap
        # honoured on only one of the two paths is a cap you cannot rely on.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            async def _chunks() -> Any:
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="hi", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )

            provider = OpenAIAIProvider(_config(max_tokens=4096, use_max_completion_tokens=False))
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_chunks())

            async for _ in provider.generate_structured_stream(_context()):
                pass

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_official_default_model_uses_compatible_request_shape(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            await provider.generate(_context(max_tokens=321, temperature=0.7))

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["max_completion_tokens"] == 321
            assert "max_tokens" not in call_kwargs
            assert "temperature" not in call_kwargs

    @pytest.mark.asyncio
    async def test_temperature_omitted_when_unsupported(self) -> None:
        # Reasoning models accept only temperature=1; supports_custom_temperature
        # =False must drop the param entirely rather than send a rejected value.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config(supports_custom_temperature=False))
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
            await provider.generate(_context(temperature=0.7))
            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert "temperature" not in call_kwargs

    @pytest.mark.asyncio
    async def test_reasoning_effort_sent_only_when_configured(self) -> None:
        # reasoning_effort rides the request only when set on the config;
        # default (None) omits it so non-reasoning models aren't rejected.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            for effort, present in (("high", True), (None, False)):
                provider = OpenAIAIProvider(_config(reasoning_effort=effort))
                provider._client = MagicMock()
                provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
                await provider.generate(_context())
                call_kwargs = provider._client.chat.completions.create.call_args[1]
                assert ("reasoning_effort" in call_kwargs) is present
                if present:
                    assert call_kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_gpt_5_6_tools_force_effective_reasoning_none(self) -> None:
        # GPT-5.6 defaults to medium, but Chat Completions function tools only
        # accept effective reasoning none. An explicit higher config cannot be
        # allowed to turn the default tool loop into a provider-side 400.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config(reasoning_effort="high"))
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
            tool = AITool(name="get_weather", description="x", parameters={})
            await provider.generate(_context(tools=[tool]))
            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["reasoning_effort"] == "none"
            assert "tools" in call_kwargs

    @pytest.mark.asyncio
    async def test_gpt_5_6_default_tools_send_reasoning_none(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
            tool = AITool(name="get_weather", description="x", parameters={})

            await provider.generate(_context(tools=[tool]))

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["reasoning_effort"] == "none"

    @pytest.mark.asyncio
    async def test_custom_gpt_5_6_compatible_endpoint_is_not_force_profiled(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(
                _config(model="gpt-5.6-local", base_url="http://localhost:8000/v1")
            )
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
            tool = AITool(name="get_weather", description="x", parameters={})

            await provider.generate(_context(tools=[tool]))

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert "reasoning_effort" not in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_maps_usage(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(prompt_tokens=42, completion_tokens=7)
            )

            result = await provider.generate(_context())

            assert result.usage == {"input_tokens": 42, "output_tokens": 7}

    @pytest.mark.asyncio
    async def test_generate_reports_the_cached_prefix_apart_from_fresh_input(self) -> None:
        # OpenAI's prompt_tokens counts the cached prefix; reporting both
        # whole would have a cost dashboard bill those tokens twice, at the
        # full input rate on top of the cached one.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(
                    prompt_tokens=1_000, completion_tokens=20, cached_tokens=800
                )
            )

            result = await provider.generate(_context())

            assert result.usage == {
                "input_tokens": 200,
                "output_tokens": 20,
                "cache_read_input_tokens": 800,
            }

    @pytest.mark.asyncio
    async def test_generate_reports_cache_writes_apart_from_fresh_input(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(
                    prompt_tokens=1_000,
                    completion_tokens=20,
                    cached_tokens=300,
                    cache_write_tokens=500,
                )
            )

            result = await provider.generate(_context())

            assert result.usage == {
                "input_tokens": 200,
                "output_tokens": 20,
                "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 500,
            }

    @pytest.mark.asyncio
    async def test_generate_api_error(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )

            with pytest.raises(Exception, match="API rate limit"):
                await provider.generate(_context())

    @pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-astra"])
    def test_config_defaults(self, model: str) -> None:
        cfg = _config(model=model)
        assert cfg.model == model
        assert cfg.max_tokens == 1024
        assert cfg.base_url is None
        assert cfg.use_max_completion_tokens is True
        assert cfg.supports_custom_temperature is False

    def test_config_profiles_old_and_custom_endpoints_conservatively(self) -> None:
        old = _config(model="gpt-4o")
        custom = _config(base_url="http://localhost:11434/v1")
        explicit = _config(
            use_max_completion_tokens=False,
            supports_custom_temperature=True,
        )

        assert old.use_max_completion_tokens is False
        assert old.supports_custom_temperature is True
        assert custom.use_max_completion_tokens is False
        assert custom.supports_custom_temperature is True
        assert explicit.use_max_completion_tokens is False
        assert explicit.supports_custom_temperature is True

    def test_config_with_base_url(self) -> None:
        cfg = _config(base_url="http://localhost:11434/v1")
        assert cfg.base_url == "http://localhost:11434/v1"

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped_in_provider_error(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()

            exc = _FakeAPIStatusError("rate limited", status_code=429)
            provider._client.chat.completions.create = AsyncMock(side_effect=exc)

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is True
            assert exc_info.value.provider == "openai"
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_sdk_error_non_retryable(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()

            exc = _FakeAPIStatusError("bad request", status_code=400)
            provider._client.chat.completions.create = AsyncMock(side_effect=exc)

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is False
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_sdk_error_no_status_code(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import ProviderError
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("connection lost")
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.generate(_context())

            assert exc_info.value.retryable is False
            assert exc_info.value.status_code is None

    def test_lazy_import_error(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import roomkit.providers.openai.ai as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="openai is required"):
                mod.OpenAIAIProvider(_config())

    @pytest.mark.asyncio
    async def test_generate_with_tools(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())

            ctx = _context(
                tools=[
                    AITool(
                        name="search",
                        description="Search for info",
                        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
                    )
                ]
            )
            await provider.generate(ctx)

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) == 1
            assert call_kwargs["tools"][0]["type"] == "function"
            assert call_kwargs["tools"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_generate_extracts_tool_calls(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(
                    text="",
                    tool_calls=[
                        {"id": "call_abc", "name": "search", "arguments": '{"query": "cats"}'}
                    ],
                )
            )

            result = await provider.generate(_context())

            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "call_abc"
            assert result.tool_calls[0].name == "search"
            assert result.tool_calls[0].arguments == {"query": "cats"}

    @pytest.mark.asyncio
    async def test_generate_extracts_think_tags(self) -> None:
        """generate() strips <think> tags and populates AIResponse.thinking."""
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(
                    text="<think>Let me reason about this.</think>The answer is 42."
                )
            )

            result = await provider.generate(_context())

            assert result.thinking == "Let me reason about this."
            assert result.content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_generate_no_think_tags(self) -> None:
        """generate() returns None thinking when no <think> tags present."""
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="Just a normal response.")
            )

            result = await provider.generate(_context())

            assert result.thinking is None
            assert result.content == "Just a normal response."

    @pytest.mark.asyncio
    async def test_generate_reads_reasoning_content_field(self) -> None:
        """generate() surfaces a trace carried beside the content, not inline.

        Regression: the non-streaming path only ever parsed ``<think>`` tags,
        so every server using the dedicated field — DeepSeek, Qwen, vLLM with a
        reasoning parser, OpenRouter — silently dropped its reasoning here
        while the streaming path surfaced it.
        """
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(
                    text="The answer is 42.", reasoning_content="Let me reason."
                )
            )

            result = await provider.generate(_context())

            assert result.thinking == "Let me reason."
            assert result.content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_generate_reads_openrouter_reasoning_field(self) -> None:
        """OpenRouter names the same field ``reasoning``."""
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="42.", reasoning="Thinking out loud.")
            )

            result = await provider.generate(_context())

            assert result.thinking == "Thinking out loud."

    @pytest.mark.asyncio
    async def test_generate_merges_both_reasoning_conventions(self) -> None:
        """A server emitting tags *and* a field loses neither trace."""
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(
                    text="<think>inline.</think>42.", reasoning_content="field."
                )
            )

            result = await provider.generate(_context())

            assert result.thinking == "inline.field."
            assert result.content == "42."

    @pytest.mark.asyncio
    async def test_structured_stream_with_think_tags(self) -> None:
        """generate_structured_stream() yields thinking then text deltas."""
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()

            # Simulate streaming chunks: "<think>reason</think>answer"
            chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=c, tool_calls=None),
                            finish_reason=None,
                        )
                    ]
                )
                for c in ["<think>", "reason", "</think>", "answer"]
            ]
            # Add final chunk with finish_reason
            chunks.append(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None, tool_calls=None),
                            finish_reason="stop",
                        )
                    ]
                )
            )

            async def _fake_stream() -> Any:
                for c in chunks:
                    yield c

            provider._client.chat.completions.create = AsyncMock(return_value=_fake_stream())

            events = []
            async for ev in provider.generate_structured_stream(_context()):
                events.append(ev)

            thinking_events = [e for e in events if isinstance(e, StreamThinkingDelta)]
            text_events = [e for e in events if isinstance(e, StreamTextDelta)]

            assert "".join(e.thinking for e in thinking_events) == "reason"
            assert "".join(e.text for e in text_events) == "answer"

            # Thinking must come before text
            first_thinking = next(
                i for i, e in enumerate(events) if isinstance(e, StreamThinkingDelta)
            )
            first_text = next(i for i, e in enumerate(events) if isinstance(e, StreamTextDelta))
            assert first_thinking < first_text

    @pytest.mark.asyncio
    async def test_structured_stream_with_reasoning_content_field(self) -> None:
        """Reasoning via a dedicated reasoning_content field (DeepSeek-R1, vLLM)."""
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()

            chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None, reasoning_content="thinking", tool_calls=None
                            ),
                            finish_reason=None,
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="answer", tool_calls=None),
                            finish_reason="stop",
                        )
                    ]
                ),
            ]

            async def _fake_stream() -> Any:
                for c in chunks:
                    yield c

            provider._client.chat.completions.create = AsyncMock(return_value=_fake_stream())

            events = [e async for e in provider.generate_structured_stream(_context())]
            thinking = [e.thinking for e in events if isinstance(e, StreamThinkingDelta)]
            text = [e.text for e in events if isinstance(e, StreamTextDelta)]
            assert thinking == ["thinking"]
            assert text == ["answer"]

    @pytest.mark.asyncio
    async def test_thinking_part_round_trip(self) -> None:
        """AIThinkingPart in history is re-wrapped as <think> tags."""
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=_mock_response(text="result")
            )

            ctx = _context(
                messages=[
                    AIMessage(role="user", content="What is 2+2?"),
                    AIMessage(
                        role="assistant",
                        content=[
                            AIThinkingPart(thinking="I need to add 2 and 2"),
                            AIToolCallPart(id="tc1", name="calc", arguments={"expr": "2+2"}),
                        ],
                    ),
                ],
            )
            await provider.generate(ctx)

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assistant_msg = call_kwargs["messages"][1]
            assert assistant_msg["role"] == "assistant"
            # Thinking is prepended as <think> tags to the content
            assert "<think>I need to add 2 and 2</think>" in assistant_msg["content"]

    @pytest.mark.asyncio
    async def test_structured_stream_with_tool_calls(self) -> None:
        """generate_structured_stream() yields tool calls from streamed chunks."""
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()

            # Simulate tool call chunks
            chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call_1",
                                        function=SimpleNamespace(
                                            name="search",
                                            arguments='{"q":',
                                        ),
                                    )
                                ],
                            ),
                            finish_reason=None,
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id=None,
                                        function=SimpleNamespace(
                                            name=None,
                                            arguments='"cats"}',
                                        ),
                                    )
                                ],
                            ),
                            finish_reason=None,
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None, tool_calls=None),
                            finish_reason="tool_calls",
                        )
                    ]
                ),
            ]

            async def _fake_stream() -> Any:
                for c in chunks:
                    yield c

            provider._client.chat.completions.create = AsyncMock(return_value=_fake_stream())

            events = []
            async for ev in provider.generate_structured_stream(_context()):
                events.append(ev)

            tool_events = [e for e in events if isinstance(e, StreamToolCall)]
            assert len(tool_events) == 1
            assert tool_events[0].name == "search"
            assert tool_events[0].arguments == {"q": "cats"}
            assert tool_events[0].id == "call_1"

            # Each fragment is surfaced as it arrives, so a host can show what
            # is being composed instead of waiting out the whole composition.
            deltas = [e for e in events if isinstance(e, StreamToolCallDelta)]
            assert [d.arguments_delta for d in deltas] == ['{"q":', '"cats"}']
            assert all(d.name == "search" and d.id == "call_1" for d in deltas)
            assert events.index(deltas[-1]) < events.index(tool_events[0])

    @pytest.mark.asyncio
    async def test_structured_stream_tool_call_arguments_that_never_parse(self) -> None:
        """Unparseable arguments still stream, and still fall back to {"raw": ...}.

        The fragments are surfaced as they arrive, which means they are
        surfaced before anything could know whether the whole will parse.
        """
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()

            def _chunk(name: str | None, arguments: str, finish: str | None = None) -> Any:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call_1" if name else None,
                                        function=SimpleNamespace(name=name, arguments=arguments),
                                    )
                                ],
                            ),
                            finish_reason=finish,
                        )
                    ]
                )

            chunks = [_chunk("search", "{not"), _chunk(None, " json")]

            async def _fake_stream() -> Any:
                for c in chunks:
                    yield c

            provider._client.chat.completions.create = AsyncMock(return_value=_fake_stream())

            events = [e async for e in provider.generate_structured_stream(_context())]

            deltas = [e for e in events if isinstance(e, StreamToolCallDelta)]
            assert [d.arguments_delta for d in deltas] == ["{not", " json"]

            calls = [e for e in events if isinstance(e, StreamToolCall)]
            assert len(calls) == 1
            assert calls[0].arguments == {"raw": "{not json"}


class TestOpenAIToolResultImages:
    """Image tool results (screenshot-style output) reach the model as a real
    image on a synthetic user message — Chat Completions keeps tool messages
    text-only."""

    def test_tool_result_string_passes_through(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import AIToolResultPart
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            messages = provider._build_messages(
                [
                    AIMessage(
                        role="tool",
                        content=[AIToolResultPart(tool_call_id="t1", name="foo", result="hello")],
                    )
                ]
            )
            assert messages == [{"role": "tool", "tool_call_id": "t1", "content": "hello"}]

    def test_tool_result_image_splits_to_user_message(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import (
                AIImagePart,
                AITextPart,
                AIToolResultPart,
            )
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            messages = provider._build_messages(
                [
                    AIMessage(
                        role="tool",
                        content=[
                            AIToolResultPart(
                                tool_call_id="t1",
                                name="screenshot",
                                result=[
                                    AITextPart(text="the screen"),
                                    AIImagePart(url="data:image/png;base64,SU1HREFUQQ=="),
                                ],
                            )
                        ],
                    )
                ]
            )
            assert messages == [
                {"role": "tool", "tool_call_id": "t1", "content": "the screen"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,SU1HREFUQQ=="},
                        }
                    ],
                },
            ]


class TestThinkTagParser:
    """Unit tests for the streaming <think> tag parser."""

    def test_no_tags(self) -> None:
        parser = ThinkTagParser()
        result = parser.feed("hello world")
        assert result == [("text", "hello world")]
        assert parser.flush() == []

    def test_complete_tag_single_chunk(self) -> None:
        parser = ThinkTagParser()
        result = parser.feed("<think>reasoning</think>answer")
        assert result == [("thinking", "reasoning"), ("text", "answer")]

    def test_tag_split_across_chunks(self) -> None:
        parser = ThinkTagParser()
        r1 = parser.feed("<thi")
        r2 = parser.feed("nk>reason")
        r3 = parser.feed("</think>ans")
        r4 = parser.flush()

        thinking = "".join(s for k, s in r1 + r2 + r3 + r4 if k == "thinking")
        text = "".join(s for k, s in r1 + r2 + r3 + r4 if k == "text")
        assert thinking == "reason"
        assert text == "ans"

    def test_close_tag_split(self) -> None:
        parser = ThinkTagParser()
        r1 = parser.feed("<think>ok</th")
        r2 = parser.feed("ink>done")
        r3 = parser.flush()

        thinking = "".join(s for k, s in r1 + r2 + r3 if k == "thinking")
        text = "".join(s for k, s in r1 + r2 + r3 if k == "text")
        assert thinking == "ok"
        assert text == "done"

    def test_empty_think_block(self) -> None:
        parser = ThinkTagParser()
        result = parser.feed("<think></think>answer")
        assert result == [("text", "answer")]

    def test_only_thinking_no_text(self) -> None:
        parser = ThinkTagParser()
        r1 = parser.feed("<think>just thinking")
        r2 = parser.flush()
        thinking = "".join(s for k, s in r1 + r2 if k == "thinking")
        assert thinking == "just thinking"

    def test_multiple_small_chunks(self) -> None:
        parser = ThinkTagParser()
        all_results = []
        for char in "<think>abc</think>xyz":
            all_results.extend(parser.feed(char))
        all_results.extend(parser.flush())

        thinking = "".join(s for k, s in all_results if k == "thinking")
        text = "".join(s for k, s in all_results if k == "text")
        assert thinking == "abc"
        assert text == "xyz"


class TestExtractThinkTags:
    """Unit tests for the non-streaming <think> tag extraction."""

    def test_no_tags(self) -> None:
        thinking, text = extract_think_tags("plain response")
        assert thinking is None
        assert text == "plain response"

    def test_basic_extraction(self) -> None:
        thinking, text = extract_think_tags("<think>Let me think</think>The answer is 42.")
        assert thinking == "Let me think"
        assert text == "The answer is 42."

    def test_multiline_thinking(self) -> None:
        thinking, text = extract_think_tags("<think>Step 1: analyze\nStep 2: solve</think>Done.")
        assert thinking == "Step 1: analyze\nStep 2: solve"
        assert text == "Done."

    def test_empty_think_block(self) -> None:
        thinking, text = extract_think_tags("<think></think>answer")
        assert thinking is None
        assert text == "answer"

    def test_a_response_reads_as_its_stream_reads(self) -> None:
        """What the parser hands a stream, spaces included (RFC §6.4)."""
        thinking, text = extract_think_tags("<think>  \n  </think>answer")
        assert thinking == "  \n  "
        assert text == "answer"

        thinking, text = extract_think_tags("<think> weighing it \n</think>\n\nThe answer.  ")
        assert thinking == " weighing it \n"
        assert text == "\n\nThe answer.  "


class TestOpenAIHeadersAndExtraBody:
    def test_default_headers_passed_to_client(self) -> None:
        mod = _mock_openai_module()
        with patch.dict("sys.modules", {"openai": mod}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            OpenAIAIProvider(_config(default_headers={"X-Proxy": "v1"}))
            assert mod.AsyncOpenAI.call_args.kwargs["default_headers"] == {"X-Proxy": "v1"}

    def test_default_headers_none_by_default(self) -> None:
        mod = _mock_openai_module()
        with patch.dict("sys.modules", {"openai": mod}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            OpenAIAIProvider(_config())
            assert mod.AsyncOpenAI.call_args.kwargs["default_headers"] is None

    @pytest.mark.asyncio
    async def test_extra_body_sent_on_generate(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config(extra_body={"guided_choice": ["yes", "no"]}))
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
            await provider.generate(_context())
            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["extra_body"] == {"guided_choice": ["yes", "no"]}

    @pytest.mark.asyncio
    async def test_extra_body_omitted_when_unset(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
            await provider.generate(_context())
            assert "extra_body" not in provider._client.chat.completions.create.call_args[1]

    @pytest.mark.asyncio
    async def test_extra_body_sent_on_stream(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config(extra_body={"top_k": 20}))
            provider._client = MagicMock()

            async def _fake_stream() -> Any:
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="hi", tool_calls=None),
                            finish_reason="stop",
                        )
                    ]
                )

            provider._client.chat.completions.create = AsyncMock(return_value=_fake_stream())
            async for _ in provider.generate_structured_stream(_context()):
                pass
            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert call_kwargs["extra_body"] == {"top_k": 20}


class TestOpenAIImageDataURIs:
    """A data: URI is normalised by the shared reader, and refused before the request."""

    def test_a_data_uri_is_rebuilt_with_the_part_s_mime_type(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import AIImagePart
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            [message] = provider._build_messages(
                [
                    AIMessage(
                        role="user",
                        content=[AIImagePart(url="data:;base64,QUJDMTIz", mime_type="image/png")],
                    )
                ]
            )
            assert message["content"] == [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDMTIz"}}
            ]

    def test_a_malformed_payload_is_refused_before_the_request(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import AIImagePart, ProviderError
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            with pytest.raises(ProviderError, match="not valid base64") as excinfo:
                provider._build_messages(
                    [
                        AIMessage(
                            role="user",
                            content=[AIImagePart(url="data:image/png;base64,not*base64")],
                        )
                    ]
                )
            assert excinfo.value.retryable is False
            assert excinfo.value.provider == "openai"

    def test_a_remote_url_passes_through(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import AIImagePart
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            [message] = provider._build_messages(
                [AIMessage(role="user", content=[AIImagePart(url="https://example.com/a.png")])]
            )
            assert message["content"] == [
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
            ]

    def test_a_tool_result_image_is_rebuilt_too(self) -> None:
        # The image a tool returned rides a user message after the tool
        # message; it goes through the same reader as a user's image.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import AIImagePart, AIToolResultPart
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            messages = provider._build_messages(
                [
                    AIMessage(
                        role="tool",
                        content=[
                            AIToolResultPart(
                                tool_call_id="t1",
                                name="screenshot",
                                result=[
                                    AIImagePart(url="data:;base64,QUJDMTIz", mime_type="image/png")
                                ],
                            )
                        ],
                    )
                ]
            )
            assert messages[-1]["content"] == [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDMTIz"}}
            ]

    async def test_a_malformed_image_never_reaches_the_client(self) -> None:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.ai.base import AIImagePart, ProviderError
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config())
            message = AIMessage(
                role="user", content=[AIImagePart(url="data:image/png;base64,not*base64")]
            )
            with pytest.raises(ProviderError, match="not valid base64"):
                await provider.generate(_context(messages=[message]))
            provider._client.chat.completions.create.assert_not_called()


_VERDICT: dict[str, Any] = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["label"],
    "additionalProperties": False,
}


async def _openai_chunks(text: str, *, refusal: str | None = None) -> Any:
    """A Chat Completions stream: the text in one delta, then the stop."""
    delta = SimpleNamespace(content=text or None, tool_calls=None, refusal=refusal)
    yield SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None)
    yield SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop"
            )
        ],
        usage=None,
    )


class TestOpenAIResponseSchema:
    """RFC §6.7: the schema rides a strict ``json_schema`` response format."""

    @staticmethod
    def _provider(response: SimpleNamespace | None = None, **config: Any) -> Any:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config(**config))
        provider._client = MagicMock()
        provider._client.chat.completions.create = AsyncMock(
            return_value=response or _mock_response(text='{"label": "yes"}')
        )
        return provider

    async def test_the_schema_rides_a_strict_json_schema_response_format(self) -> None:
        provider = self._provider()

        result = await provider.generate(_context(response_schema=_VERDICT))

        assert result.content == '{"label": "yes"}'
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": _VERDICT, "strict": True},
        }

    async def test_no_schema_sends_no_response_format(self) -> None:
        provider = self._provider()

        await provider.generate(_context())

        assert "response_format" not in provider._client.chat.completions.create.call_args.kwargs

    async def test_think_tags_are_split_off_before_the_document_is_checked(self) -> None:
        provider = self._provider(_mock_response(text='<think>easy</think>{"label": "no"}'))

        result = await provider.generate(_context(response_schema=_VERDICT))

        assert result.content == '{"label": "no"}'
        assert result.thinking == "easy"

    async def test_a_refusal_field_raises_refusal(self) -> None:
        response = _mock_response(text="")
        response.choices[0].message.refusal = "I can't help with that."
        provider = self._provider(response)

        with pytest.raises(ResponseSchemaError, match="can't help") as exc:
            await provider.generate(_context(response_schema=_VERDICT))

        assert exc.value.reason == "refusal"

    @pytest.mark.parametrize(
        ("text", "finish", "reason"),
        [
            ("", "content_filter", "refusal"),
            ('{"label": "y', "length", "truncated"),
            ("Yes, it is.", "stop", "invalid_json"),
            ('{"dept": "billing"}', "stop", "invalid_json"),
        ],
    )
    async def test_an_answer_without_its_document_raises(
        self, text: str, finish: str, reason: str
    ) -> None:
        provider = self._provider(_mock_response(text=text, finish_reason=finish))

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.generate(_context(response_schema=_VERDICT))

        assert exc.value.reason == reason
        assert exc.value.provider == "openai"

    async def test_a_server_declared_without_support_is_refused_before_the_call(self) -> None:
        provider = self._provider(base_url="http://local:8000/v1", supports_response_schema=False)

        assert provider.supports_response_schema is False
        with pytest.raises(ResponseSchemaError) as exc:
            await provider.generate(_context(response_schema=_VERDICT))

        assert exc.value.reason == "unsupported"
        provider._client.chat.completions.create.assert_not_called()

    @staticmethod
    async def _drain(stream: Any) -> tuple[list[Any], ResponseSchemaError | None]:
        events: list[Any] = []
        try:
            async for event in stream:
                events.append(event)
        except ResponseSchemaError as exc:
            return events, exc
        return events, None

    async def test_a_streamed_answer_is_checked_before_its_done_event(self) -> None:
        provider = self._provider()
        provider._client.chat.completions.create = AsyncMock(
            return_value=_openai_chunks('{"label": "yes"}')
        )

        events, error = await self._drain(
            provider.generate_structured_stream(_context(response_schema=_VERDICT))
        )

        assert error is None
        assert isinstance(events[-1], StreamDone)
        assert (
            "".join(e.text for e in events if isinstance(e, StreamTextDelta)) == '{"label": "yes"}'
        )
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["json_schema"]["schema"] == _VERDICT

    async def test_a_streamed_answer_that_is_not_the_document_raises_instead_of_done(
        self,
    ) -> None:
        provider = self._provider()
        provider._client.chat.completions.create = AsyncMock(return_value=_openai_chunks("Yes."))

        events, error = await self._drain(
            provider.generate_structured_stream(_context(response_schema=_VERDICT))
        )

        assert error is not None and error.reason == "invalid_json"
        assert not any(isinstance(e, StreamDone) for e in events)

    async def test_a_streamed_refusal_is_a_refusal(self) -> None:
        provider = self._provider()
        provider._client.chat.completions.create = AsyncMock(
            return_value=_openai_chunks("", refusal="I can't help with that.")
        )

        _events, error = await self._drain(
            provider.generate_structured_stream(_context(response_schema=_VERDICT))
        )

        assert error is not None and error.reason == "refusal"
        assert "can't help" in str(error)


class TestOpenAIResponseSchemaWithTools:
    """RFC §6.7: OpenAI's own endpoint combines a schema with tools; a server
    behind base_url only when its config says so."""

    @staticmethod
    def _provider(**config: Any) -> Any:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config(**config))
        provider._client = MagicMock()
        return provider

    def test_on_for_the_official_endpoint_off_behind_a_base_url(self) -> None:
        assert self._provider().supports_response_schema_with_tools is True
        assert (
            self._provider(base_url="http://vllm:8000/v1").supports_response_schema_with_tools
            is False
        )
        assert (
            self._provider(
                base_url="http://proxy/v1", supports_response_schema_with_tools=True
            ).supports_response_schema_with_tools
            is True
        )

    async def test_a_tool_round_is_not_checked_and_the_final_answer_is(self) -> None:
        provider = self._provider()
        tool = AITool(name="lookup", description="Look it up")
        provider._client.chat.completions.create = AsyncMock(
            side_effect=[
                _mock_response(text="", tool_calls=[{"name": "lookup"}]),
                _mock_response(text="Yes."),
            ]
        )
        context = _context(response_schema=_VERDICT, tools=[tool])

        first = await provider.generate(context)
        with pytest.raises(ResponseSchemaError) as exc:
            await provider.generate(context)

        assert [call.name for call in first.tool_calls] == ["lookup"]
        assert exc.value.reason == "invalid_json"
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert "tools" in kwargs and "response_format" in kwargs

    async def test_a_server_behind_base_url_refuses_the_pair_before_the_call(self) -> None:
        provider = self._provider(base_url="http://vllm:8000/v1")
        provider._client.chat.completions.create = AsyncMock()
        tool = AITool(name="lookup", description="Look it up")

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.generate(_context(response_schema=_VERDICT, tools=[tool]))

        assert exc.value.reason == "unsupported"
        provider._client.chat.completions.create.assert_not_called()


class TestOpenAIToolTurnReasoning:
    """A turn with tools sends what Chat Completions accepts there (RFC §6.7).

    Before GPT-5.4 a reasoning model takes the effort alongside function
    tools; from GPT-5.4 on only ``none`` passes. The catalogue says which.
    """

    _TOOL = AITool(name="lookup", description="x", parameters={})

    def _sent(self, context: AIContext, **cfg: Any) -> Any:
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider.__new__(OpenAIAIProvider)
            provider._config = _config(**cfg)
            kwargs: dict[str, Any] = {}
            provider._apply_sampling_kwargs(kwargs, context)
            return kwargs.get("reasoning_effort")

    @pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-5.1", "o4-mini"])
    def test_a_model_before_gpt_5_4_takes_the_effort_with_tools(self, model: str) -> None:
        tools = _context(tools=[self._TOOL])

        assert self._sent(tools, model=model, reasoning_effort="low") == "low"
        assert self._sent(tools, model=model) is None

    def test_the_turn_effort_outranks_the_config_on_a_tool_turn(self) -> None:
        context = _context(tools=[self._TOOL], reasoning_effort="minimal")

        assert self._sent(context, model="gpt-5-mini", reasoning_effort="high") == "minimal"

    @pytest.mark.parametrize("model", ["gpt-5.4-mini", "gpt-5.5", "gpt-6-sol"])
    def test_from_gpt_5_4_a_tool_turn_sends_none(self, model: str) -> None:
        tools = _context(tools=[self._TOOL], reasoning_effort="high")

        assert self._sent(tools, model=model) == "none"
        assert self._sent(_context(reasoning_effort="high"), model=model) == "high"

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-7-preview",  # not in the catalogue
            "gpt-4.1",  # not a reasoning model
            "gpt-6-astra",  # refuses function tools on Chat Completions
            "gpt-5.5-pro",  # served by the Responses API only
        ],
    )
    def test_a_model_the_catalogue_does_not_tag_omits_it_with_tools(self, model: str) -> None:
        tools = _context(tools=[self._TOOL])

        assert self._sent(tools, model=model, reasoning_effort="low") is None
        assert self._sent(_context(), model=model, reasoning_effort="low") == "low"

    def test_behind_a_base_url_the_model_is_unknown_and_it_is_omitted(self) -> None:
        tools = _context(tools=[self._TOOL])

        sent = self._sent(
            tools, model="gpt-5-mini", base_url="http://localhost:8000/v1", reasoning_effort="low"
        )

        assert sent is None
