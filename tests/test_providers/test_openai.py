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
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)
from roomkit.providers.openai.ai import _extract_think_tags, _ThinkTagParser
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
    defaults: dict[str, Any] = {"api_key": "sk-test-key"}
    defaults.update(overrides)
    return OpenAIConfig(**defaults)


def _mock_response(
    text: str = "Hello!",
    finish_reason: str = "stop",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 25,
    tool_calls: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Build a fake OpenAI chat completion response."""
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
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=mock_tool_calls),
                finish_reason=finish_reason,
            ),
        ],
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
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
        # Default sends the deprecated max_tokens (OpenAI-compatible servers).
        # use_max_completion_tokens flips it to the name OpenAI's newer models
        # require, and never sends both.
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
    async def test_reasoning_effort_dropped_when_tools_present(self) -> None:
        # Chat Completions rejects reasoning_effort + function tools for some
        # models (gpt-5.5), so it's omitted whenever the turn carries tools.
        with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
            from roomkit.providers.openai.ai import OpenAIAIProvider

            provider = OpenAIAIProvider(_config(reasoning_effort="high"))
            provider._client = MagicMock()
            provider._client.chat.completions.create = AsyncMock(return_value=_mock_response())
            tool = AITool(name="get_weather", description="x", parameters={})
            await provider.generate(_context(tools=[tool]))
            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assert "reasoning_effort" not in call_kwargs
            assert "tools" in call_kwargs

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

    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.model == "gpt-4o"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.7
        assert cfg.base_url is None

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
                                    AIImagePart(url="data:image/png;base64,IMGDATA"),
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
                            "image_url": {"url": "data:image/png;base64,IMGDATA"},
                        }
                    ],
                },
            ]


class TestThinkTagParser:
    """Unit tests for the streaming <think> tag parser."""

    def test_no_tags(self) -> None:
        parser = _ThinkTagParser()
        result = parser.feed("hello world")
        assert result == [("text", "hello world")]
        assert parser.flush() == []

    def test_complete_tag_single_chunk(self) -> None:
        parser = _ThinkTagParser()
        result = parser.feed("<think>reasoning</think>answer")
        assert result == [("thinking", "reasoning"), ("text", "answer")]

    def test_tag_split_across_chunks(self) -> None:
        parser = _ThinkTagParser()
        r1 = parser.feed("<thi")
        r2 = parser.feed("nk>reason")
        r3 = parser.feed("</think>ans")
        r4 = parser.flush()

        thinking = "".join(s for k, s in r1 + r2 + r3 + r4 if k == "thinking")
        text = "".join(s for k, s in r1 + r2 + r3 + r4 if k == "text")
        assert thinking == "reason"
        assert text == "ans"

    def test_close_tag_split(self) -> None:
        parser = _ThinkTagParser()
        r1 = parser.feed("<think>ok</th")
        r2 = parser.feed("ink>done")
        r3 = parser.flush()

        thinking = "".join(s for k, s in r1 + r2 + r3 if k == "thinking")
        text = "".join(s for k, s in r1 + r2 + r3 if k == "text")
        assert thinking == "ok"
        assert text == "done"

    def test_empty_think_block(self) -> None:
        parser = _ThinkTagParser()
        result = parser.feed("<think></think>answer")
        assert result == [("text", "answer")]

    def test_only_thinking_no_text(self) -> None:
        parser = _ThinkTagParser()
        r1 = parser.feed("<think>just thinking")
        r2 = parser.flush()
        thinking = "".join(s for k, s in r1 + r2 if k == "thinking")
        assert thinking == "just thinking"

    def test_multiple_small_chunks(self) -> None:
        parser = _ThinkTagParser()
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
        thinking, text = _extract_think_tags("plain response")
        assert thinking is None
        assert text == "plain response"

    def test_basic_extraction(self) -> None:
        thinking, text = _extract_think_tags("<think>Let me think</think>The answer is 42.")
        assert thinking == "Let me think"
        assert text == "The answer is 42."

    def test_multiline_thinking(self) -> None:
        thinking, text = _extract_think_tags("<think>Step 1: analyze\nStep 2: solve</think>Done.")
        assert thinking == "Step 1: analyze\nStep 2: solve"
        assert text == "Done."

    def test_empty_think_block(self) -> None:
        thinking, text = _extract_think_tags("<think></think>answer")
        assert thinking is None
        assert text == "answer"

    def test_whitespace_only_think_block(self) -> None:
        thinking, text = _extract_think_tags("<think>  \n  </think>answer")
        assert thinking is None
        assert text == "answer"


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
