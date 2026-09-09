"""Cerebras contract tests using the real SDK with an offline HTTP transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from roomkit import CerebrasAIProvider, CerebrasConfig
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
)

_MODEL = "gpt-oss-120b"
_TOOL = AITool(
    name="weather",
    description="Read the weather",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)
_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 30,
    "total_tokens": 130,
    "prompt_tokens_details": {"cached_tokens": 80},
}


def _response(**message: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Sunny.", **message},
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": _USAGE,
        },
    )


def _stream_response(*deltas: dict[str, Any], finish: str = "stop") -> httpx.Response:
    frames = [
        {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas
    ]
    frames.extend(
        [
            {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {"choices": [], "usage": _USAGE},
        ]
    )
    payload = "".join(
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": _MODEL,
                **frame,
            }
        )
        + "\n\n"
        for frame in frames
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=payload + "data: [DONE]\n\n",
    )


@asynccontextmanager
async def _provider(
    handler: Callable[[httpx.Request], httpx.Response], **config: Any
) -> AsyncIterator[CerebrasAIProvider]:
    sdk = pytest.importorskip("openai")
    constructor = sdk.AsyncOpenAI
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with patch(
            "openai.AsyncOpenAI",
            side_effect=lambda **kwargs: constructor(http_client=client, **kwargs),
        ):
            provider = CerebrasAIProvider(
                CerebrasConfig(**{"api_key": "test-key", "model": _MODEL, **config})
            )
        try:
            yield provider
        finally:
            await provider.close()


def _context(**overrides: Any) -> AIContext:
    return AIContext(**{"messages": [AIMessage(role="user", content="Hello")], **overrides})


class TestCerebrasConfig:
    def test_model_required_and_key_redacted(self) -> None:
        with pytest.raises(ValueError, match="model"):
            CerebrasConfig(api_key="test-key")  # type: ignore[call-arg]
        assert "test-key" not in repr(CerebrasConfig(api_key="test-key", model=_MODEL))

    def test_import_hint_names_cerebras_extra(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match=r"roomkit\[cerebras\]"):
                CerebrasAIProvider(CerebrasConfig(api_key="test-key", model=_MODEL))
            assert CerebrasAIProvider.available_models()


class TestCerebrasRequests:
    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize("tools", [[], [_TOOL]])
    async def test_reasoning_sampling_and_usage_on_wire(
        self, streaming: bool, tools: list[AITool]
    ) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "https://api.cerebras.ai/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer test-key"
            assert request.extensions["timeout"]["connect"] == 3.0
            assert request.extensions["timeout"]["read"] == 42.0
            body = json.loads(request.content)
            assert body["model"] == _MODEL
            assert body["max_completion_tokens"] == 512
            assert "max_tokens" not in body
            assert body["temperature"] == 0.2
            assert body["reasoning_effort"] == "low"
            assert body["reasoning_format"] == "parsed"
            assert body["seed"] == 7
            assert "clear_thinking" not in body
            assert "stream_options" not in body
            if tools:
                assert body["tools"][0]["function"]["name"] == "weather"
            if streaming:
                assert body["stream"] is True
                return _stream_response({"reasoning": "Consider."}, {"content": "Sunny."})
            return _response(reasoning="Consider.")

        async with _provider(
            handle,
            reasoning_effort="high",
            extra_body={"seed": 7},
            timeout=42.0,
            connect_timeout=3.0,
        ) as provider:
            telemetry = MagicMock()
            provider._telemetry = telemetry
            context = _context(
                tools=tools, max_tokens=512, temperature=0.2, reasoning_effort="low"
            )
            if streaming:
                events = [event async for event in provider.generate_structured_stream(context)]
                assert events[0] == StreamThinkingDelta(thinking="Consider.")
                assert events[1] == StreamTextDelta(text="Sunny.")
                assert isinstance(events[-1], StreamDone)
                usage = events[-1].usage
            else:
                result = await provider.generate(context)
                assert result.content == "Sunny."
                assert result.thinking == "Consider."
                usage = result.usage
            assert usage == {
                "input_tokens": 20,
                "output_tokens": 30,
                "cache_read_input_tokens": 80,
            }
            entry = provider.catalog_entry()
            assert entry is not None and entry.pricing is not None
            assert entry.pricing.cost_for(usage) == pytest.approx((100 * 0.35 + 30 * 0.75) / 1e6)
            assert telemetry.record_metric.call_args.kwargs["attributes"]["provider"] == "cerebras"

    @pytest.mark.parametrize("streaming", [False, True])
    async def test_tool_round_preserves_reasoning_and_call_results(self, streaming: bool) -> None:
        requests: list[dict[str, Any]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                if streaming:
                    return _stream_response(
                        {"reasoning": "Check weather."},
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": '{"city":'},
                                }
                            ]
                        },
                        {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
                        finish="tool_calls",
                    )
                return _response(
                    content=None,
                    reasoning="Check weather.",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                )
            assistant = body["messages"][2]
            assert assistant["reasoning"] == "Check weather."
            assert assistant["content"] is None
            assert assistant["tool_calls"][0]["id"] == "call-1"
            assert body["messages"][3] == {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "Sunny.",
            }
            assert "<think>" not in request.content.decode()
            return _stream_response({"content": "Sunny."}) if streaming else _response()

        async with _provider(handle) as provider:
            context = _context(tools=[_TOOL], system_prompt="Weather assistant")
            if streaming:
                events = [event async for event in provider.generate_structured_stream(context)]
                assert any(isinstance(e, StreamToolCallDelta) for e in events)
                call = next(e for e in events if isinstance(e, StreamToolCall))
                thinking = next(e.thinking for e in events if isinstance(e, StreamThinkingDelta))
            else:
                result = await provider.generate(context)
                call = result.tool_calls[0]
                thinking = result.thinking
            assert call.arguments == {"city": "Paris"}
            context.messages.extend(
                [
                    AIMessage(
                        role="assistant",
                        content=[
                            AIThinkingPart(thinking=thinking or ""),
                            AIToolCallPart(id=call.id, name=call.name, arguments=call.arguments),
                        ],
                    ),
                    AIMessage(
                        role="tool",
                        content=[
                            AIToolResultPart(tool_call_id=call.id, name=call.name, result="Sunny.")
                        ],
                    ),
                ]
            )
            original = context.model_dump()
            if streaming:
                assert (
                    "".join([text async for text in provider.generate_stream(context)]) == "Sunny."
                )
            else:
                assert (await provider.generate(context)).content == "Sunny."
            assert context.model_dump() == original

    async def test_qwen_options_image_and_plain_assistant_history(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert str(request.url).startswith("https://proxy.example/v1/")
            assert request.headers["x-routing"] == "cerebras"
            assert body["clear_thinking"] is False
            assert body["reasoning_effort"] == "none"
            assert body["reasoning_format"] == "parsed"
            assert "temperature" not in body
            assert "thinking_budget" not in body
            assert (
                body["messages"][0]["content"][0]["image_url"]["url"]
                == "https://example.com/image.png"
            )
            assert body["messages"][1] == {
                "role": "assistant",
                "content": [{"type": "text", "text": "A tree."}],
                "reasoning": "Look.Closer.",
            }
            assert body["messages"][2] == {
                "role": "assistant",
                "content": "",
                "reasoning": "Again.",
            }
            return _response()

        async with _provider(
            handle,
            model="qwen-3.8-27b",
            clear_thinking=False,
            reasoning_effort="none",
            supports_custom_temperature=False,
            base_url="https://proxy.example/v1",
            default_headers={"x-routing": "cerebras"},
        ) as provider:
            assert provider.supports_vision is True
            await provider.generate(
                _context(
                    thinking_budget=200,
                    messages=[
                        AIMessage(
                            role="user", content=[AIImagePart(url="https://example.com/image.png")]
                        ),
                        AIMessage(
                            role="assistant",
                            content=[
                                AIThinkingPart(thinking="Look."),
                                AITextPart(text="A tree."),
                                AIThinkingPart(thinking="Closer."),
                            ],
                        ),
                        AIMessage(role="assistant", content=[AIThinkingPart(thinking="Again.")]),
                    ],
                )
            )

    async def test_unconfigured_effort_and_extra_body(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert "reasoning_effort" not in body
            assert body["reasoning_format"] == "hidden"
            assert body["max_tokens"] == 256
            assert "max_completion_tokens" not in body
            return _response()

        async with _provider(
            handle,
            reasoning_format=None,
            extra_body={"reasoning_format": "hidden"},
            use_max_completion_tokens=False,
            max_tokens=256,
        ) as provider:
            await provider.generate(_context())


class TestCerebrasErrors:
    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize(("status", "retryable"), [(401, False), (429, True), (503, True)])
    async def test_errors_are_attributed_without_sdk_retries(
        self, streaming: bool, status: int, retryable: bool
    ) -> None:
        calls = 0

        def handle(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(status, json={"error": {"message": "Request failed"}})

        async with _provider(handle) as provider:
            with pytest.raises(ProviderError) as exc:
                if streaming:
                    _ = [event async for event in provider.generate_structured_stream(_context())]
                else:
                    await provider.generate(_context())
            assert exc.value.provider == "cerebras"
            assert exc.value.status_code == status
            assert exc.value.retryable is retryable
            assert calls == 1

    async def test_connection_failure_is_retryable(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Unavailable", request=request)

        async with _provider(handle) as provider:
            with pytest.raises(ProviderError) as exc:
                await provider.generate(_context())
            assert exc.value.provider == "cerebras"
            assert exc.value.retryable is True


class TestCerebrasModels:
    async def test_live_discovery_uses_account_and_backfills_only_known_ids(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "https://api.cerebras.ai/v1/models"
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": _MODEL, "object": "model", "created": 0, "owned_by": "Cerebras"},
                        {
                            "id": "private-model",
                            "object": "model",
                            "created": 0,
                            "owned_by": "Cerebras",
                        },
                    ],
                },
            )

        async with _provider(handle, model="private-model") as provider:
            models = await provider.list_models()
            assert [model.id for model in models] == [_MODEL, "private-model"]
            assert models[0].context_window == 131_072
            assert models[0].pricing is not None
            assert models[1].context_window is None
            assert models[1].pricing is None
            assert provider.supports_vision is False

    @pytest.mark.parametrize(
        ("model", "vision"),
        [(_MODEL, False), ("qwen-3.8-27b", True), ("gemma-4-31b", True), ("gpt-5-private", False)],
    )
    async def test_vision_is_cerebras_specific(self, model: str, vision: bool) -> None:
        async with _provider(lambda request: _response(), model=model) as provider:
            assert provider.supports_vision is vision
