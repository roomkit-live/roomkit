"""Tests for the PolarGrid AI provider."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import polargrid
import pytest

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
from roomkit.providers.ai.response_schema import ResponseSchemaError
from roomkit.providers.polargrid.config import PolarGridConfig

# ---------------------------------------------------------------------------
# Fake polargrid module
# ---------------------------------------------------------------------------


# Builds and checks request bodies only; nothing listens at its address.
_BUILDER = polargrid.PolarGrid(api_key="k", base_url="http://127.0.0.1:1")


def _mock_polargrid_module() -> MagicMock:
    """Return a MagicMock that behaves like the polargrid module."""
    mod = MagicMock()
    mod.AuthenticationError = polargrid.AuthenticationError
    mod.BillingError = polargrid.BillingError
    mod.ValidationError = polargrid.ValidationError
    mod.RateLimitError = polargrid.RateLimitError
    mod.NetworkError = polargrid.NetworkError
    mod.TimeoutError = polargrid.TimeoutError
    mod.NotFoundError = polargrid.NotFoundError
    mod.ServerError = polargrid.ServerError

    # A streamed chat goes through providers/polargrid/sdk_patch.py, which
    # uses the SDK's own types and request builders: the real ones, so only
    # the server's lines (``_serve``) are faked.
    mod.Message = polargrid.Message
    mod.ChatCompletionRequest = polargrid.ChatCompletionRequest
    mod.ChatCompletionChunk = polargrid.ChatCompletionChunk
    mod.TokenUsage = polargrid.TokenUsage

    client = MagicMock()
    # A chat request goes through the SDK's own checks, body builder and
    # response converter (sdk_patch.py); only its HTTP call is faked.
    client._make_request = AsyncMock()
    client._convert_chat_completion_response = _BUILDER._convert_chat_completion_response
    client._validate_chat_completion_request = _BUILDER._validate_chat_completion_request
    client._build_chat_completion_body = _BUILDER._build_chat_completion_body
    client.list_models = AsyncMock()
    client.get_region_id = MagicMock(return_value="yul-02")
    client.get_region_name = MagicMock(return_value="Montreal 02")
    client.close = AsyncMock()

    # Async constructor (PolarGrid.create) and sync constructor both
    # need to return our client. AsyncMock for create; regular call
    # for the sync constructor.
    mod.PolarGrid = MagicMock()
    mod.PolarGrid.create = AsyncMock(return_value=client)
    mod.PolarGrid.return_value = client
    # Expose the client so tests can configure return values without
    # walking through the MagicMock chain.
    mod._client = client
    return mod


def _config(**overrides: Any) -> PolarGridConfig:
    defaults: dict[str, Any] = {"api_key": "pg_test", "model": "qwen-3.5-27b"}
    defaults.update(overrides)
    return PolarGridConfig(**defaults)


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {
        "messages": [AIMessage(role="user", content="Hi")],
        "system_prompt": "You are helpful.",
        "max_tokens": 256,
        "temperature": 0.5,
    }
    defaults.update(overrides)
    return AIContext(**defaults)


def _response_obj(
    *,
    content: str | None = "",
    finish_reason: str = "stop",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    model: str = "qwen-3.5-27b",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A non-streamed answer as PolarGrid's server writes it."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-0",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _raw_chunk(
    choices: list[dict[str, Any]], usage: dict[str, int] | None = None
) -> dict[str, Any]:
    """One streamed line as PolarGrid's server writes it."""
    chunk: dict[str, Any] = {
        "id": "chatcmpl-0",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "qwen-3.5-27b",
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _stream_chunk(
    *, content: str | None = None, finish_reason: str | None = None
) -> dict[str, Any]:
    delta = {"content": content} if content is not None else {}
    return _raw_chunk([{"index": 0, "delta": delta, "finish_reason": finish_reason}])


def _usage_chunk(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    """The last line when the request asks for the usage: no choices."""
    return _raw_chunk([], _usage(prompt_tokens, completion_tokens))


def _tool_chunk(
    *,
    index: int = 0,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """A line carrying a fragmented tool-call delta: arguments arrive in
    fragments to be concatenated per ``index``."""
    func: dict[str, Any] = {}
    if name is not None:
        func["name"] = name
    if arguments is not None:
        func["arguments"] = arguments
    call: dict[str, Any] = {"index": index, "type": "function", "function": func}
    if id is not None:
        call["id"] = id
    delta = {"tool_calls": [call]}
    return _raw_chunk([{"index": index, "delta": delta, "finish_reason": finish_reason}])


def _respond(mod: MagicMock, raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Have the client answer *raw*, as PolarGrid's server writes it, through
    the SDK's own request checks and response types; return the bodies sent."""
    sent: list[dict[str, Any]] = []
    edge = polargrid.PolarGrid(api_key="k", base_url="http://127.0.0.1:1")

    async def make_request(
        endpoint: str,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        sent.append(body or {})
        return raw

    edge._make_request = make_request  # type: ignore[method-assign]
    mod._client._make_request = make_request
    mod._client.list_models = edge.list_models
    return sent


def _completion(
    *, message: dict[str, Any] | None = None, finish_reason: str = "stop"
) -> dict[str, Any]:
    """A non-streamed answer as PolarGrid's server writes it."""
    choices = (
        []
        if message is None
        else [
            {
                "index": 0,
                "message": {"role": "assistant", **message},
                "finish_reason": finish_reason,
            }
        ]
    )
    return {
        "id": "chatcmpl-0",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen-3.8-27b",
        "choices": choices,
        "usage": _usage(4, 2),
    }


def _serve(mod: MagicMock, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Have the client's stream answer *chunks*; return the bodies it is sent."""
    sent: list[dict[str, Any]] = []

    async def stream_post(endpoint: str, body: dict[str, Any]) -> Any:
        sent.append(body)
        for chunk in chunks:
            yield chunk

    mod._client._stream_post = stream_post
    return sent


def _tool_call_obj(*, id: str, name: str, arguments: str) -> dict[str, Any]:
    """A non-streamed tool call as the server writes it: its arguments a JSON
    string."""
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _provider(mod: MagicMock | None = None, **config_overrides: Any) -> tuple[Any, MagicMock]:
    mod = mod or _mock_polargrid_module()
    with patch.dict("sys.modules", {"polargrid": mod}):
        from roomkit.providers.polargrid.ai import PolarGridAIProvider

        return PolarGridAIProvider(_config(**config_overrides)), mod


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestPolarGridGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(content="Hello!")

        resp = await provider.generate(_context())

        assert resp.content == "Hello!"
        assert resp.finish_reason == "stop"
        assert resp.usage == {"input_tokens": 11, "output_tokens": 7}

    @pytest.mark.asyncio
    async def test_generate_builds_messages_with_system_prompt(self) -> None:
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(content="ok")

        await provider.generate(_context(system_prompt="Be terse."))

        request = mod._client._make_request.await_args.kwargs["body"]
        assert request["messages"][0] == {"role": "system", "content": "Be terse."}
        assert request["messages"][1] == {"role": "user", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_a_request_without_a_cap_asks_for_the_apis_maximum(self) -> None:
        # Left out, the SDK sends 150 and the server stops near 200 tokens,
        # mid-sentence, under a "stop" finish.
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(content="ok")
        sent = _serve(mod, [_stream_chunk(content="ok", finish_reason="stop")])

        await provider.generate(_context(max_tokens=None))
        _ = [e async for e in provider.generate_structured_stream(_context(max_tokens=None))]

        assert mod._client._make_request.await_args.kwargs["body"]["max_tokens"] == 4096
        assert sent[0]["max_tokens"] == 4096

    @pytest.mark.asyncio
    @pytest.mark.parametrize("where", ["turn", "config"])
    async def test_a_cap_above_the_sdks_maximum_is_sent_as_the_maximum(
        self, where: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The stream path runs the SDK's own validator, which refuses more
        # than 4096: the turn goes through instead of failing.
        provider, mod = _provider(**({"max_tokens": 8192} if where == "config" else {}))
        mod._client._make_request.return_value = _response_obj(content="ok")
        sent = _serve(mod, [_stream_chunk(content="ok", finish_reason="stop")])
        context = _context(max_tokens=8192 if where == "turn" else None)

        with caplog.at_level(logging.WARNING, logger="roomkit.providers.polargrid"):
            await provider.generate(context)
            _ = [e async for e in provider.generate_structured_stream(context)]

        assert mod._client._make_request.await_args.kwargs["body"]["max_tokens"] == 4096
        assert sent[0]["max_tokens"] == 4096
        assert caplog.text.count("caps max_tokens at 4096") == 1

    @pytest.mark.asyncio
    async def test_a_small_window_bounds_the_default_cap(self) -> None:
        # The pilot model serves 8192 tokens: a 4096-token answer behind a
        # long prompt would overflow it, which the server refuses.
        provider, mod = _provider(model="qwen-3.6-35b-a3b")
        mod._client._make_request.return_value = _response_obj(content="ok")
        long_prompt = [AIMessage(role="user", content="word " * 3000)]

        await provider.generate(_context(messages=long_prompt, max_tokens=None))

        cap = mod._client._make_request.await_args.kwargs["body"]["max_tokens"]
        assert 0 < cap < 4096
        assert cap + len("word " * 3000) // 4 <= 8192

    @pytest.mark.asyncio
    async def test_a_zero_temperature_goes_out_as_zero(self) -> None:
        # The SDK's body builder reads a 0 as unset and sends 0.7.
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(content="ok")
        sent = _serve(mod, [_stream_chunk(content="ok", finish_reason="stop")])

        await provider.generate(_context(temperature=0.0))
        _ = [e async for e in provider.generate_structured_stream(_context(temperature=0.0))]

        assert mod._client._make_request.await_args.kwargs["body"]["temperature"] == 0.0
        assert sent[0]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_generate_passes_temperature_and_max_tokens(self) -> None:
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(content="ok")

        await provider.generate(_context(max_tokens=512, temperature=0.2))

        request = mod._client._make_request.await_args.kwargs["body"]
        assert request["model"] == "qwen-3.5-27b"
        assert request["temperature"] == 0.2
        assert request["max_tokens"] == 512
        assert request["top_p"] == 0.9
        assert request["stream"] is False

    @pytest.mark.asyncio
    async def test_generate_forwards_tools(self) -> None:
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(content="ok")

        tool = AITool(
            name="get_weather",
            description="Get current weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        await provider.generate(_context(tools=[tool]))

        request = mod._client._make_request.await_args.kwargs["body"]
        assert request["tools"][0]["type"] == "function"
        assert request["tools"][0]["function"]["name"] == "get_weather"
        assert request["tools"][0]["function"]["parameters"]["required"] == ["city"]
        # No tool_choice in AIContext — leave it unset so PolarGrid defaults to auto.
        assert "tool_choice" not in request

    @pytest.mark.asyncio
    async def test_debug_logs_full_request(self, caplog: pytest.LogCaptureFixture) -> None:
        provider, mod = _provider(thinking=True)
        mod._client._make_request.return_value = _response_obj(content="ok")

        with caplog.at_level(logging.DEBUG, logger="roomkit.providers.polargrid"):
            await provider.generate(_context())

        logged = [r.message for r in caplog.records if "PolarGrid request:" in r.message]
        assert logged, "expected the outgoing request to be logged at DEBUG"
        # The enable_thinking flag and model are visible in the logged payload.
        assert '"enable_thinking": true' in logged[0]
        assert '"model": "qwen-3.5-27b"' in logged[0]

    @pytest.mark.asyncio
    async def test_generate_extracts_tool_calls(self) -> None:
        provider, mod = _provider()
        call = _tool_call_obj(
            id="call_1", name="get_weather", arguments=json.dumps({"city": "Montreal"})
        )
        mod._client._make_request.return_value = _response_obj(
            content=None, finish_reason="tool_calls", tool_calls=[call]
        )

        resp = await provider.generate(
            _context(tools=[AITool(name="get_weather", description="x")])
        )

        assert resp.content == ""  # content=None coerces to ""
        assert resp.finish_reason == "tool_calls"
        assert len(resp.tool_calls) == 1
        call = resp.tool_calls[0]
        assert call.id == "call_1"
        assert call.name == "get_weather"
        # JSON-string arguments parsed into a dict for RoomKit.
        assert call.arguments == {"city": "Montreal"}

    @pytest.mark.asyncio
    async def test_generate_tool_call_malformed_args_preserved(self) -> None:
        provider, mod = _provider()
        call = {"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{x"}}
        _respond(
            mod,
            _completion(
                message={"content": None, "tool_calls": [call]}, finish_reason="tool_calls"
            ),
        )

        resp = await provider.generate(_context())

        assert resp.tool_calls[0].arguments == {"raw": "{x"}

    @pytest.mark.asyncio
    async def test_generate_empty_choices_returns_empty_content(self) -> None:
        provider, mod = _provider()
        _respond(mod, _completion())

        resp = await provider.generate(_context())

        assert resp.content == ""


# ---------------------------------------------------------------------------
# Region routing
# ---------------------------------------------------------------------------


class TestPolarGridRegionRouting:
    @pytest.mark.asyncio
    async def test_region_none_uses_async_create(self) -> None:
        provider, mod = _provider()  # region defaults to None
        mod._client._make_request.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        mod.PolarGrid.create.assert_awaited_once()
        call_kwargs = mod.PolarGrid.create.await_args.kwargs
        assert "region" not in call_kwargs
        assert call_kwargs["api_key"] == "pg_test"
        # Auto-routing is model-aware (polargrid-sdk 0.10.0): the autorouter
        # picks an edge that already serves the configured model.
        assert call_kwargs["routing_model"] == "qwen-3.5-27b"

    @pytest.mark.asyncio
    async def test_region_pinned_uses_sync_constructor(self) -> None:
        provider, mod = _provider(region="toronto")
        mod._client._make_request.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        mod.PolarGrid.assert_called_once()
        call_kwargs = mod.PolarGrid.call_args.kwargs
        assert call_kwargs["region"] == "toronto"
        # A pinned region bypasses the autorouter entirely — no routing hint.
        assert "routing_model" not in call_kwargs
        mod.PolarGrid.create.assert_not_called()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestPolarGridStreaming:
    @pytest.mark.asyncio
    async def test_streams_text_deltas_and_done(self) -> None:
        provider, mod = _provider()
        _serve(
            mod,
            [
                _stream_chunk(content="42"),
                _stream_chunk(content=" is "),
                _stream_chunk(content="it.", finish_reason="stop"),
                # The usage, last, with no choices: the request asked for it.
                _usage_chunk(12, 5),
            ],
        )

        events = [e async for e in provider.generate_structured_stream(_context())]
        text_events = [e for e in events if isinstance(e, StreamTextDelta)]
        done_events = [e for e in events if isinstance(e, StreamDone)]

        assert [e.text for e in text_events] == ["42", " is ", "it."]
        assert len(done_events) == 1
        assert done_events[0].finish_reason == "stop"
        assert done_events[0].usage == {"input_tokens": 12, "output_tokens": 5}

    @pytest.mark.asyncio
    async def test_streaming_asks_for_a_stream_and_its_usage(self) -> None:
        provider, mod = _provider()
        sent = _serve(mod, [_stream_chunk(content="a", finish_reason="stop")])

        async for _ in provider.generate_structured_stream(_context()):
            pass

        assert sent[0]["stream"] is True
        assert sent[0]["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_generate_stream_yields_text_only(self) -> None:
        provider, mod = _provider()
        _serve(
            mod,
            [
                _stream_chunk(content="a"),
                _stream_chunk(content="b", finish_reason="stop"),
            ],
        )

        chunks = [c async for c in provider.generate_stream(_context())]
        assert chunks == ["a", "b"]

    @pytest.mark.asyncio
    async def test_streaming_emits_tool_calls(self) -> None:
        provider, mod = _provider()
        _serve(
            mod,
            [
                # id + name + first arg fragment, then the tail fragment.
                _tool_chunk(index=0, id="call_1", name="get_weather", arguments='{"ci'),
                _tool_chunk(index=0, arguments='ty": "Montreal"}'),
                # Finish marker, then the usage.
                _raw_chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
                _usage_chunk(4, 2),
            ],
        )

        events = [e async for e in provider.generate_structured_stream(_context())]
        tool_events = [e for e in events if isinstance(e, StreamToolCall)]
        done_events = [e for e in events if isinstance(e, StreamDone)]

        assert len(tool_events) == 1
        assert tool_events[0].id == "call_1"
        assert tool_events[0].name == "get_weather"
        # Fragmented arguments concatenated then parsed to a dict.
        assert tool_events[0].arguments == {"city": "Montreal"}
        assert done_events[0].finish_reason == "tool_calls"
        assert done_events[0].usage == {"input_tokens": 4, "output_tokens": 2}

        # One composition event per fragment, all before the completed call.
        deltas = [e for e in events if isinstance(e, StreamToolCallDelta)]
        assert [d.arguments_delta for d in deltas] == ['{"ci', 'ty": "Montreal"}']
        assert all(d.name == "get_weather" and d.id == "call_1" for d in deltas)
        assert events.index(deltas[-1]) < events.index(tool_events[0])

    @pytest.mark.asyncio
    async def test_streaming_text_then_tool_call_ordering(self) -> None:
        provider, mod = _provider()
        _serve(
            mod,
            [
                _stream_chunk(content="Let me check. "),
                _tool_chunk(index=0, id="call_9", name="lookup", arguments="{}"),
                _raw_chunk([]),
            ],
        )

        events = [e async for e in provider.generate_structured_stream(_context())]
        kinds = [type(e).__name__ for e in events]
        # Text deltas come before tool calls, StreamDone last. A call's
        # arguments are surfaced as they are composed, so its fragments sit
        # between the text and the completed call.
        assert kinds == [
            "StreamTextDelta",
            "StreamToolCallDelta",
            "StreamToolCall",
            "StreamDone",
        ]


# ---------------------------------------------------------------------------
# Thinking / reasoning (<think> tags)
# ---------------------------------------------------------------------------


class TestPolarGridThinking:
    @pytest.mark.asyncio
    async def test_generate_extracts_thinking(self) -> None:
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(
            content="<think>Let me reason about it.</think>The answer is 42."
        )

        resp = await provider.generate(_context())

        assert resp.thinking == "Let me reason about it."
        assert resp.content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_generate_no_thinking_leaves_content(self) -> None:
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(content="Just an answer.")

        resp = await provider.generate(_context())

        assert resp.thinking is None
        assert resp.content == "Just an answer."

    @pytest.mark.asyncio
    async def test_streaming_emits_thinking_then_text(self) -> None:
        provider, mod = _provider()
        _serve(
            mod,
            [
                _stream_chunk(content="<think>"),
                _stream_chunk(content="reasoning here"),
                _stream_chunk(content="</think>"),
                _stream_chunk(content="final answer", finish_reason="stop"),
            ],
        )

        events = [e async for e in provider.generate_structured_stream(_context())]
        thinking = "".join(e.thinking for e in events if isinstance(e, StreamThinkingDelta))
        text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))

        assert thinking == "reasoning here"
        assert text == "final answer"
        # Order: thinking deltas precede text deltas.
        kinds = [
            type(e).__name__
            for e in events
            if isinstance(e, StreamThinkingDelta | StreamTextDelta)
        ]
        assert kinds == ["StreamThinkingDelta", "StreamTextDelta"]

    @pytest.mark.asyncio
    async def test_streaming_thinking_tag_split_across_chunks(self) -> None:
        provider, mod = _provider()
        _serve(
            mod,
            [
                _stream_chunk(content="<th"),
                _stream_chunk(content="ink>deep "),
                _stream_chunk(content="thoughts</thi"),
                _stream_chunk(content="nk>the answer", finish_reason="stop"),
            ],
        )

        events = [e async for e in provider.generate_structured_stream(_context())]
        thinking = "".join(e.thinking for e in events if isinstance(e, StreamThinkingDelta))
        text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))

        assert thinking == "deep thoughts"
        assert text == "the answer"

    @pytest.mark.asyncio
    async def test_generate_stream_filters_out_thinking(self) -> None:
        provider, mod = _provider()
        _serve(mod, [_stream_chunk(content="<think>hidden</think>visible", finish_reason="stop")])

        chunks = [c async for c in provider.generate_stream(_context())]

        assert "".join(chunks) == "visible"

    @pytest.mark.asyncio
    async def test_thinking_true_sets_enable_thinking(self) -> None:
        provider, mod = _provider(thinking=True)
        mod._client._make_request.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        request = mod._client._make_request.await_args.kwargs["body"]
        assert request["enable_thinking"] is True
        # The toggle rides on enable_thinking, so the user message is untouched.
        user = [m for m in request["messages"] if m["role"] == "user"][-1]
        assert user["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_thinking_false_sets_enable_thinking_false(self) -> None:
        provider, mod = _provider(thinking=False)
        mod._client._make_request.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        request = mod._client._make_request.await_args.kwargs["body"]
        assert request["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_thinking_none_omits_enable_thinking(self) -> None:
        provider, mod = _provider()  # thinking defaults to None
        mod._client._make_request.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        request = mod._client._make_request.await_args.kwargs["body"]
        assert "enable_thinking" not in request

    @pytest.mark.asyncio
    async def test_assistant_thinking_not_round_tripped(self) -> None:
        # qwen echoes any wrapper we feed back, so prior reasoning must be
        # dropped from history — not re-sent as [thinking] text.
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(content="ok")

        messages = [
            AIMessage(role="user", content="hi"),
            AIMessage(
                role="assistant",
                content=[
                    AIThinkingPart(thinking="secret chain of thought"),
                    AITextPart(text="Hello!"),
                ],
            ),
            AIMessage(role="user", content="more"),
        ]
        await provider.generate(_context(messages=messages, system_prompt=None))

        request = mod._client._make_request.await_args.kwargs["body"]
        blob = json.dumps(request)
        assert "[thinking]" not in blob
        assert "secret chain of thought" not in blob
        # The assistant's actual text is still sent.
        assistant = [m for m in request["messages"] if m["role"] == "assistant"][0]
        assert assistant["content"] == "Hello!"


# ---------------------------------------------------------------------------
# Multi-turn tool messages
# ---------------------------------------------------------------------------


class TestPolarGridToolMessages:
    @pytest.mark.asyncio
    async def test_renders_tool_call_and_result_messages(self) -> None:
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(content="ok")

        messages = [
            AIMessage(role="user", content="Weather in Montreal?"),
            AIMessage(
                role="assistant",
                content=[
                    AIToolCallPart(
                        id="call_1", name="get_weather", arguments={"city": "Montreal"}
                    ),
                ],
            ),
            AIMessage(
                role="tool",
                content=[
                    AIToolResultPart(
                        tool_call_id="call_1", name="get_weather", result="12C, sunny"
                    ),
                ],
            ),
        ]
        await provider.generate(_context(messages=messages, system_prompt=None))

        msgs = mod._client._make_request.await_args.kwargs["body"]["messages"]
        assistant = next(m for m in msgs if m["role"] == "assistant")
        assert assistant["tool_calls"][0]["id"] == "call_1"
        assert assistant["tool_calls"][0]["type"] == "function"
        assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
        # Arguments rendered back as a JSON string for the wire.
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
            "city": "Montreal"
        }

        tool_msg = next(m for m in msgs if m["role"] == "tool")
        assert tool_msg["content"] == "12C, sunny"
        assert tool_msg["tool_call_id"] == "call_1"
        assert tool_msg["name"] == "get_weather"

    def test_image_tool_result_splits_to_user_message(self) -> None:
        # polargrid-sdk 0.9.0 accepts multimodal chat content. A tool message
        # can't carry an image, so it stays text-only and the image rides on a
        # synthetic user message right after (OpenAI-shaped image_url).
        provider, _ = _provider()
        messages = provider._build_messages(
            [
                AIMessage(
                    role="tool",
                    content=[
                        AIToolResultPart(
                            tool_call_id="call_1",
                            name="screenshot",
                            result=[
                                AITextPart(text="the screen"),
                                AIImagePart(url="data:image/png;base64,SU1HREFUQQ=="),
                            ],
                        )
                    ],
                )
            ],
            None,
        )
        assert messages == [
            {
                "role": "tool",
                "content": "the screen",
                "tool_call_id": "call_1",
                "name": "screenshot",
            },
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

    def test_renders_user_image_as_image_url(self) -> None:
        # A user turn with a text + image part renders as an OpenAI-shaped
        # multimodal content list (text block + image_url block, order kept).
        provider, _ = _provider()
        messages = provider._build_messages(
            [
                AIMessage(
                    role="user",
                    content=[
                        AITextPart(text="what is this?"),
                        AIImagePart(url="data:image/png;base64,SU1HREFUQQ=="),
                    ],
                )
            ],
            None,
        )
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,SU1HREFUQQ=="},
                    },
                ],
            }
        ]


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


class TestPolarGridModels:
    def test_available_models_catalog(self) -> None:
        provider, _ = _provider()
        models = provider.available_models()
        by_id = {m.id: m for m in models}
        # The public lineup is one model. qwen-3.5-27b was retired on
        # 2026-08-20 (404 model_not_loaded everywhere) and qwen-3.6-35b-a3b
        # is a customer pilot served from no public edge: neither is offered.
        assert set(by_id) == {"qwen-3.8-27b"}
        qwen38 = by_id["qwen-3.8-27b"]
        assert qwen38.display_name == "Qwen 3.8 27B"
        assert qwen38.context_window == 262_144
        # Thinking validated live on yto-01; image input refused server-side
        # ("does not support image input"), so text-only.
        assert "thinking" in qwen38.capabilities
        assert qwen38.supports_vision is False
        # The vendor publishes a list price for it (models page).
        assert qwen38.pricing is not None
        assert qwen38.pricing.input_per_million == 0.20
        assert qwen38.pricing.output_per_million == 0.75

    def test_available_models_is_offline_classmethod(self) -> None:
        # Callable on the class without an SDK/instance (no network/key).
        from roomkit.providers.polargrid.ai import PolarGridAIProvider

        ids = [m.id for m in PolarGridAIProvider.available_models()]
        assert ids == ["qwen-3.8-27b"]

    def test_pilot_model_is_recognised_but_not_advertised(self) -> None:
        from roomkit.providers.polargrid.ai import PolarGridAIProvider
        from roomkit.providers.polargrid.models import MODELS_BY_ID, PILOT_MODELS

        pilot = {m.id for m in PILOT_MODELS}
        assert pilot == {"qwen-3.6-35b-a3b"}
        assert MODELS_BY_ID["qwen-3.6-35b-a3b"].supports_vision is True
        assert "thinking" in MODELS_BY_ID["qwen-3.6-35b-a3b"].capabilities
        assert not any(m.id in pilot for m in PolarGridAIProvider.available_models())

    def test_supports_vision_is_model_driven(self) -> None:
        # Per-model: only the pilot qwen-3.6-35b-a3b reads images (verified
        # live on yul-02 while it was public); 3.8, a retired id and any
        # unknown model are text-only.
        vision, _ = _provider(model="qwen-3.6-35b-a3b")
        assert vision.supports_vision is True

        text_35, _ = _provider(model="qwen-3.5-27b")
        assert text_35.supports_vision is False

        text_default, _ = _provider(model="qwen-3.8-27b")
        assert text_default.supports_vision is False

        unknown, _ = _provider(model="some-text-only-model")
        assert unknown.supports_vision is False

    @pytest.mark.asyncio
    async def test_list_models_tags_what_the_edge_lists(self) -> None:
        provider, mod = _provider()
        # As the edge answers (measured 2026-10-02, yul-01): no model type.
        listed = [
            "qwen-3.6-35b-a3b",
            "kokoro-82m",
            "whisper-large-v3-turbo",
            "tada-3b-ml",
            "cohere-transcribe-03-2026",
            "whisper-small",
            "mystery-1b",
        ]
        data = [
            {"id": i, "object": "model", "created": 0, "owned_by": "triton", "root": i}
            for i in listed
        ]
        _respond(mod, {"object": "list", "data": data})

        by_id = {m.id: m for m in await provider.list_models()}

        # Every model the edge lists is returned (chat + STT/TTS).
        assert set(by_id) == set(listed)
        # The catalog backfills a chat model (a pilot one too) and tags a
        # speech model; an id it does not know stays unknown.
        assert by_id["qwen-3.6-35b-a3b"].display_name == "Qwen 3.6 35B-A3B"
        assert by_id["qwen-3.6-35b-a3b"].supports_vision is True
        assert by_id["qwen-3.6-35b-a3b"].capabilities == [
            "completion",
            "tools",
            "thinking",
            "vision",
        ]
        assert by_id["kokoro-82m"].capabilities == ["speech"]
        assert by_id["tada-3b-ml"].capabilities == ["speech"]
        assert by_id["whisper-large-v3-turbo"].capabilities == ["transcription"]
        assert by_id["cohere-transcribe-03-2026"].capabilities == ["transcription"]
        # An id the catalog does not know is read by its name.
        assert by_id["whisper-small"].capabilities == ["transcription"]
        assert by_id["mystery-1b"].capabilities == []

    def test_available_regions_catalog(self) -> None:
        provider, _ = _provider()
        regions = provider.available_regions()
        by_id = {r.id: r for r in regions}
        # Every edge the SDK can route, with the Canada/US residency split.
        # yul-02 left the vendor's published list with the qwen-3.6 pilot but
        # the SDK still routes it and it still answers, so it stays.
        assert len(regions) == 16
        assert by_id["yul-02"].name == "Montreal 02"
        assert by_id["yul-02"].location == "Canada East"
        canadian = [r.id for r in regions if (r.location or "").startswith("Canada")]
        assert set(canadian) == {"yto-01", "yul-01", "yul-02", "yvr-02"}
        # The edges that came online after the first snapshot. roomkit refused
        # them until 2026-08-05 even though the SDK routes them, which made a
        # valid region look like a typo.
        assert {"was-01", "mia-01", "chi-01", "sfo-03", "lax-01", "sea-01", "phx-01"} <= set(by_id)
        assert all((by_id[r].location or "").startswith("US") for r in ("lax-01", "mia-01"))

    def test_region_catalog_matches_what_the_sdk_can_route(self) -> None:
        # The list is an offline mirror of polargrid.client.POLARGRID_REGIONS,
        # and a mirror that drifts is worse than no mirror: an edge missing
        # here is rejected by resolve_region_id even though the SDK would
        # route it. Skipped when the optional SDK is absent.
        client = pytest.importorskip("polargrid.client")

        from roomkit.providers.polargrid.models import REGIONS

        assert {r.id for r in REGIONS} == set(client.POLARGRID_REGIONS)

    def test_new_edges_resolve(self) -> None:
        from roomkit.providers.polargrid.models import resolve_region_id

        assert resolve_region_id("lax-01") == "lax-01"
        assert resolve_region_id("SFO-03") == "sfo-03"  # case-insensitive
        assert resolve_region_id("yul-2") is None  # still rejects a typo

    @pytest.mark.asyncio
    async def test_connected_region_reports_edge_with_location(self) -> None:
        provider, mod = _provider(region="toronto")
        mod._client.get_region_id.return_value = "yto-01"
        mod._client.get_region_name.return_value = "Toronto 01"

        region = await provider.connected_region()

        assert region.id == "yto-01"
        assert region.name == "Toronto 01"  # SDK name preferred
        assert region.location == "Canada Central"  # backfilled from the catalog


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


async def _read_all(stream: Any) -> None:
    _ = [event async for event in stream]


# Each path that reaches the server: how to call it, and where its client fails.
_CALLS: dict[str, Any] = {
    "generate": lambda provider: provider.generate(_context()),
    "stream": lambda provider: _read_all(provider.generate_structured_stream(_context())),
    "list_models": lambda provider: provider.list_models(),
}


def _failing_stream(error: Exception) -> Any:
    async def stream_post(endpoint: str, body: dict[str, Any]) -> Any:
        raise error
        yield  # an async generator, as the SDK's is

    return stream_post


_FAILURES: dict[str, Any] = {
    "generate": lambda client, error: setattr(client._make_request, "side_effect", error),
    "stream": lambda client, error: setattr(client, "_stream_post", _failing_stream(error)),
    "list_models": lambda client, error: setattr(client.list_models, "side_effect", error),
}


async def _fail(provider: Any, mod: MagicMock, path: str, error: Exception) -> ProviderError:
    """The error *path* surfaces when its client raises *error*."""
    _FAILURES[path](mod._client, error)
    with pytest.raises(ProviderError) as exc:
        await _CALLS[path](provider)
    return exc.value


class TestPolarGridErrors:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["generate", "stream", "list_models"])
    @pytest.mark.parametrize(
        ("status", "retryable"),
        [(401, False), (402, False), (400, False), (404, False), (429, True), (503, True)],
    )
    async def test_an_http_error_keeps_its_status(
        self, path: str, status: int, retryable: bool
    ) -> None:
        """Each error as the SDK builds it from the server's status, on every
        path that reaches the server."""
        provider, mod = _provider()
        error = polargrid.create_error_from_response(status, None, "refused", None, "req_1")

        failed = await _fail(provider, mod, path, error)

        assert (failed.status_code, failed.retryable) == (status, retryable)
        assert failed.provider == "polargrid"

    @pytest.mark.asyncio
    async def test_a_request_the_sdk_refused_itself_has_no_status(self) -> None:
        provider, mod = _provider()
        refused = polargrid.ValidationError("max_tokens must be at most 4096")

        failed = await _fail(provider, mod, "generate", refused)

        assert (failed.status_code, failed.retryable) == (None, False)

    @pytest.mark.asyncio
    async def test_an_unknown_error_reads_as_on_every_provider(self) -> None:
        """No status and no lost connection: final, as the shared rule reads
        it (RMK-524), no longer retried by a default of PolarGrid's own."""
        provider, mod = _provider()
        mod._client._make_request.side_effect = RuntimeError("???")

        with pytest.raises(ProviderError) as exc:
            await provider.generate(_context())

        assert exc.value.retryable is False


# ---------------------------------------------------------------------------
# Config + lazy import
# ---------------------------------------------------------------------------


class TestPolarGridConfig:
    def test_defaults(self) -> None:
        config = PolarGridConfig(api_key="pg_test")
        assert config.model == "qwen-3.8-27b"
        assert config.region is None
        assert config.top_p == 0.9
        assert config.timeout == 30.0
        assert config.max_retries == 0
        assert config.debug is False

    def test_overrides(self) -> None:
        config = PolarGridConfig(
            api_key="pg_test",
            model="qwen-3.5-27b",
            region="vancouver",
            max_tokens=2048,
            debug=True,
        )
        assert config.model == "qwen-3.5-27b"
        assert config.region == "vancouver"
        assert config.max_tokens == 2048
        assert config.debug is True

    @pytest.mark.parametrize("region", ["yul-02", "toronto", "MONTREAL", "sf", None])
    def test_valid_region_accepted(self, region: str | None) -> None:
        # Edge ids, friendly aliases (case-insensitive), and None all pass.
        assert PolarGridConfig(api_key="pg_test", region=region).region == region

    @pytest.mark.parametrize("region", ["yul-2", "quebec", "yto-99", ""])
    def test_unknown_region_rejected(self, region: str) -> None:
        # A typo like "yul-2" must fail loudly at construction, not later with
        # an opaque DNS error from an unroutable host.
        with pytest.raises(ValueError, match=rf"unknown PolarGrid region {region!r}"):
            PolarGridConfig(api_key="pg_test", region=region)


class TestPolarGridLazyImport:
    def test_import_error_message(self) -> None:
        with patch.dict("sys.modules", {"polargrid": None}):
            import importlib

            import roomkit.providers.polargrid.ai as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match=r"pip install roomkit\[polargrid\]"):
                mod.PolarGridAIProvider(_config())


class TestPolarGridImageDataURIs:
    def test_a_malformed_payload_is_refused_before_the_request(self) -> None:
        provider, _ = _provider()
        with pytest.raises(ProviderError, match="not valid base64") as excinfo:
            provider._build_messages(
                [
                    AIMessage(
                        role="user",
                        content=[AIImagePart(url="data:image/png;base64,not*base64")],
                    )
                ],
                None,
            )
        assert excinfo.value.retryable is False
        assert excinfo.value.provider == provider._provider_name

    def test_a_tool_result_image_is_rebuilt_too(self) -> None:
        provider, _ = _provider()
        messages = provider._build_messages(
            [
                AIMessage(
                    role="tool",
                    content=[
                        AIToolResultPart(
                            tool_call_id="call_1",
                            name="screenshot",
                            result=[
                                AIImagePart(url="data:;base64,QUJDMTIz", mime_type="image/png")
                            ],
                        )
                    ],
                )
            ],
            None,
        )
        assert messages[-1]["content"] == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDMTIz"}}
        ]


_VERDICT: dict[str, Any] = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["label"],
    "additionalProperties": False,
}


class TestPolarGridResponseSchema:
    """RFC §6.7: the schema rides a ``json_schema`` response format."""

    async def test_the_schema_rides_the_response_format(self) -> None:
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(content='{"label": "yes"}')

        result = await provider.generate(_context(response_schema=_VERDICT))

        assert result.content == '{"label": "yes"}'
        request = mod._client._make_request.await_args.kwargs["body"]
        assert request["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": _VERDICT, "strict": True},
        }

    @pytest.mark.parametrize(
        ("content", "finish_reason", "reason"),
        [
            ("", "content_filter", "refusal"),
            ('{"label": "y', "length", "truncated"),
            ("Yes.", "stop", "invalid_json"),
        ],
    )
    async def test_an_answer_without_its_document_raises(
        self, content: str, finish_reason: str, reason: str
    ) -> None:
        provider, mod = _provider()
        mod._client._make_request.return_value = _response_obj(
            content=content, finish_reason=finish_reason
        )

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.generate(_context(response_schema=_VERDICT))

        assert exc.value.reason == reason

    async def test_no_choice_at_all_is_not_a_document(self) -> None:
        provider, mod = _provider()
        _respond(mod, _completion())

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.generate(_context(response_schema=_VERDICT))

        assert exc.value.reason == "invalid_json"

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
        provider, mod = _provider()
        sent = _serve(
            mod, [_stream_chunk(content='{"label": "yes"}'), _stream_chunk(finish_reason="stop")]
        )

        events, error = await self._drain(
            provider.generate_structured_stream(_context(response_schema=_VERDICT))
        )

        assert error is None
        assert isinstance(events[-1], StreamDone)
        assert (
            "".join(e.text for e in events if isinstance(e, StreamTextDelta)) == '{"label": "yes"}'
        )
        assert sent[0]["response_format"]["json_schema"]["schema"] == _VERDICT

    async def test_a_streamed_answer_that_is_not_the_document_raises_instead_of_done(
        self,
    ) -> None:
        provider, mod = _provider()
        _serve(mod, [_stream_chunk(content="Yes."), _stream_chunk(finish_reason="stop")])

        events, error = await self._drain(
            provider.generate_structured_stream(_context(response_schema=_VERDICT))
        )

        assert error is not None and error.reason == "invalid_json"
        assert not any(isinstance(e, StreamDone) for e in events)
