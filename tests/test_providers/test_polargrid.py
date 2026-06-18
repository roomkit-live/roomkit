"""Tests for the PolarGrid AI provider."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import (
    AIContext,
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
)
from roomkit.providers.polargrid.config import PolarGridConfig

# ---------------------------------------------------------------------------
# Fake polargrid module
# ---------------------------------------------------------------------------


class _PGError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _AuthError(_PGError):
    pass


class _ValidationError(_PGError):
    pass


class _RateLimitError(_PGError):
    pass


class _NetworkError(_PGError):
    pass


class _TimeoutError(_PGError):
    pass


class _NotFoundError(_PGError):
    pass


class _ServerError(_PGError):
    pass


def _mock_polargrid_module() -> MagicMock:
    """Return a MagicMock that behaves like the polargrid module."""
    mod = MagicMock()
    mod.AuthenticationError = _AuthError
    mod.ValidationError = _ValidationError
    mod.RateLimitError = _RateLimitError
    mod.NetworkError = _NetworkError
    mod.TimeoutError = _TimeoutError
    mod.NotFoundError = _NotFoundError
    mod.ServerError = _ServerError

    client = MagicMock()
    client.chat_completion = AsyncMock()
    # chat_completion_stream is sync-returning-async-iterable; we set
    # its return value per-test to an _FakeStream instance.
    client.chat_completion_stream = MagicMock()
    client.list_models = AsyncMock()
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
    content: str = "",
    finish_reason: str = "stop",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    model: str = "qwen-3.5-27b",
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class _FakeStream:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks
        self._i = 0

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk


def _stream_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content) if content is not None else SimpleNamespace()
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    fields: dict[str, Any] = {"choices": [choice]}
    if prompt_tokens is not None or completion_tokens is not None:
        fields["usage"] = SimpleNamespace(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
        )
    return SimpleNamespace(**fields)


def _tool_chunk(
    *,
    index: int = 0,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """A streaming chunk carrying a fragmented tool-call delta.

    Mirrors polargrid-sdk's ``ToolCallDelta``: ``function`` is a dict and
    arguments arrive in fragments to be concatenated per ``index``.
    """
    func: dict[str, Any] = {}
    if name is not None:
        func["name"] = name
    if arguments is not None:
        func["arguments"] = arguments
    tc = SimpleNamespace(index=index, id=id, type="function", function=func or None)
    delta = SimpleNamespace(content=None, tool_calls=[tc])
    return SimpleNamespace(
        choices=[SimpleNamespace(index=index, delta=delta, finish_reason=finish_reason)]
    )


def _tool_call_obj(*, id: str, name: str, arguments: str) -> SimpleNamespace:
    """A non-streaming ``ToolCall``: ``function.arguments`` is a JSON string."""
    return SimpleNamespace(
        id=id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


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
        mod._client.chat_completion.return_value = _response_obj(content="Hello!")

        resp = await provider.generate(_context())

        assert resp.content == "Hello!"
        assert resp.finish_reason == "stop"
        assert resp.usage == {"input_tokens": 11, "output_tokens": 7}

    @pytest.mark.asyncio
    async def test_generate_builds_messages_with_system_prompt(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.return_value = _response_obj(content="ok")

        await provider.generate(_context(system_prompt="Be terse."))

        request = mod._client.chat_completion.await_args.args[0]
        assert request["messages"][0] == {"role": "system", "content": "Be terse."}
        assert request["messages"][1] == {"role": "user", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_generate_passes_temperature_and_max_tokens(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.return_value = _response_obj(content="ok")

        await provider.generate(_context(max_tokens=512, temperature=0.2))

        request = mod._client.chat_completion.await_args.args[0]
        assert request["model"] == "qwen-3.5-27b"
        assert request["temperature"] == 0.2
        assert request["max_tokens"] == 512
        assert request["top_p"] == 0.9
        assert request["stream"] is False

    @pytest.mark.asyncio
    async def test_generate_forwards_tools(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.return_value = _response_obj(content="ok")

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

        request = mod._client.chat_completion.await_args.args[0]
        assert request["tools"][0]["type"] == "function"
        assert request["tools"][0]["function"]["name"] == "get_weather"
        assert request["tools"][0]["function"]["parameters"]["required"] == ["city"]
        # No tool_choice in AIContext — leave it unset so PolarGrid defaults to auto.
        assert "tool_choice" not in request

    @pytest.mark.asyncio
    async def test_debug_logs_full_request(self, caplog: pytest.LogCaptureFixture) -> None:
        provider, mod = _provider(thinking=True)
        mod._client.chat_completion.return_value = _response_obj(content="ok")

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
        message = SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                _tool_call_obj(
                    id="call_1",
                    name="get_weather",
                    arguments=json.dumps({"city": "Montreal"}),
                )
            ],
        )
        mod._client.chat_completion.return_value = SimpleNamespace(
            model="qwen-3.5-27b",
            choices=[SimpleNamespace(index=0, message=message, finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
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
        message = SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[_tool_call_obj(id="call_1", name="t", arguments="{not json")],
        )
        mod._client.chat_completion.return_value = SimpleNamespace(
            model="qwen-3.5-27b",
            choices=[SimpleNamespace(index=0, message=message, finish_reason="tool_calls")],
            usage=None,
        )

        resp = await provider.generate(_context())

        assert resp.tool_calls[0].arguments == {"raw": "{not json"}

    @pytest.mark.asyncio
    async def test_generate_empty_choices_returns_empty_content(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.return_value = SimpleNamespace(
            choices=[], usage=None, model="qwen-3.5-27b"
        )

        resp = await provider.generate(_context())

        assert resp.content == ""


# ---------------------------------------------------------------------------
# Region routing
# ---------------------------------------------------------------------------


class TestPolarGridRegionRouting:
    @pytest.mark.asyncio
    async def test_region_none_uses_async_create(self) -> None:
        provider, mod = _provider()  # region defaults to None
        mod._client.chat_completion.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        mod.PolarGrid.create.assert_awaited_once()
        call_kwargs = mod.PolarGrid.create.await_args.kwargs
        assert "region" not in call_kwargs
        assert call_kwargs["api_key"] == "pg_test"

    @pytest.mark.asyncio
    async def test_region_pinned_uses_sync_constructor(self) -> None:
        provider, mod = _provider(region="toronto")
        mod._client.chat_completion.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        mod.PolarGrid.assert_called_once()
        call_kwargs = mod.PolarGrid.call_args.kwargs
        assert call_kwargs["region"] == "toronto"
        mod.PolarGrid.create.assert_not_called()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestPolarGridStreaming:
    @pytest.mark.asyncio
    async def test_streams_text_deltas_and_done(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion_stream.return_value = _FakeStream(
            [
                _stream_chunk(content="42"),
                _stream_chunk(content=" is "),
                _stream_chunk(content="it.", finish_reason="stop"),
                # Final usage-only chunk with no choices.
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5, total_tokens=17),
                ),
            ]
        )

        events = [e async for e in provider.generate_structured_stream(_context())]
        text_events = [e for e in events if isinstance(e, StreamTextDelta)]
        done_events = [e for e in events if isinstance(e, StreamDone)]

        assert [e.text for e in text_events] == ["42", " is ", "it."]
        assert len(done_events) == 1
        assert done_events[0].finish_reason == "stop"
        assert done_events[0].usage == {"input_tokens": 12, "output_tokens": 5}

    @pytest.mark.asyncio
    async def test_streaming_passes_stream_true(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion_stream.return_value = _FakeStream([])

        async for _ in provider.generate_structured_stream(_context()):
            pass

        request = mod._client.chat_completion_stream.call_args.args[0]
        assert request["stream"] is True

    @pytest.mark.asyncio
    async def test_generate_stream_yields_text_only(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion_stream.return_value = _FakeStream(
            [
                _stream_chunk(content="a"),
                _stream_chunk(content="b", finish_reason="stop"),
            ]
        )

        chunks = [c async for c in provider.generate_stream(_context())]
        assert chunks == ["a", "b"]

    @pytest.mark.asyncio
    async def test_streaming_emits_tool_calls(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion_stream.return_value = _FakeStream(
            [
                # id + name + first arg fragment, then the tail fragment.
                _tool_chunk(index=0, id="call_1", name="get_weather", arguments='{"ci'),
                _tool_chunk(index=0, arguments='ty": "Montreal"}'),
                # Finish marker, then a usage-only chunk.
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            index=0,
                            delta=SimpleNamespace(content=None, tool_calls=None),
                            finish_reason="tool_calls",
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
                ),
            ]
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

    @pytest.mark.asyncio
    async def test_streaming_text_then_tool_call_ordering(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion_stream.return_value = _FakeStream(
            [
                _stream_chunk(content="Let me check. "),
                _tool_chunk(index=0, id="call_9", name="lookup", arguments="{}"),
                SimpleNamespace(choices=[], usage=None),
            ]
        )

        events = [e async for e in provider.generate_structured_stream(_context())]
        kinds = [type(e).__name__ for e in events]
        # Text deltas come before tool calls, StreamDone last.
        assert kinds == ["StreamTextDelta", "StreamToolCall", "StreamDone"]


# ---------------------------------------------------------------------------
# Thinking / reasoning (<think> tags)
# ---------------------------------------------------------------------------


class TestPolarGridThinking:
    @pytest.mark.asyncio
    async def test_generate_extracts_thinking(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.return_value = _response_obj(
            content="<think>Let me reason about it.</think>The answer is 42."
        )

        resp = await provider.generate(_context())

        assert resp.thinking == "Let me reason about it."
        assert resp.content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_generate_no_thinking_leaves_content(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.return_value = _response_obj(content="Just an answer.")

        resp = await provider.generate(_context())

        assert resp.thinking is None
        assert resp.content == "Just an answer."

    @pytest.mark.asyncio
    async def test_streaming_emits_thinking_then_text(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion_stream.return_value = _FakeStream(
            [
                _stream_chunk(content="<think>"),
                _stream_chunk(content="reasoning here"),
                _stream_chunk(content="</think>"),
                _stream_chunk(content="final answer", finish_reason="stop"),
            ]
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
        mod._client.chat_completion_stream.return_value = _FakeStream(
            [
                _stream_chunk(content="<th"),
                _stream_chunk(content="ink>deep "),
                _stream_chunk(content="thoughts</thi"),
                _stream_chunk(content="nk>the answer", finish_reason="stop"),
            ]
        )

        events = [e async for e in provider.generate_structured_stream(_context())]
        thinking = "".join(e.thinking for e in events if isinstance(e, StreamThinkingDelta))
        text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))

        assert thinking == "deep thoughts"
        assert text == "the answer"

    @pytest.mark.asyncio
    async def test_generate_stream_filters_out_thinking(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion_stream.return_value = _FakeStream(
            [_stream_chunk(content="<think>hidden</think>visible", finish_reason="stop")]
        )

        chunks = [c async for c in provider.generate_stream(_context())]

        assert "".join(chunks) == "visible"

    @pytest.mark.asyncio
    async def test_thinking_true_sets_enable_thinking(self) -> None:
        provider, mod = _provider(thinking=True)
        mod._client.chat_completion.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        request = mod._client.chat_completion.await_args.args[0]
        assert request["enable_thinking"] is True
        # The toggle rides on enable_thinking, so the user message is untouched.
        user = [m for m in request["messages"] if m["role"] == "user"][-1]
        assert user["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_thinking_false_sets_enable_thinking_false(self) -> None:
        provider, mod = _provider(thinking=False)
        mod._client.chat_completion.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        request = mod._client.chat_completion.await_args.args[0]
        assert request["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_thinking_none_omits_enable_thinking(self) -> None:
        provider, mod = _provider()  # thinking defaults to None
        mod._client.chat_completion.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        request = mod._client.chat_completion.await_args.args[0]
        assert "enable_thinking" not in request

    @pytest.mark.asyncio
    async def test_assistant_thinking_not_round_tripped(self) -> None:
        # qwen echoes any wrapper we feed back, so prior reasoning must be
        # dropped from history — not re-sent as [thinking] text.
        provider, mod = _provider()
        mod._client.chat_completion.return_value = _response_obj(content="ok")

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

        request = mod._client.chat_completion.await_args.args[0]
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
        mod._client.chat_completion.return_value = _response_obj(content="ok")

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

        msgs = mod._client.chat_completion.await_args.args[0]["messages"]
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


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


class TestPolarGridModels:
    def test_available_models_catalog(self) -> None:
        provider, _ = _provider()
        models = provider.available_models()
        by_id = {m.id: m for m in models}
        assert "qwen-3.5-27b" in by_id
        assert "qwen-3.6-35b-a3b" in by_id
        assert by_id["qwen-3.5-27b"].display_name == "Qwen 3.5 27B"
        # 3.6 is the thinking-capable model (validated on yul-02).
        assert "thinking" in by_id["qwen-3.6-35b-a3b"].capabilities
        assert by_id["qwen-3.5-27b"].supports_vision is False

    def test_available_models_is_offline_classmethod(self) -> None:
        # Callable on the class without an SDK/instance (no network/key).
        from roomkit.providers.polargrid.ai import PolarGridAIProvider

        ids = [m.id for m in PolarGridAIProvider.available_models()]
        assert ids == ["qwen-3.5-27b", "qwen-3.6-35b-a3b"]

    @pytest.mark.asyncio
    async def test_list_models_maps_and_backfills(self) -> None:
        provider, mod = _provider()
        mod._client.list_models.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(id="qwen-3.6-35b-a3b", pg_model_type="llm"),
                SimpleNamespace(id="kokoro-82m", pg_model_type="tts"),
                SimpleNamespace(id="whisper-large-v3-turbo", pg_model_type=None),
            ]
        )

        models = await provider.list_models()
        by_id = {m.id: m for m in models}

        # Live edge models are all returned (chat + STT/TTS).
        assert set(by_id) == {"qwen-3.6-35b-a3b", "kokoro-82m", "whisper-large-v3-turbo"}
        # pg_model_type → capabilities; curated backfills the display name.
        assert by_id["kokoro-82m"].capabilities == ["tts"]
        assert by_id["qwen-3.6-35b-a3b"].display_name == "Qwen 3.6 35B-A3B"
        # No pg_model_type and not in catalog → empty capabilities, no crash.
        assert by_id["whisper-large-v3-turbo"].capabilities == []

    @pytest.mark.asyncio
    async def test_list_models_wraps_sdk_error(self) -> None:
        provider, mod = _provider()
        mod._client.list_models.side_effect = _ServerError("down", status_code=503)

        with pytest.raises(ProviderError) as exc:
            await provider.list_models()

        assert exc.value.retryable is True


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestPolarGridErrors:
    @pytest.mark.asyncio
    async def test_auth_error_not_retryable(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.side_effect = _AuthError("bad key", status_code=401)

        with pytest.raises(ProviderError) as exc:
            await provider.generate(_context())

        assert exc.value.retryable is False
        assert exc.value.status_code == 401
        assert exc.value.provider == "polargrid"

    @pytest.mark.asyncio
    async def test_validation_error_not_retryable(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.side_effect = _ValidationError("bad input", status_code=400)

        with pytest.raises(ProviderError) as exc:
            await provider.generate(_context())

        assert exc.value.retryable is False

    @pytest.mark.asyncio
    async def test_rate_limit_error_retryable(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.side_effect = _RateLimitError("slow down", status_code=429)

        with pytest.raises(ProviderError) as exc:
            await provider.generate(_context())

        assert exc.value.retryable is True
        assert exc.value.status_code == 429

    @pytest.mark.asyncio
    async def test_server_error_retryable(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.side_effect = _ServerError("oops", status_code=503)

        with pytest.raises(ProviderError) as exc:
            await provider.generate(_context())

        assert exc.value.retryable is True

    @pytest.mark.asyncio
    async def test_unknown_error_retryable(self) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.side_effect = RuntimeError("???")

        with pytest.raises(ProviderError) as exc:
            await provider.generate(_context())

        assert exc.value.retryable is True


# ---------------------------------------------------------------------------
# Config + lazy import
# ---------------------------------------------------------------------------


class TestPolarGridConfig:
    def test_defaults(self) -> None:
        config = PolarGridConfig(api_key="pg_test")
        assert config.model == "qwen-3.5-27b"
        assert config.region is None
        assert config.temperature == 0.7
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
            temperature=0.1,
            debug=True,
        )
        assert config.model == "qwen-3.5-27b"
        assert config.region == "vancouver"
        assert config.max_tokens == 2048
        assert config.temperature == 0.1
        assert config.debug is True


class TestPolarGridLazyImport:
    def test_import_error_message(self) -> None:
        with patch.dict("sys.modules", {"polargrid": None}):
            import importlib

            import roomkit.providers.polargrid.ai as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match=r"pip install roomkit\[polargrid\]"):
                mod.PolarGridAIProvider(_config())
