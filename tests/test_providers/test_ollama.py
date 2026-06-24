"""Tests for the Ollama AI provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
)
from roomkit.providers.ollama.config import OllamaConfig


class _FakeResponseError(Exception):
    """Stub for ollama.ResponseError used in tests."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.error = message
        self.status_code = status_code


def _mock_ollama_module() -> MagicMock:
    """Return a MagicMock that behaves like the ollama module."""
    mod = MagicMock()
    mod.ResponseError = _FakeResponseError
    client = MagicMock()
    client.chat = AsyncMock()
    mod.AsyncClient.return_value = client
    return mod


def _config(**overrides: Any) -> OllamaConfig:
    defaults: dict[str, Any] = {"host": "http://localhost:11434", "model": "qwen3:8b"}
    defaults.update(overrides)
    return OllamaConfig(**defaults)


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {
        "messages": [AIMessage(role="user", content="Hi")],
        "system_prompt": "You are helpful.",
        "max_tokens": 256,
        "temperature": 0.7,
    }
    defaults.update(overrides)
    return AIContext(**defaults)


def _response_obj(
    *,
    content: str = "",
    thinking: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    done_reason: str = "stop",
    eval_count: int = 7,
    prompt_eval_count: int = 11,
    model: str = "qwen3:8b",
) -> SimpleNamespace:
    """Build a fake ollama ChatResponse object."""
    message_fields: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking is not None:
        message_fields["thinking"] = thinking
    if tool_calls is not None:
        message_fields["tool_calls"] = [
            SimpleNamespace(
                id=tc.get("id"),
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls
        ]
    return SimpleNamespace(
        model=model,
        message=SimpleNamespace(**message_fields),
        done=True,
        done_reason=done_reason,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
    )


class _FakeStream:
    """Async iterator of ollama stream chunks."""

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
    content: str = "",
    thinking: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    done: bool = False,
    done_reason: str | None = None,
    eval_count: int | None = None,
    prompt_eval_count: int | None = None,
) -> SimpleNamespace:
    message_fields: dict[str, Any] = {"role": "assistant"}
    if content:
        message_fields["content"] = content
    if thinking:
        message_fields["thinking"] = thinking
    if tool_calls:
        message_fields["tool_calls"] = [
            SimpleNamespace(
                id=tc.get("id"),
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls
        ]
    fields: dict[str, Any] = {
        "message": SimpleNamespace(**message_fields),
        "done": done,
    }
    if done_reason is not None:
        fields["done_reason"] = done_reason
    if eval_count is not None:
        fields["eval_count"] = eval_count
    if prompt_eval_count is not None:
        fields["prompt_eval_count"] = prompt_eval_count
    return SimpleNamespace(**fields)


def _provider(mod: MagicMock | None = None, **config_overrides: Any) -> Any:
    """Instantiate OllamaAIProvider with a mocked ollama module."""
    mod = mod or _mock_ollama_module()
    with patch.dict("sys.modules", {"ollama": mod}):
        from roomkit.providers.ollama.ai import OllamaAIProvider

        return OllamaAIProvider(_config(**config_overrides)), mod


class TestOllamaAIProviderGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="Hello!")

        resp = await provider.generate(_context())

        assert resp.content == "Hello!"
        assert resp.thinking is None
        assert resp.finish_reason == "stop"
        assert resp.usage == {"input_tokens": 11, "output_tokens": 7}

    @pytest.mark.asyncio
    async def test_generate_extracts_thinking(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _response_obj(
            content="The answer is 42.",
            thinking="The user is asking a meaning-of-life question.",
        )

        resp = await provider.generate(_context())

        assert resp.content == "The answer is 42."
        assert resp.thinking == "The user is asking a meaning-of-life question."

    @pytest.mark.asyncio
    async def test_generate_extracts_tool_calls(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _response_obj(
            content="",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    }
                }
            ],
        )

        resp = await provider.generate(
            _context(tools=[AITool(name="get_weather", description="x")])
        )

        assert len(resp.tool_calls) == 1
        call = resp.tool_calls[0]
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Paris"}
        # Synthetic id since Ollama doesn't issue stable ones
        assert call.id.startswith("call_get_weather")

    @pytest.mark.asyncio
    async def test_synthesized_tool_ids_unique_across_turns(self) -> None:
        # Regression: a per-response counter restarted at 0 each turn, so
        # downstream consumers that pair START/END tool_call events by id
        # collapsed every same-named call across the conversation onto a
        # single pair. Each generate() call must mint a fresh id.
        provider, mod = _provider()
        turn = {
            "function": {"name": "ping", "arguments": {}},
        }
        mod.AsyncClient.return_value.chat.side_effect = [
            _response_obj(content="", tool_calls=[turn]),
            _response_obj(content="", tool_calls=[turn]),
            _response_obj(content="", tool_calls=[turn]),
        ]
        ctx = _context(tools=[AITool(name="ping", description="x")])
        ids = [(await provider.generate(ctx)).tool_calls[0].id for _ in range(3)]
        assert len(set(ids)) == 3, ids

    @pytest.mark.asyncio
    async def test_generate_passes_system_prompt(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context(system_prompt="Be brief."))

        kwargs = mod.AsyncClient.return_value.chat.await_args.kwargs
        assert kwargs["messages"][0] == {"role": "system", "content": "Be brief."}

    @pytest.mark.asyncio
    async def test_generate_passes_tools(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(
            _context(
                tools=[
                    AITool(
                        name="get_weather",
                        description="Get the weather",
                        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                    )
                ]
            )
        )

        kwargs = mod.AsyncClient.return_value.chat.await_args.kwargs
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_generate_maps_options(self) -> None:
        provider, mod = _provider(num_ctx=8192)
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context(max_tokens=512, temperature=0.2))

        kwargs = mod.AsyncClient.return_value.chat.await_args.kwargs
        assert kwargs["options"] == {
            "temperature": 0.2,
            "num_predict": 512,
            "num_ctx": 8192,
        }

    @pytest.mark.asyncio
    async def test_generate_maps_sampling_options(self) -> None:
        provider, mod = _provider(num_ctx=8192, top_p=0.95, top_k=40, min_p=0.2)
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context(max_tokens=512, temperature=0.2))

        kwargs = mod.AsyncClient.return_value.chat.await_args.kwargs
        assert kwargs["options"] == {
            "temperature": 0.2,
            "num_predict": 512,
            "num_ctx": 8192,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.2,
        }

    @pytest.mark.asyncio
    async def test_generate_omits_unset_sampling_options(self) -> None:
        provider, mod = _provider(num_ctx=8192)
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context(max_tokens=512, temperature=0.2))

        kwargs = mod.AsyncClient.return_value.chat.await_args.kwargs
        assert "top_p" not in kwargs["options"]
        assert "top_k" not in kwargs["options"]
        assert "min_p" not in kwargs["options"]

    @pytest.mark.asyncio
    async def test_generate_passes_keep_alive(self) -> None:
        provider, mod = _provider(keep_alive="10m")
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        kwargs = mod.AsyncClient.return_value.chat.await_args.kwargs
        assert kwargs["keep_alive"] == "10m"


class TestOllamaThinkParameter:
    @pytest.mark.asyncio
    async def test_think_config_true_passed(self) -> None:
        provider, mod = _provider(think=True)
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        assert mod.AsyncClient.return_value.chat.await_args.kwargs["think"] is True

    @pytest.mark.asyncio
    async def test_think_config_false_passed(self) -> None:
        provider, mod = _provider(think=False)
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        assert mod.AsyncClient.return_value.chat.await_args.kwargs["think"] is False

    @pytest.mark.asyncio
    async def test_think_config_none_omitted(self) -> None:
        provider, mod = _provider()  # think defaults to None
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        kwargs = mod.AsyncClient.return_value.chat.await_args.kwargs
        assert "think" not in kwargs

    @pytest.mark.asyncio
    async def test_thinking_budget_overrides_config_to_enable(self) -> None:
        # Config says don't think, but request asks for thinking budget > 0.
        provider, mod = _provider(think=False)
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context(thinking_budget=4096))

        assert mod.AsyncClient.return_value.chat.await_args.kwargs["think"] is True

    @pytest.mark.asyncio
    async def test_thinking_budget_zero_disables(self) -> None:
        # Config says think, but request explicitly budgets 0 tokens.
        provider, mod = _provider(think=True)
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context(thinking_budget=0))

        assert mod.AsyncClient.return_value.chat.await_args.kwargs["think"] is False

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    @pytest.mark.asyncio
    async def test_think_effort_string_passes_through(self, effort: str) -> None:
        provider, mod = _provider(think=effort)
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context())

        assert mod.AsyncClient.return_value.chat.await_args.kwargs["think"] == effort

    @pytest.mark.asyncio
    async def test_thinking_budget_positive_preserves_effort_string(self) -> None:
        # Config sets "high" effort; a positive budget at the channel layer
        # means "thinking on" — the configured level survives.
        provider, mod = _provider(think="high")
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context(thinking_budget=4096))

        assert mod.AsyncClient.return_value.chat.await_args.kwargs["think"] == "high"

    @pytest.mark.asyncio
    async def test_thinking_budget_zero_disables_even_with_effort_config(self) -> None:
        # Config sets "high" effort, but the channel explicitly says off.
        # Off wins — no thinking at all.
        provider, mod = _provider(think="high")
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        await provider.generate(_context(thinking_budget=0))

        assert mod.AsyncClient.return_value.chat.await_args.kwargs["think"] is False


class TestOllamaStreaming:
    @pytest.mark.asyncio
    async def test_streams_thinking_then_text(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _FakeStream(
            [
                _stream_chunk(thinking="Let me "),
                _stream_chunk(thinking="think."),
                _stream_chunk(content="42"),
                _stream_chunk(content=" is "),
                _stream_chunk(content="it."),
                _stream_chunk(
                    done=True,
                    done_reason="stop",
                    eval_count=5,
                    prompt_eval_count=12,
                ),
            ]
        )

        events = [e async for e in provider.generate_structured_stream(_context())]

        thinking_events = [e for e in events if isinstance(e, StreamThinkingDelta)]
        text_events = [e for e in events if isinstance(e, StreamTextDelta)]
        done_events = [e for e in events if isinstance(e, StreamDone)]

        assert [e.thinking for e in thinking_events] == ["Let me ", "think."]
        assert [e.text for e in text_events] == ["42", " is ", "it."]
        assert len(done_events) == 1
        assert done_events[0].finish_reason == "stop"
        assert done_events[0].usage == {"input_tokens": 12, "output_tokens": 5}

    @pytest.mark.asyncio
    async def test_streams_tool_calls_after_text(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _FakeStream(
            [
                _stream_chunk(thinking="Need a tool."),
                _stream_chunk(
                    tool_calls=[
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": {"q": "weather"},
                            }
                        }
                    ]
                ),
                _stream_chunk(done=True, done_reason="tool_calls"),
            ]
        )

        events = [e async for e in provider.generate_structured_stream(_context())]
        tool_events = [e for e in events if isinstance(e, StreamToolCall)]
        done_events = [e for e in events if isinstance(e, StreamDone)]

        assert len(tool_events) == 1
        assert tool_events[0].name == "lookup"
        assert tool_events[0].arguments == {"q": "weather"}
        assert done_events[0].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_streaming_passes_stream_true(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _FakeStream([_stream_chunk(done=True)])

        async for _ in provider.generate_structured_stream(_context()):
            pass

        assert mod.AsyncClient.return_value.chat.await_args.kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_generate_stream_yields_text_only(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _FakeStream(
            [
                _stream_chunk(thinking="Let me "),
                _stream_chunk(thinking="think."),
                _stream_chunk(content="42"),
                _stream_chunk(content=" is "),
                _stream_chunk(content="it."),
                _stream_chunk(done=True, done_reason="stop"),
            ]
        )

        chunks = [c async for c in provider.generate_stream(_context())]

        assert chunks == ["42", " is ", "it."]


class TestOllamaMessageBuilding:
    @pytest.mark.asyncio
    async def test_thinking_part_round_trip(self) -> None:
        """Prior assistant thinking should travel back to the model.

        Ollama supports a top-level ``thinking`` field on assistant
        messages, so we don't need to wrap it as <think> tags the way
        the OpenAI-compat shim does.
        """
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        prior = AIMessage(
            role="assistant",
            content=[
                AIThinkingPart(thinking="Earlier reasoning."),
                AIToolCallPart(id="x", name="t", arguments={}),
            ],
        )
        await provider.generate(_context(messages=[prior]))

        sent = mod.AsyncClient.return_value.chat.await_args.kwargs["messages"]
        # system + 1 assistant msg
        assistant_msg = sent[-1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["thinking"] == "Earlier reasoning."
        assert assistant_msg["tool_calls"] == [{"function": {"name": "t", "arguments": {}}}]

    @pytest.mark.asyncio
    async def test_tool_result_becomes_tool_role_message(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        tool_result_msg = AIMessage(
            role="tool",
            content=[AIToolResultPart(tool_call_id="call_1", name="lookup", result="42")],
        )
        await provider.generate(_context(messages=[tool_result_msg]))

        sent = mod.AsyncClient.return_value.chat.await_args.kwargs["messages"]
        tool_msg = sent[-1]
        assert tool_msg == {"role": "tool", "content": "42", "tool_name": "lookup"}

    @pytest.mark.asyncio
    async def test_data_uri_image_stripped_to_raw_base64(self) -> None:
        """A ``data:`` URI image must reach Ollama as raw base64.

        RoomKit carries images as ``data:<mime>;base64,<data>`` URIs, but
        Ollama's SDK rejects the full URI with "Invalid image data,
        expected base64 string or path to image file". The provider
        strips the prefix so the model receives a decodable payload.
        """
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
            "AAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        msg = AIMessage(
            role="user",
            content=[
                AITextPart(text="what is this?"),
                AIImagePart(url=f"data:image/png;base64,{b64}", mime_type="image/png"),
            ],
        )
        await provider.generate(_context(messages=[msg]))

        sent = mod.AsyncClient.return_value.chat.await_args.kwargs["messages"]
        user_msg = sent[-1]
        assert user_msg["content"] == "what is this?"
        assert user_msg["images"] == [b64]

    @pytest.mark.asyncio
    async def test_plain_base64_image_passes_through(self) -> None:
        """A bare base64 string (or path) is forwarded unchanged."""
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.return_value = _response_obj(content="ok")

        msg = AIMessage(
            role="user",
            content=[AIImagePart(url="/tmp/cat.png", mime_type="image/png")],
        )
        await provider.generate(_context(messages=[msg]))

        sent = mod.AsyncClient.return_value.chat.await_args.kwargs["messages"]
        assert sent[-1]["images"] == ["/tmp/cat.png"]


class TestOllamaErrors:
    @pytest.mark.asyncio
    async def test_response_error_retryable(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.side_effect = _FakeResponseError(
            "rate limit", status_code=429
        )

        with pytest.raises(ProviderError) as exc:
            await provider.generate(_context())

        assert exc.value.retryable is True
        assert exc.value.status_code == 429
        assert exc.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_response_error_non_retryable(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.side_effect = _FakeResponseError(
            "bad request", status_code=400
        )

        with pytest.raises(ProviderError) as exc:
            await provider.generate(_context())

        assert exc.value.retryable is False
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_stream_abort_without_status_retryable(self) -> None:
        """Ollama reports template parse failures of the model's own
        tool-call output (e.g. qwen emitting malformed XML) as a
        ResponseError with status -1. That's a transient generation defect
        — a retry regenerates with fresh sampling."""
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.side_effect = _FakeResponseError(
            "XML syntax error on line 7: element <parameter> closed by </function>",
            status_code=-1,
        )

        with pytest.raises(ProviderError) as exc:
            await provider.generate(_context())

        assert exc.value.retryable is True
        assert exc.value.status_code == -1

    @pytest.mark.asyncio
    async def test_connection_error_retryable(self) -> None:
        provider, mod = _provider()
        mod.AsyncClient.return_value.chat.side_effect = RuntimeError("conn refused")

        with pytest.raises(ProviderError) as exc:
            await provider.generate(_context())

        # Transport errors get marked retryable so RetryPolicy decides.
        assert exc.value.retryable is True


class TestOllamaAuth:
    def test_api_key_sets_bearer_header(self) -> None:
        _, mod = _provider(api_key="secret-token")
        headers = mod.AsyncClient.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret-token"

    def test_custom_headers_passed_through(self) -> None:
        _, mod = _provider(headers={"X-Proxy-Token": "abc"})
        headers = mod.AsyncClient.call_args.kwargs["headers"]
        assert headers == {"X-Proxy-Token": "abc"}

    def test_api_key_wins_over_headers_authorization(self) -> None:
        # The dedicated api_key field beats an Authorization smuggled in via
        # headers; unrelated custom headers are preserved alongside it.
        _, mod = _provider(
            api_key="real-key",
            headers={"Authorization": "Bearer stale", "X-Env": "prod"},
        )
        headers = mod.AsyncClient.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer real-key"
        assert headers["X-Env"] == "prod"

    def test_no_auth_passes_none_headers(self) -> None:
        # None lets the SDK keep its own OLLAMA_API_KEY env-var fallback.
        _, mod = _provider()
        assert mod.AsyncClient.call_args.kwargs["headers"] is None


class TestOllamaConfig:
    def test_defaults(self) -> None:
        config = OllamaConfig()
        assert config.host == "http://localhost:11434"
        assert config.model == "llama3.2"
        assert config.max_tokens is None
        assert config.temperature == 0.7
        assert config.timeout == 120.0
        assert config.think is None
        assert config.keep_alive is None
        assert config.num_ctx is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.min_p is None
        assert config.api_key is None
        assert config.headers is None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("-1", -1), ("0", 0), ("300", 300), (" -1 ", -1), (-1, -1), (5, 5)],
    )
    def test_numeric_keep_alive_coerced_to_int(self, raw: object, expected: int) -> None:
        # A unit-less integer must reach Ollama as an int — "-1"/"0" as a string
        # fail the server's Go-duration parser and are silently ignored.
        config = OllamaConfig(keep_alive=raw)  # type: ignore[arg-type]
        assert config.keep_alive == expected
        assert isinstance(config.keep_alive, int)

    @pytest.mark.parametrize("raw", ["5m", "30s", "1h", "1h30m"])
    def test_duration_string_keep_alive_passes_through(self, raw: str) -> None:
        config = OllamaConfig(keep_alive=raw)
        assert config.keep_alive == raw

    def test_api_key_is_secret(self) -> None:
        config = OllamaConfig(api_key="super-secret-xyz")
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == "super-secret-xyz"
        # SecretStr masks the value in repr so it never leaks into logs.
        assert "super-secret-xyz" not in repr(config)

    def test_overrides(self) -> None:
        config = OllamaConfig(
            host="http://example:11434",
            model="qwen3:8b",
            think=True,
            num_ctx=4096,
            top_p=0.95,
            top_k=40,
            min_p=0.2,
            keep_alive="5m",
        )
        assert config.host == "http://example:11434"
        assert config.model == "qwen3:8b"
        assert config.think is True
        assert config.num_ctx == 4096
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.min_p == 0.2
        assert config.keep_alive == "5m"


class TestOllamaLazyImport:
    def test_import_error_message(self) -> None:
        # Force the import inside __init__ to fail by patching sys.modules.
        with patch.dict("sys.modules", {"ollama": None}):
            # Need to reimport to trigger the lazy import inside __init__.
            import importlib

            import roomkit.providers.ollama.ai as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match="pip install roomkit\\[ollama\\]"):
                mod.OllamaAIProvider(_config())


class TestOllamaListModels:
    @pytest.mark.asyncio
    async def test_list_models_attaches_capabilities(self) -> None:
        """list_models reads /api/tags then probes /api/show for capabilities."""
        provider, mod = _provider()
        client = mod.AsyncClient.return_value
        client.list = AsyncMock(
            return_value=SimpleNamespace(
                models=[
                    SimpleNamespace(model="qwen3:8b"),
                    SimpleNamespace(model="nomic-embed-text"),
                ]
            )
        )
        caps = {"qwen3:8b": ["completion", "tools"], "nomic-embed-text": ["embedding"]}

        async def _show(model: str) -> SimpleNamespace:
            return SimpleNamespace(capabilities=caps[model])

        client.show = AsyncMock(side_effect=_show)

        models = await provider.list_models()

        by_id = {m.id: m for m in models}
        assert by_id["qwen3:8b"].capabilities == ["completion", "tools"]
        assert by_id["nomic-embed-text"].capabilities == ["embedding"]

    @pytest.mark.asyncio
    async def test_list_models_tolerates_show_failure(self) -> None:
        """A failing /api/show probe yields empty capabilities, not a crash."""
        provider, mod = _provider()
        client = mod.AsyncClient.return_value
        client.list = AsyncMock(return_value=SimpleNamespace(models=[SimpleNamespace(model="m1")]))
        client.show = AsyncMock(side_effect=RuntimeError("boom"))

        models = await provider.list_models()

        assert models[0].id == "m1"
        assert models[0].capabilities == []
