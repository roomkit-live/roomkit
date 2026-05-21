"""Tests for the PolarGrid AI provider."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AITool,
    ProviderError,
    StreamDone,
    StreamTextDelta,
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
    async def test_generate_warns_on_tools_and_drops_them(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, mod = _provider()
        mod._client.chat_completion.return_value = _response_obj(content="ok")

        with caplog.at_level(logging.WARNING, logger="roomkit.providers.polargrid"):
            await provider.generate(_context(tools=[AITool(name="t", description="x")]))

        assert any("does not support tool" in r.message for r in caplog.records)
        request = mod._client.chat_completion.await_args.args[0]
        assert "tools" not in request

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
