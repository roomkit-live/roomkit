"""A transport failure is retryable on every text provider, and a client
error is not (RMK-509).

The channel retries a provider error only when it is ``retryable``
(``RetryPolicy``). A connection refused before any status, or one the server
answered then dropped before the first event, is the same failure whichever
SDK surfaces it, so each provider says it is worth retrying, through
``generate()`` and through the stream; a 400 stays final everywhere. Each
provider is driven as its SDK surfaces the failure: a client over a mock
transport of the HTTP client the SDK runs on, where it takes one (httpx for the
OpenAI wire and Ollama, httpx2 for Mistral), and the SDK's own errors otherwise.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

import anthropic
import httpx
import httpx2
import ollama
import polargrid
import pytest
from google.genai import errors as genai_errors

from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIProvider,
    ProviderError,
    is_transport_failure,
)
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.polargrid.config import PolarGridConfig
from tests.text_conformance.mistral_wire import mistral_over

Failure = Literal["refused", "dropped", "client_error"]

_CONTEXT = AIContext(messages=[AIMessage(role="user", content="go")])
_REFUSED = "[Errno 111] Connection refused"


async def _dropped(client: ModuleType) -> AsyncIterator[bytes]:
    """A body the server stops sending before its first byte."""
    raise client.ReadTimeout("timed out")
    yield b""  # pragma: no cover


def _transport(
    failure: Failure, client: ModuleType = httpx
) -> httpx.MockTransport | httpx2.MockTransport:
    """A mock transport of the HTTP client *client* (httpx or httpx2) that
    answers *failure*."""

    def answer(request: Any) -> Any:
        if failure == "refused":
            raise client.ConnectError(_REFUSED, request=request)
        if failure == "dropped":
            headers = {"content-type": "text/event-stream"}
            return client.Response(200, headers=headers, content=_dropped(client))
        return client.Response(400, json={"error": {"message": "bad request"}})

    return client.MockTransport(answer)


def _openai(failure: Failure) -> AIProvider:
    config = OpenAIConfig(api_key="k", model="gpt-4.1", max_retries=0)
    return OpenAIAIProvider(config, transport=_transport(failure))


def _mistral(failure: Failure) -> AIProvider:
    """mistralai 3.x runs on httpx2 and lets its transport errors through."""
    return mistral_over(_transport(failure, httpx2))


def _ollama(failure: Failure) -> AIProvider:
    provider = OllamaAIProvider(OllamaConfig(model="qwen3:8b"))
    provider._client = ollama.AsyncClient(host="http://ollama.test", transport=_transport(failure))
    return provider


class _DroppedAnthropicStream:
    """A message stream the server drops before its first event: the SDK
    lets the httpx2 error through as it is."""

    async def __aenter__(self) -> _DroppedAnthropicStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        raise httpx2.RemoteProtocolError("peer closed connection")


def _anthropic(failure: Failure) -> AIProvider:
    """anthropic 1.x runs on httpx2: its client raises its own errors, a
    connection error raised from httpx2's, and lets an error of the stream's
    read through."""
    provider = AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-sonnet-5-5"))
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def stream(**kwargs: Any) -> Any:
        if failure == "dropped":
            return _DroppedAnthropicStream()
        if failure == "refused":
            raise anthropic.APIConnectionError(request=request) from httpx2.ConnectError(
                _REFUSED, request=request
            )
        response = httpx2.Response(400, request=request)
        raise anthropic.BadRequestError("bad request", response=response, body=None)

    provider._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    return provider


async def _dropped_gemini_stream() -> Any:
    raise httpx.ReadTimeout("timed out")
    yield  # pragma: no cover


def _gemini(failure: Failure) -> AIProvider:
    """google-genai lets the httpx error through (RoomKit's client sets no
    retries of its own)."""
    provider = GeminiAIProvider(GeminiConfig(api_key="k"))

    async def generate_content_stream(**kwargs: Any) -> Any:
        if failure == "dropped":
            return _dropped_gemini_stream()
        if failure == "refused":
            raise httpx.ConnectError(_REFUSED)
        raise genai_errors.ClientError(400, {"error": {"message": "bad request"}})

    models = SimpleNamespace(generate_content_stream=generate_content_stream)
    provider._client = SimpleNamespace(aio=SimpleNamespace(models=models))  # type: ignore[assignment]
    return provider


def _polargrid(failure: Failure) -> AIProvider:
    """The PolarGrid client raises its typed errors."""
    provider = PolarGridAIProvider(PolarGridConfig(api_key="k", model="qwen-3.8-27b"))
    client = polargrid.PolarGrid(api_key="k", base_url="http://127.0.0.1:1")

    def error() -> Exception:
        if failure in ("refused", "dropped"):
            # As the SDK wraps a transport failure: the httpx error under it.
            return polargrid.NetworkError(_REFUSED, httpx.ConnectError(_REFUSED), None)
        return polargrid.ValidationError("bad request", None, "req-1")

    async def make_request(*args: Any, **kwargs: Any) -> Any:
        raise error()

    async def stream_post(*args: Any, **kwargs: Any) -> Any:
        raise error()
        yield  # pragma: no cover

    client._make_request = make_request  # type: ignore[method-assign]
    client._stream_post = stream_post  # type: ignore[method-assign]
    provider._client = client
    return provider


_PROVIDERS: dict[str, Callable[[Failure], AIProvider]] = {
    "openai": _openai,
    "anthropic": _anthropic,
    "mistral": _mistral,
    "gemini": _gemini,
    "ollama": _ollama,
    "polargrid": _polargrid,
}


async def _error(provider: AIProvider, mode: str) -> ProviderError:
    with pytest.raises(ProviderError) as raised:
        if mode == "generate":
            await provider.generate(_CONTEXT)
        else:
            async for _ in provider.generate_structured_stream(_CONTEXT):
                pass
    return raised.value


@pytest.mark.parametrize("failure", ["refused", "dropped"])
@pytest.mark.parametrize("mode", ["generate", "stream"])
@pytest.mark.parametrize("name", list(_PROVIDERS))
async def test_a_transport_failure_is_retryable(name: str, mode: str, failure: Failure) -> None:
    """Refused before any status, or dropped after it before the first
    event: both are worth another try, on every provider and both modes."""
    error = await _error(_PROVIDERS[name](failure), mode)

    assert error.retryable is True
    assert error.status_code is None


@pytest.mark.parametrize("mode", ["generate", "stream"])
@pytest.mark.parametrize("name", list(_PROVIDERS))
async def test_a_client_error_stays_final(name: str, mode: str) -> None:
    error = await _error(_PROVIDERS[name]("client_error"), mode)

    assert error.retryable is False
    assert error.status_code == 400


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError(_REFUSED),
        httpx.ReadTimeout("timed out"),
        httpx2.RemoteProtocolError("peer closed connection"),
        ConnectionResetError(104, "Connection reset by peer"),
        TimeoutError(),
    ],
    ids=["httpx-connect", "httpx-timeout", "httpx2-protocol", "socket-reset", "timeout"],
)
def test_a_transport_failure_is_recognised_through_its_cause(exc: Exception) -> None:
    wrapped = RuntimeError("Connection error.")
    wrapped.__cause__ = exc

    assert is_transport_failure(exc) is True
    assert is_transport_failure(wrapped) is True


def test_an_error_that_is_not_a_transport_failure_is_not_one() -> None:
    context_only = ValueError("bad arguments")
    context_only.__context__ = httpx.ConnectError(_REFUSED)

    assert is_transport_failure(ValueError("bad arguments")) is False
    # Only what an error was raised *from* is its cause: one merely raised
    # while handling a transport failure is not that failure.
    assert is_transport_failure(context_only) is False
