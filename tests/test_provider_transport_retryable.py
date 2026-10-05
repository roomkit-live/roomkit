"""A transport failure is retryable on every text provider, and a client
error is not (RMK-509).

The channel retries a provider error only when it is ``retryable``
(``RetryPolicy``). A connection refused before any status is the same failure
whichever SDK surfaces it, so each provider says it is worth retrying, through
``generate()`` and through the stream; a 400 stays final everywhere. Each
provider is driven as its SDK surfaces the failure: a client over an httpx
transport that refuses the connection where the SDK takes one (the OpenAI
wire, Mistral, Ollama), and the SDK's own error otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Literal

import anthropic
import httpx
import httpx2
import ollama
import polargrid
import pytest
from google.genai import errors as genai_errors
from mistralai.client import Mistral

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
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.polargrid.config import PolarGridConfig

Failure = Literal["refused", "client_error"]

_CONTEXT = AIContext(messages=[AIMessage(role="user", content="go")])
_REFUSED = "[Errno 111] Connection refused"


def _transport(failure: Failure) -> httpx.MockTransport:
    def answer(request: httpx.Request) -> httpx.Response:
        if failure == "refused":
            raise httpx.ConnectError(_REFUSED, request=request)
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    return httpx.MockTransport(answer)


def _openai(failure: Failure) -> AIProvider:
    config = OpenAIConfig(api_key="k", model="gpt-4.1", max_retries=0)
    return OpenAIAIProvider(config, transport=_transport(failure))


def _mistral(failure: Failure) -> AIProvider:
    provider = MistralAIProvider(MistralConfig(api_key="k", model="mistral-large-latest"))
    http = httpx.AsyncClient(transport=_transport(failure))
    provider._client = Mistral(api_key="k", async_client=http)
    return provider


def _ollama(failure: Failure) -> AIProvider:
    provider = OllamaAIProvider(OllamaConfig(model="qwen3:8b"))
    provider._client = ollama.AsyncClient(host="http://ollama.test", transport=_transport(failure))
    return provider


def _anthropic(failure: Failure) -> AIProvider:
    """anthropic 1.x takes no httpx transport: its client raises its own
    errors, a connection error raised from httpx2's."""
    provider = AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-sonnet-5-5"))
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def stream(**kwargs: Any) -> Any:
        if failure == "refused":
            raise anthropic.APIConnectionError(request=request) from httpx2.ConnectError(
                _REFUSED, request=request
            )
        response = httpx2.Response(400, request=request)
        raise anthropic.BadRequestError("bad request", response=response, body=None)

    provider._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    return provider


def _gemini(failure: Failure) -> AIProvider:
    """google-genai re-raises the httpx error after its own retries."""
    provider = GeminiAIProvider(GeminiConfig(api_key="k"))

    async def generate_content_stream(**kwargs: Any) -> Any:
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
        if failure == "refused":
            return polargrid.NetworkError(_REFUSED, None, None)
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


@pytest.mark.parametrize("mode", ["generate", "stream"])
@pytest.mark.parametrize("name", list(_PROVIDERS))
async def test_a_refused_connection_is_retryable(name: str, mode: str) -> None:
    error = await _error(_PROVIDERS[name]("refused"), mode)

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
