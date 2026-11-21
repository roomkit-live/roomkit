"""``transport=``: every request of an OpenAI-family provider goes through it.

The seam a host puts an outbound policy in (one that judges the address
actually dialled), and a test a ``MockTransport``. Honoured by
``OpenAIAIProvider`` and every provider built on it, those that build their
own SDK client included (Azure, OpenRouter) and ``create_vllm_provider``; and
by the other clients built on the ``openai`` SDK, whose endpoint a host's
users name too: the OpenAI, Azure, xAI and Meta image providers and
``OpenAIVisionProvider``; and by the requests beside a completion, such as
LiteLLM's and OpenRouter's model listings. The SDK runs on httpx2 (openai
3.x), so these take an httpx2 transport and refuse an httpx one.
``OpenRouterImageProvider``, on httpx alone, takes the same keyword with an
httpx transport (RMK-453).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

import httpx
import httpx2
import openai
import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, ProviderError
from roomkit.providers.azure.ai import AzureAIProvider
from roomkit.providers.azure.config import AzureAIConfig, AzureImageConfig
from roomkit.providers.azure.image import AzureImageProvider
from roomkit.providers.deepseek import DeepSeekAIProvider, DeepSeekConfig
from roomkit.providers.image.base import ImageGenerationError
from roomkit.providers.litellm import LiteLLMAIProvider, LiteLLMConfig
from roomkit.providers.meta.config import MetaImageConfig
from roomkit.providers.meta.image import MetaImageProvider
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig, OpenAIImageConfig
from roomkit.providers.openai.image import OpenAIImageProvider
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig, OpenRouterImageConfig
from roomkit.providers.openrouter.image import OpenRouterImageProvider
from roomkit.providers.url_safety import validate_public_url
from roomkit.providers.vllm import create_vllm_provider
from roomkit.providers.vllm.config import VLLMConfig
from roomkit.providers.xai.config import XAIImageConfig
from roomkit.providers.xai.image import XAIImageProvider
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.openai import OpenAIVisionConfig, OpenAIVisionProvider

_COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

Factory = Callable[[httpx2.AsyncBaseTransport], Any]

_PROVIDERS: dict[str, Factory] = {
    "openai": lambda t: OpenAIAIProvider(
        OpenAIConfig(api_key="k", model="gpt-x", base_url="https://llm.example/v1"), transport=t
    ),
    "azure": lambda t: AzureAIProvider(
        AzureAIConfig(api_key="k", azure_endpoint="https://az.example", model="dep"),
        transport=t,
    ),
    "openrouter": lambda t: OpenRouterAIProvider(
        OpenRouterConfig(api_key="k", model="vendor/model"), transport=t
    ),
    "vllm": lambda t: create_vllm_provider(
        VLLMConfig(model="m", base_url="https://vllm.example/v1"), transport=t
    ),
    "inherited (deepseek)": lambda t: DeepSeekAIProvider(
        DeepSeekConfig(api_key="k", model="deepseek-chat"), transport=t
    ),
}


@pytest.mark.parametrize("kind", list(_PROVIDERS))
async def test_every_request_goes_through_the_transport(kind: str) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=_COMPLETION)

    provider = _PROVIDERS[kind](httpx2.MockTransport(handler))
    try:
        response = await provider.generate(
            AIContext(messages=[AIMessage(role="user", content="hello")])
        )
    finally:
        await provider.close()

    assert response.content == "ok"
    assert len(seen) == 1
    assert seen[0].url.path.endswith("/chat/completions")


async def test_the_sdk_defaults_stay_redirects_followed_through_the_transport() -> None:
    """The transport sits inside the SDK's own default client: redirects are
    still followed, each hop through the transport, where a policy judges it."""
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            return httpx2.Response(307, headers={"location": "https://llm.example/v2/chat"})
        return httpx2.Response(200, json=_COMPLETION)

    provider = _PROVIDERS["openai"](httpx2.MockTransport(handler))
    try:
        response = await provider.generate(
            AIContext(messages=[AIMessage(role="user", content="hello")])
        )
    finally:
        await provider.close()

    assert response.content == "ok"
    assert seen == ["/v1/chat/completions", "/v2/chat"]


_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
_IMAGES = {"created": 0, "data": [{"b64_json": _PNG_B64}]}


def _answer(http: Any, request: Any) -> Any:
    """The endpoint's answer, as a response of the *http* package: an image
    to an images call, an empty model listing to a listing, a completion
    otherwise."""
    if request.url.path.endswith(("/images/generations", "/images")):
        return http.Response(200, json=_IMAGES)
    if request.url.path.endswith(("/model/info", "/models")):
        return http.Response(200, json={"data": []})
    return http.Response(200, json=_COMPLETION)


class _Client(NamedTuple):
    """A client that takes ``transport=``, and one call it makes beside a completion."""

    build: Callable[..., Any]
    """``(origin, transport, **config)``: the client for an endpoint at this
    origin, over this transport, its config's other fields overridden."""
    call: Callable[[Any], Awaitable[object]]
    http: Any = httpx2
    """The package its HTTP client is of, whose transports it takes: httpx2
    under the openai SDK."""


def _draw(provider: Any) -> Awaitable[object]:
    return provider.generate("an origami fox")


def _list(provider: Any) -> Awaitable[object]:
    return provider.list_models()


def _look(provider: Any) -> Awaitable[object]:
    return provider.analyze_frame(
        VideoFrame(data=b"\x00\x00\x01", codec="h264", width=2, height=2)
    )


_CALLS: dict[str, _Client] = {
    "openai image": _Client(
        lambda origin, t, **c: OpenAIImageProvider(
            OpenAIImageConfig(api_key="k", model="gpt-image-2", base_url=f"{origin}/v1", **c),
            transport=t,
        ),
        _draw,
    ),
    "azure image": _Client(
        lambda origin, t, **c: AzureImageProvider(
            AzureImageConfig(api_key="k", azure_endpoint=origin, model="dep", **c), transport=t
        ),
        _draw,
    ),
    "xai image": _Client(
        lambda origin, t, **c: XAIImageProvider(
            XAIImageConfig(api_key="k", base_url=f"{origin}/v1", **c), transport=t
        ),
        _draw,
    ),
    "meta image": _Client(
        lambda origin, t, **c: MetaImageProvider(
            MetaImageConfig(api_key="k", base_url=f"{origin}/v1", **c), transport=t
        ),
        _draw,
    ),
    "openai vision": _Client(
        lambda origin, t, **c: OpenAIVisionProvider(
            OpenAIVisionConfig(base_url=f"{origin}/v1", **c), transport=t
        ),
        _look,
    ),
    "litellm models": _Client(
        lambda origin, t, **c: LiteLLMAIProvider(
            LiteLLMConfig(api_key="k", model="m", base_url=origin, **c), transport=t
        ),
        _list,
    ),
    "openrouter image": _Client(
        lambda origin, t, **c: OpenRouterImageProvider(
            OpenRouterImageConfig(
                api_key="k", model="vendor/painter", base_url=f"{origin}/api/v1", **c
            ),
            transport=t,
        ),
        _draw,
        http=httpx,
    ),
    "openrouter models": _Client(
        lambda origin, t, **c: OpenRouterAIProvider(
            OpenRouterConfig(api_key="k", model="vendor/model", base_url=f"{origin}/api/v1", **c),
            transport=t,
        ),
        _list,
    ),
}


def _recording(http: Any, seen: list[Any]) -> Any:
    """A mock transport of the *http* package that answers, and keeps each request."""

    def handler(request: Any) -> Any:
        seen.append(request)
        return _answer(http, request)

    return http.MockTransport(handler)


def _public_only(http: Any, inner: Any) -> Any:
    """A host's policy, as a transport of the *http* package: refuses a
    private or reserved address, then hands on to *inner*.

    The refusal is raised as a connect failure, which the clients read as a
    request that never left."""

    class PublicOnly(http.AsyncBaseTransport):
        async def handle_async_request(self, request: Any) -> Any:
            try:
                validate_public_url(str(request.url), resolve_dns=False)
            except ValueError as refusal:
                raise http.ConnectError(str(refusal), request=request) from refusal
            return await inner.handle_async_request(request)

    return PublicOnly()


class _Scripted(httpx2.AsyncBaseTransport):
    """Fails each request with the next step of *script*, a transport error
    raised or an error status answered, then answers."""

    def __init__(self, *script: type[httpx2.TransportError] | int) -> None:
        self._script = list(script)
        self.requests = 0

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests += 1
        if not self._script:
            return _answer(httpx2, request)
        step = self._script.pop(0)
        if isinstance(step, int):
            return httpx2.Response(step, json={"error": {"message": "scripted"}})
        raise step("scripted", request=request)


def _refusing() -> httpx2.AsyncBaseTransport:
    return _public_only(httpx2, _recording(httpx2, []))


def _timing_out() -> httpx2.AsyncBaseTransport:
    return _Scripted(httpx2.ReadTimeout)


@pytest.mark.parametrize("kind", list(_CALLS))
async def test_each_call_sends_its_request_through_the_transport(kind: str) -> None:
    seen: list[Any] = []
    client = _CALLS[kind]
    provider = client.build("https://models.example", _recording(client.http, seen))
    try:
        await client.call(provider)
    finally:
        await provider.close()

    assert [request.url.host for request in seen] == ["models.example"]


@pytest.mark.parametrize("kind", list(_CALLS))
async def test_a_policy_in_the_transport_refuses_a_private_endpoint_before_any_request(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The vision client keeps the SDK's default retries, each one refused by
    # the policy too; none waits here.
    monkeypatch.setattr("openai._base_client.INITIAL_RETRY_DELAY", 0.0)
    sent: list[Any] = []
    client = _CALLS[kind]
    policy = _public_only(client.http, _recording(client.http, sent))
    provider = client.build("https://169.254.169.254", policy)
    try:
        with pytest.raises((ProviderError, openai.APIConnectionError)) as caught:
            await client.call(provider)
    finally:
        await provider.close()

    assert sent == []
    cause: BaseException = caught.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, ValueError)
    assert "169.254.169.254" in str(cause)


@pytest.mark.parametrize(
    ("kind", "transport", "status"),
    [
        ("openai image", _refusing, "failed"),
        ("azure image", _refusing, "failed"),
        ("openai image", _timing_out, "unknown"),
        ("azure image", _timing_out, "unknown"),
    ],
    ids=["openai-refused", "azure-refused", "openai-timed-out", "azure-timed-out"],
)
async def test_an_image_attempt_is_unknown_once_its_request_may_have_left(
    kind: str, transport: Callable[[], httpx2.AsyncBaseTransport], status: str
) -> None:
    """A request the policy refused reached nothing, so nothing ran or was
    billed: ``failed``. One that timed out may have run: ``unknown``
    (RFC §25.2)."""
    client = _CALLS[kind]
    # The scripted transports answer whatever the address; the policy refuses this one.
    provider = client.build("https://169.254.169.254", transport())
    try:
        with pytest.raises(ImageGenerationError) as caught:
            await client.call(provider)
    finally:
        await provider.close()

    [attempt] = caught.value.attempts
    assert attempt.status == status


@pytest.mark.parametrize("then", [200, 503], ids=["then-200", "then-503"])
@pytest.mark.parametrize("kind", ["openai image", "azure image", "xai image", "meta image"])
async def test_a_generation_that_may_have_run_is_never_sent_again(
    kind: str, then: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paid generation is never retried automatically (RFC §25.2), and no
    config asks the SDK to, whose own default would: the first request timed
    out, and may have run and been billed. It is sent once, and its error is
    ambiguous: the attempt ``unknown``, the error not retryable."""
    monkeypatch.setattr("openai._base_client.INITIAL_RETRY_DELAY", 0.0)
    transport = _Scripted(httpx2.ReadTimeout, then)
    client = _CALLS[kind]
    provider = client.build("https://models.example", transport)
    try:
        with pytest.raises(ProviderError) as caught:
            await client.call(provider)
    finally:
        await provider.close()

    assert transport.requests == 1
    assert caught.value.retryable is False
    if isinstance(caught.value, ImageGenerationError):
        assert [attempt.status for attempt in caught.value.attempts] == ["unknown"]


@pytest.mark.parametrize("kind", ["litellm models", "openrouter models"])
async def test_a_model_listing_that_is_not_json_raises(kind: str) -> None:
    """A 200 carrying a login page is no listing: an error, not an empty one."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, text="<html>Sign in</html>", headers={"content-type": "text/html"}
        )

    client = _CALLS[kind]
    provider = client.build("https://models.example", httpx2.MockTransport(handler))
    try:
        with pytest.raises(json.JSONDecodeError):
            await client.call(provider)
    finally:
        await provider.close()


@pytest.mark.parametrize("kind", ["openai image", "azure image", "xai image", "meta image"])
async def test_an_image_status_error_says_what_the_server_answered(kind: str) -> None:
    """The openai SDK's own text holds ``error.message`` alone (3.57.0, where
    3.0.0 wrote the whole body): the image providers' error carries the body,
    so what a server words elsewhere in it can be read."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(402, json={"detail": "Image credit exhausted"})

    client = _CALLS[kind]
    provider = client.build("https://models.example", httpx2.MockTransport(handler))
    try:
        with pytest.raises(ProviderError, match="Image credit exhausted") as caught:
            await client.call(provider)
    finally:
        await provider.close()

    if isinstance(caught.value, ImageGenerationError):
        assert "Image credit exhausted" in (caught.value.attempts[0].error or "")


_ON_THE_SDK: dict[str, Callable[[Any], Any]] = {
    **_PROVIDERS,
    **{
        kind: (lambda t, client=client: client.build("https://models.example", t))
        for kind, client in _CALLS.items()
        if client.http is httpx2
    },
}


@pytest.mark.parametrize("kind", list(_ON_THE_SDK))
def test_an_httpx_transport_is_refused_when_the_client_is_built(kind: str) -> None:
    """The SDK's client is an httpx2 one. Given an httpx transport it lets the
    request out and fails on the response with a bare ``AssertionError``, so
    the transport is refused before any request, by a message naming httpx2."""
    with pytest.raises(TypeError, match="httpx2.AsyncBaseTransport"):
        _ON_THE_SDK[kind](_recording(httpx, []))


@pytest.mark.parametrize("kind", list(_ON_THE_SDK))
async def test_an_openai_sdk_older_than_3_is_refused_with_its_version(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RoomKit installed without its extra can sit over openai 2.x, which runs
    on httpx: there a client built, and its requests through an httpx2
    transport failed as a lost connection. It is refused where the SDK is
    imported, by a message naming the installed version and the extra."""
    monkeypatch.setattr(openai, "__version__", "2.54.0")

    with pytest.raises(ImportError, match=r"openai 2\.54\.0 is installed.*'roomkit\[\w+\]'"):
        provider = _ON_THE_SDK[kind](_recording(httpx2, []))
        if kind in _CALLS:  # the vision provider imports the SDK on first use
            await _CALLS[kind].call(provider)


async def test_a_transport_of_both_packages_is_taken() -> None:
    """One class serving both clients is an httpx2 transport too: not refused."""

    class Both(httpx.AsyncBaseTransport, httpx2.AsyncBaseTransport):
        async def handle_async_request(self, request: Any) -> Any:
            return httpx2.Response(200, json=_COMPLETION)

    provider = _PROVIDERS["openai"](Both())
    try:
        response = await provider.generate(
            AIContext(messages=[AIMessage(role="user", content="hello")])
        )
    finally:
        await provider.close()

    assert response.content == "ok"
