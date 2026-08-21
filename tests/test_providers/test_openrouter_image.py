"""Tests for the OpenRouter image provider (RFC §25).

Offline: the HTTP client is a stub, so request building — the pixels-through
``size``, the ``input_references`` blocks, the per-request fan-out for ``n`` —
response mapping and error translation are exercised without a key or a
network. ``httpx`` itself is imported for its exception types only.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from roomkit.providers.ai.base import AIImagePart, ProviderError
from roomkit.providers.image import to_data_uri
from roomkit.providers.openrouter.config import OpenRouterImageConfig
from roomkit.providers.openrouter.image import OpenRouterImageProvider

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG).decode("ascii")
JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 20).decode("ascii")


def _provider(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {"api_key": "or-test", "model": "google/gemini-3.1-flash-image"}
    defaults.update(overrides)
    with patch("httpx.AsyncClient"):
        return OpenRouterImageProvider(OpenRouterImageConfig(**defaults))


def _payload(
    b64: str | None = PNG_B64,
    *,
    media_type: str | None = "image/png",
    usage: dict[str, Any] | None = None,
    count: int = 1,
    revised_prompt: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    if b64 is not None:
        entry["b64_json"] = b64
    if media_type is not None:
        entry["media_type"] = media_type
    if revised_prompt is not None:
        entry["revised_prompt"] = revised_prompt
    payload: dict[str, Any] = {"created": 1755750000, "data": [dict(entry) for _ in range(count)]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def _respond(provider: Any, payload: dict[str, Any]) -> AsyncMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=payload)
    provider._http.post = AsyncMock(return_value=response)
    return provider._http.post


async def test_generate_posts_one_image_request() -> None:
    provider = _provider()
    post = _respond(provider, _payload())

    [result] = await provider.generate("an origami fox")

    assert result.mime_type == "image/png"
    assert result.decoded() == PNG
    args, kwargs = post.await_args
    assert args == ("/images",)
    assert kwargs["json"] == {"model": "google/gemini-3.1-flash-image", "prompt": "an origami fox"}


async def test_size_goes_through_as_pixels() -> None:
    provider = _provider(quality="high", output_format="webp", background="transparent")
    post = _respond(provider, _payload(media_type="image/webp"))

    [result] = await provider.generate("a fox", size="1664X936")

    body = post.await_args.kwargs["json"]
    assert body["size"] == "1664x936"
    assert body["quality"] == "high"
    assert body["output_format"] == "webp"
    assert body["background"] == "transparent"
    assert result.mime_type == "image/webp"


async def test_a_malformed_size_is_refused_locally() -> None:
    provider = _provider()
    post = _respond(provider, _payload())

    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        await provider.generate("a fox", size="big")
    post.assert_not_awaited()


async def test_n_fans_out_into_n_billed_requests() -> None:
    """Per-model batch caps vary from 1 to 10, so singles are the one universal form."""
    provider = _provider()
    post = _respond(provider, _payload(usage={"completion_tokens": 100, "cost": 0.01}))

    results = await provider.generate("three foxes", n=3)

    assert len(results) == 3
    assert post.await_count == 3
    assert all(r.usage == {"output_image_tokens": 100, "cost": 0.01} for r in results)


async def test_generate_rejects_n_below_one() -> None:
    provider = _provider()
    with pytest.raises(ValueError, match="between 1 and 10"):
        await provider.generate("a fox", n=0)


async def test_generate_rejects_n_above_the_batch_cap() -> None:
    """Every unit of ``n`` is an immediately spawned billed request, so the
    fan-out is refused past the lineup's largest batch cap — before the first
    call is made, not after ten of eleven succeeded."""
    provider = _provider()
    post = _respond(provider, _payload())

    with pytest.raises(ValueError, match="between 1 and 10"):
        await provider.generate("a fox", n=11)
    post.assert_not_awaited()


async def test_more_than_one_image_per_request_is_an_error() -> None:
    provider = _provider()
    _respond(provider, _payload(count=2))

    with pytest.raises(ProviderError, match="2 image"):
        await provider.generate("a fox")


async def test_a_result_without_inline_bytes_is_an_error() -> None:
    provider = _provider()
    _respond(provider, _payload(b64=None))

    with pytest.raises(ProviderError, match="without inline bytes"):
        await provider.generate("a fox")


async def test_an_omitted_media_type_is_sniffed_from_the_bytes() -> None:
    provider = _provider()
    _respond(provider, _payload(b64=JPEG_B64, media_type=None))

    [result] = await provider.generate("a fox")

    assert result.mime_type == "image/jpeg"


async def test_revised_prompt_is_carried_when_reported() -> None:
    provider = _provider()
    _respond(provider, _payload(revised_prompt="a fox folded from red paper"))

    [result] = await provider.generate("a fox")

    assert result.revised_prompt == "a fox folded from red paper"


def test_attribution_and_override_headers_are_sent() -> None:
    config = OpenRouterImageConfig(
        api_key="or-test",
        model="openai/gpt-image-2",
        site_url="https://example.app",
        app_name="Example",
        default_headers={"X-Extra": "1"},
    )
    headers = OpenRouterImageProvider._headers(config)
    assert headers["Authorization"] == "Bearer or-test"
    assert headers["HTTP-Referer"] == "https://example.app"
    assert headers["X-Title"] == "Example"
    assert headers["X-Extra"] == "1"


# --- Editing -------------------------------------------------------------------


async def test_references_ride_the_same_request() -> None:
    provider = _provider()
    post = _respond(provider, _payload())
    inline = AIImagePart(url=to_data_uri(PNG, "image/png"))
    remote = AIImagePart(url="https://example.com/fox.png")

    await provider.generate("blend them", reference_images=[inline, remote])

    body = post.await_args.kwargs["json"]
    assert body["input_references"] == [
        {"type": "image_url", "image_url": {"url": to_data_uri(PNG, "image/png")}},
        {"type": "image_url", "image_url": {"url": "https://example.com/fox.png"}},
    ]


async def test_a_corrupt_reference_is_the_callers_error() -> None:
    provider = _provider()
    post = _respond(provider, _payload())

    with pytest.raises(ValueError, match="reference image 0"):
        await provider.generate(
            "make it blue",
            reference_images=[AIImagePart(url="data:image/png;base64,!!!")],
        )
    post.assert_not_awaited()


# --- Usage ---------------------------------------------------------------------


async def test_usage_maps_counters_and_carries_the_bill() -> None:
    provider = _provider()
    _respond(
        provider,
        _payload(usage={"prompt_tokens": 12, "completion_tokens": 4175, "cost": 0.04}),
    )

    [result] = await provider.generate("a fox")

    assert result.usage == {"input_tokens": 12, "output_image_tokens": 4175, "cost": 0.04}


async def test_usage_is_empty_when_the_vendor_reports_none() -> None:
    provider = _provider()
    _respond(provider, _payload())

    [result] = await provider.generate("a fox")

    assert result.usage == {}


# --- Errors --------------------------------------------------------------------


def _status_response(provider: Any, status: int, text: str = "err") -> None:
    response = MagicMock()
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "boom",
            request=MagicMock(),
            response=SimpleNamespace(status_code=status, text=text),
        )
    )
    provider._http.post = AsyncMock(return_value=response)


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (400, False)])
async def test_status_errors_map_to_retryability(status: int, retryable: bool) -> None:
    provider = _provider()
    _status_response(provider, status, "the model refused")

    with pytest.raises(ProviderError, match="the model refused") as exc_info:
        await provider.generate("a fox")
    assert exc_info.value.retryable is retryable
    assert exc_info.value.status_code == status
    assert exc_info.value.provider == "openrouter"


async def test_a_transport_failure_is_retryable() -> None:
    provider = _provider()
    provider._http.post = AsyncMock(side_effect=httpx.ConnectError("down"))

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("a fox")
    assert exc_info.value.retryable is True


async def test_an_unexpected_error_is_wrapped_not_retryable() -> None:
    provider = _provider()
    provider._http.post = AsyncMock(side_effect=RuntimeError("weird"))

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("a fox")
    assert exc_info.value.retryable is False


async def test_a_fan_out_failure_surfaces_as_one_provider_error() -> None:
    """The caller gets the failure, not an exception group to unpack."""
    provider = _provider()
    _status_response(provider, 500)

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("three foxes", n=3)
    assert not isinstance(exc_info.value, BaseExceptionGroup)


async def test_a_non_object_payload_is_an_error() -> None:
    provider = _provider()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=["not", "an", "object"])
    provider._http.post = AsyncMock(return_value=response)

    with pytest.raises(ProviderError, match="non-object"):
        await provider.generate("a fox")


def test_missing_httpx_names_the_extra() -> None:
    with (
        patch.dict("sys.modules", {"httpx": None}),
        pytest.raises(ImportError, match=r"roomkit\[openrouter\]"),
    ):
        OpenRouterImageProvider(OpenRouterImageConfig(api_key="k", model="openai/gpt-image-2"))
