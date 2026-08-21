"""Tests for the xAI (Grok Imagine) image provider (RFC §25).

Offline: the ``openai`` module is replaced by a stub, so request building —
including the size-to-ratio translation and the JSON edits path the SDK's
multipart ``images.edit`` cannot express — response mapping and error
translation are exercised without a key or a network.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import AIImagePart, ProviderError
from roomkit.providers.image import to_data_uri
from roomkit.providers.xai.config import XAIImageConfig

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG).decode("ascii")
JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 20).decode("ascii")


class _FakeAPIStatusError(Exception):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeAPIConnectionError(Exception):
    pass


def _mock_openai_module() -> MagicMock:
    mod = MagicMock()
    mod.APIStatusError = _FakeAPIStatusError
    mod.APIConnectionError = _FakeAPIConnectionError
    return mod


def _provider(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {"api_key": "xai-test"}
    defaults.update(overrides)
    with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
        from roomkit.providers.xai.image import XAIImageProvider

        return XAIImageProvider(XAIImageConfig(**defaults))


def _image(b64: str | None = PNG_B64, mime_type: str | None = "image/png") -> SimpleNamespace:
    image = SimpleNamespace(b64_json=b64)
    if mime_type is not None:
        image.mime_type = mime_type
    return image


def _response(count: int = 1, *, usage: Any = None, **image_kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(data=[_image(**image_kwargs) for _ in range(count)], usage=usage)


async def test_generate_requests_inline_bytes() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    [result] = await provider.generate("an origami fox")

    assert result.mime_type == "image/png"
    assert result.decoded() == PNG
    assert provider._client.images.generate.await_args.kwargs == {
        "model": "grok-imagine-image-2.0",
        "prompt": "an origami fox",
        "n": 1,
        "response_format": "b64_json",
    }


async def test_a_size_becomes_a_ratio_and_a_tier() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    await provider.generate("a fox", size="1024x768")

    kwargs = provider._client.images.generate.await_args.kwargs
    assert kwargs["aspect_ratio"] == "4:3"
    assert kwargs["resolution"] == "1k"


async def test_a_large_size_lands_on_the_2k_tier() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    await provider.generate("a fox", size="2048x2048")

    kwargs = provider._client.images.generate.await_args.kwargs
    assert kwargs["aspect_ratio"] == "1:1"
    assert kwargs["resolution"] == "2k"


async def test_an_unoffered_ratio_is_refused_not_rounded() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    with pytest.raises(ValueError, match="aspect ratio 1000:999"):
        await provider.generate("a fox", size="1000x999")
    provider._client.images.generate.assert_not_awaited()


async def test_a_size_beyond_the_largest_tier_is_refused() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    with pytest.raises(ValueError, match="largest tier"):
        await provider.generate("a fox", size="4096x4096")


async def test_the_configured_tier_applies_when_no_size_is_named() -> None:
    provider = _provider(resolution="2k", quality="low")
    provider._client.images.generate = AsyncMock(return_value=_response())

    await provider.generate("a fox")

    kwargs = provider._client.images.generate.await_args.kwargs
    assert kwargs["resolution"] == "2k"
    assert kwargs["quality"] == "low"
    assert "aspect_ratio" not in kwargs


async def test_a_named_size_wins_over_the_configured_tier() -> None:
    provider = _provider(resolution="2k")
    provider._client.images.generate = AsyncMock(return_value=_response())

    await provider.generate("a fox", size="512x512")

    assert provider._client.images.generate.await_args.kwargs["resolution"] == "1k"


# --- Response mapping ----------------------------------------------------------


async def test_generate_returns_exactly_n_results() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response(count=3))

    results = await provider.generate("three foxes", n=3)

    assert len(results) == 3
    assert provider._client.images.generate.await_args.kwargs["n"] == 3


async def test_a_short_response_is_an_error_not_a_short_list() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response(count=2))

    with pytest.raises(ProviderError, match="2 image"):
        await provider.generate("three foxes", n=3)


async def test_generate_rejects_n_below_one() -> None:
    provider = _provider()
    with pytest.raises(ValueError, match="at least 1"):
        await provider.generate("a fox", n=0)


async def test_a_result_without_inline_bytes_is_an_error() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response(b64=None))

    with pytest.raises(ProviderError, match="b64_json"):
        await provider.generate("a fox")


async def test_an_undeclared_mime_type_is_sniffed_from_the_bytes() -> None:
    """Grok Imagine answers JPEG by default; a fixed PNG fallback would mislabel it."""
    provider = _provider()
    provider._client.images.generate = AsyncMock(
        return_value=_response(b64=JPEG_B64, mime_type=None)
    )

    [result] = await provider.generate("a fox")

    assert result.mime_type == "image/jpeg"
    assert result.data.startswith("data:image/jpeg;base64,")


# --- Editing -------------------------------------------------------------------


async def test_one_reference_posts_json_to_the_edits_path() -> None:
    provider = _provider()
    provider._client.post = AsyncMock(return_value=_response())
    provider._client.images.generate = AsyncMock(return_value=_response())
    reference = AIImagePart(url=to_data_uri(PNG, "image/png"))

    await provider.generate("make it blue", reference_images=[reference])

    provider._client.images.generate.assert_not_awaited()
    args, kwargs = provider._client.post.await_args
    assert args == ("/images/edits",)
    body = kwargs["body"]
    assert body["image"] == {"url": to_data_uri(PNG, "image/png"), "type": "image_url"}
    assert "images" not in body
    assert body["response_format"] == "b64_json"


async def test_several_references_ride_the_images_array() -> None:
    provider = _provider()
    provider._client.post = AsyncMock(return_value=_response())
    inline = AIImagePart(url=to_data_uri(PNG, "image/png"))
    remote = AIImagePart(url="https://example.com/fox.png")

    await provider.generate("blend them", reference_images=[inline, remote])

    body = provider._client.post.await_args.kwargs["body"]
    assert "image" not in body
    assert body["images"] == [
        {"url": to_data_uri(PNG, "image/png"), "type": "image_url"},
        {"url": "https://example.com/fox.png", "type": "image_url"},
    ]


async def test_a_corrupt_reference_is_the_callers_error() -> None:
    provider = _provider()
    provider._client.post = AsyncMock(return_value=_response())

    with pytest.raises(ValueError, match="reference image 0"):
        await provider.generate(
            "make it blue",
            reference_images=[AIImagePart(url="data:image/png;base64,!!!")],
        )
    provider._client.post.assert_not_awaited()


async def test_a_model_without_the_edit_tag_refuses_references() -> None:
    """RFC §25.4 — silently redrawing from the prompt alone would drop the reference."""
    provider = _provider(model="grok-imagine-image")
    provider._client.post = AsyncMock(return_value=_response())

    assert provider.supports_editing is False
    with pytest.raises(ValueError, match="does not take reference images"):
        await provider.generate("edit", reference_images=[AIImagePart(url="https://x/y.png")])
    provider._client.post.assert_not_awaited()


def test_an_uncatalogued_model_is_presumed_to_edit() -> None:
    assert _provider(model="grok-imagine-image-3.0").supports_editing is True


# --- Usage ---------------------------------------------------------------------


async def test_usage_lands_on_the_image_counter() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(
        return_value=_response(usage=SimpleNamespace(input_tokens=12, output_tokens=1024))
    )

    [result] = await provider.generate("a fox")

    assert result.usage == {"input_tokens": 12, "output_image_tokens": 1024}


async def test_usage_rides_the_first_result_only() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(
        return_value=_response(count=2, usage=SimpleNamespace(input_tokens=1, output_tokens=2))
    )

    first, second = await provider.generate("two foxes", n=2)

    assert first.usage["output_image_tokens"] == 2
    assert second.usage == {}


async def test_usage_is_empty_when_the_vendor_reports_none() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response(usage=None))

    [result] = await provider.generate("a fox")

    assert result.usage == {}


# --- Errors --------------------------------------------------------------------


async def test_a_connection_failure_is_retryable() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(side_effect=_FakeAPIConnectionError("down"))

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("a fox")
    assert exc_info.value.retryable is True
    assert exc_info.value.provider == "xai"


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (400, False)])
async def test_status_errors_map_to_retryability(status: int, retryable: bool) -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(
        side_effect=_FakeAPIStatusError("boom", status_code=status)
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("a fox")
    assert exc_info.value.retryable is retryable
    assert exc_info.value.status_code == status


async def test_an_unexpected_error_is_wrapped_not_retryable() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(side_effect=RuntimeError("weird"))

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("a fox")
    assert exc_info.value.retryable is False


def test_missing_sdk_names_the_extra() -> None:
    with (
        patch.dict("sys.modules", {"openai": None}),
        pytest.raises(ImportError, match=r"roomkit\[xai\]"),
    ):
        from roomkit.providers.xai.image import XAIImageProvider

        XAIImageProvider(XAIImageConfig(api_key="k"))
