"""Tests for the OpenAI image provider (RFC §25).

Offline: the ``openai`` module is replaced by a stub, so the provider's
request-building, response mapping and error translation are exercised without
a key or a network.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import AIImagePart, ProviderError
from roomkit.providers.image import to_data_uri
from roomkit.providers.openai.config import OpenAIImageConfig

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG).decode("ascii")


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
    """Build the provider against a stubbed SDK, returning it ready to drive."""
    defaults: dict[str, Any] = {"api_key": "sk-test", "model": "gpt-image-2"}
    defaults.update(overrides)
    with patch.dict("sys.modules", {"openai": _mock_openai_module()}):
        from roomkit.providers.openai.image import OpenAIImageProvider

        return OpenAIImageProvider(OpenAIImageConfig(**defaults))


def _response(
    count: int = 1,
    *,
    output_format: str | None = "png",
    revised_prompt: str | None = None,
    b64: str | None = PNG_B64,
    usage: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=b64, revised_prompt=revised_prompt) for _ in range(count)],
        output_format=output_format,
        usage=usage,
    )


def _usage(input_tokens: int, image_tokens: int, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        input_tokens_details=SimpleNamespace(image_tokens=image_tokens),
        output_tokens=output_tokens,
    )


async def test_generate_returns_a_decodable_data_uri() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    [result] = await provider.generate("an origami fox")

    assert result.mime_type == "image/png"
    assert result.decoded() == PNG
    assert provider._client.images.generate.await_args.kwargs == {
        "model": "gpt-image-2",
        "prompt": "an origami fox",
        "n": 1,
    }


async def test_generate_forwards_size_quality_and_format() -> None:
    provider = _provider(quality="high", background="transparent", output_format="webp")
    provider._client.images.generate = AsyncMock(return_value=_response(output_format="webp"))

    [result] = await provider.generate("a fox", size="1536x1024")

    kwargs = provider._client.images.generate.await_args.kwargs
    assert kwargs["size"] == "1536x1024"
    assert kwargs["quality"] == "high"
    assert kwargs["background"] == "transparent"
    assert kwargs["output_format"] == "webp"
    assert result.mime_type == "image/webp"


async def test_a_size_off_the_classic_menu_goes_to_the_vendor() -> None:
    """gpt-image-2 takes near-arbitrary geometry, so no local list can judge."""
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    await provider.generate("a fox", size="3840X2160")

    assert provider._client.images.generate.await_args.kwargs["size"] == "3840x2160"


async def test_a_malformed_size_is_still_refused_locally() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        await provider.generate("a fox", size="huge")
    provider._client.images.generate.assert_not_awaited()


async def test_generate_returns_exactly_n_results() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response(count=3))

    results = await provider.generate("three foxes", n=3)

    assert len(results) == 3
    assert provider._client.images.generate.await_args.kwargs["n"] == 3


async def test_a_short_response_is_an_error_not_a_short_list() -> None:
    """RFC §25.2 — fewer results than asked for must raise, not be discovered later."""
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

    with pytest.raises(ProviderError, match="without inline bytes"):
        await provider.generate("a fox")


async def test_revised_prompt_is_carried_when_reported() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(
        return_value=_response(revised_prompt="a fox folded from red paper")
    )

    [result] = await provider.generate("a fox")

    assert result.revised_prompt == "a fox folded from red paper"


async def test_revised_prompt_is_none_rather_than_an_echo() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response())

    [result] = await provider.generate("a fox")

    assert result.revised_prompt is None


# --- Editing -------------------------------------------------------------------


async def test_reference_images_route_to_the_edit_endpoint() -> None:
    provider = _provider()
    provider._client.images.edit = AsyncMock(return_value=_response())
    provider._client.images.generate = AsyncMock(return_value=_response())
    reference = AIImagePart(url=to_data_uri(PNG, "image/png"))

    await provider.generate("make it blue", reference_images=[reference])

    provider._client.images.generate.assert_not_awaited()
    filename, data, mime_type = provider._client.images.edit.await_args.kwargs["image"][0]
    assert filename == "reference-0.png"
    assert data == PNG
    assert mime_type == "image/png"


async def test_an_edit_reference_must_carry_inline_bytes() -> None:
    provider = _provider()
    provider._client.images.edit = AsyncMock(return_value=_response())

    with pytest.raises(ValueError, match="reference image 0: expected a data: URI, got a https"):
        await provider.generate(
            "make it blue",
            reference_images=[AIImagePart(url="https://example.com/fox.png")],
        )


async def test_an_edit_reference_with_a_corrupt_payload_is_refused() -> None:
    provider = _provider()
    provider._client.images.edit = AsyncMock(return_value=_response())

    with pytest.raises(ValueError, match="not valid base64"):
        await provider.generate(
            "make it blue",
            reference_images=[AIImagePart(url="data:image/png;base64,!!!")],
        )


# --- Usage ---------------------------------------------------------------------


async def test_usage_counters_are_disjoint() -> None:
    """OpenAI nests image tokens inside the input total; the split must not double-bill."""
    provider = _provider()
    provider._client.images.generate = AsyncMock(
        return_value=_response(
            usage=_usage(input_tokens=150, image_tokens=100, output_tokens=1024)
        )
    )

    [result] = await provider.generate("a fox")

    assert result.usage == {
        "input_tokens": 50,
        "input_image_tokens": 100,
        "output_tokens": 0,
        "output_image_tokens": 1024,
    }


async def test_usage_rides_the_first_result_only() -> None:
    """The counters describe the call, so copying them per image would bill n times."""
    provider = _provider()
    provider._client.images.generate = AsyncMock(
        return_value=_response(count=2, usage=_usage(10, 0, 2048))
    )

    first, second = await provider.generate("two foxes", n=2)

    assert first.usage["output_image_tokens"] == 2048
    assert second.usage == {}


async def test_usage_is_empty_when_the_vendor_reports_none() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(return_value=_response(usage=None))

    [result] = await provider.generate("a fox")

    assert result.usage == {}


async def test_the_reported_usage_prices_against_the_catalog() -> None:
    """End to end: the counters this provider emits are the ones the catalog rates."""
    from roomkit.providers.openai.image import OpenAIImageProvider

    provider = _provider()
    provider._client.images.generate = AsyncMock(
        return_value=_response(usage=_usage(input_tokens=100, image_tokens=0, output_tokens=1000))
    )
    [result] = await provider.generate("a fox")

    entry = next(m for m in OpenAIImageProvider.available_models() if m.id == "gpt-image-2")
    assert entry.pricing is not None
    assert entry.pricing.cost_for(result.usage) == pytest.approx((100 * 5.0 + 1000 * 30.0) / 1e6)


# --- Errors --------------------------------------------------------------------


async def test_a_connection_failure_is_retryable() -> None:
    provider = _provider()
    provider._client.images.generate = AsyncMock(side_effect=_FakeAPIConnectionError("down"))

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("a fox")
    assert exc_info.value.retryable is True
    assert exc_info.value.provider == "openai"


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


async def test_missing_sdk_names_the_extra() -> None:
    with (
        patch.dict("sys.modules", {"openai": None}),
        pytest.raises(ImportError, match=r"roomkit\[openai\]"),
    ):
        from roomkit.providers.openai.image import OpenAIImageProvider

        OpenAIImageProvider(OpenAIImageConfig(api_key="sk", model="gpt-image-2"))
