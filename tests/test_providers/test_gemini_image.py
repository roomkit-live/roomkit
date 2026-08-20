"""Tests for the Gemini image provider (RFC §25).

Offline: ``google.genai`` is replaced by a stub, so geometry translation,
response mapping, the modality split of usage and error translation are all
exercised without a key or a network.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.ai.base import AIImagePart, ProviderError
from roomkit.providers.gemini.config import GeminiImageConfig
from roomkit.providers.image import to_data_uri

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG).decode("ascii")


def _mock_genai_module() -> MagicMock:
    """A stub for ``from google import genai``."""
    google = MagicMock()
    google.genai = MagicMock()
    return google


def _provider(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {"api_key": "key", "model": "gemini-3.1-flash-image"}
    defaults.update(overrides)
    google = _mock_genai_module()
    with patch.dict("sys.modules", {"google": google, "google.genai": google.genai}):
        from roomkit.providers.gemini.image import GeminiImageProvider

        return GeminiImageProvider(GeminiImageConfig(**defaults))


def _interaction(
    *,
    data: str | None = PNG_B64,
    mime_type: str = "image/png",
    usage: Any = None,
    status: str = "completed",
) -> SimpleNamespace:
    image = None if data is None else SimpleNamespace(data=data, mime_type=mime_type)
    return SimpleNamespace(output_image=image, usage=usage, status=status)


def _usage(
    *,
    input_total: int = 0,
    output_total: int = 0,
    thoughts: int = 0,
    input_image: int = 0,
    output_image: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        total_input_tokens=input_total,
        total_output_tokens=output_total,
        total_thought_tokens=thoughts,
        input_tokens_by_modality=[SimpleNamespace(modality="image", tokens=input_image)],
        output_tokens_by_modality=[
            SimpleNamespace(modality="image", tokens=output_image),
            SimpleNamespace(modality="text", tokens=output_total - output_image),
        ],
    )


def _arm(provider: Any, *interactions: SimpleNamespace) -> AsyncMock:
    create = AsyncMock(side_effect=list(interactions) or [_interaction()])
    provider._client.aio.interactions.create = create
    return create


async def test_generate_returns_a_decodable_data_uri() -> None:
    provider = _provider()
    _arm(provider, _interaction())

    [result] = await provider.generate("an origami fox")

    assert result.mime_type == "image/png"
    assert result.decoded() == PNG


async def test_generate_names_no_delivery_mode() -> None:
    """Naming one is what fails the call.

    ``delivery`` passes the schema — the API validates its enum, refusing
    ``b64_json`` with the supported values — and is then refused per model:
    every image model answers "Image delivery mode is not supported" to
    ``inline`` and to ``uri`` alike. The default is the inline payload, which
    is what §25.3 wants; the invariant is enforced on the response instead.
    """
    provider = _provider()
    create = _arm(provider, _interaction())

    await provider.generate("a fox")

    response_format = create.await_args.kwargs["response_format"]
    assert "delivery" not in response_format
    assert response_format["type"] == "image"


async def test_a_uri_delivered_image_is_refused_by_its_own_name() -> None:
    """§25.3 enforced where it is checkable: a link never becomes an ImageResult."""
    provider = _provider()
    linked = SimpleNamespace(
        output_image=SimpleNamespace(data=None, uri="https://example.test/fox.png"),
        usage=None,
        status="completed",
    )
    _arm(provider, linked)

    with pytest.raises(ProviderError, match="URI"):
        await provider.generate("a fox")


async def test_the_prompt_becomes_a_text_content_block() -> None:
    provider = _provider()
    create = _arm(provider, _interaction())

    await provider.generate("an origami fox")

    assert create.await_args.kwargs["input"] == [{"type": "text", "text": "an origami fox"}]
    assert create.await_args.kwargs["model"] == "gemini-3.1-flash-image"


@pytest.mark.parametrize(
    ("size", "aspect_ratio", "tier"),
    [
        ("1024x1024", "1:1", "1K"),
        ("1920x1080", "16:9", "2K"),
        ("512x512", "1:1", "512"),
        ("1024x1536", "2:3", "2K"),
        ("3840x2160", "16:9", "4K"),
    ],
)
async def test_size_translates_to_gemini_geometry(size: str, aspect_ratio: str, tier: str) -> None:
    """RFC §25.2 — the caller passes pixels; the provider speaks its vendor's dialect."""
    provider = _provider()
    create = _arm(provider, _interaction())

    await provider.generate("a fox", size=size)

    response_format = create.await_args.kwargs["response_format"]
    assert response_format["aspect_ratio"] == aspect_ratio
    assert response_format["image_size"] == tier


async def test_an_unofferable_ratio_is_refused_not_rounded() -> None:
    provider = _provider()
    create = _arm(provider, _interaction())

    with pytest.raises(ValueError, match="aspect ratio 7:3"):
        await provider.generate("a fox", size="700x300")
    create.assert_not_awaited()


async def test_a_size_beyond_the_largest_tier_is_refused() -> None:
    provider = _provider()
    _arm(provider, _interaction())

    with pytest.raises(ValueError, match="largest tier"):
        await provider.generate("a fox", size="8000x8000")


async def test_a_requested_size_wins_over_the_configured_tier() -> None:
    provider = _provider(image_size="4K")
    create = _arm(provider, _interaction())

    await provider.generate("a fox", size="1024x1024")

    assert create.await_args.kwargs["response_format"]["image_size"] == "1K"


async def test_the_configured_tier_applies_when_no_size_is_asked_for() -> None:
    provider = _provider(image_size="2K")
    create = _arm(provider, _interaction())

    await provider.generate("a fox")

    response_format = create.await_args.kwargs["response_format"]
    assert response_format["image_size"] == "2K"
    assert "aspect_ratio" not in response_format


async def test_the_configured_output_type_is_forwarded() -> None:
    provider = _provider(output_mime_type="image/jpeg")
    create = _arm(provider, _interaction(mime_type="image/jpeg"))

    [result] = await provider.generate("a fox")

    assert create.await_args.kwargs["response_format"]["mime_type"] == "image/jpeg"
    assert result.mime_type == "image/jpeg"


async def test_n_images_are_n_interactions() -> None:
    """One interaction yields one image, so n is n calls — not a silent single result."""
    provider = _provider()
    create = _arm(provider, _interaction(), _interaction(), _interaction())

    results = await provider.generate("three foxes", n=3)

    assert len(results) == 3
    assert create.await_count == 3


async def test_one_failure_among_several_surfaces_as_a_provider_error() -> None:
    """The task group must not hand its caller an ExceptionGroup to unpack."""
    provider = _provider()
    provider._client.aio.interactions.create = AsyncMock(
        side_effect=[_interaction(), RuntimeError("429 rate limit exceeded"), _interaction()]
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("three foxes", n=3)
    assert exc_info.value.retryable is True
    # The SDK exception underneath survives the unwrapping.
    assert isinstance(exc_info.value.__cause__, RuntimeError)


async def test_generate_rejects_n_below_one() -> None:
    provider = _provider()
    create = _arm(provider, _interaction())

    with pytest.raises(ValueError, match="at least 1"):
        await provider.generate("a fox", n=0)
    create.assert_not_awaited()


async def test_an_interaction_without_an_image_is_an_error() -> None:
    provider = _provider()
    _arm(provider, _interaction(data=None, status="blocked"))

    with pytest.raises(ProviderError, match="no image"):
        await provider.generate("a fox")


async def test_gemini_reports_no_revised_prompt() -> None:
    provider = _provider()
    _arm(provider, _interaction())

    [result] = await provider.generate("a fox")

    assert result.revised_prompt is None


# --- Editing -------------------------------------------------------------------


async def test_reference_images_ride_the_same_call_as_inline_bytes() -> None:
    provider = _provider()
    create = _arm(provider, _interaction())
    reference = AIImagePart(url=to_data_uri(PNG, "image/png"))

    await provider.generate("make it blue", reference_images=[reference])

    content = create.await_args.kwargs["input"]
    assert content[0] == {"type": "text", "text": "make it blue"}
    assert content[1] == {"type": "image", "data": PNG_B64, "mime_type": "image/png"}


async def test_a_remote_reference_is_forwarded_as_a_uri() -> None:
    provider = _provider()
    create = _arm(provider, _interaction())
    reference = AIImagePart(url="https://example.com/fox.png", mime_type="image/png")

    await provider.generate("make it blue", reference_images=[reference])

    assert create.await_args.kwargs["input"][1] == {
        "type": "image",
        "uri": "https://example.com/fox.png",
        "mime_type": "image/png",
    }


async def test_a_corrupt_reference_payload_is_refused() -> None:
    provider = _provider()
    create = _arm(provider, _interaction())

    with pytest.raises(ValueError, match="not valid base64"):
        await provider.generate(
            "make it blue",
            reference_images=[AIImagePart(url="data:image/png;base64,!!!")],
        )
    create.assert_not_awaited()


# --- Usage ---------------------------------------------------------------------


async def test_usage_counters_are_disjoint() -> None:
    """The modality breakdown is subtracted from the totals, never added to them."""
    provider = _provider()
    _arm(
        provider,
        _interaction(
            usage=_usage(input_total=350, input_image=300, output_total=1200, output_image=1120)
        ),
    )

    [result] = await provider.generate("a fox")

    assert result.usage == {
        "input_tokens": 50,
        "input_image_tokens": 300,
        "output_tokens": 80,
        "output_image_tokens": 1120,
    }


async def test_thinking_tokens_join_the_text_output_counter() -> None:
    """Google bills thinking at the "text and thinking" output rate."""
    provider = _provider()
    _arm(provider, _interaction(usage=_usage(output_total=1120, output_image=1120, thoughts=200)))

    [result] = await provider.generate("a fox")

    assert result.usage["output_tokens"] == 200
    assert result.usage["output_image_tokens"] == 1120


async def test_usage_is_empty_when_the_vendor_reports_none() -> None:
    provider = _provider()
    _arm(provider, _interaction(usage=None))

    [result] = await provider.generate("a fox")

    assert result.usage == {}


async def test_the_reported_usage_prices_against_the_catalog() -> None:
    from roomkit.providers.gemini.image import GeminiImageProvider

    provider = _provider()
    _arm(
        provider, _interaction(usage=_usage(input_total=100, output_total=1120, output_image=1120))
    )
    [result] = await provider.generate("a fox")

    entry = next(
        m for m in GeminiImageProvider.available_models() if m.id == "gemini-3.1-flash-image"
    )
    assert entry.pricing is not None
    assert entry.pricing.cost_for(result.usage) == pytest.approx((100 * 0.5 + 1120 * 60.0) / 1e6)


# --- Errors --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "retryable"),
    [("429 rate limit exceeded", True), ("400 invalid argument", False)],
)
async def test_sdk_errors_map_to_retryability(message: str, retryable: bool) -> None:
    provider = _provider()
    provider._client.aio.interactions.create = AsyncMock(side_effect=RuntimeError(message))

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("a fox")
    assert exc_info.value.retryable is retryable
    assert exc_info.value.provider == "gemini"


async def test_a_status_code_drives_retryability_when_present() -> None:
    error = RuntimeError("boom")
    error.code = 503  # type: ignore[attr-defined]
    provider = _provider()
    provider._client.aio.interactions.create = AsyncMock(side_effect=error)

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("a fox")
    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 503


async def test_missing_sdk_names_the_extra() -> None:
    # Blocking ``google`` itself, not ``google.genai``: ``from google import
    # genai`` resolves the submodule as an attribute of an already-imported
    # ``google`` package, so masking only the submodule passes in isolation and
    # fails once anything else in the suite has imported the SDK.
    with (
        patch.dict("sys.modules", {"google": None}),
        pytest.raises(ImportError, match=r"roomkit\[gemini\]"),
    ):
        from roomkit.providers.gemini.image import GeminiImageProvider

        GeminiImageProvider(GeminiImageConfig(api_key="k", model="gemini-3.1-flash-image"))
