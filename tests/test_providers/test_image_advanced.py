"""Image controls are validated before spend; partial spend remains observable."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from roomkit.providers.ai.base import AIImagePart
from roomkit.providers.image import ImageAttempt, ImageGenerationError, ImageOptions
from roomkit.providers.image.usage import gemini_image_usage, openai_image_usage
from tests.test_providers.test_gemini_image import _interaction
from tests.test_providers.test_gemini_image import _provider as gemini
from tests.test_providers.test_gemini_image import _usage as gemini_usage
from tests.test_providers.test_openai_image import PNG_B64, _response
from tests.test_providers.test_openai_image import _provider as openai
from tests.test_providers.test_openai_image import _usage as openai_usage


@pytest.mark.parametrize(
    ("model", "quality"),
    [
        ("gpt-image-2.5-flare", "max"),
        ("gpt-image-2.5-sunburst-2026-09-08", "xhigh"),
    ],
)
async def test_openai_options_and_aliases_reach_the_sdk(model: str, quality: str) -> None:
    provider = openai(model=model)
    create = AsyncMock(return_value=_response(output_format="webp"))
    provider._client.images.generate = create
    [result] = await provider.generate_with_options(
        "a fox",
        size="auto",
        options=ImageOptions(
            quality=quality,
            output_format="webp",
            background="transparent",
            output_compression=75,
        ),
    )
    assert create.await_args.kwargs["quality"] == quality
    assert create.await_args.kwargs["output_compression"] == 75
    assert result.mime_type == "image/webp"
    assert result.effective_options["size"] == "auto"


@pytest.mark.parametrize(
    ("options", "size"),
    [
        (ImageOptions(quality="max"), None),
        (ImageOptions(output_format="jpeg", background="transparent"), None),
        (ImageOptions(output_compression=50), None),
        (ImageOptions(aspect_ratio="16:9"), None),
        (ImageOptions(input_fidelity="high"), None),
        (ImageOptions(), "1000x1000"),
        (ImageOptions(), "4096x4096"),
        (ImageOptions(), "64x64"),
    ],
)
async def test_openai_incompatible_options_never_spend(
    options: ImageOptions, size: str | None
) -> None:
    provider = openai()
    create = AsyncMock()
    provider._client.images.generate = create
    with pytest.raises(ValueError):
        await provider.generate_with_options("a fox", options=options, size=size)
    create.assert_not_awaited()


async def test_mask_and_ordered_references_use_edit() -> None:
    provider = openai()
    provider._client.images.edit = AsyncMock(return_value=_response())
    part = AIImagePart(url=f"data:image/png;base64,{PNG_B64}", mime_type="image/png")
    await provider.generate_with_options(
        "change the background", reference_images=[part, part], mask=part
    )
    kwargs = provider._client.images.edit.await_args.kwargs
    assert len(kwargs["image"]) == 2
    assert kwargs["mask"][2] == "image/png"


async def test_openai_short_or_corrupted_response_keeps_billed_usage_and_good_pixels() -> None:
    provider = openai()
    response = _response(count=1, usage=openai_usage(120, 20, 1000))
    response._request_id = "request-one"
    provider._client.images.generate = AsyncMock(return_value=response)
    with pytest.raises(ImageGenerationError) as caught:
        await provider.generate_with_options("two foxes", n=2)
    error = caught.value
    assert not error.retryable
    assert len(error.results) == 1
    assert error.attempts[0].usage["output_image_tokens"] == 1000
    assert error.attempts[0].raw_usage["input_tokens"] == 120
    assert error.attempts[0].provider_request_id == "request-one"


async def test_gemini_native_geometry_continuity_thinking_and_grounding() -> None:
    provider = gemini()
    response = _interaction(usage=gemini_usage(output_total=1120, output_image=1120))
    response.id = "interaction-one"
    response.steps = [{"type": "google_search_result", "search_suggestions": "<p>Sources</p>"}]
    provider._client.aio.interactions.create = AsyncMock(return_value=response)
    [result] = await provider.generate_with_options(
        "edit the fox",
        options=ImageOptions(
            aspect_ratio="16:9",
            image_size="2K",
            output_format="jpeg",
            previous_interaction_id="interaction-zero",
            search_types=["web_search", "image_search"],
            thinking_level="high",
        ),
    )
    request = provider._client.aio.interactions.create.await_args.kwargs
    assert request["response_format"] == {
        "type": "image",
        "aspect_ratio": "16:9",
        "image_size": "2K",
        "mime_type": "image/jpeg",
    }
    assert request["previous_interaction_id"] == "interaction-zero"
    assert request["tools"][0]["search_types"] == ["web_search", "image_search"]
    assert request["generation_config"] == {"thinking_level": "high"}
    assert result.provider_request_id == "interaction-one"
    assert result.metadata["steps"][0]["search_suggestions"] == "<p>Sources</p>"


async def test_gemini_nano_banana_2_1_takes_flash_controls_at_its_sizes() -> None:
    """Nano Banana 2.1 is catalogued (RMK-652): an extreme ratio, 4K, image
    search and a thinking level reach the wire instead of being refused as the
    controls of a model with unknown capabilities."""
    provider = gemini(model="gemini-nano-banana-2.1")
    # A 4K image on Nano Banana 2.1 is 3780 output tokens (measured 2026-10-09).
    response = _interaction(usage=gemini_usage(output_total=3780, output_image=3780))
    provider._client.aio.interactions.create = AsyncMock(return_value=response)
    options = ImageOptions(
        aspect_ratio="8:1", image_size="4K", search_types=["image_search"], thinking_level="high"
    )

    await provider.generate_with_options("a banner", options=options)

    request = provider._client.aio.interactions.create.await_args.kwargs
    assert request["response_format"]["aspect_ratio"] == "8:1"
    assert request["response_format"]["image_size"] == "4K"
    assert request["tools"][0]["search_types"] == ["image_search"]
    assert request["generation_config"] == {"thinking_level": "high"}


@pytest.mark.parametrize(
    ("model", "options"),
    [
        ("gemini-3.1-flash-lite-image", ImageOptions(image_size="4K")),
        ("gemini-3.1-flash-lite-image", ImageOptions(search_types=["web_search"])),
        ("gemini-3-pro-image", ImageOptions(image_size="512")),
        ("gemini-3-pro-image", ImageOptions(thinking_level="minimal")),
        ("gemini-3-pro-image", ImageOptions(search_types=["image_search"])),
        ("gemini-nano-banana-2.1", ImageOptions(image_size="512")),
    ],
)
async def test_gemini_capabilities_belong_to_the_selected_model(
    model: str, options: ImageOptions
) -> None:
    provider = gemini(model=model)
    create = AsyncMock()
    provider._client.aio.interactions.create = create
    with pytest.raises(ValueError):
        await provider.generate_with_options("a fox", options=options)
    create.assert_not_awaited()


async def test_gemini_failed_conversion_and_success_both_keep_their_usage() -> None:
    provider = gemini()
    provider._client.aio.interactions.create = AsyncMock(
        side_effect=[
            _interaction(usage=gemini_usage(output_total=1120, output_image=1120)),
            _interaction(data=None, usage=gemini_usage(thoughts=45)),
        ]
    )
    events: list[ImageAttempt] = []

    async def progress(event: ImageAttempt) -> None:
        events.append(event)

    with pytest.raises(ImageGenerationError) as caught:
        await provider.generate_with_options("two foxes", n=2, on_progress=progress)
    assert len(caught.value.results) == 1
    assert len(caught.value.attempts) == 2
    assert caught.value.attempts[1].usage["output_tokens"] == 45
    assert [event.status for event in events].count("succeeded") == 1
    assert [event.status for event in events].count("failed") == 1


async def test_cancellation_preserves_completed_siblings_and_marks_inflight_unknown() -> None:
    provider = gemini()
    ready = asyncio.Event()
    events: list[ImageAttempt] = []
    count = 0

    async def create(**_kwargs: object) -> object:
        nonlocal count
        count += 1
        if count == 1:
            return _interaction()
        ready.set()
        await asyncio.Event().wait()

    async def progress(event: ImageAttempt) -> None:
        events.append(event)

    provider._client.aio.interactions.create = create
    task = asyncio.create_task(provider.generate_with_options("foxes", n=2, on_progress=progress))
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert any(event.status == "succeeded" and event.results for event in events)
    assert any(event.status == "unknown" for event in events)


async def test_callback_failure_keeps_the_vendor_result() -> None:
    provider = gemini()
    provider._client.aio.interactions.create = AsyncMock(return_value=_interaction())

    async def progress(event: ImageAttempt) -> None:
        if event.status == "succeeded":
            raise RuntimeError("storage unavailable")

    with pytest.raises(ImageGenerationError) as caught:
        await provider.generate_with_options("a fox", on_progress=progress)
    assert len(caught.value.results) == 1
    assert not caught.value.retryable


def test_empty_usage_objects_do_not_invent_zero_counters() -> None:
    assert openai_image_usage(SimpleNamespace()) == {}
    assert gemini_image_usage(SimpleNamespace()) == {}
    assert openai_image_usage(SimpleNamespace(input_tokens=42)) == {
        "unclassified_input_tokens": 42
    }


async def test_invalid_first_openai_result_does_not_hide_a_valid_later_image() -> None:
    provider = openai()
    response = _response(count=2, usage=openai_usage(100, 0, 2200))
    response.data[0].b64_json = None
    provider._client.images.generate = AsyncMock(return_value=response)
    with pytest.raises(ImageGenerationError) as caught:
        await provider.generate_with_options("two foxes", n=2)
    assert len(caught.value.results) == 1
    assert caught.value.results[0].usage["output_image_tokens"] == 2200


@pytest.mark.parametrize("store", [False, True, None])
@pytest.mark.parametrize("model", ["gemini-3.1-flash-image", "custom-image-model"])
async def test_gemini_explicit_storage_choice_reaches_the_request(
    store: bool | None, model: str
) -> None:
    provider = gemini(model=model)
    provider._client.aio.interactions.create = AsyncMock(return_value=_interaction())
    await provider.generate_with_options("a fox", options=ImageOptions(store=store))
    request = provider._client.aio.interactions.create.await_args.kwargs
    if store is None:
        assert "store" not in request
    else:
        assert request["store"] is store


def test_gemini_portable_geometry_preflight_is_public_and_matches_execution() -> None:
    provider = gemini()
    assert provider.resolve_size("2048x2048") == ("1:1", "2K")
    assert provider._geometry("2048x2048") == provider.resolve_size("2048x2048")
    with pytest.raises(ValueError):
        provider.resolve_size("bad-size")
