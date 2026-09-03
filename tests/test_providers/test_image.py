"""Tests for the image-generation surface (RFC §25).

The ABC, ``ImageResult``'s data-URI invariant, the mock, the catalogs, and the
image counters ``ModelPricing`` learned. Everything here is offline: catalogs
are classmethods and the mock needs no SDK.
"""

from __future__ import annotations

import base64
from datetime import date

import pytest
from pydantic import SecretStr, ValidationError

from roomkit.providers.ai.base import AIImagePart, ModelPricing
from roomkit.providers.gemini.image import GeminiImageProvider
from roomkit.providers.image import (
    IMAGE_GEN_CAPABILITY,
    ImageProvider,
    ImageResult,
    MockImageProvider,
    parse_data_uri,
    parse_size,
    sniff_mime_type,
    to_data_uri,
)
from roomkit.providers.openai.config import OpenAIImageConfig
from roomkit.providers.openai.image import OpenAIImageProvider
from roomkit.providers.openrouter.image import OpenRouterImageProvider
from roomkit.providers.xai.image import XAIImageProvider

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

# Catalogs whose vendors meter per token, and so carry ModelPricing.
PRICED_IMAGE_CATALOGS = [OpenAIImageProvider, GeminiImageProvider]
# Catalogs whose vendors bill a flat amount per image — a unit ModelPricing
# cannot state, so their entries deliberately carry no pricing.
PER_IMAGE_BILLED_CATALOGS = [XAIImageProvider, OpenRouterImageProvider]
IMAGE_CATALOGS = PRICED_IMAGE_CATALOGS + PER_IMAGE_BILLED_CATALOGS


# --- parse_size / to_data_uri --------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1024x1024", (1024, 1024)), ("1536X1024", (1536, 1024)), (" 512x512 ", (512, 512))],
)
def test_parse_size_accepts_wxh(value: str, expected: tuple[int, int]) -> None:
    assert parse_size(value) == expected


@pytest.mark.parametrize("value", ["1024", "1024*1024", "axb", "", "1024x", "-1x10", "0x100"])
def test_parse_size_rejects_everything_else(value: str) -> None:
    with pytest.raises(ValueError, match="size"):
        parse_size(value)


def test_to_data_uri_round_trips() -> None:
    uri = to_data_uri(PNG, "image/png")
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.partition(",")[2]) == PNG


# --- ImageResult ---------------------------------------------------------------


def test_image_result_decodes_to_the_original_bytes() -> None:
    result = ImageResult(data=to_data_uri(PNG, "image/png"), mime_type="image/png")
    assert result.decoded() == PNG


@pytest.mark.parametrize(
    "data",
    [
        "iVBORw0KGgo=",  # bare base64 — the ambiguity the invariant exists to remove
        "https://example.com/cat.png",
        "data:image/png;base64,",  # header, no payload
        "data:image/png,notbase64",  # not declared base64
    ],
)
def test_image_result_refuses_anything_but_a_data_uri(data: str) -> None:
    with pytest.raises(ValidationError):
        ImageResult(data=data, mime_type="image/png")


def test_image_result_decoded_reports_a_corrupt_payload() -> None:
    result = ImageResult(data="data:image/png;base64,!!!not-base64!!!", mime_type="image/png")
    with pytest.raises(ValueError, match="not valid base64"):
        result.decoded()


def test_image_result_becomes_a_message_part() -> None:
    result = ImageResult(data=to_data_uri(PNG, "image/png"), mime_type="image/png")
    part = result.to_image_part()
    assert isinstance(part, AIImagePart)
    assert part.url == result.data
    assert part.mime_type == "image/png"


def test_image_result_round_trips_through_a_message_part() -> None:
    """A generated image is directly re-usable as the next edit's reference."""
    result = ImageResult(data=to_data_uri(PNG, "image/png"), mime_type="image/png")
    back = ImageResult(data=result.to_image_part().url, mime_type="image/png")
    assert back.decoded() == PNG


# --- MockImageProvider ---------------------------------------------------------


async def test_mock_returns_exactly_n_decodable_images() -> None:
    provider = MockImageProvider()
    results = await provider.generate("an origami fox", n=3)
    assert len(results) == 3
    assert all(r.decoded() == PNG for r in results)


async def test_mock_records_its_calls() -> None:
    provider = MockImageProvider()
    reference = AIImagePart(url=to_data_uri(PNG, "image/png"), mime_type="image/png")
    await provider.generate("redraw it", size="1024x1024", reference_images=[reference])
    assert provider.calls == [("redraw it", "1024x1024", 1, [reference])]


async def test_mock_cycles_through_its_images() -> None:
    first, second = PNG, PNG + b"\x00"
    provider = MockImageProvider(images=[first, second])
    results = await provider.generate("two", n=3)
    assert [r.decoded() for r in results] == [first, second, first]


async def test_mock_reports_disjoint_usage_counters() -> None:
    provider = MockImageProvider()
    reference = AIImagePart(url=to_data_uri(PNG, "image/png"))
    [result] = await provider.generate("a b c", reference_images=[reference])
    assert result.usage == {
        "input_tokens": 3,
        "input_image_tokens": 100,
        "output_tokens": 0,
        "output_image_tokens": 1024,
    }


async def test_mock_refuses_editing_when_it_says_it_cannot() -> None:
    provider = MockImageProvider(supports_editing=False)
    assert provider.supports_editing is False
    with pytest.raises(ValueError, match="does not support editing"):
        await provider.generate(
            "edit", reference_images=[AIImagePart(url=to_data_uri(PNG, "image/png"))]
        )


async def test_mock_rejects_a_malformed_size() -> None:
    provider = MockImageProvider()
    with pytest.raises(ValueError, match="size"):
        await provider.generate("hi", size="huge")


async def test_mock_rejects_n_below_one() -> None:
    provider = MockImageProvider()
    with pytest.raises(ValueError, match="at least 1"):
        await provider.generate("hi", n=0)


# --- sniff_mime_type -----------------------------------------------------------


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\x89PNG\r\n\x1a\n rest", "image/png"),
        (b"\xff\xd8\xff\xe0 rest", "image/jpeg"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"GIF89a rest", "image/gif"),
    ],
)
def test_sniff_mime_type_reads_the_magic_number(head: bytes, expected: str) -> None:
    assert sniff_mime_type(head) == expected


def test_sniff_mime_type_falls_back_when_bytes_say_nothing() -> None:
    assert sniff_mime_type(b"not an image") == "image/png"
    assert sniff_mime_type(b"", fallback="image/jpeg") == "image/jpeg"


# --- ImageProvider base --------------------------------------------------------


def test_base_catalog_is_empty_and_lookup_returns_none() -> None:
    class Bare(ImageProvider):
        @property
        def model_name(self) -> str:
            return "nothing"

        async def generate(self, prompt: str, **kwargs: object) -> list[ImageResult]:
            return []

    assert ImageProvider.available_models() == []
    assert Bare().catalog_entry() is None
    assert Bare().supports_editing is False
    assert Bare().name == "Bare"


def test_catalog_entry_resolves_the_active_model() -> None:
    provider = OpenAIImageProvider.__new__(OpenAIImageProvider)
    provider._config = OpenAIImageConfig(api_key=SecretStr("k"), model="gpt-image-2")
    entry = provider.catalog_entry()
    assert entry is not None
    assert entry.display_name == "GPT Image 2"


# --- Catalogs ------------------------------------------------------------------


@pytest.mark.parametrize("provider", IMAGE_CATALOGS)
def test_image_catalog_ids_are_unique(provider: type[ImageProvider]) -> None:
    ids = [m.id for m in provider.available_models()]
    assert len(ids) == len(set(ids))
    assert ids


@pytest.mark.parametrize("provider", IMAGE_CATALOGS)
def test_image_catalog_entries_are_tagged(provider: type[ImageProvider]) -> None:
    for model in provider.available_models():
        assert IMAGE_GEN_CAPABILITY in model.capabilities, model.id


@pytest.mark.parametrize("provider", PRICED_IMAGE_CATALOGS)
def test_image_catalog_entries_price_the_pixels(provider: type[ImageProvider]) -> None:
    """A catalog that omits the image-output rate bills a generation as free."""
    for model in provider.available_models():
        assert model.pricing is not None, model.id
        assert model.pricing.image_output_per_million, model.id
        assert model.pricing.image_input_per_million, model.id


@pytest.mark.parametrize("provider", PER_IMAGE_BILLED_CATALOGS)
def test_per_image_billed_catalogs_state_no_rate(provider: type[ImageProvider]) -> None:
    """A flat per-image charge restated per token would be a wrong number, not a missing one."""
    for model in provider.available_models():
        assert model.pricing is None, model.id


def test_image_catalog_is_disjoint_from_the_conversational_one() -> None:
    """RFC §25.6 — no id draws and converses, and neither list carries the other's."""
    from roomkit.providers.gemini.ai import GeminiAIProvider
    from roomkit.providers.openai.ai import OpenAIAIProvider
    from roomkit.providers.openrouter.ai import OpenRouterAIProvider
    from roomkit.providers.xai.ai import XAIAIProvider

    for chat, image in (
        (OpenAIAIProvider, OpenAIImageProvider),
        (GeminiAIProvider, GeminiImageProvider),
        (XAIAIProvider, XAIImageProvider),
        (OpenRouterAIProvider, OpenRouterImageProvider),
    ):
        chat_ids = {m.id for m in chat.available_models()}
        image_ids = {m.id for m in image.available_models()}
        assert not chat_ids & image_ids


def test_conversational_catalogs_carry_no_image_rates() -> None:
    """The new fields stay ``None`` where nothing generates an image."""
    from roomkit.providers.gemini.ai import GeminiAIProvider
    from roomkit.providers.openai.ai import OpenAIAIProvider

    for provider in (OpenAIAIProvider, GeminiAIProvider):
        for model in provider.available_models():
            assert model.pricing is not None
            assert model.pricing.image_output_per_million is None, model.id


# --- ModelPricing image counters -----------------------------------------------


def _pricing(**overrides: object) -> ModelPricing:
    base: dict[str, object] = {
        "input_per_million": 2.0,
        "output_per_million": 12.0,
        "image_input_per_million": 2.0,
        "image_output_per_million": 120.0,
        "verified": date(2026, 8, 7),
    }
    base.update(overrides)
    return ModelPricing(**base)  # type: ignore[arg-type]


def test_cost_for_prices_image_tokens_on_their_own_meter() -> None:
    pricing = _pricing()
    cost = pricing.cost_for(
        {
            "input_tokens": 1_000_000,
            "input_image_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "output_image_tokens": 1_000_000,
        }
    )
    assert cost == pytest.approx(2.0 + 2.0 + 12.0 + 120.0)


def test_cost_for_prices_one_nano_banana_pro_image() -> None:
    """1120 output-image tokens at $120/M is the $0.134 Google advertises."""
    assert _pricing().cost_for({"output_image_tokens": 1120}) == pytest.approx(0.1344)


def test_cost_for_ignores_image_counters_without_rates() -> None:
    """A text-only catalog entry bills nothing for counters it does not represent."""
    pricing = _pricing(image_input_per_million=None, image_output_per_million=None)
    assert pricing.cost_for({"input_image_tokens": 10_000, "output_image_tokens": 10_000}) == 0.0


def test_cost_for_is_unchanged_for_text_only_usage() -> None:
    """Non-regression: the counters that existed before still price the same."""
    pricing = ModelPricing(
        input_per_million=5.0,
        output_per_million=30.0,
        cache_read_per_million=0.5,
        verified=date(2026, 8, 5),
    )
    usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 2000}
    assert pricing.cost_for(usage) == pytest.approx((1000 * 5 + 500 * 30 + 2000 * 0.5) / 1_000_000)


def test_image_input_tokens_count_toward_the_long_context_threshold() -> None:
    pricing = _pricing(
        long_context_threshold_tokens=100,
        long_context_input_multiplier=2.0,
        long_context_output_multiplier=1.5,
    )
    cost = pricing.cost_for({"input_image_tokens": 200, "output_image_tokens": 100})
    assert cost == pytest.approx((200 * 2.0 * 2.0 + 100 * 120.0 * 1.5) / 1_000_000)


def test_image_counters_must_be_non_negative_integers() -> None:
    with pytest.raises(ValueError, match="output_image_tokens"):
        _pricing().cost_for({"output_image_tokens": -1})


def test_image_rates_reject_negative_values() -> None:
    with pytest.raises(ValidationError):
        _pricing(image_output_per_million=-1.0)


# --- parse_data_uri ------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        "QUJDMTIz",  # canonical
        "QUJD\nMTIz",  # an encoder that wrapped its lines
        "QUJDMTIz\r\n",  # trailing newline
        " QUJD MTIz ",  # spaced
    ],
)
def test_parse_data_uri_repairs_whitespace(payload: str) -> None:
    assert parse_data_uri(f"data:image/png;base64,{payload}") == ("image/png", b"ABC123")


def test_parse_data_uri_repairs_missing_padding() -> None:
    assert parse_data_uri("data:image/png;base64,QUJDMQ") == ("image/png", b"ABC1")
    assert parse_data_uri("data:image/png;base64,QUJDMQ==") == ("image/png", b"ABC1")


def test_parse_data_uri_takes_the_media_type_from_the_header_then_the_fallback() -> None:
    assert (
        parse_data_uri("data:image/jpeg;base64,QUJD", fallback_mime="image/png")[0] == "image/jpeg"
    )
    assert parse_data_uri("data:;base64,QUJD", fallback_mime="image/webp")[0] == "image/webp"
    assert parse_data_uri("data:;base64,QUJD")[0] == "image/png"


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://example.com/a.png", "expected a data: URI"),
        ("data:image/png;base64,", "no payload"),
        ("data:image/png;base64,  \n ", "no payload"),
        ("data:image/png;base64,not*base64", "not valid base64"),
        ("data:image/png;base64,QUJDM", "not valid base64"),  # a length no padding completes
        ("data:image/png;base64,QUJD-_==", "not valid base64"),  # the URL-safe alphabet
    ],
)
def test_parse_data_uri_refuses_what_it_cannot_repair(url: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_data_uri(url)
