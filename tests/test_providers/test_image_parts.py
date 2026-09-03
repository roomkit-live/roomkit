"""The one reading of an image part, in the three shapes the AI providers need."""

from __future__ import annotations

import pytest

from roomkit.providers.ai import (
    AIImagePart,
    ProviderError,
    image_part_base64,
    image_part_payload,
    image_part_uri,
)


def test_payload_takes_the_media_type_from_the_part_when_the_header_has_none() -> None:
    part = AIImagePart(url="data:;base64,QUJDMTIz", mime_type="image/png")
    assert image_part_payload(part, provider="x") == ("image/png", b"ABC123")


def test_base64_is_canonical_whatever_the_uri_carried() -> None:
    part = AIImagePart(url="data:image/png;base64,QUJD\nMTI", mime_type=None)
    assert image_part_base64(part, provider="x") == ("image/png", "QUJDMTI=")


def test_uri_is_rebuilt_for_a_data_uri_and_untouched_for_a_url() -> None:
    part = AIImagePart(url="data:;base64,QUJD\nMTIz", mime_type="image/png")
    assert image_part_uri(part, provider="x") == "data:image/png;base64,QUJDMTIz"
    assert image_part_uri(AIImagePart(url="https://example.com/a.png"), provider="x") == (
        "https://example.com/a.png"
    )


@pytest.mark.parametrize("read", [image_part_payload, image_part_base64, image_part_uri])
def test_a_malformed_uri_is_a_non_retryable_provider_error_naming_the_cause(read) -> None:
    with pytest.raises(ProviderError, match="invalid image part: .*not valid base64") as excinfo:
        read(AIImagePart(url="data:image/png;base64,not*base64"), provider="acme")
    assert excinfo.value.retryable is False
    assert excinfo.value.provider == "acme"
