"""Tests for the Meta Muse Image provider (RFC §25).

Offline: the ``openai`` module is replaced by a stub, so request building —
the tools sent off by default, Meta's fields riding ``extra_body``, the JSON
edits path — response mapping and error translation run without a key.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from roomkit.providers.ai.base import AIImagePart, ProviderError
from roomkit.providers.image import to_data_uri
from roomkit.providers.image.options import ImageModelInfo, ImageOptions
from roomkit.providers.meta.config import MetaImageConfig

WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 24
WEBP_B64 = base64.b64encode(WEBP).decode("ascii")
JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 20).decode("ascii")
_TOOLS_OFF = {"enable_web_search": False, "enable_image_search": False, "enable_shell": False}


class _FakeAPIStatusError(Exception):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeAPIConnectionError(Exception):
    pass


def _provider(**overrides: Any) -> Any:
    mod = MagicMock()
    mod.APIStatusError = _FakeAPIStatusError
    mod.APIConnectionError = _FakeAPIConnectionError
    with patch.dict("sys.modules", {"openai": mod}):
        from roomkit.providers.meta.image import MetaImageProvider

        return MetaImageProvider(MetaImageConfig(api_key=SecretStr("k"), **overrides))


def _response(count: int = 1, *, b64: str = WEBP_B64, usage: Any = None) -> SimpleNamespace:
    """The live response's shape (2026-09-27): no mime type, no revised prompt."""
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=b64, revised_prompt=None) for _ in range(count)],
        usage=usage,
        output_format="webp",
    )


class TestGeneration:
    async def test_tools_are_sent_off_and_meta_fields_ride_extra_body(self) -> None:
        provider = _provider()
        provider._client.images.generate = AsyncMock(return_value=_response())

        [result] = await provider.generate("un phare breton")

        # Meta enables all three tools when a request says nothing.
        assert provider._client.images.generate.await_args.kwargs == {
            "model": "muse-image-1.0",
            "prompt": "un phare breton",
            "n": 1,
            "response_format": "b64_json",
            "extra_body": {"tool_enablement": _TOOLS_OFF},
        }
        assert result.mime_type == "image/webp"  # read off the bytes: WebP is the default
        assert result.decoded() == WEBP

    async def test_configured_defaults_and_call_options(self) -> None:
        provider = _provider(tools=["image_search"], reasoning_strength="high", moderation="low")
        provider._client.images.generate = AsyncMock(return_value=_response(b64=JPEG_B64))

        [result] = await provider.generate_with_options(
            "un phare",
            size="1536x1024",
            options=ImageOptions(
                output_format="jpeg", thinking_level="minimal", search_types=["web_search"]
            ),
        )

        kwargs = provider._client.images.generate.await_args.kwargs
        assert kwargs["size"] == "1536x1024"
        assert kwargs["moderation"] == "low"
        assert kwargs["output_format"] == "jpeg"
        assert kwargs["extra_body"] == {
            "tool_enablement": {
                "enable_web_search": True,
                "enable_image_search": True,
                "enable_shell": False,
            },
            "reasoning_strength": "low",  # the call's "minimal" wins over the config
        }
        assert result.mime_type == "image/jpeg"

    @pytest.mark.parametrize(
        ("size", "sent"), [("auto", "auto"), ("AUTO", "auto"), ("1000x333", "1000x333")]
    )
    async def test_size_is_a_ratio_or_auto(self, size: str, sent: str) -> None:
        provider = _provider()
        provider._client.images.generate = AsyncMock(return_value=_response())
        await provider.generate("x", size=size)
        assert provider._client.images.generate.await_args.kwargs["size"] == sent

    async def test_malformed_size_is_refused_before_the_call(self) -> None:
        provider = _provider()
        provider._client.images.generate = AsyncMock()
        with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
            await provider.generate("x", size="big")
        provider._client.images.generate.assert_not_awaited()

    @pytest.mark.parametrize(
        "options",
        [ImageOptions(quality="high"), ImageOptions(background="transparent")],
        ids=["quality", "background"],
    )
    async def test_unsupported_options_are_refused(self, options: ImageOptions) -> None:
        provider = _provider()
        provider._client.images.generate = AsyncMock()
        with pytest.raises(ValueError, match="Unsupported image options"):
            await provider.generate_with_options("x", options=options)
        provider._client.images.generate.assert_not_awaited()

    async def test_mask_and_progress_are_refused(self) -> None:
        provider = _provider()
        part = AIImagePart(url=to_data_uri(WEBP, "image/webp"), mime_type="image/webp")
        with pytest.raises(ValueError, match="mask"):
            await provider.generate_with_options("x", reference_images=[part], mask=part)
        with pytest.raises(ValueError, match="previews"):
            await provider.generate_with_options("x", on_progress=AsyncMock())

    async def test_n_images_and_usage_on_the_first_only(self) -> None:
        usage = SimpleNamespace(
            input_tokens=8012,
            output_tokens=1875,
            input_tokens_details=SimpleNamespace(text_tokens=12, image_tokens=8000),
        )
        provider = _provider()
        provider._client.images.generate = AsyncMock(return_value=_response(2, usage=usage))

        results = await provider.generate("x", n=2)

        assert [r.usage for r in results] == [
            {"input_tokens": 12, "input_image_tokens": 8000, "output_image_tokens": 1875},
            {},
        ]

    async def test_a_short_answer_is_an_error(self) -> None:
        provider = _provider()
        provider._client.images.generate = AsyncMock(return_value=_response(1))
        with pytest.raises(ProviderError, match="1 image"):
            await provider.generate("x", n=2)


class TestEditing:
    async def test_references_are_posted_as_json_image_urls(self) -> None:
        provider = _provider()
        provider._client.post = AsyncMock(return_value=_response())
        inline = AIImagePart(url=to_data_uri(WEBP, "image/webp"), mime_type="image/webp")
        remote = AIImagePart(url="https://example.com/a.png", mime_type="image/png")

        await provider.generate("fusionne", reference_images=[inline, remote])

        path = provider._client.post.await_args.args[0]
        body = provider._client.post.await_args.kwargs["body"]
        assert path == "/images/edits"
        assert body["images"] == [
            {"image_url": to_data_uri(WEBP, "image/webp")},
            {"image_url": "https://example.com/a.png"},
        ]
        assert body["tool_enablement"] == _TOOLS_OFF

    async def test_a_corrupt_reference_is_the_callers_error(self) -> None:
        provider = _provider()
        provider._client.post = AsyncMock()
        bad = AIImagePart(url="data:image/png;base64,!!!", mime_type="image/png")
        with pytest.raises(ValueError, match="reference image 0"):
            await provider.generate("x", reference_images=[bad])
        provider._client.post.assert_not_awaited()


class TestErrors:
    @pytest.mark.parametrize(
        ("status", "retryable"),
        [(402, False), (408, True), (429, True), (503, True), (500, False), (504, False)],
    )
    async def test_status_errors(self, status: int, retryable: bool) -> None:
        provider = _provider()
        provider._client.images.generate = AsyncMock(
            side_effect=_FakeAPIStatusError("billing_not_configured", status_code=status)
        )
        with pytest.raises(ProviderError) as info:
            await provider.generate("x")
        assert info.value.status_code == status
        assert info.value.retryable is retryable

    async def test_connection_errors_are_retryable(self) -> None:
        provider = _provider()
        provider._client.images.generate = AsyncMock(side_effect=_FakeAPIConnectionError("down"))
        with pytest.raises(ProviderError) as info:
            await provider.generate("x")
        assert info.value.retryable is True


def test_a_key_no_header_can_carry_is_refused_without_echoing_it() -> None:
    with pytest.raises(ValueError, match="non-ASCII") as info:
        MetaImageConfig(api_key=SecretStr("sk-secret…"))
    assert "sk-secret" not in str(info.value)


def test_catalog_describes_muse_image() -> None:
    provider = _provider()
    entry = provider.catalog_entry()
    assert isinstance(entry, ImageModelInfo)
    assert entry.image.formats == ["png", "jpeg", "webp"]
    assert provider.supports_editing is True
