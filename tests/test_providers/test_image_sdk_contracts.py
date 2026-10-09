"""Exercise optional real SDK serialization offline, including retry boundaries."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from roomkit.providers.ai.base import AIImagePart
from roomkit.providers.image import ImageGenerationError, ImageOptions
from tests.test_providers.test_gemini_image import _interaction
from tests.test_providers.test_gemini_image import _provider as gemini
from tests.test_providers.test_openai_image import PNG_B64, _response
from tests.test_providers.test_openai_image import _provider as openai


@pytest.mark.parametrize("failure", ["200", "503", "timeout"])
async def test_real_gemini_sdk_does_not_repeat_503(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    pytest.importorskip("google.genai")
    from roomkit.providers.gemini.config import GeminiImageConfig
    from roomkit.providers.gemini.image import GeminiImageProvider

    calls = []

    async def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("interrupted", request=request)
        if failure == "200":
            return httpx.Response(
                200,
                json={
                    "id": "img-ok",
                    "status": "completed",
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [
                                {"type": "image", "data": PNG_B64, "mime_type": "image/png"}
                            ],
                        }
                    ],
                },
            )
        return httpx.Response(503, json={"error": {"code": 503, "message": "unavailable"}})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client_class(**kw, transport=httpx.MockTransport(respond)),
    )
    # Pinned: the default (Nano Banana 2.1) returns JPEG only, and this answers PNG.
    provider = GeminiImageProvider(
        GeminiImageConfig(api_key="test", model="gemini-3.1-flash-image")
    )
    try:
        if failure == "200":
            [result] = await provider.generate("a fox")
            assert result.decoded() == base64.b64decode(PNG_B64)
        else:
            with pytest.raises(ImageGenerationError):
                await provider.generate("a fox")
        assert len(calls) == 1
    finally:
        await provider.close()


async def test_real_openai_sdk_edit_moderation_multipart() -> None:
    sdk = pytest.importorskip("openai")
    calls = []

    async def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": PNG_B64}]})

    provider = openai()
    provider._client = sdk.AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    try:
        [result] = await provider.generate_with_options(
            "a fox",
            reference_images=[AIImagePart(url=f"data:image/png;base64,{PNG_B64}")],
            options=ImageOptions(moderation="low"),
        )
        assert result.decoded() == base64.b64decode(PNG_B64)
        assert b'name="moderation"\r\n\r\nlow' in calls[0].content
        assert result.effective_options["moderation"] == "low"
    finally:
        await provider.close()


@pytest.mark.parametrize("factory", [openai, gemini])
async def test_cancellation_survives_failed_notification(factory) -> None:
    provider = factory()
    entered = asyncio.Event()

    async def blocked(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    if factory is openai:
        provider._client.images.generate = blocked
    else:
        provider._client.aio.interactions.create = blocked

    async def progress(attempt):
        if attempt.status == "unknown":
            raise RuntimeError("database offline")

    task = asyncio.create_task(provider.generate_with_options("a fox", on_progress=progress))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_legacy_geometry_and_gemini_batch_remain_supported() -> None:
    provider = openai(model="gpt-image-1")
    provider._client.images.generate = AsyncMock(return_value=_response())
    await provider.generate("a fox", size=" 1024X1024 ")
    assert provider._client.images.generate.await_args.kwargs["size"] == "1024x1024"
    provider = gemini(model="custom-image")
    provider._client.aio.interactions.create = AsyncMock(return_value=_interaction())
    assert len(await provider.generate("a fox", n=11, size="1024x1024")) == 11


async def test_mini_refuses_fidelity_before_started() -> None:
    provider = openai(model="gpt-image-1-mini")
    callback = AsyncMock()
    with pytest.raises(ValueError, match="input_fidelity"):
        await provider.generate_with_options(
            "a fox",
            options=ImageOptions(input_fidelity="high"),
            reference_images=[AIImagePart(url=f"data:image/png;base64,{PNG_B64}")],
            on_progress=callback,
        )
    callback.assert_not_called()


async def test_stream_network_loss_keeps_request_id_and_unknown_status() -> None:
    class Stream:
        response = SimpleNamespace(headers={"x-request-id": "req-image"})

        async def __aiter__(self):
            yield SimpleNamespace(type="image_generation.partial_image", b64_json=PNG_B64)
            raise httpx.ReadError("connection lost")

        async def close(self):
            pass

    provider = openai()
    provider._client.images.generate = AsyncMock(return_value=Stream())
    with pytest.raises(ImageGenerationError) as caught:
        await provider.generate_with_options(
            "a fox", options=ImageOptions(partial_images=1), on_progress=AsyncMock()
        )
    [attempt] = caught.value.attempts
    assert attempt.status == "unknown"
    assert attempt.provider_request_id == "req-image"
