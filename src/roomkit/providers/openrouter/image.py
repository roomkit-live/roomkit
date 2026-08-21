"""OpenRouter image provider — draws via OpenRouter's Image API (RFC §25).

Not the OpenAI images path: OpenRouter's image surface is its own endpoint
(``POST /api/v1/images``) with its own request shape — reference images ride
the same request as ``input_references``, geometry may be pixels or a tier,
and the response states the amount billed. One request yields one model's
answer, so ``n`` images are ``n`` concurrent requests: per-model batch caps
vary from 1 to 10 across the aggregated lineup, and concurrent singles are the
one form every routed model accepts.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from typing import Any

from roomkit.providers.ai.base import (
    RETRYABLE_STATUS_CODES,
    AIImagePart,
    ModelInfo,
    ProviderError,
)
from roomkit.providers.image.base import (
    ImageProvider,
    ImageResult,
    parse_data_uri,
    parse_size,
    sniff_mime_type,
    to_data_uri,
)
from roomkit.providers.openrouter.config import OpenRouterImageConfig
from roomkit.providers.openrouter.image_models import MODELS


class OpenRouterImageProvider(ImageProvider):
    """Image provider using OpenRouter's Image API."""

    def __init__(self, config: OpenRouterImageConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for OpenRouterImageProvider. "
                "Install it with: pip install roomkit[openrouter]"
            ) from exc
        self._config = config
        self._status_error = _httpx.HTTPStatusError
        self._transport_error = _httpx.TransportError
        self._http = _httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout,
            headers=self._headers(config),
        )

    @staticmethod
    def _headers(config: OpenRouterImageConfig) -> dict[str, str]:
        """Auth plus OpenRouter's attribution headers, plus caller overrides.

        Attribution mirrors :class:`~roomkit.providers.openrouter.ai.OpenRouterAIProvider`;
        ``default_headers`` layers on top and wins on key collisions.
        """
        headers = {"Authorization": f"Bearer {config.api_key.get_secret_value()}"}
        if config.site_url:
            headers["HTTP-Referer"] = config.site_url
        if config.app_name:
            headers["X-Title"] = config.app_name
        headers.update(config.default_headers or {})
        return headers

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_editing(self) -> bool:
        return True

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline slice of OpenRouter image model slugs.

        A small representative sample — OpenRouter's public
        ``GET /api/v1/images/models`` is the discovery surface for the full,
        always-current set, with each model's live parameter constraints.
        """
        return list(MODELS)

    async def generate(
        self,
        prompt: str,
        *,
        size: str | None = None,
        n: int = 1,
        reference_images: list[AIImagePart] | None = None,
    ) -> list[ImageResult]:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        body = self._build_body(prompt, size, reference_images or [])
        # A task group rather than ``gather``: when one request fails the
        # siblings are cancelled instead of running to completion for a result
        # nobody will read. Each one is a billed image.
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(self._create(body)) for _ in range(n)]
        except BaseExceptionGroup as failures:
            # The caller asked for images and gets the failure that stopped
            # them, not a group wrapper it would have to unpack to find the
            # ProviderError its retry policy reads. Re-chained to its own cause
            # so the exception underneath survives the unwrapping.
            first = failures.exceptions[0]
            raise first from first.__cause__
        return [self._result(task.result()) for task in tasks]

    def _build_body(
        self,
        prompt: str,
        size: str | None,
        reference_images: list[AIImagePart],
    ) -> dict[str, Any]:
        """Assemble one ``/images`` body, reused across the ``n`` requests.

        ``size`` goes through as pixels: OpenRouter's Image API takes an
        explicit ``"WIDTHxHEIGHT"`` and maps or refuses it per routed model,
        so no ratio-and-tier translation is needed here. A size the model
        cannot produce comes back as the vendor's error, never as a silently
        substituted geometry.
        """
        body: dict[str, Any] = {"model": self._config.model, "prompt": prompt}
        if size is not None:
            width, height = parse_size(size)
            body["size"] = f"{width}x{height}"
        if self._config.quality is not None:
            body["quality"] = self._config.quality
        if self._config.background is not None:
            body["background"] = self._config.background
        if self._config.output_format is not None:
            body["output_format"] = self._config.output_format
        if reference_images:
            body["input_references"] = [
                self._reference(part, index) for index, part in enumerate(reference_images)
            ]
        return body

    @staticmethod
    def _reference(part: AIImagePart, index: int) -> dict[str, Any]:
        """Turn a reference image into the content block ``input_references`` takes.

        A remote URL is forwarded as-is — OpenRouter dereferences it, roomkit
        never does. Inline bytes make the round trip through
        :func:`parse_data_uri`: decoding is what proves the payload is valid
        before it reaches the wire, and re-encoding from those bytes is what
        guarantees the URI sent is canonical base64 even when the caller's
        carried line breaks or padding of its own.
        """
        if not part.url.startswith("data:"):
            return {"type": "image_url", "image_url": {"url": part.url}}
        try:
            mime_type, data = parse_data_uri(part.url, fallback_mime=part.mime_type)
        except ValueError as exc:
            raise ValueError(f"reference image {index}: {exc}") from exc
        return {"type": "image_url", "image_url": {"url": to_data_uri(data, mime_type)}}

    async def _create(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST one image request and return the decoded JSON payload."""
        try:
            response = await self._http.post("/images", json=body)
            response.raise_for_status()
            payload = response.json()
        except ProviderError:
            raise
        except self._status_error as exc:
            status = exc.response.status_code
            raise ProviderError(
                f"OpenRouter images API returned {status}: {exc.response.text[:500]}",
                retryable=status in RETRYABLE_STATUS_CODES,
                provider="openrouter",
                status_code=status,
            ) from exc
        except self._transport_error as exc:
            raise ProviderError(str(exc), retryable=True, provider="openrouter") from exc
        except Exception as exc:
            raise ProviderError(str(exc), retryable=False, provider="openrouter") from exc
        if not isinstance(payload, dict):
            raise ProviderError(
                "OpenRouter images API returned a non-object payload",
                retryable=False,
                provider="openrouter",
            )
        return payload

    def _result(self, payload: dict[str, Any]) -> ImageResult:
        """Map one ``/images`` response onto an :class:`ImageResult`.

        Each of the ``n`` requests is its own billed call with its own usage,
        so every result carries its own counters — no first-result convention
        to remember here.
        """
        images = payload.get("data") or []
        if len(images) != 1:
            raise ProviderError(
                f"OpenRouter returned {len(images)} image(s) for a request of 1",
                retryable=False,
                provider="openrouter",
            )
        image = images[0]
        data = image.get("b64_json")
        if not data:
            raise ProviderError(
                "OpenRouter returned an image without inline bytes; the Image API "
                "answers in base64, so an empty payload is a failed generation",
                retryable=False,
                provider="openrouter",
            )
        mime_type = self._mime_type(image, data)
        return ImageResult(
            data=f"data:{mime_type};base64,{data}",
            mime_type=mime_type,
            revised_prompt=image.get("revised_prompt"),
            usage=self._usage(payload),
        )

    @staticmethod
    def _mime_type(image: dict[str, Any], payload: str) -> str:
        """The media type of one returned image.

        OpenRouter declares ``media_type`` per image but omits it when the
        routed model's output is unidentifiable; then the bytes answer for
        themselves.
        """
        declared = image.get("media_type")
        if declared:
            return str(declared)
        try:
            # 32 base64 chars decode the 24 leading bytes every magic number
            # fits in; the full payload is only decoded when a consumer reads it.
            head = base64.b64decode(payload[:32], validate=True)
        except (binascii.Error, ValueError):
            return "image/png"
        return sniff_mime_type(head)

    @staticmethod
    def _usage(payload: dict[str, Any]) -> dict[str, Any]:
        """Map OpenRouter's usage onto the counters RFC §25.5 names — plus the bill.

        ``completion_tokens`` on this endpoint counts the generated image (the
        Image API returns nothing else), so it lands on the image counter, and
        ``prompt_tokens`` is the text input. What has no counter is ``cost``:
        OpenRouter states the amount it billed for the call, in USD, and for
        the mostly per-image-priced lineup behind this API that figure is the
        one authoritative price — the catalog deliberately carries no per-token
        rates to derive it from. It rides under its own key, a report alongside
        the token counters, not one of them.
        """
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return {}
        counters: dict[str, Any] = {}
        if (prompt_tokens := usage.get("prompt_tokens")) is not None:
            counters["input_tokens"] = int(prompt_tokens)
        if (completion_tokens := usage.get("completion_tokens")) is not None:
            counters["output_image_tokens"] = int(completion_tokens)
        if (cost := usage.get("cost")) is not None:
            counters["cost"] = float(cost)
        return counters

    async def close(self) -> None:
        await self._http.aclose()
