"""OpenAI image provider — draws via the OpenAI Images API (RFC §25)."""

from __future__ import annotations

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
)
from roomkit.providers.openai.config import OpenAIImageConfig
from roomkit.providers.openai.image_models import MODELS
from roomkit.providers.utils import http_timeout

_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


class OpenAIImageProvider(ImageProvider):
    """Image provider using the OpenAI Images API.

    ``generate`` calls ``images.generate``; a call carrying reference images
    calls ``images.edit`` instead. That split is OpenAI's, not the caller's —
    RFC §25.4 requires the provider to absorb it.
    """

    def __init__(self, config: OpenAIImageConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIImageProvider. "
                "Install it with: pip install roomkit[openai]"
            ) from exc
        self._config = config
        self._api_status_error = _openai.APIStatusError
        self._api_connection_error = _openai.APIConnectionError
        self._client = _openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=http_timeout(config),
            max_retries=config.max_retries,
            default_headers=config.default_headers,
        )

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "openai"

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_editing(self) -> bool:
        return True

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of OpenAI image models."""
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
        references = list(reference_images or [])
        kwargs: dict[str, Any] = {"model": self._config.model, "prompt": prompt, "n": n}
        if size is not None:
            kwargs["size"] = self._validated_size(size)
        if self._config.quality is not None:
            kwargs["quality"] = self._config.quality
        if self._config.background is not None:
            kwargs["background"] = self._config.background
        if self._config.output_format is not None:
            kwargs["output_format"] = self._config.output_format
        # Built before the try: a malformed reference is the caller's error and
        # must stay a ValueError, not be relabelled as a provider failure by the
        # catch-all below — and not be retried as one either.
        if references:
            kwargs["image"] = [
                self._as_upload(part, index) for index, part in enumerate(references)
            ]

        try:
            if references:
                response = await self._client.images.edit(**kwargs)
            else:
                response = await self._client.images.generate(**kwargs)
        except ProviderError:
            raise
        except self._api_connection_error as exc:
            raise ProviderError(str(exc), retryable=True, provider=self._provider_name) from exc
        except self._api_status_error as exc:
            raise ProviderError(
                str(exc),
                retryable=exc.status_code in RETRYABLE_STATUS_CODES,
                provider=self._provider_name,
                status_code=exc.status_code,
            ) from exc
        except Exception as exc:
            raise ProviderError(str(exc), retryable=False, provider=self._provider_name) from exc

        return self._results(response, n)

    @staticmethod
    def _validated_size(size: str) -> str:
        """Normalize a ``"WIDTHxHEIGHT"`` request and send it as-is.

        This used to refuse anything off a fixed list, and the list went
        stale: ``gpt-image-2`` takes near-arbitrary geometry (edges in
        multiples of 16, long edge up to 3840, ratio up to 3:1 — the SDK types
        ``size`` as an open string) while the ``gpt-image-1`` series keeps a
        fixed menu, so no one list is right across the lineup this provider
        configures. The vendor judges instead; its rejection still raises
        rather than substituting another geometry (RFC §25.2).
        """
        width, height = parse_size(size)
        return f"{width}x{height}"

    @staticmethod
    def _as_upload(part: AIImagePart, index: int) -> tuple[str, bytes, str]:
        """Turn an image part into the ``(filename, bytes, mime)`` tuple the SDK uploads.

        ``images.edit`` is a multipart endpoint: it takes file content, not a
        URL, so a reference that is not inline bytes cannot be forwarded.
        """
        try:
            mime_type, data = parse_data_uri(part.url, fallback_mime=part.mime_type)
        except ValueError as exc:
            raise ValueError(
                f"reference image {index}: {exc}. OpenAI image editing uploads file "
                "content, so a reference must carry inline bytes as a data: URI."
            ) from exc
        return (f"reference-{index}.{_EXTENSIONS.get(mime_type, 'png')}", data, mime_type)

    def _results(self, response: Any, expected: int) -> list[ImageResult]:
        """Map an ``ImagesResponse`` onto :class:`ImageResult` objects."""
        images = list(getattr(response, "data", None) or [])
        if len(images) != expected:
            raise ProviderError(
                f"{self._provider_name} returned {len(images)} image(s) "
                f"for a request of {expected}",
                retryable=False,
                provider=self._provider_name,
            )
        mime_type = self._response_mime_type(response)
        # The usage counters describe the whole call, not one image; splitting
        # them across n results would invent per-image numbers the vendor never
        # reported, so they ride the first result only and the rest report none.
        usage = self._usage(response)
        results: list[ImageResult] = []
        for index, image in enumerate(images):
            payload = getattr(image, "b64_json", None)
            if not payload:
                raise ProviderError(
                    f"{self._provider_name} returned image {index} without inline bytes. The GPT "
                    "image models on this endpoint always answer in base64; a model that "
                    "answers with an expiring URL instead is not one roomkit supports here.",
                    retryable=False,
                    provider=self._provider_name,
                )
            results.append(
                ImageResult(
                    data=f"data:{mime_type};base64,{payload}",
                    mime_type=mime_type,
                    revised_prompt=getattr(image, "revised_prompt", None),
                    usage=usage if index == 0 else {},
                )
            )
        return results

    def _response_mime_type(self, response: Any) -> str:
        """The media type of the returned bytes, as the response reports it."""
        output_format = getattr(response, "output_format", None) or self._config.output_format
        return {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}.get(
            output_format or "png", "image/png"
        )

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        """Split OpenAI's usage into the disjoint counters RFC §25.5 requires.

        ``input_tokens`` is the total, with ``input_tokens_details.image_tokens``
        a subset of it — so the image share is subtracted before the text
        counter is reported, and summing the counters bills each token once.
        On the generation endpoint every output token is an image token.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        details = getattr(usage, "input_tokens_details", None)
        input_image_tokens = int(getattr(details, "image_tokens", 0) or 0)
        return {
            "input_tokens": max(input_tokens - input_image_tokens, 0),
            "input_image_tokens": input_image_tokens,
            "output_tokens": 0,
            "output_image_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }

    async def close(self) -> None:
        await self._client.close()
