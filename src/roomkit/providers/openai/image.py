"""OpenAI image provider — draws via the OpenAI Images API (RFC §25)."""

from __future__ import annotations

import asyncio
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal

from roomkit.providers.ai.base import (
    AIImagePart,
    ModelInfo,
    ProviderError,
)
from roomkit.providers.ai.openai_dialect import status_error_text
from roomkit.providers.image.base import (
    ImageAttempt,
    ImageGenerationError,
    ImageProgressCallback,
    ImageProvider,
    ImageResult,
    never_sent,
    notify_image_progress,
    parse_data_uri,
    parse_size,
)
from roomkit.providers.image.options import ImageModelInfo, ImageOptions, plain_metadata
from roomkit.providers.image.usage import openai_image_usage
from roomkit.providers.openai.config import OpenAIImageConfig
from roomkit.providers.openai.image_models import MODELS
from roomkit.providers.openai.image_stream import consume_image_stream
from roomkit.providers.utils import import_openai, openai_http_client, openai_timeout

if TYPE_CHECKING:
    import httpx2

_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


class OpenAIImageProvider(ImageProvider):
    """Image provider using the OpenAI Images API.

    ``generate`` calls ``images.generate``; a call carrying reference images
    calls ``images.edit`` instead. That split is OpenAI's, not the caller's —
    RFC §25.4 requires the provider to absorb it.
    """

    def __init__(
        self, config: OpenAIImageConfig, *, transport: httpx2.AsyncBaseTransport | None = None
    ) -> None:
        """*transport* carries every request, as for
        :class:`~roomkit.providers.openai.ai.OpenAIAIProvider`: the seam for a
        policy that judges the address dialled when users name ``base_url``."""
        _openai = import_openai("OpenAIImageProvider", "openai")
        self._config = config
        # The client openai 3.x runs on, installed with it: its errors are read.
        self._httpx2 = import_module("httpx2")
        # No SDK retry: a paid generation is never retried automatically (RFC §25.2).
        self._sdk_retries = 0
        self._api_status_error = _openai.APIStatusError
        self._api_connection_error = (_openai.APIConnectionError, self._httpx2.TransportError)
        self._client = _openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=openai_timeout(_openai, config),
            max_retries=self._sdk_retries,
            default_headers=config.default_headers,
            http_client=openai_http_client(_openai, transport),
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
        return await self.generate_with_options(
            prompt, size=size, n=n, reference_images=reference_images
        )

    async def generate_with_options(
        self,
        prompt: str,
        *,
        size: str | None = None,
        n: int = 1,
        reference_images: list[AIImagePart] | None = None,
        options: ImageOptions | None = None,
        mask: AIImagePart | None = None,
        on_progress: ImageProgressCallback | None = None,
    ) -> list[ImageResult]:
        size = self._validated_size(size) if size is not None else None
        references = list(reference_images or [])
        defaults = {
            key: getattr(self._config, key)
            for key in ("quality", "background", "output_format")
            if getattr(self._config, key) is not None
        }
        options = ImageOptions.model_validate(
            {**defaults, **(options or ImageOptions()).model_dump(exclude_none=True)}
        )
        entry = self.catalog_entry()
        if isinstance(entry, ImageModelInfo):
            entry.image.validate_request(
                options, size=size, n=n, references=len(references), mask=mask is not None
            )
        elif options.model_dump(exclude_none=True) or mask:
            # User-named Azure deployments retain their configured controls;
            # new controls cannot be advertised without a known model contract.
            unknown = set(options.model_dump(exclude_none=True)) - set(defaults)
            if unknown or mask:
                raise ValueError("Advanced controls require a model with known image capabilities")
        if not 1 <= n <= 10:
            raise ValueError("n must be at least 1 and at most 10")
        if options.input_fidelity and not references:
            raise ValueError("input_fidelity requires reference images")
        if options.partial_images is not None and not on_progress:
            raise ValueError("Streaming previews require on_progress")
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "n": n,
            **options.model_dump(exclude_none=True),
        }
        if size is not None:
            kwargs["size"] = self._validated_size(size)
        if references:
            kwargs["image"] = [
                self._as_upload(part, index) for index, part in enumerate(references)
            ]
        if mask:
            kwargs["mask"] = self._as_upload(mask, 0)
            if kwargs["mask"][2] != "image/png":
                raise ValueError("A mask must be a PNG image")
        attempt = ImageAttempt(
            effective_options={
                key: value
                for key, value in kwargs.items()
                if key not in {"prompt", "image", "mask"}
            }
        )
        if references and "moderation" in kwargs:
            kwargs["extra_body"] = {"moderation": kwargs.pop("moderation")}
        if on_progress:
            await notify_image_progress(on_progress, attempt, provider=self._provider_name)
        try:
            method = self._client.images.edit if references else self._client.images.generate
            if options.partial_images is not None:
                response = await consume_image_stream(method, kwargs, attempt, on_progress)
                attempt.results = []
            else:
                response = await method(**kwargs)
            attempt.provider_request_id = (
                getattr(response, "_request_id", None) or attempt.provider_request_id
            )
            attempt.usage = self._usage(response)
            attempt.raw_usage = plain_metadata(getattr(response, "usage", None)) or {}
            attempt.effective_options.update(
                {
                    key: value
                    for key in ("size", "quality", "background", "output_format")
                    if (value := getattr(response, key, None)) is not None
                }
            )
            attempt.results = self._results(response, n, attempt)
            attempt.status = "succeeded"
        except asyncio.CancelledError as cancelled:
            attempt.status = "unknown"
            attempt.error = "Cancelled locally; the provider may still bill this request"
            try:
                await notify_image_progress(on_progress, attempt, provider=self._provider_name)
            except Exception as exc:
                cancelled.add_note(f"Failed to report unknown image outcome: {exc}")
            raise
        except Exception as exc:
            attempt.status = self._failure_status(exc)
            attempt.error = self._error_text(exc)
            attempt.status_code = getattr(exc, "status_code", None)
            if on_progress:
                await notify_image_progress(on_progress, attempt, provider=self._provider_name)
            raise ImageGenerationError(
                attempt.error, provider=self._provider_name, attempts=[attempt]
            ) from exc
        if on_progress:
            await notify_image_progress(on_progress, attempt, provider=self._provider_name)
        return attempt.results

    def _error_text(self, exc: Exception) -> str:
        """What *exc* says: for a status error, the whole body the server
        answered with, on every SDK release (:func:`status_error_text`)."""
        return status_error_text(exc) if isinstance(exc, self._api_status_error) else str(exc)

    def _failure_status(self, exc: Exception) -> Literal["failed", "unknown"]:
        """``unknown`` when the vendor may have run, and billed, the request: a
        transport failure once it left, or any failure after an SDK retry,
        status included (an earlier attempt may have left). One that never
        left (a connect failure, a host's policy refusing the address as one)
        ``failed`` (RFC §25.2)."""
        if self._sdk_retries > 0:
            return "unknown"
        if not isinstance(exc, self._api_connection_error):
            return "failed"
        return "failed" if never_sent(exc, self._httpx2, retries=0) else "unknown"

    @staticmethod
    def _validated_size(size: str) -> str:
        """Normalize pixel geometry before model-specific capability validation."""
        if size == "auto":
            return size
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

    def _results(
        self, response: Any, expected: int, attempt: ImageAttempt | None = None
    ) -> list[ImageResult]:
        """Map an ``ImagesResponse`` onto :class:`ImageResult` objects."""
        images = list(getattr(response, "data", None) or [])
        mime_type = self._response_mime_type(response)
        if attempt and not getattr(response, "output_format", None):
            mime_type = "image/" + str(attempt.effective_options.get("output_format", "png"))
        # The usage counters describe the whole call, not one image; splitting
        # them across n results would invent per-image numbers the vendor never
        # reported, so they ride the first result only and the rest report none.
        usage = self._usage(response)
        results: list[ImageResult] = [] if attempt is None else attempt.results
        errors: list[str] = []
        for index, image in enumerate(images):
            payload = getattr(image, "b64_json", None)
            if not payload:
                errors.append(f"{self._provider_name} returned image {index} without inline bytes")
                continue
            results.append(
                ImageResult(
                    data=f"data:{mime_type};base64,{payload}",
                    mime_type=mime_type,
                    revised_prompt=getattr(image, "revised_prompt", None),
                    usage=usage if not results else {},
                    attempt_id=attempt.id if attempt else None,
                    provider_request_id=attempt.provider_request_id if attempt else None,
                    raw_usage=attempt.raw_usage if attempt and not results else {},
                    effective_options=attempt.effective_options if attempt else {},
                )
            )
        if errors:
            raise ProviderError("; ".join(errors), retryable=False, provider=self._provider_name)
        if len(images) != expected:
            raise ProviderError(
                f"{self._provider_name} returned {len(images)} images for a request of {expected}",
                retryable=False,
                provider=self._provider_name,
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
        return openai_image_usage(getattr(response, "usage", None))

    async def close(self) -> None:
        await self._client.close()
