"""Meta Muse Image provider — draws and edits via the Meta Model API (RFC §25).

``muse-image-1.0`` answers on the OpenAI-shaped ``/v1/images/generations``
and ``/v1/images/edits``, with three Meta fields on top, all sent explicitly:

* ``tool_enablement`` — the generator may search the web, fetch reference
  images and run code while it draws, and Meta turns all three on when the
  request says nothing. RoomKit sends them off unless
  :attr:`MetaImageConfig.tools` (or a call's ``search_types``) asks, so a
  prompt does not leave for the web by default.
* ``reasoning_strength`` — ``high`` (several refinement passes) or ``low``;
  :class:`~roomkit.providers.image.options.ImageOptions.thinking_level`
  ``minimal`` maps to ``low``.
* ``moderation``.

Edits post JSON (``images: [{"image_url": ...}]``) through the same client, as
the xAI provider does: the OpenAI SDK's ``images.edit`` uploads multipart.

``size`` sets the aspect ratio, not the pixels: the service draws at its own
resolution (``1536x1024`` came back 1920x1280, measured 2026-09-27). The
default format is WebP.
"""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import (
    AIImagePart,
    ModelInfo,
    ProviderError,
)
from roomkit.providers.image.base import (
    PAID_GENERATION_RETRYABLE,
    ImageProvider,
    ImageResult,
    never_sent,
    parse_data_uri,
    parse_size,
    payload_mime_type,
    to_data_uri,
)
from roomkit.providers.image.options import ImageModelInfo, ImageOptions
from roomkit.providers.meta.config import MetaImageConfig
from roomkit.providers.meta.image_models import MODELS
from roomkit.providers.utils import http_timeout

_REASONING = {"minimal": "low", "high": "high"}


class MetaImageProvider(ImageProvider):
    """Image provider on Meta's Muse Image."""

    def __init__(self, config: MetaImageConfig) -> None:
        try:
            import httpx as _httpx
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for MetaImageProvider. "
                "Install it with: pip install roomkit[meta]"
            ) from exc
        self._config = config
        self._api_status_error = _openai.APIStatusError
        self._api_connection_error = _openai.APIConnectionError
        self._httpx = _httpx
        self._images_response_cls = _openai.types.ImagesResponse
        self._client = _openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=http_timeout(config),
            max_retries=config.max_retries,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_editing(self) -> bool:
        return True

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Meta's image models."""
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
        on_progress: Any = None,
    ) -> list[ImageResult]:
        """Draw ``n`` images, or edit ``reference_images``.

        Supported options: ``output_format``, ``moderation``,
        ``thinking_level`` (``minimal`` → ``low``, ``high``) and
        ``search_types`` (``web_search``, ``image_search``, added to the
        configured tools for this call). Anything else is refused before the
        billable call, as are a mask and progress callbacks, which Meta does
        not document.
        """
        if mask is not None:
            raise ValueError("Muse Image does not take a mask")
        if on_progress is not None:
            raise ValueError("MetaImageProvider does not stream previews")
        options = options or ImageOptions()
        references = list(reference_images or [])
        entry = self.catalog_entry()
        if isinstance(entry, ImageModelInfo):
            entry.image.validate_request(
                options, size=None, n=n, references=len(references), mask=False
            )
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        body = self._build_body(prompt, size, n, options)
        if references:
            body["images"] = [self._reference(part, i) for i, part in enumerate(references)]
        response = await self._call(body, edit=bool(references))
        return self._results(response, n)

    def _build_body(
        self, prompt: str, size: str | None, n: int, options: ImageOptions
    ) -> dict[str, Any]:
        """The request fields, the call's options over the configured defaults.

        ``b64_json`` is asked for explicitly: RFC §25.3 wants bytes that
        outlive the call.
        """
        config = self._config
        tools = set(config.tools) | set(options.search_types or [])
        body: dict[str, Any] = {
            "model": config.model,
            "prompt": prompt,
            "n": n,
            "response_format": "b64_json",
            "tool_enablement": {
                "enable_web_search": "web_search" in tools,
                "enable_image_search": "image_search" in tools,
                "enable_shell": "shell" in tools,
            },
        }
        if size is not None:
            body["size"] = self._size(size)
        reasoning = (
            _REASONING[options.thinking_level]
            if options.thinking_level
            else config.reasoning_strength
        )
        for key, value in (
            ("reasoning_strength", reasoning),
            ("moderation", options.moderation or config.moderation),
            ("output_format", options.output_format or config.output_format),
        ):
            if value is not None:
                body[key] = value
        return body

    @staticmethod
    def _size(size: str) -> str:
        """``"auto"`` or a well-formed ``"WIDTHxHEIGHT"``, which Meta reads as a ratio."""
        if size.strip().lower() == "auto":
            return "auto"
        width, height = parse_size(size)
        return f"{width}x{height}"

    async def _call(self, body: dict[str, Any], *, edit: bool) -> Any:
        """One billable request, with the SDK's errors as :class:`ProviderError`."""
        try:
            if edit:
                return await self._client.post(
                    "/images/edits", body=body, cast_to=self._images_response_cls
                )
            generation = dict(body)
            extra = {
                key: generation.pop(key)
                for key in ("tool_enablement", "reasoning_strength")
                if key in generation
            }
            return await self._client.images.generate(**generation, extra_body=extra)
        except self._api_connection_error as exc:
            raise ProviderError(
                str(exc), retryable=never_sent(exc, self._httpx), provider="meta"
            ) from exc
        except self._api_status_error as exc:
            raise ProviderError(
                str(exc),
                retryable=exc.status_code in PAID_GENERATION_RETRYABLE,
                provider="meta",
                status_code=exc.status_code,
            ) from exc

    @staticmethod
    def _reference(part: AIImagePart, index: int) -> dict[str, str]:
        """A reference image as the ``{"image_url": ...}`` item edits take.

        A remote URL is forwarded as-is — Meta dereferences it, RoomKit never
        does. Inline bytes are decoded and re-encoded, which proves the
        payload valid before it reaches the wire.
        """
        if not part.url.startswith("data:"):
            return {"image_url": part.url}
        try:
            mime_type, data = parse_data_uri(part.url, fallback_mime=part.mime_type)
        except ValueError as exc:
            raise ValueError(f"reference image {index}: {exc}") from exc
        return {"image_url": to_data_uri(data, mime_type)}

    def _results(self, response: Any, expected: int) -> list[ImageResult]:
        """Map an images response onto :class:`ImageResult` objects."""
        images = list(getattr(response, "data", None) or [])
        if len(images) != expected:
            raise ProviderError(
                f"Meta returned {len(images)} image(s) for a request of {expected}",
                retryable=False,
                provider="meta",
            )
        # The usage describes the whole call; it rides the first result only,
        # rather than inventing a per-image split the vendor never reported.
        usage = self._usage(response)
        results: list[ImageResult] = []
        for index, image in enumerate(images):
            payload = getattr(image, "b64_json", None)
            if not payload:
                raise ProviderError(
                    f"Meta returned image {index} without inline bytes despite the request "
                    "naming b64_json (RFC §25.3)",
                    retryable=False,
                    provider="meta",
                )
            mime_type = self._mime_type(payload)
            results.append(
                ImageResult(
                    data=f"data:{mime_type};base64,{payload}",
                    mime_type=mime_type,
                    revised_prompt=getattr(image, "revised_prompt", None),
                    usage=usage if index == 0 else {},
                )
            )
        return results

    @staticmethod
    def _mime_type(payload: str) -> str:
        """The media type read off the bytes: the response declares none per image."""
        return payload_mime_type(None, payload, fallback="image/webp")

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        """Meta's token counters, mapped to RFC §25.5's names.

        ``output_tokens`` counts the generated image. ``input_tokens`` covers
        the prompt and, on an edit, the references; the text/image breakdown
        is used when the response carries one. Billing is per image regardless.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        counters: dict[str, int] = {}
        details = getattr(usage, "input_tokens_details", None)
        text = getattr(details, "text_tokens", None) if details is not None else None
        images = getattr(details, "image_tokens", None) if details is not None else None
        if text is not None:
            counters["input_tokens"] = int(text)
            if images is not None:
                counters["input_image_tokens"] = int(images)
        elif (input_tokens := getattr(usage, "input_tokens", None)) is not None:
            counters["input_tokens"] = int(input_tokens)
        if (output_tokens := getattr(usage, "output_tokens", None)) is not None:
            counters["output_image_tokens"] = int(output_tokens)
        return counters

    async def close(self) -> None:
        await self._client.close()
