"""Google Gemini image provider — draws via the Interactions API (RFC §25).

Not ``generateContent``. Google's image-generation surface moved to the
Interactions API (``client.interactions.create`` with an image
``response_format``), and that is what the models docs describe and the
``google-genai`` SDK ships resources for. One interaction yields one image, so
``n`` images are ``n`` concurrent interactions — the vendor bills per image
either way.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from roomkit.providers.ai.base import AIImagePart, ModelInfo, ProviderError
from roomkit.providers.gemini.config import GeminiImageConfig
from roomkit.providers.gemini.errors import wrap_gemini_error
from roomkit.providers.gemini.image_geometry import ASPECT_RATIOS, TIERS, size_geometry
from roomkit.providers.gemini.image_models import MODELS
from roomkit.providers.gemini.sdk import build_genai_client, close_genai_client
from roomkit.providers.image.base import (
    ImageAttempt,
    ImageGenerationError,
    ImageProgressCallback,
    ImageProvider,
    ImageResult,
    notify_image_progress,
    parse_data_uri,
    payload_mime_type,
)
from roomkit.providers.image.options import (
    ImageCapabilities,
    ImageModelInfo,
    ImageOptions,
    plain_metadata,
)
from roomkit.providers.image.usage import gemini_image_usage


class GeminiImageProvider(ImageProvider):
    """Image provider using the Gemini Interactions API."""

    def __init__(self, config: GeminiImageConfig) -> None:
        self._config = config
        # The client carries the connect/read split; see ``build_genai_client``
        # for why it cannot go on the request.
        built = build_genai_client(
            config,
            provider="GeminiImageProvider",
            api_key=config.api_key.get_secret_value(),
            disable_retries=True,
        )
        self._client, self._http = built.client, built.http

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_editing(self) -> bool:
        return True

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Gemini image models."""
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
        if n < 1:
            raise ValueError("n must be at least 1")
        if mask:
            raise ValueError("Gemini does not support explicit masks")
        references = reference_images or []
        resolved = self._call_options(size, n, len(references), options or ImageOptions())
        request = self._build_request(prompt, references, resolved)
        # Each call settles independently. A rejected sibling does not erase a
        # successful billed image or cancel a call whose outcome is still unknown.
        outcomes = await asyncio.gather(
            *[self._attempt(request, index, on_progress) for index in range(n)],
            return_exceptions=True,
        )
        attempts: list[ImageAttempt] = []
        failures: list[BaseException] = []
        for outcome in outcomes:
            if isinstance(outcome, ImageAttempt):
                attempts.append(outcome)
            elif isinstance(outcome, ImageGenerationError):
                attempts.extend(outcome.attempts)
                failures.append(outcome)
            elif isinstance(outcome, BaseException):
                failures.append(outcome)
        if failures:
            first = failures[0]
            raise ImageGenerationError(str(first), provider="gemini", attempts=attempts) from (
                first.__cause__ or first
            )
        return [result for attempt in attempts for result in attempt.results]

    async def _attempt(
        self, request: dict[str, Any], index: int, on_progress: ImageProgressCallback | None
    ) -> ImageAttempt:
        attempt = ImageAttempt(
            index=index,
            effective_options={key: value for key, value in request.items() if key != "input"},
        )
        if on_progress:
            await notify_image_progress(on_progress, attempt, provider="gemini")
        interaction = None
        failure: Exception | None = None
        try:
            interaction = await self._create(request)
            attempt.provider_request_id = getattr(interaction, "id", None)
            attempt.usage = self._usage(interaction)
            attempt.raw_usage = plain_metadata(getattr(interaction, "usage", None)) or {}
            attempt.metadata = {
                "steps": plain_metadata(getattr(interaction, "steps", [])),
                "output_text": getattr(interaction, "output_text", None),
            }
            result = self._result(interaction)
            attempt.results = [
                result.model_copy(
                    update={
                        "attempt_id": attempt.id,
                        "provider_request_id": attempt.provider_request_id,
                        "raw_usage": attempt.raw_usage,
                        "effective_options": attempt.effective_options,
                        "metadata": attempt.metadata,
                    }
                )
            ]
            attempt.status = "succeeded"
        except asyncio.CancelledError as cancelled:
            attempt.status = "unknown"
            attempt.error = "Cancelled locally; the provider may still bill this request"
            try:
                await notify_image_progress(on_progress, attempt, provider="gemini")
            except Exception as exc:
                cancelled.add_note(f"Failed to report unknown image outcome: {exc}")
            raise
        except Exception as exc:
            failure = exc
            attempt.status = (
                "failed"
                if interaction is not None or getattr(exc, "status_code", None)
                else "unknown"
            )
            attempt.error = str(exc)
            attempt.status_code = getattr(exc, "status_code", None)
        if on_progress:
            await notify_image_progress(on_progress, attempt, provider="gemini")
        if failure:
            raise ImageGenerationError(
                str(failure), provider="gemini", attempts=[attempt]
            ) from failure
        return attempt

    def _call_options(
        self, size: str | None, n: int, references: int, options: ImageOptions
    ) -> ImageOptions:
        """The options a call runs with: the config's defaults under the caller's,
        checked against the model's capabilities before any billable request."""
        if size and (options.aspect_ratio or options.image_size):
            raise ValueError("Use size or aspect_ratio/image_size, not both")
        entry = self.catalog_entry()
        image = entry.image if isinstance(entry, ImageModelInfo) else None
        defaults = self._inherited_options(size, image)
        resolved = ImageOptions.model_validate(
            {**defaults, **options.model_dump(exclude_none=True)}
        )
        if image is not None:
            image.validate_request(resolved, size=None, n=n, references=references, mask=False)
        elif set(resolved.model_dump(exclude_none=True)) - set(defaults) - {"store"}:
            raise ValueError("Advanced controls require a model with known image capabilities")
        return resolved

    def _inherited_options(
        self, size: str | None, image: ImageCapabilities | None
    ) -> dict[str, Any]:
        """The options a call inherits from the config and from a portable size,
        each refused under the name the caller set when the model lacks it."""
        defaults: dict[str, Any] = {}
        if self._config.image_size:
            defaults["image_size"] = self._config.image_size
        if mime := self._config.output_mime_type:
            defaults["output_format"] = mime.removeprefix("image/")
            if image is not None and defaults["output_format"] not in image.formats:
                offered = ", ".join(f"image/{fmt}" for fmt in image.formats) or "none"
                raise ValueError(
                    f"output_mime_type {mime!r} is not offered by {self._config.model} "
                    f"(its types: {offered})"
                )
        if size:
            # The requested pixels win over the deployment default: a caller
            # naming a geometry is more specific than a configured tier.
            defaults["aspect_ratio"], defaults["image_size"] = (
                size_geometry(size, image.aspect_ratios, image.image_sizes, self._config.model)
                if image is not None
                else self.resolve_size(size)
            )
        return defaults

    def _build_request(
        self,
        prompt: str,
        reference_images: list[AIImagePart],
        options: ImageOptions,
    ) -> dict[str, Any]:
        """Assemble one ``interactions.create`` body from the prompt, the
        references and the call's resolved options, reused across the ``n`` calls."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            self._image_content(part, index) for index, part in enumerate(reference_images)
        )

        # No ``delivery``: the field is schema-valid (the API validates its
        # enum) and then refused per model — every image model answers
        # "Image delivery mode is not supported" to ``inline`` and to ``uri``
        # alike, so naming the mode we want is what makes the call fail. The
        # default is the inline payload anyway, and RFC §25.3's "never a link"
        # is enforced where it can actually be checked: on the response, in
        # :meth:`_result`.
        response_format: dict[str, Any] = {"type": "image"}
        for field in ("aspect_ratio", "image_size"):
            if (value := getattr(options, field)) is not None:
                response_format[field] = value
        if options.output_format:
            response_format["mime_type"] = "image/" + options.output_format
        request: dict[str, Any] = {
            "model": self._config.model,
            "input": content,
            "response_format": response_format,
        }
        if options.store is not None:
            request["store"] = options.store
        if options.previous_interaction_id:
            request["previous_interaction_id"] = options.previous_interaction_id
        if options.search_types:
            request["tools"] = [{"type": "google_search", "search_types": options.search_types}]
        if options.thinking_level:
            request["generation_config"] = {"thinking_level": options.thinking_level}
        return request

    async def _create(self, request: dict[str, Any]) -> Any:
        try:
            return await self._client.aio.interactions.create(**request)  # ty: ignore[unresolved-attribute]
        except ProviderError:
            raise
        except Exception as exc:
            raise wrap_gemini_error(exc) from exc

    @staticmethod
    def resolve_size(size: str) -> tuple[str, str]:
        """Translate ``"WIDTHxHEIGHT"`` into Gemini's aspect ratio and size tier.

        Gemini expresses geometry as a named ratio and a resolution tier, not
        as pixels. Translating here is what lets a caller pass one size string
        to every provider (RFC §25.2) instead of learning each vendor's form.
        """
        return size_geometry(size, ASPECT_RATIOS, TIERS, "Gemini")

    _geometry = resolve_size

    @staticmethod
    def _image_content(part: AIImagePart, index: int) -> dict[str, Any]:
        """Turn a reference image into an Interactions image content block.

        A remote URI is forwarded as-is — Google dereferences it, roomkit never
        does. Inline bytes make the round trip through :func:`parse_data_uri`
        rather than being copied across: decoding is what proves the payload is
        valid before it reaches the wire, and re-encoding from those bytes is
        what guarantees the string sent is canonical base64 even when the
        caller's URI carried line breaks or padding of its own.
        """
        if not part.url.startswith("data:"):
            return {"type": "image", "uri": part.url, "mime_type": part.mime_type or "image/png"}
        try:
            mime_type, data = parse_data_uri(part.url, fallback_mime=part.mime_type)
        except ValueError as exc:
            raise ValueError(f"reference image {index}: {exc}") from exc
        return {
            "type": "image",
            "data": base64.b64encode(data).decode("ascii"),
            "mime_type": mime_type,
        }

    def _result(self, interaction: Any) -> ImageResult:
        """Map one ``Interaction`` onto an :class:`ImageResult`."""
        image = getattr(interaction, "output_image", None)
        payload = getattr(image, "data", None) if image is not None else None
        if not payload:
            # A link instead of bytes is its own failure, and worth its own
            # words: it is where RFC §25.3 is enforced, the request having no
            # say in the delivery mode (see :meth:`_build_request`). Final: the
            # image was generated and billed, and generating it again is the
            # host's call (RFC §25.2).
            if image is not None and getattr(image, "uri", None):
                raise ProviderError(
                    "Gemini delivered the image as a URI; roomkit returns inline bytes "
                    "only, since a link expires while an ImageResult is expected to "
                    "outlive the call",
                    retryable=False,
                    provider="gemini",
                )
            raise ProviderError(
                "Gemini returned an interaction with no image; "
                f"status={getattr(interaction, 'status', None)!r}",
                retryable=False,
                provider="gemini",
            )
        # The SDK types ``mime_type`` as optional; Nano Banana 2.1 returns JPEG
        # only, so an undeclared type is read off the bytes, not assumed PNG.
        mime_type = payload_mime_type(getattr(image, "mime_type", None), payload)
        return ImageResult(
            data=f"data:{mime_type};base64,{payload}",
            mime_type=mime_type,
            # Gemini reports no rewritten prompt; ``output_text`` is the model's
            # commentary about the picture, not the prompt it drew from, so it
            # is not passed off as one.
            revised_prompt=None,
            usage=self._usage(interaction),
        )

    @staticmethod
    def _usage(interaction: Any) -> dict[str, int]:
        """Split Gemini's usage into the disjoint counters RFC §25.5 requires.

        ``*_tokens_by_modality`` breaks each total down, so the image share is
        read from there and the text counter is the remainder — never the total
        again, which would bill the pixels twice. Thought tokens are billed at
        the text output rate, so they join the text output counter.
        """
        return gemini_image_usage(getattr(interaction, "usage", None))

    async def close(self) -> None:
        """Close the SDK and the httpx client it was given."""
        client, self._client = self._client, None
        http, self._http = self._http, None
        await close_genai_client(client, http)
