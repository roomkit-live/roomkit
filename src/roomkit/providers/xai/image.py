"""xAI (Grok Imagine) image provider — draws via xAI's images API (RFC §25).

Generation goes through the OpenAI-compatible ``/v1/images/generations`` and
the SDK's ``images.generate``. Editing does not: xAI's ``/v1/images/edits``
takes references as JSON objects where the OpenAI SDK's ``images.edit`` uploads
multipart file content, so edits are posted as JSON through the same client.
That split is xAI's, not the caller's — RFC §25.4 requires the provider to
absorb it.
"""

from __future__ import annotations

import base64
import binascii
from math import gcd
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
from roomkit.providers.xai.config import XAIImageConfig
from roomkit.providers.xai.image_models import MODELS

_MODELS_BY_ID: dict[str, ModelInfo] = {m.id: m for m in MODELS}

# Aspect ratios the images API accepts. An unlisted ratio is refused rather
# than rounded to a neighbour, because a silently different geometry is a
# failure the caller can neither see nor correct (RFC §25.2).
_ASPECT_RATIOS = frozenset(
    {
        "1:1",
        "16:9",
        "9:16",
        "4:3",
        "3:4",
        "3:2",
        "2:3",
        "2:1",
        "1:2",
        "20:9",
        "9:20",
        "19.5:9",
        "9:19.5",
    }
)

# Two ratios xAI spells fractionally: 19.5:9 *is* 13:6 (both 2.1666…), just
# under another name, so the gcd's integer spelling is renamed — not rounded —
# before the lookup. Without this, 1300x600 reduces to a "13:6" the API does
# not take and an exactly-offered geometry is refused.
_RATIO_ALIASES = {"13:6": "19.5:9", "6:13": "9:19.5"}

# Resolution tiers, keyed by the largest dimension the caller asked for. xAI
# names tiers, not pixel counts, so the request is mapped to the smallest tier
# that covers it.
_SIZE_TIERS: tuple[tuple[int, str], ...] = ((1024, "1k"), (2048, "2k"))


class XAIImageProvider(ImageProvider):
    """Image provider using xAI's Grok Imagine images API."""

    def __init__(self, config: XAIImageConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for XAIImageProvider. "
                "Install it with: pip install roomkit[xai]"
            ) from exc
        self._config = config
        self._api_status_error = _openai.APIStatusError
        self._api_connection_error = _openai.APIConnectionError
        self._images_response_cls = _openai.types.ImagesResponse
        self._client = _openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_editing(self) -> bool:
        """Whether the configured model takes reference images.

        Read off the catalog's ``edit`` tag. An id the catalog does not know
        (a model newer than the snapshot) defaults to ``True`` rather than
        refusing a capability the current lineup mostly has — the same
        permissive default the chat provider uses for vision.
        """
        info = _MODELS_BY_ID.get(self._config.model)
        if info is None:
            return True
        return "edit" in info.capabilities

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Grok Imagine image models."""
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
        if references and not self.supports_editing:
            raise ValueError(
                f"model {self._config.model!r} does not take reference images; "
                "an edit silently redrawn from the prompt alone would drop them (RFC §25.4)"
            )
        body = self._build_body(prompt, size, n)
        # Built before the try, like every reference below: a malformed data
        # URI is the caller's error and must stay a ValueError, not be
        # relabelled as a provider failure by the catch-all.
        if references:
            parts = [self._reference(part, index) for index, part in enumerate(references)]
            if len(parts) == 1:
                body["image"] = parts[0]
            else:
                body["images"] = parts

        try:
            if references:
                response = await self._client.post(
                    "/images/edits", body=body, cast_to=self._images_response_cls
                )
            else:
                response = await self._client.images.generate(**body)
        except ProviderError:
            raise
        except self._api_connection_error as exc:
            raise ProviderError(str(exc), retryable=True, provider="xai") from exc
        except self._api_status_error as exc:
            raise ProviderError(
                str(exc),
                retryable=exc.status_code in RETRYABLE_STATUS_CODES,
                provider="xai",
                status_code=exc.status_code,
            ) from exc
        except Exception as exc:
            raise ProviderError(str(exc), retryable=False, provider="xai") from exc

        return self._results(response, n)

    def _build_body(self, prompt: str, size: str | None, n: int) -> dict[str, Any]:
        """Assemble the request fields generation and editing share.

        ``b64_json`` is requested explicitly: xAI's default delivery is an
        expiring URL, and RFC §25.3 wants bytes that outlive the call.
        """
        body: dict[str, Any] = {
            "model": self._config.model,
            "prompt": prompt,
            "n": n,
            "response_format": "b64_json",
        }
        if size is not None:
            # The requested pixels win over the deployment default: a caller
            # naming a geometry is more specific than a configured tier.
            aspect_ratio, tier = self._geometry(size)
            body["aspect_ratio"] = aspect_ratio
            body["resolution"] = tier
        elif self._config.resolution is not None:
            body["resolution"] = self._config.resolution
        if self._config.quality is not None:
            body["quality"] = self._config.quality
        return body

    @staticmethod
    def _geometry(size: str) -> tuple[str, str]:
        """Translate ``"WIDTHxHEIGHT"`` into xAI's aspect ratio and resolution tier.

        xAI expresses geometry as a named ratio and a resolution tier, not as
        pixels. Translating here is what lets a caller pass one size string to
        every provider (RFC §25.2) instead of learning each vendor's form.
        """
        width, height = parse_size(size)
        divisor = gcd(width, height)
        reduced = f"{width // divisor}:{height // divisor}"
        aspect_ratio = _RATIO_ALIASES.get(reduced, reduced)
        if aspect_ratio not in _ASPECT_RATIOS:
            raise ValueError(
                f"size {size!r} reduces to aspect ratio {aspect_ratio}, which xAI does not "
                f"offer; supported ratios are {', '.join(sorted(_ASPECT_RATIOS))}"
            )
        largest = max(width, height)
        for ceiling, tier in _SIZE_TIERS:
            if largest <= ceiling:
                return aspect_ratio, tier
        raise ValueError(f"size {size!r} exceeds xAI's largest tier (2k)")

    @staticmethod
    def _reference(part: AIImagePart, index: int) -> dict[str, Any]:
        """Turn a reference image into the ``{"url", "type"}`` object edits take.

        A remote URL is forwarded as-is — xAI dereferences it, roomkit never
        does. Inline bytes make the round trip through :func:`parse_data_uri`:
        decoding is what proves the payload is valid before it reaches the
        wire, and re-encoding from those bytes is what guarantees the URI sent
        is canonical base64 even when the caller's carried line breaks or
        padding of its own.
        """
        if not part.url.startswith("data:"):
            return {"url": part.url, "type": "image_url"}
        try:
            mime_type, data = parse_data_uri(part.url, fallback_mime=part.mime_type)
        except ValueError as exc:
            raise ValueError(f"reference image {index}: {exc}") from exc
        return {"url": to_data_uri(data, mime_type), "type": "image_url"}

    def _results(self, response: Any, expected: int) -> list[ImageResult]:
        """Map an images response onto :class:`ImageResult` objects."""
        images = list(getattr(response, "data", None) or [])
        if len(images) != expected:
            raise ProviderError(
                f"xAI returned {len(images)} image(s) for a request of {expected}",
                retryable=False,
                provider="xai",
            )
        # The usage counters describe the whole call, not one image; splitting
        # them across n results would invent per-image numbers the vendor never
        # reported, so they ride the first result only and the rest report none.
        usage = self._usage(response)
        results: list[ImageResult] = []
        for index, image in enumerate(images):
            payload = getattr(image, "b64_json", None)
            if not payload:
                raise ProviderError(
                    f"xAI returned image {index} without inline bytes despite the request "
                    "naming b64_json; an expiring URL is not a result roomkit returns "
                    "(RFC §25.3)",
                    retryable=False,
                    provider="xai",
                )
            mime_type = self._mime_type(image, payload)
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
    def _mime_type(image: Any, payload: str) -> str:
        """The media type of one returned image.

        xAI declares ``mime_type`` per image; when the field is absent the
        bytes answer for themselves — Grok Imagine returns JPEG by default,
        so a fixed PNG fallback would mislabel most images.
        """
        declared = getattr(image, "mime_type", None)
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
    def _usage(response: Any) -> dict[str, int]:
        """Map xAI's usage onto the counters RFC §25.5 names.

        xAI reports two totals. ``output_tokens`` counts the generated image —
        this endpoint emits no text — so it lands on the image counter.
        ``input_tokens`` is reported as text input; on an edit the vendor
        publishes no breakdown that would let a reference's share move to
        ``input_image_tokens``, and a counter the vendor does not report is
        absent rather than invented. Billing is per image regardless — the
        counters here are the report, not the price.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        counters: dict[str, int] = {}
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is not None:
            counters["input_tokens"] = int(input_tokens)
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is not None:
            counters["output_image_tokens"] = int(output_tokens)
        return counters

    async def close(self) -> None:
        await self._client.close()
