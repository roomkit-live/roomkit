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
import binascii
from math import gcd
from typing import Any

from roomkit.providers.ai.base import AIImagePart, ModelInfo, ProviderError
from roomkit.providers.gemini.config import GeminiImageConfig
from roomkit.providers.gemini.errors import wrap_gemini_error
from roomkit.providers.gemini.image_models import MODELS
from roomkit.providers.image.base import ImageProvider, ImageResult, parse_size

# Aspect ratios the Interactions image format accepts. A requested size is
# reduced to its ratio and looked up here; an unlisted one is refused rather
# than rounded to a neighbour, because a silently different geometry is a
# failure the caller can neither see nor correct (RFC §25.2).
_ASPECT_RATIOS = frozenset(
    {
        "1:1",
        "2:3",
        "3:2",
        "3:4",
        "4:3",
        "4:5",
        "5:4",
        "9:16",
        "16:9",
        "21:9",
        "1:8",
        "8:1",
        "1:4",
        "4:1",
    }
)

# Resolution tiers, keyed by the largest dimension the caller asked for. Google
# names tiers, not pixel counts, so the request is mapped to the smallest tier
# that covers it.
_SIZE_TIERS: tuple[tuple[int, str], ...] = ((512, "512"), (1024, "1K"), (2048, "2K"), (4096, "4K"))


class GeminiImageProvider(ImageProvider):
    """Image provider using the Gemini Interactions API."""

    def __init__(self, config: GeminiImageConfig) -> None:
        try:
            from google import genai as _genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiImageProvider. "
                "Install it with: pip install roomkit[gemini]"
            ) from exc

        self._config = config
        self._client = _genai.Client(api_key=config.api_key.get_secret_value())

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
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        request = self._build_request(prompt, size, reference_images or [])
        interactions = await asyncio.gather(
            *(self._create(request) for _ in range(n)),
        )
        return [self._result(interaction) for interaction in interactions]

    def _build_request(
        self,
        prompt: str,
        size: str | None,
        reference_images: list[AIImagePart],
    ) -> dict[str, Any]:
        """Assemble one ``interactions.create`` body, reused across the ``n`` calls."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            self._image_content(part, index) for index, part in enumerate(reference_images)
        )

        response_format: dict[str, Any] = {"type": "image", "delivery": "inline"}
        if size is not None:
            # The requested pixels win over the deployment default: a caller
            # naming a geometry is more specific than a configured tier.
            aspect_ratio, tier = self._geometry(size)
            response_format["aspect_ratio"] = aspect_ratio
            response_format["image_size"] = tier
        elif self._config.image_size is not None:
            response_format["image_size"] = self._config.image_size
        if self._config.output_mime_type is not None:
            response_format["mime_type"] = self._config.output_mime_type

        return {
            "model": self._config.model,
            "input": content,
            "response_format": response_format,
        }

    async def _create(self, request: dict[str, Any]) -> Any:
        try:
            return await self._client.aio.interactions.create(**request)
        except ProviderError:
            raise
        except Exception as exc:
            raise wrap_gemini_error(exc) from exc

    @staticmethod
    def _geometry(size: str) -> tuple[str, str]:
        """Translate ``"WIDTHxHEIGHT"`` into Gemini's aspect ratio and size tier.

        Gemini expresses geometry as a named ratio and a resolution tier, not
        as pixels. Translating here is what lets a caller pass one size string
        to every provider (RFC §25.2) instead of learning each vendor's form.
        """
        width, height = parse_size(size)
        divisor = gcd(width, height)
        aspect_ratio = f"{width // divisor}:{height // divisor}"
        if aspect_ratio not in _ASPECT_RATIOS:
            raise ValueError(
                f"size {size!r} reduces to aspect ratio {aspect_ratio}, which Gemini does not "
                f"offer; supported ratios are {', '.join(sorted(_ASPECT_RATIOS))}"
            )
        largest = max(width, height)
        for ceiling, tier in _SIZE_TIERS:
            if largest <= ceiling:
                return aspect_ratio, tier
        raise ValueError(f"size {size!r} exceeds Gemini's largest tier (4K)")

    @staticmethod
    def _image_content(part: AIImagePart, index: int) -> dict[str, Any]:
        """Turn a reference image into an Interactions image content block."""
        url = part.url
        if not url.startswith("data:"):
            return {"type": "image", "uri": url, "mime_type": part.mime_type or "image/png"}
        header, _, payload = url.partition(",")
        mime_type = header[len("data:") :].split(";", 1)[0] or part.mime_type or "image/png"
        try:
            base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"reference image {index} is not valid base64") from exc
        return {"type": "image", "data": payload, "mime_type": mime_type}

    def _result(self, interaction: Any) -> ImageResult:
        """Map one ``Interaction`` onto an :class:`ImageResult`."""
        image = getattr(interaction, "output_image", None)
        payload = getattr(image, "data", None) if image is not None else None
        if not payload:
            raise ProviderError(
                "Gemini returned an interaction with no image; "
                f"status={getattr(interaction, 'status', None)!r}",
                retryable=False,
                provider="gemini",
            )
        mime_type = str(getattr(image, "mime_type", None) or "image/png")
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
        usage = getattr(interaction, "usage", None)
        if usage is None:
            return {}
        input_total = int(getattr(usage, "total_input_tokens", 0) or 0)
        output_total = int(getattr(usage, "total_output_tokens", 0) or 0)
        thoughts = int(getattr(usage, "total_thought_tokens", 0) or 0)
        input_image = _modality_tokens(getattr(usage, "input_tokens_by_modality", None), "image")
        output_image = _modality_tokens(getattr(usage, "output_tokens_by_modality", None), "image")
        return {
            "input_tokens": max(input_total - input_image, 0),
            "input_image_tokens": input_image,
            "output_tokens": max(output_total - output_image, 0) + thoughts,
            "output_image_tokens": output_image,
        }

    async def close(self) -> None:
        await self._client.aio.aclose()


def _modality_tokens(breakdown: Any, modality: str) -> int:
    """Sum the token counts one modality contributes to a usage breakdown."""
    if not breakdown:
        return 0
    total = 0
    for entry in breakdown:
        if str(getattr(entry, "modality", "")).lower() == modality:
            total += int(getattr(entry, "tokens", 0) or 0)
    return total
