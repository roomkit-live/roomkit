"""Image generation provider ABC — RFC §25.

Image synthesis is a capability a small number of models hold, and the model
holding a conversation is rarely one of them. So it lives here, on its own
surface, the way :class:`~roomkit.voice.stt.base.STTProvider` and
:class:`~roomkit.voice.tts.base.TTSProvider` do — an agent conversing through
Anthropic draws through Gemini exactly as it transcribes through Deepgram.

The alternative — image parts on ``AIResponse`` — was rejected twice over:
``content: str`` is read by every consumer of the framework, and reaching image
generation *through* the conversational response would confine the capability
to the conversations already held by a model that draws.
"""

from __future__ import annotations

import base64
import binascii
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, field_validator

from roomkit.providers.ai.base import AIImagePart, ModelInfo

IMAGE_GEN_CAPABILITY = "image_gen"
"""``ModelInfo.capabilities`` tag marking an entry as an image-generating model.

The image catalog and the conversational one are disjoint sets — see
:meth:`ImageProvider.available_models` — so nothing in roomkit needs this tag
to tell them apart. It is there for a consumer that deliberately merges the two
lists and then has to.
"""

_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")


def parse_size(size: str) -> tuple[int, int]:
    """Parse a ``"WIDTHxHEIGHT"`` size string into its two integers.

    Args:
        size: Geometry as the :class:`ImageProvider` surface spells it.

    Returns:
        ``(width, height)``.

    Raises:
        ValueError: If *size* is not two positive integers joined by ``x``.
    """
    match = _SIZE_RE.match(size.strip().lower())
    if match is None:
        raise ValueError(f"size must be 'WIDTHxHEIGHT' (e.g. '1024x1024'), got {size!r}")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"size must have positive dimensions, got {size!r}")
    return width, height


def to_data_uri(data: bytes, mime_type: str) -> str:
    """Encode raw image bytes as the data URI :attr:`ImageResult.data` requires."""
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


class ImageResult(BaseModel):
    """One generated image.

    Attributes:
        data: The image as a ``data:<mime_type>;base64,<payload>`` URI — always,
            never bare base64 and never a remote URL. A field documented as
            "one or the other" makes every consumer sniff the value before it
            can use it, and one consumer sniffs it wrong. The invariant is also
            what makes a result immediately usable: a data URI is what
            ``MediaContent.url`` and :class:`AIImagePart` already accept, so a
            generated image enters a room, or comes back as a reference for the
            next edit, without conversion.
        mime_type: Media type of the payload. Always equal to the one spelled
            in :attr:`data`; carried separately so a caller can branch on it
            without parsing the URI.
        revised_prompt: The prompt as the model rewrote it, where the vendor
            reports one — ``None`` otherwise. Never a copy of the caller's
            prompt: echoing the input back would conceal exactly the divergence
            this field exists to reveal.
        usage: Token counters for the call, disjoint by construction —
            ``input_tokens`` (text in), ``input_image_tokens`` (reference
            images), ``output_tokens`` (text out, where billed) and
            ``output_image_tokens`` (the generated image). Priced by
            :meth:`~roomkit.providers.ai.base.ModelPricing.cost_for`. Empty
            when the vendor reports nothing.
    """

    data: str
    mime_type: str
    revised_prompt: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def _validate_data_uri(cls, value: str) -> str:
        if not value.startswith("data:"):
            raise ValueError("ImageResult.data must be a data URI (data:<mime>;base64,<payload>)")
        header, separator, payload = value.partition(",")
        if not separator or not payload:
            raise ValueError("ImageResult.data is missing its base64 payload")
        if ";base64" not in header:
            raise ValueError("ImageResult.data must carry a base64 payload")
        return value

    def decoded(self) -> bytes:
        """The raw image bytes.

        Raises:
            ValueError: If the payload is not valid base64 — a corrupted result
                is worth an error at the point it is read, not a truncated file
                on disk.
        """
        payload = self.data.partition(",")[2]
        try:
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("ImageResult.data payload is not valid base64") from exc

    def to_image_part(self) -> AIImagePart:
        """The result as a message part — an AI input, or the next edit's reference."""
        return AIImagePart(url=self.data, mime_type=self.mime_type)


class ImageProvider(ABC):
    """Generates images from a prompt, decoupled from the conversation (RFC §25)."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'OpenAIImageProvider')."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier (e.g. 'gpt-image-2', 'gemini-3-pro-image')."""
        ...

    @property
    def supports_editing(self) -> bool:
        """Whether ``reference_images`` is honoured. ``False`` refuses them outright."""
        return False

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Offline metadata for the image models this provider can describe.

        Deliberately *not* folded into
        :meth:`~roomkit.providers.ai.base.AIProvider.available_models`. The two
        catalogs are disjoint sets — ``gpt-image-2`` is not a conversational
        model and no chat id draws — so merging them saves no maintenance, the
        entries being written once either way, while obliging every consumer of
        the conversational catalog to filter out a class of models it can never
        use. Entries carry :data:`IMAGE_GEN_CAPABILITY` for a consumer that
        merges the lists on purpose.

        The base returns an empty list; providers override it.
        """
        return []

    def catalog_entry(self) -> ModelInfo | None:
        """The offline :class:`ModelInfo` for the active model, if the catalog has it."""
        name = self.model_name
        for model in type(self).available_models():
            if model.id == name:
                return model
        return None

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        size: str | None = None,
        n: int = 1,
        reference_images: list[AIImagePart] | None = None,
    ) -> list[ImageResult]:
        """Generate ``n`` images from ``prompt``.

        Args:
            prompt: What to draw.
            size: Geometry as ``"WIDTHxHEIGHT"`` (e.g. ``"1024x1024"``), or
                ``None`` for the model's default. A provider whose API speaks
                aspect ratios translates; the caller never has to know which
                form its vendor wants. A size the model cannot produce raises
                rather than silently becoming another one.
            n: How many images. Must be at least 1. A provider whose API has no
                batch parameter issues concurrent calls — the vendor bills per
                image either way.
            reference_images: Images to edit or draw from. Non-empty makes this
                an edit; a provider that reports ``supports_editing`` as False
                raises rather than quietly generating from the prompt alone.

        Returns:
            Exactly ``n`` results, or an exception. Never fewer without error:
            a caller discovers a short list by indexing it, in production.

        Raises:
            ProviderError: The vendor call failed.
            ValueError: The request is one this provider cannot express.
        """
        ...

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""
