"""Typed image controls and offline capabilities, shared by image adapters."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from roomkit.providers.ai.base import ModelInfo


class ImageOptions(BaseModel):
    """Per-request controls. Omission inherits the provider's configured default.

    ``size`` remains on ``generate`` for portable pixel geometry. ``aspect_ratio``
    and ``image_size`` request native geometry without promising exact pixels.
    ``partial_images`` requests streaming previews and requires a progress callback.
    """

    model_config = ConfigDict(extra="forbid")

    quality: Literal["auto", "low", "medium", "high", "xhigh", "max"] | None = None
    background: Literal["auto", "opaque", "transparent"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    output_compression: int | None = Field(default=None, ge=0, le=100, strict=True)
    moderation: Literal["auto", "low"] | None = None
    input_fidelity: Literal["low", "high"] | None = None
    aspect_ratio: str | None = None
    image_size: Literal["512", "1K", "2K", "4K"] | None = None
    thinking_level: Literal["minimal", "medium", "high"] | None = None
    previous_interaction_id: str | None = Field(default=None, min_length=1)
    store: bool | None = Field(default=None, strict=True)
    search_types: list[Literal["web_search", "image_search"]] | None = None
    partial_images: int | None = Field(default=None, ge=0, le=3, strict=True)


class ImageCapabilities(BaseModel):
    """The controls verified for one model on its image endpoint.

    An absent entry is unknown, not the capabilities of a neighbouring model.
    Lists name supported values; an empty list means the control is unsupported.
    """

    options: list[str] = Field(default_factory=list)
    qualities: list[str] = Field(default_factory=list)
    formats: list[str] = Field(default_factory=list)
    backgrounds: list[str] = Field(default_factory=list)
    aspect_ratios: list[str] = Field(default_factory=list)
    image_sizes: list[str] = Field(default_factory=list)
    thinking_levels: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)
    flexible_size: bool = False
    max_images: int | None = 10
    max_references: int = 0
    mask: bool = False
    continuity: bool = False
    search_types: list[str] = Field(default_factory=list)
    streaming: bool = False
    retirement_date: date | None = None
    verified: date

    def validate_request(
        self, options: ImageOptions, *, size: str | None, n: int, references: int, mask: bool
    ) -> None:
        """Reject incompatible requests before any billable call."""
        if n < 1 or (self.max_images is not None and n > self.max_images):
            raise ValueError(f"n must be at least 1; model limit: {self.max_images}")
        if references > self.max_references:
            raise ValueError(f"This model accepts at most {self.max_references} references")
        if mask and (not self.mask or not references):
            raise ValueError("A mask requires a reference image and a model supporting masks")
        supplied = options.model_dump(exclude_none=True)
        unsupported = set(supplied) - set(self.options)
        if unsupported:
            raise ValueError(f"Unsupported image options: {', '.join(sorted(unsupported))}")
        for field, allowed in (
            ("quality", self.qualities),
            ("output_format", self.formats),
            ("background", self.backgrounds),
            ("aspect_ratio", self.aspect_ratios),
            ("image_size", self.image_sizes),
            ("thinking_level", self.thinking_levels),
        ):
            value = supplied.get(field)
            if value is not None and value not in allowed:
                raise ValueError(f"{field} must be one of {', '.join(allowed)}")
        if set(options.search_types or []) - set(self.search_types):
            raise ValueError("This model does not support the requested search types")
        if options.previous_interaction_id and not self.continuity:
            raise ValueError("This model does not support interaction continuity")
        if options.background == "transparent" and options.output_format == "jpeg":
            raise ValueError("Transparent output requires PNG or WebP")
        if options.output_compression is not None and options.output_format not in (
            "jpeg",
            "webp",
        ):
            raise ValueError("output_compression requires an explicit JPEG or WebP format")
        if size and (options.aspect_ratio or options.image_size):
            raise ValueError("Use size or aspect_ratio/image_size, not both")
        if size:
            self.validate_size(size)

    def validate_size(self, size: str) -> None:
        if size in self.sizes:
            return
        if not self.flexible_size:
            raise ValueError(f"size must be one of {', '.join(self.sizes)}")
        match = re.fullmatch(r"(\d+)[xX](\d+)", size)
        if not match:
            raise ValueError("size must be WIDTHxHEIGHT or auto")
        width, height = (int(value) for value in match.groups())
        if (
            min(width, height) < 16
            or width % 16
            or height % 16
            or max(width, height) > 3840
            or max(width, height) > 3 * min(width, height)
            or not 655360 <= width * height <= 8294400
        ):
            raise ValueError(
                "Image dimensions must be multiples of 16, at most 3840 per edge, "
                "at most 3:1, and between 655360 and 8294400 pixels"
            )


class ImageModelInfo(ModelInfo):
    """An image catalogue entry with its endpoint-specific capabilities."""

    image: ImageCapabilities
    aliases: list[str] = Field(default_factory=list)


def image_model_entry(models: list[ModelInfo], model: str) -> ModelInfo | None:
    """Resolve only exact ids and explicitly verified snapshot aliases."""
    return next(
        (
            entry
            for entry in models
            if entry.id == model or (isinstance(entry, ImageModelInfo) and model in entry.aliases)
        ),
        None,
    )


def plain_metadata(value: Any) -> Any:
    """Serialize SDK metadata while excluding image payloads and credentials."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    elif hasattr(value, "__dict__"):
        value = vars(value)
    if isinstance(value, dict):
        return {
            key: plain_metadata(item)
            for key, item in value.items()
            if key not in {"data", "b64_json", "api_key", "authorization"}
            and not key.startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [plain_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
