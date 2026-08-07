"""Google Gemini provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class GeminiConfig(BaseModel):
    """Google Gemini AI provider configuration."""

    api_key: SecretStr
    model: str = "gemini-3.1-flash-lite"
    max_tokens: int = 1024
    temperature: float = 1.0  # Gemini default
    thinking_level: str | None = None
    """Thinking level for Gemini 3.1 models: minimal, low, medium, high."""


class GeminiImageConfig(BaseModel):
    """Google Gemini image-generation provider configuration (RFC §25).

    Separate from :class:`GeminiConfig` because it configures a disjoint model
    lineup — the ``*-image`` models the chat catalog explicitly excludes — and
    a different set of knobs: geometry and output encoding rather than
    temperature and thinking level.

    Attributes:
        api_key: API key for authentication.
        model: Image model identifier (e.g. ``"gemini-3-pro-image"``).
        image_size: Default resolution tier — ``"512"`` | ``"1K"`` | ``"2K"`` |
            ``"4K"``. A per-call ``size`` wins over it, since the caller asking
            for specific pixels is more specific than a deployment default.
            ``None`` leaves the model's own default.
        output_mime_type: ``"image/jpeg"`` to ask for JPEG. ``None`` leaves the
            vendor default (PNG); the response reports what it actually
            produced and the provider reads that rather than assuming. Gemini
            offers no other selectable output type here.
    """

    api_key: SecretStr
    model: str = "gemini-3.1-flash-image"
    image_size: str | None = None
    output_mime_type: str | None = None
