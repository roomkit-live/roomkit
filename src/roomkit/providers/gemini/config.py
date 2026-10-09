"""Google Gemini provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class GeminiConfig(BaseModel):
    """Google Gemini AI provider configuration."""

    api_key: SecretStr
    model: str = "gemini-3.8-flash"
    max_tokens: int = 1024
    thinking_level: str | None = None
    """Thinking level for Gemini 3 models: minimal, low, medium, high.
    ``gemini-3.8-flash``, the default, refuses ``minimal`` with a 400:
    ``low`` is its lowest level."""
    timeout: float = 60.0
    """Read budget in seconds: how long the first chunk, and each one after
    it, may take. Generation streams, so a stalled answer fails here instead
    of holding the turn open."""
    connect_timeout: float = 5.0
    """TCP connect timeout in seconds, kept apart from ``timeout`` so a host
    that no longer accepts connections is given up on in seconds rather
    than after the read budget. The default across RoomKit's providers."""


class GeminiImageConfig(BaseModel):
    """Google Gemini image-generation provider configuration (RFC §25).

    Separate from :class:`GeminiConfig` because it configures a disjoint model
    lineup — the ``*-image`` models the chat catalog explicitly excludes — and
    a different set of knobs: geometry and output encoding rather than
    thinking level.

    Attributes:
        api_key: API key for authentication.
        model: Image model identifier. Defaults to ``"gemini-nano-banana-2.1"``,
            the model Google recommends for new projects (RMK-656); it has no
            ``"512"`` tier, which ``"gemini-3.1-flash-image"`` keeps.
        image_size: Default resolution tier — ``"512"`` | ``"1K"`` | ``"2K"`` |
            ``"4K"``. A per-call ``size`` wins over it, since the caller asking
            for specific pixels is more specific than a deployment default.
            ``None`` leaves the model's own default.
        output_mime_type: ``"image/jpeg"`` to ask for JPEG. ``None`` leaves the
            vendor default: PNG on ``gemini-3.1-flash-image``, JPEG on
            ``gemini-nano-banana-2.1``, which offers no other. The response
            reports what it actually produced and the provider reads that
            rather than assuming. Gemini offers no other selectable output
            type here.
        timeout: Read budget in seconds for one interaction. Higher than the
            chat default because an image is produced whole: nothing streams
            before it.
        connect_timeout: TCP connect timeout in seconds, kept apart from
            ``timeout`` so a host that no longer accepts connections is given
            up on in seconds rather than after the read budget.
    """

    api_key: SecretStr
    model: str = "gemini-nano-banana-2.1"
    image_size: str | None = None
    output_mime_type: str | None = None
    timeout: float = 120.0
    connect_timeout: float = 5.0
