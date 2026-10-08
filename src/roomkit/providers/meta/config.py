"""Meta Model API provider configuration — chat (Muse Spark) and images (Muse Image)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from roomkit.providers.openai.config import OpenAIConfig

MetaImageTool = Literal["web_search", "image_search", "shell"]


# A validation error echoes its input: with the key among the fields, never.
_KEEP_KEY_OUT_OF_ERRORS = ConfigDict(hide_input_in_errors=True)


def _header_safe(key: SecretStr) -> SecretStr:
    """The key, if an HTTP header can carry it: a ``Bearer`` value is ASCII."""
    if not key.get_secret_value().isascii():
        raise ValueError(
            "api_key holds a non-ASCII character (a placeholder such as '…' copied "
            "as is?): an HTTP Authorization header cannot carry it"
        )
    return key


class MetaImageConfig(BaseModel):
    """Meta Muse Image provider configuration (RFC §25).

    Attributes:
        api_key: Meta Model API key.
        base_url: Meta Model API endpoint. Override only to point at a proxy.
        model: Image model id — see :mod:`roomkit.providers.meta.image_models`.
        tools: What the image generator may do on its own while it draws:
            ``"web_search"`` (look up facts), ``"image_search"`` (fetch visual
            references), ``"shell"`` (run code for charts and layouts). Meta
            enables all three when a request says nothing, so RoomKit always
            says: none unless listed here. A prompt sent with a search tool
            leaves Meta for the web.
        reasoning_strength: ``"high"`` (Meta's default: several refinement
            passes) or ``"low"`` (one pass), billed the same per image.
            ``None`` leaves the service default.
        moderation: ``"auto"`` or ``"low"``, or ``None`` for the default.
        output_format: ``"png"``, ``"jpeg"`` or ``"webp"``; ``None`` leaves
            Meta's default, WebP.
        timeout: HTTP request timeout in seconds; a drawing takes ~10 s at
            ``reasoning_strength="low"`` (measured 2026-09-27).
        connect_timeout: TCP connect timeout in seconds.
        max_retries: SDK-level retry count. 0 because RoomKit's RetryPolicy
            handles retries at the right layer.
    """

    api_key: SecretStr
    base_url: str = "https://api.meta.ai/v1"
    model: str = "muse-image-1.0"
    tools: list[MetaImageTool] = []
    reasoning_strength: Literal["low", "high"] | None = None
    moderation: Literal["auto", "low"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    timeout: float = 120.0
    connect_timeout: float = 5.0
    max_retries: int = 0

    model_config = _KEEP_KEY_OUT_OF_ERRORS
    _check_api_key = field_validator("api_key")(_header_safe)


class MetaConfig(OpenAIConfig):
    """Meta Muse Spark chat provider configuration.

    The Meta Model API serves an OpenAI-compatible Chat Completions API at
    ``https://api.meta.ai/v1``, so this subclasses :class:`OpenAIConfig` and
    inherits every request field the inherited provider reads
    (``reasoning_effort``, ``extra_body`` …). Only the
    endpoint, the model and two defaults change.

    ``reasoning_effort`` takes ``minimal``, ``low``, ``medium``, ``high`` or
    ``xhigh``. Muse Spark always reasons: ``"none"`` is refused by the service
    and sent as ``"minimal"``, its lightest effort.

    The ``-contributor`` model ids cost a fraction of the standard ones because
    Meta trains its models on their prompts and completions. They are never a
    default; choose one knowingly.
    """

    _vendor_rule_on_tool_turns: ClassVar[bool] = True

    base_url: str = "https://api.meta.ai/v1"
    """Meta Model API endpoint. Override only to point at a proxy."""

    model: str = "muse-spark-1.3"
    """Muse Spark model id — see :mod:`roomkit.providers.meta.models`."""

    use_max_completion_tokens: bool = True
    """Meta documents ``max_completion_tokens`` for Chat Completions."""

    include_stream_usage: bool = True
    """Meta reports usage on the last streamed chunk, reasoning tokens included."""

    model_config = _KEEP_KEY_OUT_OF_ERRORS
    _check_api_key = field_validator("api_key")(_header_safe)
