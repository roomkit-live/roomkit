"""Kimi provider configuration."""

from __future__ import annotations

from typing import ClassVar

from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.vendor_endpoint import KIMI_BASE_URL


class KimiConfig(OpenAIConfig):
    """Kimi chat configuration, on Kimi's API or on a server of your own.

    Inherits connection settings, retries and ``extra_body`` from
    :class:`OpenAIConfig`. ``model`` is required, so upgrading RoomKit never
    silently selects another model. See :mod:`roomkit.providers.kimi.models` for
    the curated catalogue.
    """

    _vendor_rule_on_tool_turns: ClassVar[bool] = True

    base_url: str = KIMI_BASE_URL
    """Kimi's OpenAI-compatible endpoint. Point it at a vLLM or a gateway serving
    Kimi's weights: the reasoning goes back the same way."""

    max_tokens: int | None = None
    """No cap unless one is set, which leaves K3's own default, 131,072. Reasoning
    counts toward it: a small cap cuts an answer after its reasoning."""

    use_max_completion_tokens: bool = True
    """A cap goes as ``max_completion_tokens``, since Kimi deprecates
    ``max_tokens``; set False for a server that reads ``max_tokens`` alone."""

    supports_custom_temperature: bool = False
    """Kimi's K3 models fix ``temperature`` and ask that it be left out. Set True
    for a server of your own that honours it."""

    reasoning_effort: str | None = None
    """``"low"`` | ``"high"`` | ``"max"``, also on a turn with tools. ``None``
    leaves the model's default, ``"max"``. The turn's own effort takes precedence."""
