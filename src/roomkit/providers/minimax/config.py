"""MiniMax provider configuration."""

from __future__ import annotations

from typing import ClassVar

from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.vendor_endpoint import MINIMAX_BASE_URL


class MiniMaxConfig(OpenAIConfig):
    """MiniMax chat configuration, on MiniMax's API or on a server of your own.

    Inherits connection settings, sampling, retries and ``extra_body`` from
    :class:`OpenAIConfig`. ``model`` is required, so upgrading RoomKit never
    silently selects another model. See :mod:`roomkit.providers.minimax.models`
    for the curated catalogue.
    """

    _vendor_rule_on_tool_turns: ClassVar[bool] = True

    base_url: str = MINIMAX_BASE_URL
    """MiniMax's OpenAI-compatible endpoint. Point it at a vLLM, a llama.cpp or
    a gateway serving MiniMax's weights: the reasoning goes back the same way."""

    max_tokens: int | None = None
    """No cap unless one is set. Reasoning counts toward it, and MiniMax documents
    that a cap too small yields ``finish_reason: "length"`` with an empty answer."""

    reasoning_effort: str | None = None
    """``"low"`` | ``"medium"`` | ``"high"`` | ``"xhigh"`` | ``"max"``, effective
    for MiniMax-M3.1-Flash only, also on a turn with tools. ``None`` leaves the
    model's default, ``"max"``. The turn's own effort takes precedence."""
