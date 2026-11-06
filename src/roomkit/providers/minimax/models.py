"""Offline metadata for MiniMax chat models, read 2026-11-04.

Sources: https://platform.minimax.io/docs/api-reference/text-openai-api (ids,
context windows, what a thinking switch does, image input) and
https://platform.minimax.io/docs/guides/pricing-paygo (rates).

MiniMax-M3's rates are its standard tier: past 512k input tokens they double,
which its ``long_context_*`` fields carry. Its ``priority`` tier costs 1.5 times
as much, which :class:`~roomkit.providers.ai.base.ModelPricing` has no place for.
No cache write is priced. MiniMax documents no ``/models`` listing.

The ``reasoning_floor_<level>`` tags come from the same API page, not from the
wire: M3.1-Flash answers 400 to ``none`` and starts at ``low``, and M3 can stop
reasoning, which :class:`~roomkit.providers.minimax.ai.MiniMaxAIProvider` sends as
``thinking: {"type": "disabled"}``.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 11, 4)

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="MiniMax-M3.1-Flash",
        display_name="MiniMax M3.1 Flash",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools", "thinking", "reasoning_floor_low"],
        pricing=ModelPricing(
            input_per_million=0.15,
            output_per_million=0.60,
            cache_read_per_million=0.015,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="MiniMax-M3",
        display_name="MiniMax M3",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools", "thinking", "reasoning_floor_none"],
        pricing=ModelPricing(
            input_per_million=0.30,
            output_per_million=1.20,
            cache_read_per_million=0.06,
            long_context_threshold_tokens=512_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=2.0,
            verified=_VERIFIED,
        ),
    ),
]
