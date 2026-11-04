"""Offline metadata for Kimi chat models, read 2026-11-04.

Sources: https://platform.kimi.ai/docs/models (ids, K2 series deprecated) and
https://platform.kimi.ai/docs/pricing/chat (context windows, rates). Account
availability is answered by ``list_models()``, not this snapshot. The K2 series
stays out: closed to new users, its retirement planned for late December 2026.

``cache_write_per_million`` is the 5-minute write, the default tier. A 1-hour
write costs twice as much, which
:class:`~roomkit.providers.ai.base.ModelPricing` has no place for.

``reasoning_floor_low`` comes from the API page, not from the wire: K3 always
thinks, and K3.1 Pro answers 400 to a request to stop; ``low`` is the lowest
``reasoning_effort`` documented.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 11, 4)
_CONTEXT_WINDOW = 1_048_576
_CAPABILITIES = ["tools", "thinking", "reasoning_floor_low"]

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="kimi-k3.1-pro",
        display_name="Kimi K3.1 Pro",
        context_window=_CONTEXT_WINDOW,
        supports_vision=True,
        capabilities=_CAPABILITIES,
        pricing=ModelPricing(
            input_per_million=2.00,
            output_per_million=10.00,
            cache_read_per_million=0.20,
            cache_write_per_million=2.00,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="kimi-k3.1-pro-highspeed",
        display_name="Kimi K3.1 Pro Highspeed",
        context_window=_CONTEXT_WINDOW,
        supports_vision=True,
        capabilities=_CAPABILITIES,
        pricing=ModelPricing(
            input_per_million=4.00,
            output_per_million=20.00,
            cache_read_per_million=0.20,
            cache_write_per_million=4.00,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="kimi-k3",
        display_name="Kimi K3",
        context_window=_CONTEXT_WINDOW,
        supports_vision=True,
        capabilities=_CAPABILITIES,
        pricing=ModelPricing(
            input_per_million=3.00,
            output_per_million=15.00,
            cache_read_per_million=0.30,
            cache_write_per_million=3.00,
            verified=_VERIFIED,
        ),
    ),
]
