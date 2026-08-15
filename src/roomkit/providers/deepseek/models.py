"""Offline metadata for DeepSeek chat models.

Hand-maintained list returned by ``DeepSeekAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what DeepSeek currently offers. Call
:meth:`~roomkit.providers.deepseek.ai.DeepSeekAIProvider.list_models` for that;
it reads the account's ``/v1/models``.

Sourced from DeepSeek's models & pricing page (api-docs.deepseek.com,
``quick_start/pricing``), verified 2026-08-14. The lineup is two models, both
served from the same endpoint and both switchable between thinking and
non-thinking modes — the mode is a request parameter, not a separate id, so it
does not double the catalog. Neither reads images: DeepSeek's API is text-only.

Context windows are stated as "1M" by the vendor and published as 1,048,576 by
every mirror, which is the value here.

Prices are DeepSeek's own per-million rates, read 2026-08-14, and they are the
rates in force *until* 2026-08-16 16:00 UTC. From that moment DeepSeek bills
peak/off-peak — peak 01:00-04:00 and 06:00-10:00 UTC, off-peak at half — with
new list rates (flash $0.44/$1.32 peak, pro $1.32/$3.96 peak). Two facts keep
those out of this snapshot: :class:`~roomkit.providers.ai.base.ModelPricing`
carries one rate per model with no time dimension, so a peak/off-peak split has
nowhere to live, and this catalog states what is billed today. The next
catalog refresh carries the peak column, which is the conservative half of the
pair. ``cache_write`` is unset because DeepSeek populates its context cache
automatically and bills nothing for the write.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 8, 14)

#: Both models expose the same 1M window, published as 2**20 by the mirrors.
_CONTEXT_WINDOW = 1_048_576

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="deepseek-v4-pro",
        display_name="DeepSeek V4 Pro",
        context_window=_CONTEXT_WINDOW,
        supports_vision=False,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=0.435,
            output_per_million=0.87,
            cache_read_per_million=0.003625,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="deepseek-v4-flash",
        display_name="DeepSeek V4 Flash",
        context_window=_CONTEXT_WINDOW,
        supports_vision=False,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=0.14,
            output_per_million=0.28,
            cache_read_per_million=0.0028,
            verified=_VERIFIED,
        ),
    ),
]
