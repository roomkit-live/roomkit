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
**peak** column of the peak/off-peak schedule that takes effect 2026-08-16
16:00 UTC — peak 01:00-04:00 and 06:00-10:00 UTC, off-peak at exactly half.
Two choices are folded into that sentence. The off-peak half has nowhere to
live: :class:`~roomkit.providers.ai.base.ModelPricing` carries one rate per
model with no time dimension, the same way Anthropic's Batch discount and
fast-mode premium are absent by construction, so the catalog states the
undiscounted rate and an off-peak call bills less than quoted rather than more.
And the peak column is stated *ahead* of its effective date — the rates in
force until then are lower (flash $0.14/$0.28, pro $0.435/$0.87) — because a
release cut after the 16th would otherwise ship rates that quietly understate
every call by 3x, and nothing would say so: the price gate cannot compare these
entries against the upstream mirror, which quotes a routed third-party host
instead of DeepSeek's own endpoint.

``cache_write`` is unset because DeepSeek populates its context cache
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
            input_per_million=1.32,
            output_per_million=3.96,
            cache_read_per_million=0.044,
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
            input_per_million=0.44,
            output_per_million=1.32,
            cache_read_per_million=0.014,
            verified=_VERIFIED,
        ),
    ),
]
