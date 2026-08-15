"""Offline metadata for Alibaba's hosted Qwen models.

Hand-maintained list returned by ``QwenAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what Model Studio currently offers. Unlike every other catalog here it is
also the *only* answer available: the OpenAI-compatible endpoint serves
``/chat/completions`` and nothing else, so ``list_models()`` has no upstream to
ask and returns this same list.

Sourced from Alibaba Cloud Model Studio's model list and billing pages
(alibabacloud.com/help/en/model-studio, ``models`` and
``billing-for-model-studio``), verified 2026-08-14.

Scope is the **commercial Qwen ids Model Studio hosts on its international
deployment**. Two things are deliberately outside it: the third-party models
(DeepSeek, Kimi, GLM, MiniMax) that share the endpoint but not the lineup —
:class:`~roomkit.providers.deepseek.ai.DeepSeekAIProvider` speaks to DeepSeek's
own API instead — and the open-weight ``qwen3-*`` checkpoints anyone can self
host, which are a vLLM or Ollama deployment rather than an id this endpoint
answers to.

Prices are the international **list** prices, per million tokens, read
2026-08-14. Alibaba runs near-permanent limited-time promotions (50% off the
3.7-max line, 20% off 3.7-plus at the time of reading) and quotes different
rates per deployment — Beijing bills qwen3.7-max at $1.65/$4.951, Hong Kong
adds night/day tiers. Neither has a home in
:class:`~roomkit.providers.ai.base.ModelPricing`, which carries one list rate
per model, so the catalog states the list price for the deployment this
provider defaults to and the promotions land as an under-charge rather than an
over-charge. ``cache_read`` is 10% of the input rate, Alibaba's published cache
hit discount; ``cache_write`` is unset because that 125% charge applies to
*explicit* cache creation, which this provider never requests — Model Studio's
caching is implicit here.

``pricing=None`` on two entries is a representation limit, not a free model:
Alibaba tiers ``qwen3-coder-plus`` across four input-length bands ($1/$5 up to
$6/$60) and ``qwen3-vl-plus`` across three, while ``ModelPricing`` carries a
single threshold. Encoding one band would understate a long-context bill by up
to 12x, so those two report no rate rather than a wrong one.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_VERIFIED = date(2026, 8, 14)

#: Alibaba states its pricing tiers in decimal units — "K means 1,000 and M
#: means 1,000,000" — so a 1M window is 1,000,000 tokens, not 2**20.
_LONG_CONTEXT_THRESHOLD = 256_000

MODELS: list[ModelInfo] = [
    ModelInfo(
        id="qwen3.7-max",
        display_name="Qwen3.7 Max",
        context_window=1_000_000,
        # The flagship is the one text-only model of the four: Alibaba's
        # multimodal capacity sits in the plus/flash/vl lines.
        supports_vision=False,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=2.5,
            output_per_million=7.5,
            cache_read_per_million=0.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="qwen3.7-plus",
        display_name="Qwen3.7 Plus",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=0.4,
            output_per_million=1.6,
            cache_read_per_million=0.04,
            # Past 256K both rates triple: $0.4/$1.6 → $1.2/$4.8.
            long_context_threshold_tokens=_LONG_CONTEXT_THRESHOLD,
            long_context_input_multiplier=3.0,
            long_context_output_multiplier=3.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="qwen3.6-flash",
        display_name="Qwen3.6 Flash",
        context_window=1_000_000,
        supports_vision=True,
        capabilities=["tools", "thinking"],
        pricing=ModelPricing(
            input_per_million=0.25,
            output_per_million=1.5,
            cache_read_per_million=0.025,
            # Past 256K: $0.25 → $1.00 input, $1.50 → $4.00 output.
            long_context_threshold_tokens=_LONG_CONTEXT_THRESHOLD,
            long_context_input_multiplier=4.0,
            long_context_output_multiplier=8 / 3,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="qwen3-coder-plus",
        display_name="Qwen3 Coder Plus",
        context_window=1_000_000,
        supports_vision=False,
        capabilities=["tools"],
    ),
    ModelInfo(
        id="qwen3-vl-plus",
        display_name="Qwen3 VL Plus",
        context_window=_LONG_CONTEXT_THRESHOLD,
        supports_vision=True,
        capabilities=["tools", "thinking"],
    ),
]
