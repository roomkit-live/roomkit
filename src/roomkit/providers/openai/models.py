"""Offline metadata for OpenAI chat/multimodal models.

Hand-maintained list returned by ``OpenAIAIProvider.available_models`` — the
context windows roomkit needs before it can make a network call, not a claim
about what OpenAI currently offers. Call ``OpenAIAIProvider.list_models()``
for that; it queries the account's ``/v1/models``.

Sourced from the OpenAI models and deprecations docs
(developers.openai.com/api/docs/models, .../deprecations), verified 2026-08-05.

Scope is the chat/responses-capable text + multimodal models; embeddings,
audio (whisper/tts), and image-generation models are intentionally omitted.

``deprecated=True`` marks a model OpenAI has given a shutdown date. Models
already past their shutdown are removed outright rather than flagged — a dead
id is a 404, and keeping it here only invites one.

The GPT-5.6 tier reaches its "pro" depth through ``reasoning.mode: "pro"`` on
the base model rather than through a separate id, which is why — unlike 5.4
and 5.5 — it has no ``-pro`` entries.

Prices are the standard synchronous rates from OpenAI's pricing page
(developers.openai.com/api/docs/pricing), read 2026-08-05 — not the Batch
column, which is half of them, and not Flex. ``cache_read`` is OpenAI's
cached-input rate, applied automatically to a repeated prefix; the ``pro``
tiers publish none because they do not cache. GPT-5.6 also publishes an
explicit cache-write rate. Sol and Terra have higher prices beyond 272k input
tokens; Luna has a smaller 400k context and no published long-context tier.
Earlier models leave ``cache_write`` unset.
"""

from __future__ import annotations

from datetime import date

from roomkit.providers.ai.base import ModelInfo, ModelPricing

_CTX_1M = 1_050_000
# What a turn with function tools accepts of ``reasoning_effort`` on Chat
# Completions (RFC §6.7), read by ``OpenAIAIProvider._tool_turn_effort``:
# the turn's effort for the reasoning models before GPT-5.4, only ``none``
# for GPT-5.4 to GPT-5.6 and GPT-6 Sol and Luna (OpenAI's migration guide to
# the Responses API). Every tagged entry checked on the wire 2026-09-30: the
# first group takes ``low`` with tools, the second takes ``none`` (gpt-5.4-mini
# answers 400 to ``low``; gpt-6-sol and gpt-6-luna answer 400 when the effort
# is left out). An untagged entry gets no effort on such a turn.
#
# What a turn that switches reasoning off sends (``reasoning_floor_<level>``,
# read by ``roomkit.providers.ai.reasoning.reasoning_floor``): ``none`` where
# the model takes it, else its lowest level. Checked on the wire 2026-10-04:
# GPT-5.1 and later take ``none``; GPT-5 and its mini and nano answer 400 to
# it and take ``minimal``; o3 and o4-mini take neither and start at ``low``,
# with function tools as without. An entry without the tag is sent nothing.
_TOOLS_REASONING_EFFORT_OFF = ["tools_reasoning_effort", "reasoning_floor_none"]
_TOOLS_REASONING_EFFORT_MINIMAL = ["tools_reasoning_effort", "reasoning_floor_minimal"]
_TOOLS_REASONING_EFFORT_LOW = ["tools_reasoning_effort", "reasoning_floor_low"]
_TOOLS_REASONING_NONE = ["tools_reasoning_none", "reasoning_floor_none"]
# What Chat Completions refuses a model (RFC §6.7), read by
# ``OpenAIAIProvider._check_model_serves`` before the request. GPT-6 Astra and
# GPT-6.1 Sol refuse function tools there whatever the effort (400, checked on
# the wire 2026-10-02: ``none`` and ``minimal`` are not values they take, and
# any other is refused with tools); the ``-pro`` models are served by the
# Responses API only (404 "not a chat model", 2026-10-02).
_CHAT_TOOLS_REFUSED = ["chat_tools_refused"]
_RESPONSES_ONLY = ["responses_only"]
_VERIFIED = date(2026, 8, 5)

# Astra and Sol prices rechecked 2026-09-08:
# https://developers.openai.com/api/docs/models/gpt-6-astra
# https://developers.openai.com/api/docs/models/gpt-5.6-sol
# GPT-6 Sol and Luna added 2026-09-22 from their model pages and the pricing
# page; like Astra they apply 2x input / 1.5x output above 272k input tokens:
# https://developers.openai.com/api/docs/models/gpt-6-sol
# https://developers.openai.com/api/docs/models/gpt-6-luna
# GPT-6.1 Sol added 2026-09-30 from its model page and the pricing page, same
# long-context rule; its cached input is half GPT-6 Sol's:
# https://developers.openai.com/api/docs/models/gpt-6.1-sol
MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-6.1-sol",
        display_name="GPT-6.1 Sol",
        capabilities=_CHAT_TOOLS_REFUSED,
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=10.0,
            cache_read_per_million=0.1,
            cache_write_per_million=2.5,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
            verified=date(2026, 9, 30),
        ),
    ),
    ModelInfo(
        id="gpt-6-astra",
        display_name="GPT-6 Astra",
        capabilities=_CHAT_TOOLS_REFUSED,
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=10.0,
            output_per_million=50.0,
            cache_read_per_million=1.0,
            cache_write_per_million=12.5,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
            verified=date(2026, 9, 8),
        ),
    ),
    ModelInfo(
        id="gpt-6-sol",
        display_name="GPT-6 Sol",
        capabilities=_TOOLS_REASONING_NONE,
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=10.0,
            cache_read_per_million=0.2,
            cache_write_per_million=2.5,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
            verified=date(2026, 9, 22),
        ),
    ),
    ModelInfo(
        id="gpt-6-luna",
        display_name="GPT-6 Luna",
        capabilities=_TOOLS_REASONING_NONE,
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.1,
            output_per_million=0.5,
            cache_read_per_million=0.01,
            cache_write_per_million=0.125,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
            verified=date(2026, 9, 22),
        ),
    ),
    ModelInfo(
        id="gpt-5.6-sol",
        display_name="GPT-5.6 Sol",
        capabilities=_TOOLS_REASONING_NONE,
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=4.0,
            output_per_million=20.0,
            cache_read_per_million=0.4,
            cache_write_per_million=5.0,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
            verified=date(2026, 9, 8),
        ),
    ),
    ModelInfo(
        id="gpt-5.6-terra",
        display_name="GPT-5.6 Terra",
        capabilities=_TOOLS_REASONING_NONE,
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=12.0,
            cache_read_per_million=0.2,
            cache_write_per_million=2.5,
            long_context_threshold_tokens=272_000,
            long_context_input_multiplier=2.0,
            long_context_output_multiplier=1.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.6-luna",
        display_name="GPT-5.6 Luna",
        capabilities=_TOOLS_REASONING_NONE,
        context_window=400_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.2,
            output_per_million=1.2,
            cache_read_per_million=0.02,
            cache_write_per_million=0.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.5",
        display_name="GPT-5.5",
        capabilities=_TOOLS_REASONING_NONE,
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=5.0,
            output_per_million=30.0,
            cache_read_per_million=0.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.5-pro",
        display_name="GPT-5.5 Pro",
        capabilities=_RESPONSES_ONLY,
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=30.0,
            output_per_million=180.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.4",
        display_name="GPT-5.4",
        capabilities=_TOOLS_REASONING_NONE,
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.5,
            output_per_million=15.0,
            cache_read_per_million=0.25,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.4-pro",
        display_name="GPT-5.4 Pro",
        capabilities=_RESPONSES_ONLY,
        context_window=_CTX_1M,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=30.0,
            output_per_million=180.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.4-mini",
        display_name="GPT-5.4 mini",
        capabilities=_TOOLS_REASONING_NONE,
        context_window=400_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.75,
            output_per_million=4.5,
            cache_read_per_million=0.075,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.4-nano",
        display_name="GPT-5.4 nano",
        capabilities=_TOOLS_REASONING_NONE,
        context_window=400_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.2,
            output_per_million=1.25,
            cache_read_per_million=0.02,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.1",
        display_name="GPT-5.1",
        capabilities=_TOOLS_REASONING_EFFORT_OFF,
        context_window=400_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=10.0,
            cache_read_per_million=0.125,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-4.1",
        display_name="GPT-4.1",
        context_window=1_047_576,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=8.0,
            cache_read_per_million=0.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-4.1-mini",
        display_name="GPT-4.1 mini",
        context_window=1_047_576,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.4,
            output_per_million=1.6,
            cache_read_per_million=0.1,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-4.1-nano",
        display_name="GPT-4.1 nano",
        context_window=1_047_576,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=0.1,
            output_per_million=0.4,
            cache_read_per_million=0.025,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-4o",
        display_name="GPT-4o",
        context_window=128_000,
        supports_vision=True,
        pricing=ModelPricing(
            input_per_million=2.5,
            output_per_million=10.0,
            cache_read_per_million=1.25,
            verified=_VERIFIED,
        ),
    ),
    # --- Deprecated: OpenAI has announced a shutdown date -----------------
    ModelInfo(
        id="gpt-5",
        display_name="GPT-5",
        capabilities=_TOOLS_REASONING_EFFORT_MINIMAL,
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=1.25,
            output_per_million=10.0,
            cache_read_per_million=0.125,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5-mini",
        display_name="GPT-5 mini",
        capabilities=_TOOLS_REASONING_EFFORT_MINIMAL,
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=0.25,
            output_per_million=2.0,
            cache_read_per_million=0.025,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5-nano",
        display_name="GPT-5 nano",
        capabilities=_TOOLS_REASONING_EFFORT_MINIMAL,
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=0.05,
            output_per_million=0.4,
            cache_read_per_million=0.005,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="gpt-5.2",
        display_name="GPT-5.2",
        capabilities=_TOOLS_REASONING_EFFORT_OFF,
        context_window=400_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=1.75,
            output_per_million=14.0,
            cache_read_per_million=0.175,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="o3",
        display_name="o3",
        capabilities=_TOOLS_REASONING_EFFORT_LOW,
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=2.0,
            output_per_million=8.0,
            cache_read_per_million=0.5,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="o3-pro",
        display_name="o3-pro",
        capabilities=_RESPONSES_ONLY,
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=20.0,
            output_per_million=80.0,
            verified=_VERIFIED,
        ),
    ),
    ModelInfo(
        id="o4-mini",
        display_name="o4-mini",
        capabilities=_TOOLS_REASONING_EFFORT_LOW,
        context_window=200_000,
        supports_vision=True,
        deprecated=True,
        pricing=ModelPricing(
            input_per_million=1.1,
            output_per_million=4.4,
            cache_read_per_million=0.275,
            verified=_VERIFIED,
        ),
    ),
]
