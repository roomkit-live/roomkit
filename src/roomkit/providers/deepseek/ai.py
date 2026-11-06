"""DeepSeek AI provider — generates responses via DeepSeek's OpenAI-compatible API."""

from __future__ import annotations

from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.ai.chat_request import ChatDialect
from roomkit.providers.ai.reasoning import thinking_switch, turn_setting
from roomkit.providers.ai.tool_declaration import ToolNameRule
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.deepseek.models import MODELS
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.vendor_endpoint import DEEPSEEK_BASE_URLS, is_vendor_endpoint

# Measured 2026-10-02: a dot or a colon, or a 129th character, is a 400.
DEEPSEEK_TOOL_NAMES = ToolNameRule("deepseek", r"[A-Za-z0-9_-]{1,128}")

# Earlier reasoning goes back in ``reasoning_content``, on every round that
# called tools: in thinking mode DeepSeek refuses (400) a round of the turn in
# progress without it, absent or null, unless it issued the round's call ids
# itself a moment ago, and takes an empty one (measured 2026-10-02 on
# deepseek-v4-pro and -flash). A <think> block in the content would be read as
# text the model said, and billed beside the reasoning DeepSeek recovers.
DEEPSEEK_CHAT = ChatDialect(thinking_field="reasoning_content", round_thinking_required=True)


class DeepSeekAIProvider(OpenAIAIProvider):
    """AI provider using DeepSeek's OpenAI-compatible Chat Completions API.

    Subclasses :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` —
    DeepSeek speaks the Chat Completions wire format verbatim, so message
    building, tool handling, response parsing, streaming, ``/v1/models``
    discovery, and client construction are all inherited unchanged. What is
    genuinely DeepSeek's own lives here: which models exist, which tool names
    it accepts, that it constrains no output to a schema, how a thinking
    request is spelled, where earlier reasoning goes back, and how a cache hit
    is reported.

    Example::

        provider = DeepSeekAIProvider(
            DeepSeekConfig(api_key="sk-...", model="deepseek-v4-pro")
        )
    """

    _config: DeepSeekConfig
    _install_extra: ClassVar[str] = "deepseek"
    _chat_dialect: ClassVar[ChatDialect] = DEEPSEEK_CHAT
    # DeepSeek's JSON output is the free-form ``json_object`` mode, which
    # constrains nothing to a schema: a response schema is refused up front.
    _response_schema_default: ClassVar[bool] = False

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "deepseek"

    @property
    def _tool_name_rule(self) -> ToolNameRule | None:
        """DeepSeek's rule on its own endpoint; behind another URL the server
        decides (RFC §6.7)."""
        own = is_vendor_endpoint(self._config.base_url, *DEEPSEEK_BASE_URLS)
        return DEEPSEEK_TOOL_NAMES if own else None

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of DeepSeek chat models."""
        return list(MODELS)

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Add temperature and DeepSeek's nested ``thinking`` object.

        Where the OpenAI parent sends a top-level ``reasoning_effort`` string,
        DeepSeek takes ``thinking: {"type": ..., "reasoning_effort": ...}``.
        The two are not interchangeable: a top-level field is silently ignored
        here, so inheriting the parent's version would leave every reasoning
        setting with no effect on the wire. It rides the SDK's ``extra_body``
        passthrough because the OpenAI schema has no such field.

        Unlike the parent, the object is sent on tool turns too — DeepSeek
        documents thinking and tool calls as compatible, and an agentic turn is
        exactly where the cost of reasoning is worth steering.
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        thinking = self._resolve_thinking(context)
        if thinking is not None:
            kwargs.setdefault("extra_body", {})["thinking"] = thinking

    def _resolve_thinking(self, context: AIContext) -> dict[str, Any] | None:
        """Build DeepSeek's ``thinking`` object for this turn, or ``None`` to omit it.

        The switch is the turn's as :func:`thinking_switch` reads it (RFC
        §6.7), before the configured ``enable_thinking``: a ``reasoning_effort``
        of ``none`` turns thinking off, and a ``thinking_budget``'s *size* is
        deliberately dropped — DeepSeek's API ignores token budgets, and
        translating one into an effort tier would invent a mapping the vendor
        does not publish. The effort, the turn's before the configured one,
        rides the object. With nothing set the request stays silent so the
        model's own default (thinking on) applies.
        """
        enabled = thinking_switch(context, self._config.enable_thinking)
        effort = turn_setting(context.reasoning_effort, self._config.reasoning_effort)
        if enabled is False:
            return {"type": "disabled"}
        if enabled is None and effort is None:
            return None
        thinking: dict[str, Any] = {"type": "enabled"}
        if effort is not None:
            thinking["reasoning_effort"] = effort
        return thinking

    @staticmethod
    def _usage_from(raw: Any) -> dict[str, int]:
        """Map DeepSeek's usage object to roomkit's canonical counters.

        DeepSeek reports cache hits as ``prompt_cache_hit_tokens`` /
        ``prompt_cache_miss_tokens`` rather than OpenAI's
        ``prompt_tokens_details.cached_tokens``. The inherited implementation
        looks only at the OpenAI shape, so every cached token would be counted
        — and priced — as ordinary input: a thirty-fold overcharge at DeepSeek's
        published rates. An endpoint that reports neither shape (a proxy) falls
        back to the parent. A counter it leaves out stays unknown, never 0
        (RFC §6.4): without a miss count nor a prompt, there is no input count.
        """
        hit = getattr(raw, "prompt_cache_hit_tokens", None)
        if hit is None:
            return OpenAIAIProvider._usage_from(raw)
        miss = getattr(raw, "prompt_cache_miss_tokens", None)
        prompt = getattr(raw, "prompt_tokens", None)
        if miss is None and prompt is not None:
            miss = max(prompt - hit, 0)
        completion = getattr(raw, "completion_tokens", None)
        usage: dict[str, int] = {}
        if miss is not None:
            usage["input_tokens"] = miss
        if completion is not None:
            usage["output_tokens"] = completion
        if hit:
            usage["cache_read_input_tokens"] = hit
        return {**usage, **OpenAIAIProvider._reasoning_detail(raw)}
