"""Kimi AI provider, on Kimi's OpenAI-compatible API or a server of your own."""

from __future__ import annotations

from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.ai.chat_request import ChatDialect
from roomkit.providers.ai.reasoning import floored_effort, reasoning_floor
from roomkit.providers.ai.tool_declaration import ToolNameRule
from roomkit.providers.kimi.config import KimiConfig
from roomkit.providers.kimi.models import MODELS
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.vendor_endpoint import KIMI_BASE_URL, is_vendor_endpoint

# From Kimi's API reference, not measured.
KIMI_TOOL_NAMES = ToolNameRule("kimi", r"[a-zA-Z_][a-zA-Z0-9_-]{0,127}")

# Earlier reasoning goes back in ``reasoning_content``, on every round that
# called tools: Kimi answers 400 to one without it ("thinking is enabled but
# reasoning_content is missing in assistant tool call message", its error
# reference), and a later turn's rounds hold no reasoning (RFC §6.4). An empty
# field reads as a missing one, so a round with no reasoning sends a single
# space: "the minimum value the API accepts", LiteLLM's Moonshot transformation
# says; not checked on Kimi's wire.
KIMI_CHAT = ChatDialect(
    thinking_field="reasoning_content",
    round_thinking_required=True,
    round_thinking_placeholder=" ",
)


class KimiAIProvider(OpenAIAIProvider):
    """Chat with Kimi's models, a round's reasoning sent back to them.

    The generic :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` sends no
    reasoning back (RFC §6.4). This class is the choice to send it, on Kimi's
    own API or wherever its weights are served::

        KimiAIProvider(KimiConfig(api_key="...", model="kimi-k3.1-pro"))
        KimiAIProvider(
            KimiConfig(
                api_key="unused",
                base_url="http://gpu:8000/v1",
                model="moonshotai/Kimi-K3",
            )
        )
    """

    _config: KimiConfig
    _install_extra: ClassVar[str] = "kimi"
    _chat_dialect: ClassVar[ChatDialect] = KIMI_CHAT

    @property
    def name(self) -> str:
        """Stable provider name in streaming and non-streaming telemetry."""
        return "kimi"

    @property
    def _provider_name(self) -> str:
        return self.name

    @property
    def _tool_name_rule(self) -> ToolNameRule | None:
        """Kimi's rule on its own endpoint; behind another URL the server
        decides (RFC §6.7)."""
        own = is_vendor_endpoint(self._config.base_url, KIMI_BASE_URL)
        return KIMI_TOOL_NAMES if own else None

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Offline metadata; use ``list_models()`` for account availability."""
        return list(MODELS)

    @property
    def supports_vision(self) -> bool:
        """What the catalogue states of the model; a model it does not name is
        the server's to decide behind a ``base_url`` (RFC §6.7)."""
        entry = self.catalog_entry()
        if entry is not None and entry.supports_vision is not None:
            return entry.supports_vision
        return not is_vendor_endpoint(self._config.base_url, KIMI_BASE_URL)

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Temperature where the config says the server takes it, and reasoning
        effort on a turn with tools as on any other; a turn that states off
        sends the model's floor, K3 always thinking (RFC §6.7)."""
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        floor = reasoning_floor(self.catalog_entry())
        effort = floored_effort(context, self._config.reasoning_effort, floor)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
