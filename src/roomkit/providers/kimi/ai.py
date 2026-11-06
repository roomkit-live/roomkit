"""Kimi AI provider, on Kimi's OpenAI-compatible API or a server of your own."""

from __future__ import annotations

from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.ai.chat_request import ChatDialect
from roomkit.providers.ai.reasoning import floored_effort, nearest_level, reasoning_floor
from roomkit.providers.ai.tool_declaration import ToolNameRule
from roomkit.providers.kimi.config import KimiConfig
from roomkit.providers.kimi.models import MODELS
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.vendor_endpoint import KIMI_BASE_URL, is_vendor_endpoint

# From Kimi's API reference, not measured.
KIMI_TOOL_NAMES = ToolNameRule("kimi", r"[a-zA-Z_][a-zA-Z0-9_-]{0,127}")

# The efforts K3 takes, least to most (Kimi's API reference); another goes as
# the nearest of them (RFC §6.7), on the scale it is read on.
_LEVELS = ("low", "high", "max")
_SCALE = ("minimal", "low", "medium", "high", "xhigh", "max")

# Earlier reasoning goes back in ``reasoning_content``, on every round that
# called tools: Kimi's platform says it is required in a tool loop, Kimi Code's
# error reference gives the 400 ("thinking is enabled but reasoning_content is
# missing in assistant tool call message"), and a later turn's rounds hold no
# reasoning (RFC §6.4). An empty field reads as a missing one, so a round with
# no reasoning sends a single space: "the minimum value the API accepts",
# LiteLLM's Moonshot transformation says. Not checked on Kimi's wire.
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
        effort on a turn with tools as on any other (RFC §6.7). On a model the
        catalogue names, a turn that states off sends its floor, K3 always
        thinking, and another effort the nearest level K3 takes; on any other
        model the effort goes as stated."""
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        entry = self.catalog_entry()
        effort = floored_effort(context, self._config.reasoning_effort, reasoning_floor(entry))
        if entry is not None:
            effort = nearest_level(effort, _LEVELS, scale=_SCALE)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
