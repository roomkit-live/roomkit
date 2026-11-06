"""MiniMax AI provider, on MiniMax's OpenAI-compatible API or a server of your own."""

from __future__ import annotations

from typing import Any, ClassVar

from roomkit.providers.ai.base import AIContext, ModelInfo
from roomkit.providers.ai.chat_request import ChatDialect
from roomkit.providers.ai.reasoning import floored_effort, nearest_level, reasoning_floor
from roomkit.providers.minimax.config import MiniMaxConfig
from roomkit.providers.minimax.models import MODELS
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.vendor_endpoint import MINIMAX_BASE_URL, is_vendor_endpoint

# Earlier reasoning goes back in ``reasoning_content``: MiniMax asks for it in a
# tool loop as "essential for achieving the model's best performance", and
# documents no refusal, so a round that did not reason sends no field. Its chat
# templates read that field, M2's and M3's alike, and a <think> block in the
# content only as a fallback under M2's tag; M3's is <mm:think>.
MINIMAX_CHAT = ChatDialect(thinking_field="reasoning_content")

# The efforts M3.1-Flash takes, least to most (MiniMax's API reference); another
# goes as the nearest of them (RFC §6.7).
_LEVELS = ("low", "medium", "high", "xhigh", "max")
_SCALE = ("minimal", "low", "medium", "high", "xhigh", "max")


class MiniMaxAIProvider(OpenAIAIProvider):
    """Chat with MiniMax's models, a round's reasoning sent back to them.

    The generic :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` sends no
    reasoning back (RFC §6.4). This class is the choice to send it, on
    MiniMax's own API or wherever its weights are served::

        MiniMaxAIProvider(MiniMaxConfig(api_key="...", model="MiniMax-M3.1-Flash"))
        MiniMaxAIProvider(
            MiniMaxConfig(
                api_key="unused",
                base_url="http://gpu:8000/v1",
                model="MiniMaxAI/MiniMax-M3",
            )
        )
    """

    _config: MiniMaxConfig
    _install_extra: ClassVar[str] = "minimax"
    _chat_dialect: ClassVar[ChatDialect] = MINIMAX_CHAT
    # Its API documents response_format for MiniMax-Text-01 alone, which it no
    # longer lists: a response schema is refused up front.
    _response_schema_default: ClassVar[bool] = False

    @property
    def name(self) -> str:
        """Stable provider name in streaming and non-streaming telemetry."""
        return "minimax"

    @property
    def _provider_name(self) -> str:
        return self.name

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Offline metadata. ``list_models()`` asks the endpoint's ``/models``,
        which MiniMax's API documents none of."""
        return list(MODELS)

    @property
    def supports_vision(self) -> bool:
        """What the catalogue states of the model; a model it does not name is
        the server's to decide behind a ``base_url`` (RFC §6.7)."""
        entry = self.catalog_entry()
        if entry is not None and entry.supports_vision is not None:
            return entry.supports_vision
        return not is_vendor_endpoint(self._config.base_url, MINIMAX_BASE_URL)

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Temperature where the config says the server takes it, and reasoning
        effort on a turn with tools as on any other (RFC §6.7).

        On a model the catalogue names, a turn that states off sends its floor:
        ``low`` on M3.1-Flash, which cannot stop reasoning, and on M3 MiniMax's
        own switch, since its ``reasoning_effort`` is effective for M3.1-Flash
        alone; another effort goes as the nearest level M3.1-Flash takes. On any
        other model the effort goes as stated.
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature
        entry = self.catalog_entry()
        floor = reasoning_floor(entry)
        effort = floored_effort(context, self._config.reasoning_effort, floor)
        if effort == "none" and floor == "none":
            kwargs.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}
            return
        if entry is not None:
            effort = nearest_level(effort, _LEVELS, scale=_SCALE)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
