"""Per-turn channel configuration resolved by an application callback.

Channel config (system prompt, tools, sampling) is often dynamic in real
deployments — admin edits, per-user gating, feature flags. Snapshotting it
into the channel object or the binding metadata at attach time creates a
second source of truth that goes stale. ``AIChannel(config_provider=...)``
lets the application resolve the current config at the start of every turn
instead; binding-metadata overrides still win on top (they are explicit
per-room operator intent).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.channels.ai import ToolHandler
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.providers.ai.base import AITool


@dataclass(slots=True)
class AIChannelTurnConfig:
    """Config for one generation turn. ``None`` fields keep the channel
    default (or the binding-metadata override when present)."""

    system_prompt: str | None = None
    tools: list[AITool] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    response_schema: dict[str, Any] | None = None
    """JSON Schema the turn's answer must satisfy (RFC §6.7): the final message
    is then one JSON document, or the turn fails with ``ResponseSchemaError``."""
    turn_budget_tokens: int | None = None
    """Billed tokens the turn may spend, cache included (RFC §6.4)."""
    turn_budget_usd: float | None = None
    """What the turn may cost at the model's catalogue price (RFC §6.4)."""
    tool_handler: ToolHandler | None = None
    """Serve the host's tools for this turn. ``None`` keeps the channel's
    handler. Channel-managed tools keep their own dispatch; a turn's handler
    is never installed on the shared channel or reused by another turn."""


ConfigProvider = Callable[["ChannelBinding", "RoomContext"], Awaitable[AIChannelTurnConfig | None]]
