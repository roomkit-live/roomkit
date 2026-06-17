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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


ConfigProvider = Callable[["ChannelBinding", "RoomContext"], Awaitable[AIChannelTurnConfig | None]]
