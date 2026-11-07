"""The callbacks the framework hands an AIChannel when it registers it, for
those no public alias already names (``BeforeToolCallback``,
``PlanUpdatedCallback`` and ``AfterResponseCallback`` do).

``register_channel`` builds each one from the room's hooks and sets it on the
channel; the mixins that call them read these types.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.core.hooks import SyncPipelineResult
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.models.tool_call import AIGenerationEvent, ToolRoundEvent
    from roomkit.speaking.base import SpeakDecision, SpeakDecisionEvent
    from roomkit.speaking.thought import ThoughtEvent

BeforeGenerationHook = Callable[["AIGenerationEvent"], Awaitable["SyncPipelineResult"]]
"""BEFORE_AI_GENERATION for a turn's context: allowed or blocked, and why. The
public ``BeforeGenerationCallback`` leaves the result untyped, which models
cannot name without importing core."""

AfterToolRoundHook = Callable[["ToolRoundEvent"], Awaitable[None]]
"""AFTER_TOOL_ROUND for a round the channel ran: the room's hooks act on the
event, the loop then applies what they asked."""

ThinkingHook = Callable[[str, str, int], Awaitable[None]]
"""ON_AI_THINKING: ``(room_id, thinking, round_idx)``."""

SpeakDecisionHook = Callable[["SpeakDecisionEvent"], Awaitable["SpeakDecision"]]
"""ON_SPEAK_DECISION for one channel's decision on one event (RFC §6.4): the
decision that stands once the room's SYNC hooks had their say."""

ThoughtHook = Callable[["ThoughtEvent"], Awaitable[None]]
"""ON_THOUGHT for one channel's new thought in one room (RFC §6.4)."""

ToolUsageLoader = Callable[[str], Awaitable[list[dict[str, Any]]]]
"""A room's stored tool calls, read to rebuild the channel's tool memory."""

ToolHistoryLoader = Callable[[str, int], Awaitable[list["RoomEvent"]]]
"""The channel's stored tool rows in a room after a timeline index, oldest
first, read for the turn's history (RFC §6.4)."""

RoomTasksLoader = Callable[[str], Awaitable[list[dict[str, Any]]]]
"""A room's background tasks as the StatusBus lists them, read for the turn's
notes (RFC §23.4)."""

RoomVisionLoader = Callable[["RoomContext"], Awaitable[str | None]]
"""What a video channel bound to the turn's room last saw, as the note the
turn carries (RFC §12.8.7); ``None`` when none has a live session's result."""
