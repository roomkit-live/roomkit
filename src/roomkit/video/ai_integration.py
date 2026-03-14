"""Wire video vision results into AIChannel conversation context.

Provides a one-call setup that injects vision descriptions into
the AI's system prompt so it can "see" the video feed.

Usage::

    from roomkit.video.ai_integration import setup_video_vision

    kit = RoomKit()
    # ... register video channel and AI channel ...
    setup_video_vision(kit, room_id="my-room", ai_channel_id="ai")

When a vision result arrives, the AI's system prompt is augmented
with the latest camera description. The AI can then reference
what it "sees" in its responses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.framework_event import FrameworkEvent

logger = logging.getLogger("roomkit.video.ai_integration")


def setup_video_vision(
    kit: RoomKit,
    room_id: str,
    ai_channel_id: str,
    *,
    context_prefix: str = "You can see a live camera feed. Current view:",
) -> None:
    """Wire video vision results into an AIChannel's context.

    .. deprecated::
        Auto-injection via ``VideoHooksMixin._update_ai_vision_context``
        is now the default for any room with a video+AI channel pair.
        Use this function only for custom ``context_prefix`` or to
        target a specific AI channel in multi-AI setups.

    Registers a framework event handler that listens for
    ``video_vision_result`` events and updates the AI channel's
    binding metadata with the latest vision description. The AI
    will include this context in its next response.

    Args:
        kit: The RoomKit instance.
        room_id: The room where video and AI channels are attached.
        ai_channel_id: The AI channel to receive vision context.
        context_prefix: Text prepended to the vision description
            in the system prompt supplement.
    """
    import warnings

    warnings.warn(
        "setup_video_vision() is deprecated — auto-injection via "
        "VideoHooksMixin._update_ai_vision_context is now the default. "
        "Use this only for custom context_prefix or multi-AI targeting.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Cache the base system prompt (resolved once on first vision event)
    _base_prompt: list[str | None] = [None]

    async def _on_vision(event: FrameworkEvent) -> None:
        if event.room_id != room_id:
            return
        description = event.data.get("description", "")
        if not description:
            return

        labels = event.data.get("labels", [])
        text = event.data.get("text")

        # Build vision context block
        parts = [f"{context_prefix} {description}"]
        if labels:
            parts.append(f"Objects detected: {', '.join(labels)}")
        if text:
            parts.append(f"Text visible: {text}")
        vision_context = "\n".join(parts)

        try:
            binding = await kit._store.get_binding(room_id, ai_channel_id)
            if binding is None:
                return
            meta = dict(binding.metadata)

            # Resolve base prompt once: binding metadata > AIChannel instance
            if _base_prompt[0] is None:
                stored = meta.get("system_prompt")
                if stored:
                    _base_prompt[0] = stored
                else:
                    ch = kit._channels.get(ai_channel_id)
                    _base_prompt[0] = getattr(ch, "_system_prompt", "") or ""

            base = _base_prompt[0] or ""
            meta["system_prompt"] = f"{base}\n\n{vision_context}" if base else vision_context
            updated = binding.model_copy(update={"metadata": meta})
            await kit._store.update_binding(updated)
        except Exception:
            logger.exception("Failed to inject vision context into AI channel")

    kit.on("video_vision_result")(_on_vision)
