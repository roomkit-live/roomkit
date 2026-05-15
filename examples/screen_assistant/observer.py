"""Wire ``video_vision_result`` events into a ``RealtimeVoiceChannel``.

This is a richer variant of :func:`roomkit.video.setup_realtime_vision`
(in ``src/roomkit/video/ai_integration.py``). The framework helper dedups
on exact-string equality and always injects silently; we dedup on a
*term-set diff* (app names + URLs) so the agent reacts to meaningful
changes (e.g. a new app in the foreground) but stays quiet on cosmetic
ones.

Usage::

    setup_screen_vision(kit, room_id, voice_channel, state)

Registers a ``kit.on("video_vision_result")`` handler — same API shape as
the framework's ``setup_realtime_vision``.
"""

from __future__ import annotations

import logging

from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.framework import RoomKit
from roomkit.models.enums import HookExecution, HookTrigger

from .state import ScreenAssistantState
from .vision import build_change_context

logger = logging.getLogger("screen_assistant.observer")

_MAX_LOG_DESC = 150


def setup_screen_vision(
    kit: RoomKit,
    room_id: str,
    voice_channel: RealtimeVoiceChannel,
    state: ScreenAssistantState,
) -> None:
    """Register a vision-injection handler for ``room_id``.

    Listens to ``video_vision_result`` framework events; on each event
    it diffs the description against the previous one and, if the change
    is meaningful, injects a ``[Screen changed] …`` context message into
    every active session on *voice_channel*. Insignificant changes are
    injected ``silent=True`` (added to context without triggering a
    response); significant ones are not silent so the agent can react.

    Barge-in safety: injections are gated on
    ``state.user_has_spoken`` (no inject until the user has produced a
    final turn — protects the greeting) AND on
    ``state.agent_speaking`` (no inject while the realtime model is
    mid-response — protects every later turn). Both gates are
    necessary because Gemini Live's ``silent=True`` is best-effort
    once mic audio is flowing.
    """

    # Track "user has spoken at least once" — opens the first gate.
    @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def _track_user_speaking(event: object, ctx: object) -> None:
        if (
            not state.user_has_spoken
            and getattr(event, "role", None) == "user"
            and getattr(event, "is_final", False)
            and getattr(event, "text", "").strip()
        ):
            state.user_has_spoken = True
            logger.info("User has spoken — vision injections unlocked")

    # Track agent speaking state — the second gate.
    def _on_response_start(_session: object) -> None:
        state.agent_speaking = True

    def _on_response_end(_session: object) -> None:
        state.agent_speaking = False

    voice_channel.provider.on_response_start(_on_response_start)
    voice_channel.provider.on_response_end(_on_response_end)

    @kit.on("video_vision_result")
    async def _on_vision(event: object) -> None:
        if event.room_id != room_id:  # type: ignore[attr-defined]
            return
        data = event.data  # type: ignore[attr-defined]
        description = data.get("description", "")
        if not description:
            return

        state.frame_count += 1
        previous = state.latest_description
        state.record_description(description)

        short = (
            description[:_MAX_LOG_DESC] + "..."
            if len(description) > _MAX_LOG_DESC
            else description
        )
        elapsed_ms = data.get("elapsed_ms", 0)
        logger.info("[Vision %d] (%dms) %s", state.frame_count, elapsed_ms, short)

        # Hard gate: do not inject anything until the user has produced a
        # final transcription. Two reasons:
        #   1. The first vision always looks "significant" (no baseline)
        #      so it would erroneously trigger non-silent injection.
        #   2. Gemini Live's silent=True is best-effort once mic audio is
        #      flowing — the provider falls back to send_realtime_input,
        #      which has no turn_complete control. Even a "silent" inject
        #      can barge in on the agent's own speech. See
        #      providers/gemini/realtime.py:667 ("silent mode is
        #      best-effort via send_realtime_input").
        # The baseline description has already been recorded above, so by
        # the time this gate opens, build_change_context will diff
        # properly against a real previous value.
        if not state.user_has_spoken:
            logger.info(
                "[Vision %d] Suppressed — user hasn't spoken yet "
                "(would risk barge-in on greeting)",
                state.frame_count,
            )
            return
        if state.agent_speaking:
            logger.info(
                "[Vision %d] Suppressed — agent mid-response (would barge in)",
                state.frame_count,
            )
            return
        if state.tool_in_progress:
            logger.info(
                "[Vision %d] Suppressed — tool call in progress "
                "(would race the tool result)",
                state.frame_count,
            )
            return

        context, significant = build_change_context(previous, description)
        if context is None:
            logger.debug(
                "[Vision %d] No meaningful change, skipping injection",
                state.frame_count,
            )
            return

        sessions = voice_channel.get_room_sessions(room_id)
        if not sessions:
            return

        if significant:
            logger.info("[Vision %d] Significant change detected", state.frame_count)

        for session in sessions:
            try:
                await voice_channel.inject_text(
                    session,
                    context,
                    role="user",
                    silent=not significant,
                )
            except Exception:
                logger.exception("[Vision %d] Failed to inject", state.frame_count)


async def send_opening_greeting(
    voice_channel: RealtimeVoiceChannel,
    room_id: str,
    instruction: str = "Begin the session — greet me as instructed in your system prompt.",
) -> None:
    """Trigger the model's first turn so it speaks before the user does.

    Realtime providers (Gemini Live, OpenAI Realtime) don't speak first
    on their own — they wait for input. This helper injects a brief
    user-side prompt that tells the model to produce its opening turn;
    the system prompt's ``## Greeting`` section dictates what it says.

    Equivalent in spirit to ``Agent(auto_greet=True, greeting=...)``,
    but works on a bare ``RealtimeVoiceChannel`` (no orchestration
    pipeline required).
    """
    for session in voice_channel.get_room_sessions(room_id):
        await voice_channel.inject_text(session, instruction, role="user")
