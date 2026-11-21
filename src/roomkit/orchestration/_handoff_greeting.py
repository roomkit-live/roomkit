"""A greeting on a handoff over a voice channel: the agent a handoff made
active introduces itself, after the outgoing agent's farewell when one is set
(RFC §10.1.1)."""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import logging
from typing import TYPE_CHECKING

from roomkit.channels.voice import VoiceChannel
from roomkit.core.hooks import HookRegistration
from roomkit.core.task_utils import hold_task
from roomkit.models.context import RoomContext
from roomkit.models.delivery import framework_message
from roomkit.models.enums import ChannelType, EventType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration")

_DEFAULT_PROMPT = (
    "[The caller has just been transferred to you — please introduce yourself briefly]"
)


class HandoffGreeting:
    """Greets on each handoff over *voice_channel_id*: the new agent is cued
    to introduce itself (*greeting_prompt*), the farewell *farewell_prompt*
    spoken before when given, the outgoing agent's speech held meanwhile."""

    def __init__(
        self,
        kit: RoomKit,
        voice_channel_id: str,
        greeting_prompt: str | None,
        farewell_prompt: str | None,
    ) -> None:
        self._kit = kit
        self._voice_channel_id = voice_channel_id
        self._prompt = greeting_prompt or _DEFAULT_PROMPT
        self._farewell = farewell_prompt
        # Rooms where a handoff is in flight — TTS blocked until the greeting
        # task starts.  The flag is set by ON_HANDOFF and cleared at the *start*
        # of _greet (before process_inbound) so the old agent's farewell is
        # only blocked during the brief window between the handoff event and
        # the greeting task starting on the next event-loop tick. The AI model
        # needs ~300-500ms to generate a farewell response, by which time the
        # flag is already cleared and the farewell plays through.
        self._pending: set[str] = set()

    def register(self, priority: int, room_id: str | None = None) -> None:
        """Register :meth:`hooks` on the kit: for every room, or for *room_id*
        alone, each named after it so its uninstall takes it back."""
        for hook in self.hooks(priority):
            if room_id is None:
                # One pipeline's greeting among others on the kit: a name of
                # its own (RFC §9.1).
                self._kit.hook(
                    hook.trigger,
                    execution=hook.execution,
                    priority=hook.priority,
                    name=f"{hook.name}_{id(self):x}",
                )(hook.fn)
            else:
                named = dataclasses.replace(hook, name=f"pipeline{hook.name}_{room_id}")
                self._kit.hook_engine.replace_room_hook(room_id, named)

    def hooks(self, priority: int) -> list[HookRegistration]:
        """Its hooks: ``ON_HANDOFF``, which greets, and, when a farewell is
        set, ``BEFORE_TTS`` at *priority* - 1, which holds the outgoing
        agent's speech while the handoff is pending."""
        hooks = [
            HookRegistration(
                trigger=HookTrigger.ON_HANDOFF,
                execution=HookExecution.ASYNC,
                fn=self._on_handoff,
                name="_on_handoff",
            )
        ]
        if self._farewell:
            hooks.append(
                HookRegistration(
                    trigger=HookTrigger.BEFORE_TTS,
                    execution=HookExecution.SYNC,
                    # BEFORE_TTS hands its hooks the text, not an event.
                    fn=self._block_farewell,  # ty: ignore[invalid-argument-type]
                    priority=priority - 1,
                    name="_block_farewell",
                )
            )
        return hooks

    async def _on_handoff(self, event: RoomEvent, _ctx: RoomContext) -> None:
        meta = event.metadata or {}
        to_agent = meta.get("to_agent", "")
        from_agent = meta.get("from_agent", "")
        logger.info("greet_on_handoff: %s → %s", from_agent, to_agent)

        self._pending.add(event.room_id)
        # Held by the kit, so its close cuts the greeting's turn as it cuts
        # every background turn, in a fresh context: an inherited room lock
        # would fake reentrancy.
        hold_task(self._kit, self._greet(event.room_id, to_agent), context=contextvars.Context())

    async def _greet(self, room_id: str, to_agent: str) -> None:
        """Cue *to_agent* to introduce itself in *room_id*, once the outgoing
        agent's speech and the farewell played."""
        # Clear the block flag *before* process_inbound so both the old
        # agent's farewell TTS and the new agent's greeting TTS can play.
        # The flag only needs to live long enough to suppress any TTS that
        # fires between ON_HANDOFF and the greeting task starting (<1ms).
        self._pending.discard(room_id)

        # Wait for the outgoing agent's TTS to finish (e.g. "let me transfer
        # you") before starting the new agent's greeting.  Without
        # this, deliver_stream() cancels the in-flight TTS immediately.
        vc = self._kit.channels.get(self._voice_channel_id)
        if isinstance(vc, VoiceChannel):
            await vc.wait_playback_done(room_id, timeout=15.0)

        # Speak farewell before the new agent greets.  This is a synthetic
        # TTS utterance — no AI generation needed — so there is always a
        # transition message regardless of what the outgoing model chose to
        # emit.
        if self._farewell and isinstance(vc, VoiceChannel):
            await self._speak_farewell(vc, room_id, self._farewell)

        try:
            result = await asyncio.wait_for(
                # The cue directs the new agent: an instruction, so the room
                # never stores it as the caller's words and only that agent
                # is asked (RFC §10.1.1).
                self._kit.process_inbound(
                    framework_message(
                        channel_id=self._voice_channel_id,
                        event_type=EventType.INSTRUCTION,
                        content=TextContent(body=self._prompt),
                        addressed_to=[to_agent] if to_agent else None,
                    ),
                    room_id=room_id,
                ),
                timeout=30.0,
            )
        except TimeoutError:
            logger.warning("Handoff greeting timed out for room %s", room_id)
        except Exception:
            logger.exception("Handoff greeting failed for room %s", room_id)
        else:
            if result.blocked:
                # A handoff to no agent the kit registers cues nobody
                # (instruction_unaddressed): said, never silent.
                logger.warning(
                    "Handoff greeting to %r refused in room %s: %s",
                    to_agent,
                    room_id,
                    result.reason,
                )

    async def _speak_farewell(self, vc: VoiceChannel, room_id: str, text: str) -> None:
        """Synthesise *text* via TTS and play it to all sessions in *room_id*."""
        bindings = await self._kit.store.list_bindings(room_id)
        voice_bindings = [b for b in bindings if b.channel_id == self._voice_channel_id]
        if not voice_bindings:
            return

        room = await self._kit.get_room(room_id)
        ctx = RoomContext(room=room, bindings=bindings)
        synth_event = RoomEvent(
            room_id=room_id,
            source=EventSource(channel_id=self._voice_channel_id, channel_type=ChannelType.VOICE),
            content=TextContent(body=text),
        )
        await vc.deliver(synth_event, voice_bindings[0], ctx)
        await vc.wait_playback_done(room_id, timeout=15.0)

    async def _block_farewell(self, text: str, ctx: RoomContext) -> HookResult:
        if ctx.room.id in self._pending:
            return HookResult.block("handoff_transition")
        return HookResult.allow()
