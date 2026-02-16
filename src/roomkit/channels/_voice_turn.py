"""VoiceChannel mixin — turn detection and text routing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.turn.base import TurnEntry

logger = logging.getLogger("roomkit.voice")

# Cap pending audio buffer at ~1MB (≈32s at 16kHz mono 16-bit).
# SmartTurnDetector uses the last 8s (≈256KB) so this is generous.
_MAX_PENDING_AUDIO_BYTES = 1_048_576


class VoiceTurnMixin:
    """Turn detection and text routing for VoiceChannel."""

    # -- attributes provided by VoiceChannel.__init__ --
    channel_id: str
    _framework: RoomKit | None
    _backend: VoiceBackend | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _pending_turns: dict[str, list[TurnEntry]]
    _pending_audio: dict[str, bytearray]

    async def _evaluate_turn(
        self,
        session: VoiceSession,
        text: str,
        room_id: str,
        context: RoomContext,
        *,
        audio_bytes: bytes | None = None,
    ) -> None:
        """Evaluate turn completion using the configured TurnDetector."""
        if not self._framework or not self._pipeline_config:
            return
        turn_detector = self._pipeline_config.turn_detector
        if turn_detector is None:
            await self._route_text(session, text, room_id)
            return

        from roomkit.voice.pipeline.turn.base import TurnContext, TurnEntry

        # Accumulate entry
        entries = self._pending_turns.setdefault(session.id, [])
        entries.append(TurnEntry(text=text, role="user"))

        # Accumulate audio for audio-native turn detectors
        if audio_bytes:
            buf = self._pending_audio.setdefault(session.id, bytearray())
            buf.extend(audio_bytes)
            if len(buf) > _MAX_PENDING_AUDIO_BYTES:
                del buf[: len(buf) - _MAX_PENDING_AUDIO_BYTES]

        accumulated_audio = bytes(self._pending_audio.get(session.id, b"")) or None
        sample_rate = session.metadata.get("input_sample_rate", 16000)

        turn_ctx = TurnContext(
            conversation_history=list(entries),
            silence_duration_ms=0.0,
            transcript=text,
            is_final=True,
            session_id=session.id,
            audio_bytes=accumulated_audio,
            audio_sample_rate=sample_rate,
        )
        decision = turn_detector.evaluate(turn_ctx)

        if decision.is_complete:
            # Fire ON_TURN_COMPLETE hook
            combined = " ".join(e.text for e in entries)
            self._pending_turns.pop(session.id, None)
            self._pending_audio.pop(session.id, None)
            try:
                from roomkit.voice.events import TurnCompleteEvent

                event = TurnCompleteEvent(
                    session=session,
                    text=combined,
                    confidence=decision.confidence,
                )
                await self._framework.hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.ON_TURN_COMPLETE,
                    event,
                    context,
                    skip_event_filter=True,
                )
            except Exception:
                logger.exception("Error firing ON_TURN_COMPLETE hook")

            await self._route_text(session, combined, room_id)
        else:
            # Fire ON_TURN_INCOMPLETE hook
            combined_so_far = " ".join(e.text for e in entries)
            try:
                from roomkit.voice.events import TurnIncompleteEvent

                incomplete_event = TurnIncompleteEvent(
                    session=session,
                    text=combined_so_far,
                    confidence=decision.confidence,
                )
                await self._framework.hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.ON_TURN_INCOMPLETE,
                    incomplete_event,
                    context,
                    skip_event_filter=True,
                )
            except Exception:
                logger.exception("Error firing ON_TURN_INCOMPLETE hook")

    async def _route_text(self, session: VoiceSession, text: str, room_id: str) -> None:
        """Route transcribed text through the inbound pipeline."""
        if not self._framework:
            return
        from roomkit.models.delivery import InboundMessage
        from roomkit.models.event import TextContent
        from roomkit.telemetry.context import reset_span, set_current_span

        inbound = InboundMessage(
            channel_id=self.channel_id,
            sender_id=session.participant_id,
            content=TextContent(body=text),
            metadata={"voice_session_id": session.id, "source": "voice"},
        )
        # Set voice session span as parent so INBOUND_PIPELINE is a child
        session_span = getattr(self, "_voice_session_spans", {}).get(session.id)
        token = set_current_span(session_span) if session_span else None
        try:
            await self._framework.process_inbound(inbound, room_id=room_id)
        finally:
            if token is not None:
                reset_span(token)
