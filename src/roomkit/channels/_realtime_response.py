"""Provider response lifecycle for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.telemetry.base import Attr, SpanKind

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")


@runtime_checkable
class RealtimeResponseHost(Protocol):
    """Contract: capabilities a host class must provide for RealtimeResponseMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_rooms: Maps session IDs to room IDs.
        _session_resamplers: Per-session (inbound, outbound) resampler pairs.
        _session_transport_rates: Negotiated transport sample rate per session.
        _audio_forward_count: Count of audio chunks forwarded per session.
        _turn_spans: Active telemetry turn span per session.
        _session_spans: Active telemetry session span per session.
        _provider_idle: Whether the provider is idle per session.
        _user_speaking: Whether the user is currently speaking per session.
        _provider: The realtime voice provider.
        _transport: The voice backend transport.
        _framework: The RoomKit framework instance (or None).
        channel_id: Channel identifier.
        _telemetry_provider: Telemetry provider for spans.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task: Schedule an async task with exception handling.
        _send_client_message: Send a JSON message to the client UI.
        _update_idle_event: Update the idle signaling event for a session.
        _set_idle: Mark a session as idle after response completes.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_resamplers: dict[str, Any]
    _session_transport_rates: dict[str, int]
    _audio_forward_count: dict[str, int]
    _turn_spans: dict[str, Any]
    _session_spans: dict[str, Any]
    _provider_idle: dict[str, bool]
    _user_speaking: dict[str, bool]
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    channel_id: str
    _telemetry_provider: Any

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    async def _send_client_message(self, session: Any, message: dict[str, Any]) -> None: ...

    def _update_idle_event(self, session_id: str) -> None: ...

    async def _set_idle(self, session: Any) -> None: ...


class RealtimeResponseMixin:
    """Response start/end, idle signaling, turn spans for RealtimeVoiceChannel.

    Host contract: :class:`RealtimeResponseHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_resamplers: dict[str, Any]
    _session_transport_rates: dict[str, int]
    _audio_forward_count: dict[str, int]
    _turn_spans: dict[str, Any]
    _session_spans: dict[str, Any]
    _provider_idle: dict[str, bool]
    _user_speaking: dict[str, bool]
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    channel_id: str
    _telemetry_provider: Any

    _track_task: Any  # see RealtimeResponseHost — cross-mixin
    _send_client_message: Any  # see RealtimeResponseHost — cross-mixin
    _update_idle_event: Any  # see RealtimeResponseHost — cross-mixin
    _set_idle: Any  # see RealtimeResponseHost — cross-mixin

    def _on_provider_response_start(self, session: VoiceSession) -> Any:
        """Handle AI response start — activate AEC, publish typing indicator."""
        with self._state_lock:
            # AI is responding → user has stopped speaking.  Clear the flag
            # so _on_provider_audio stops dropping outbound audio.
            self._user_speaking[session.id] = False
        self._provider_idle[session.id] = False
        self._update_idle_event(session.id)
        # Activate AEC: echo cancellation is bypassed until playback starts
        # to avoid the stale adaptive filter suppressing user speech.
        pipeline_cfg = getattr(self, "_pipeline_config", None)
        if pipeline_cfg is not None and pipeline_cfg.aec is not None:
            pipeline_cfg.aec.set_active(True)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._handle_response_indicator(session, is_speaking=True),
            name=f"rt_response_start:{session.id}",
        )

    def _on_provider_response_end(self, session: VoiceSession) -> Any:
        """Handle AI response end — flush, signal, clear indicator.

        IMPORTANT: end_of_response is scheduled as a task (not called
        synchronously) so it runs AFTER all pending ``_send_outbound_audio``
        tasks.  Those tasks were created via ``loop.create_task()`` in
        ``_on_provider_audio`` and asyncio processes ready tasks in FIFO
        creation order.
        """
        # Flush the outbound resampler's pending frame (sinc one-frame delay)
        with self._state_lock:
            resamplers = self._session_resamplers.get(session.id)
            transport_rate = self._session_transport_rates.get(session.id)
        if resamplers and transport_rate:
            flushed = resamplers[1].flush(transport_rate, 1, 2)
            if flushed and flushed.data:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass
                else:
                    self._track_task(
                        loop,
                        self._transport.send_audio(session, flushed.data),
                        name=f"rt_flush_audio:{session.id}",
                    )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._signal_end_of_response(session),
            name=f"rt_signal_eor:{session.id}",
        )
        self._track_task(
            loop,
            self._handle_response_indicator(session, is_speaking=False),
            name=f"rt_response_end:{session.id}",
        )
        self._track_task(
            loop,
            self._set_idle(session),
            name=f"rt_idle:{session.id}",
        )

    async def _signal_end_of_response(self, session: VoiceSession) -> None:
        """Signal end-of-response to the transport.

        Runs as an asyncio task so it executes AFTER all pending
        ``_send_outbound_audio`` tasks, preserving audio->RESPONSE_END
        ordering on the pacer queue.
        """
        self._transport.end_of_response(session)

    async def _handle_response_indicator(
        self, session: VoiceSession, *, is_speaking: bool
    ) -> None:
        """Publish ephemeral speaking indicator for the AI."""
        telemetry = self._telemetry_provider
        if not is_speaking:
            with self._state_lock:
                forwarded = self._audio_forward_count.pop(session.id, 0)
                turn_span_id = self._turn_spans.pop(session.id, None)
            if forwarded:
                logger.info(
                    "Response ended: forwarded %d audio chunks for session %s",
                    forwarded,
                    session.id,
                )
            if turn_span_id:
                turn_attrs: dict[str, Any] = {"audio_chunks_forwarded": forwarded}
                last_usage = getattr(session, "_last_usage", None)
                if last_usage:
                    turn_attrs[Attr.LLM_INPUT_TOKENS] = last_usage.get("input_tokens", 0)
                    turn_attrs[Attr.LLM_OUTPUT_TOKENS] = last_usage.get("output_tokens", 0)
                    session._last_usage = {}
                telemetry.end_span(turn_span_id, attributes=turn_attrs)
        elif is_speaking:
            with self._state_lock:
                self._audio_forward_count[session.id] = 0
                parent = self._session_spans.get(session.id)
                room_id = self._session_rooms.get(session.id)
            turn_span_id = telemetry.start_span(
                SpanKind.REALTIME_TURN,
                "realtime_turn",
                parent_id=parent,
                attributes={Attr.REALTIME_PROVIDER: self._provider.name},
                room_id=room_id,
                session_id=session.id,
                channel_id=self.channel_id,
            )
            with self._state_lock:
                self._turn_spans[session.id] = turn_span_id
        try:
            await self._send_client_message(
                session,
                {
                    "type": "speaking",
                    "speaking": is_speaking,
                    "who": "assistant",
                },
            )

            if self._framework:
                with self._state_lock:
                    room_id = self._session_rooms.get(session.id)
                if room_id:
                    await self._framework.publish_typing(
                        room_id,
                        user_id="assistant",
                        is_typing=is_speaking,
                        data={"session_id": session.id, "source": "realtime_voice"},
                    )

        except Exception:
            logger.exception("Error publishing response indicator for session %s", session.id)

    def _on_provider_error(self, session: VoiceSession, code: str, message: str) -> Any:
        """Handle provider error."""
        logger.error(
            "Realtime provider error for session %s: [%s] %s",
            session.id,
            code,
            message,
        )
