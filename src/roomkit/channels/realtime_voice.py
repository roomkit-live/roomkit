"""RealtimeVoiceChannel — wraps speech-to-speech AI APIs as a RoomKit channel."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.voice.backends.base import VoiceBackend
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.utils import rms_db

try:
    from websockets.exceptions import ConnectionClosed as _ConnectionClosed
except ImportError:  # websockets not installed
    _ConnectionClosed = ConnectionError  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

# Tool handler: async callable (session, name, arguments) -> result dict or str
ToolHandler = Callable[
    ["VoiceSession", str, dict[str, Any]],
    Awaitable[dict[str, Any] | str],
]

logger = logging.getLogger("roomkit.channels.realtime_voice")


class RealtimeVoiceChannel(Channel):
    """Real-time voice channel using speech-to-speech AI providers.

    Wraps APIs like OpenAI Realtime and Gemini Live as a first-class
    RoomKit channel. Audio flows directly between the user's browser
    and the provider; transcriptions are emitted into the Room so
    other channels (supervisor dashboards, logging) see the conversation.

    Category is TRANSPORT so that:
    - ``on_event()`` receives broadcasts (for text injection from supervisors)
    - ``deliver()`` is called but returns empty (customer is on voice)

    Example:
        from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()

        channel = RealtimeVoiceChannel(
            "realtime-1",
            provider=provider,
            transport=transport,
            system_prompt="You are a helpful agent.",
        )
        kit.register_channel(channel)
    """

    channel_type = ChannelType.REALTIME_VOICE
    category = ChannelCategory.TRANSPORT
    direction = ChannelDirection.BIDIRECTIONAL

    def __init__(
        self,
        channel_id: str,
        *,
        provider: RealtimeVoiceProvider,
        transport: VoiceBackend,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        transport_sample_rate: int | None = None,
        emit_transcription_events: bool = True,
        tool_handler: ToolHandler | None = None,
        mute_on_tool_call: bool = False,
        tool_result_max_length: int = 16384,
    ) -> None:
        """Initialize realtime voice channel.

        Args:
            channel_id: Unique channel identifier.
            provider: The realtime voice provider (OpenAI, Gemini, etc.).
            transport: The audio transport (WebSocket, etc.).
            system_prompt: Default system prompt for the AI.
            voice: Default voice ID for audio output.
            tools: Default tool/function definitions.
            temperature: Default sampling temperature.
            input_sample_rate: Default input audio sample rate (Hz).
            output_sample_rate: Default output audio sample rate (Hz).
            transport_sample_rate: Sample rate of audio from the transport (Hz).
                When set and different from provider rates, enables automatic
                resampling.  When ``None`` (default), no resampling is performed
                — backwards compatible with WebSocket transports.
            emit_transcription_events: If True, emit final transcriptions
                as RoomEvents so other channels see them.
            tool_handler: Async callable to execute tool calls.
                Signature: ``async (session, name, arguments) -> result``.
                Return a dict or JSON string.  If not set, falls back to
                ``ON_REALTIME_TOOL_CALL`` hooks.
            mute_on_tool_call: If True, mute the transport microphone during
                tool execution to prevent barge-in that causes providers
                (e.g. Gemini) to silently drop the tool result.  Defaults
                to False — use ``set_access()`` for fine-grained control.
            tool_result_max_length: Maximum character length of tool results
                before truncation.  Large results (e.g. SVG payloads) can
                overflow the provider's context window.  Defaults to 16384.
        """
        super().__init__(channel_id)
        self._provider = provider
        self._transport = transport
        self._system_prompt = system_prompt
        self._voice = voice
        self._tools = tools
        self._temperature = temperature
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._transport_sample_rate = transport_sample_rate
        self._emit_transcription_events = emit_transcription_events
        self._tool_handler = tool_handler
        self._mute_on_tool_call = mute_on_tool_call
        self._tool_result_max_length = tool_result_max_length
        self._framework: RoomKit | None = None

        # Lock for shared state accessed from both asyncio and audio threads
        self._state_lock = threading.Lock()

        # Active sessions: session_id -> (session, room_id, binding)
        self._sessions: dict[str, VoiceSession] = {}
        self._session_rooms: dict[str, str] = {}  # session_id -> room_id
        # Cached bindings for audio gating (access/muted enforcement)
        self._session_bindings: dict[str, ChannelBinding] = {}

        # Per-session resamplers: (inbound, outbound) pairs
        self._session_resamplers: dict[str, tuple[Any, Any]] = {}
        # Per-session transport sample rates (from transport metadata)
        self._session_transport_rates: dict[str, int] = {}
        # Audio forward counters (for diagnostics)
        self._audio_forward_count: dict[str, int] = {}
        # Per-session generation counter: bumped on interrupt so pending
        # send_audio tasks created before the interrupt become stale and skip.
        self._audio_generation: dict[str, int] = {}
        # Throttle audio level hooks to ~10/sec per direction
        self._last_input_level_at: float = 0.0
        self._last_output_level_at: float = 0.0
        # Cached event loop for cross-thread scheduling (e.g. PortAudio callback)
        self._event_loop: asyncio.AbstractEventLoop | None = None

        # Track fire-and-forget tasks for clean shutdown
        self._scheduled_tasks: set[asyncio.Task[Any]] = set()

        # Telemetry span tracking: session_id -> span_id
        self._session_spans: dict[str, str] = {}
        self._turn_spans: dict[str, str] = {}

        # Wire internal callbacks
        provider.on_audio(self._on_provider_audio)
        provider.on_transcription(self._on_provider_transcription)
        provider.on_speech_start(self._on_provider_speech_start)
        provider.on_speech_end(self._on_provider_speech_end)
        provider.on_tool_call(self._on_provider_tool_call)
        provider.on_response_start(self._on_provider_response_start)
        provider.on_response_end(self._on_provider_response_end)
        provider.on_error(self._on_provider_error)

        transport.on_audio_received(self._on_client_audio)
        transport.on_client_disconnected(self._on_client_disconnected)
        if transport.supports_playback_callback:
            transport.on_audio_played(self._on_transport_audio_played)

    @property
    def _telemetry_provider(self) -> NoopTelemetryProvider:
        """Access telemetry provider (set by register_channel)."""
        return getattr(self, "_telemetry", None) or NoopTelemetryProvider()

    def _rt_span_ctx(
        self, session_id: str
    ) -> tuple[str | None, contextvars.Token[str | None] | None]:
        """Set the realtime session span as current for child spans.

        Returns (parent_id, token) — caller must reset via ``reset_span(token)``
        in a finally block.
        """
        from roomkit.telemetry.context import set_current_span

        parent = self._session_spans.get(session_id)
        token = set_current_span(parent) if parent else None
        return parent, token

    def _propagate_telemetry(self) -> None:
        """Propagate telemetry to realtime provider."""
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            self._provider._telemetry = telemetry

    def set_framework(self, framework: RoomKit) -> None:
        """Set the framework reference for event routing.

        Called automatically when the channel is registered with RoomKit.
        """
        self._framework = framework
        self._sync_trace_emitter()

    def on_trace(
        self,
        callback: Any,
        *,
        protocols: list[str] | None = None,
    ) -> None:
        """Register a trace observer and bridge to the transport."""
        super().on_trace(callback, protocols=protocols)
        self._sync_trace_emitter()

    def _sync_trace_emitter(self) -> None:
        """Set or clear the transport trace emitter based on trace_enabled."""
        if self._transport is not None and hasattr(self._transport, "set_trace_emitter"):
            self._transport.set_trace_emitter(
                self.emit_trace if self.trace_enabled else None,
            )

    def resolve_trace_room(self, session_id: str | None) -> str | None:
        """Resolve room_id from realtime session mappings."""
        if session_id is None:
            return None
        return self._session_rooms.get(session_id)

    @property
    def provider_name(self) -> str | None:
        return self._transport.name if self._transport is not None else None

    @property
    def info(self) -> dict[str, Any]:
        return {
            "provider": self._provider.name,
            "transport": self._transport.name,
            "system_prompt": self._system_prompt is not None,
            "voice": self._voice,
        }

    # -- Session lifecycle --

    async def start_session(
        self,
        room_id: str,
        participant_id: str,
        connection: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        """Start a new realtime voice session.

        Connects both the transport (client audio) and the provider
        (AI service), then fires a framework event.

        Args:
            room_id: The room to join.
            participant_id: The participant's ID.
            connection: Protocol-specific connection (e.g. WebSocket).
            metadata: Optional session metadata. May include overrides
                for system_prompt, voice, tools, temperature.

        Returns:
            The created VoiceSession.
        """
        meta = metadata or {}

        session = VoiceSession(
            id=uuid4().hex,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=self.channel_id,
            state=VoiceSessionState.CONNECTING,
            metadata=meta,
        )

        # Start telemetry session span early so transport/provider connect
        # phases appear as children in Jaeger.
        telemetry = self._telemetry_provider
        session_span_id = telemetry.start_span(
            SpanKind.REALTIME_SESSION,
            "realtime_session",
            attributes={
                Attr.REALTIME_PROVIDER: self._provider.name,
                "participant_id": participant_id,
            },
            room_id=room_id,
            session_id=session.id,
            channel_id=self.channel_id,
        )
        self._session_spans[session.id] = session_span_id

        # Per-room config overrides from metadata
        system_prompt = meta.get("system_prompt", self._system_prompt)
        voice = meta.get("voice", self._voice)
        tools = meta.get("tools", self._tools)
        temperature = meta.get("temperature", self._temperature)
        provider_config = meta.get("provider_config")

        # Accept client connection (with telemetry span)
        with telemetry.span(
            SpanKind.BACKEND_CONNECT,
            "transport.accept",
            parent_id=session_span_id,
            session_id=session.id,
            attributes={Attr.BACKEND_TYPE: self._transport.name},
        ):
            await self._transport.accept(session, connection)

        # Determine transport sample rate: prefer per-session (set by
        # transport during accept, e.g. SIP codec negotiation) over the
        # channel-level default.
        transport_rate = session.metadata.get("transport_sample_rate", self._transport_sample_rate)

        # Create per-session resamplers if transport rate differs from provider
        if transport_rate is not None:
            self._session_transport_rates[session.id] = transport_rate
            needs_inbound = transport_rate != self._input_sample_rate
            needs_outbound = transport_rate != self._output_sample_rate
            if needs_inbound or needs_outbound:
                from roomkit.voice.pipeline.resampler.sinc import SincResamplerProvider

                self._session_resamplers[session.id] = (
                    SincResamplerProvider(),  # inbound: transport → provider
                    SincResamplerProvider(),  # outbound: provider → transport
                )

        # Connect to provider (with telemetry span)
        with telemetry.span(
            SpanKind.BACKEND_CONNECT,
            "provider.connect",
            parent_id=session_span_id,
            session_id=session.id,
            attributes={Attr.BACKEND_TYPE: self._provider.name},
        ):
            await self._provider.connect(
                session,
                system_prompt=system_prompt,
                voice=voice,
                tools=tools,
                temperature=temperature,
                input_sample_rate=self._input_sample_rate,
                output_sample_rate=self._output_sample_rate,
                provider_config=provider_config,
            )

        session.state = VoiceSessionState.ACTIVE
        with self._state_lock:
            self._sessions[session.id] = session
            self._session_rooms[session.id] = room_id

        # Cache the channel binding for audio gating
        if self._framework:
            stored_binding = await self._framework._store.get_binding(room_id, self.channel_id)
            if stored_binding is not None:
                with self._state_lock:
                    self._session_bindings[session.id] = stored_binding

        # Fire framework event
        if self._framework:
            await self._framework._emit_framework_event(
                "voice_session_started",
                room_id=room_id,
                channel_id=self.channel_id,
                data={
                    "session_id": session.id,
                    "participant_id": participant_id,
                    "channel_id": self.channel_id,
                    "provider": self._provider.name,
                },
            )

        logger.info(
            "Realtime session %s started: room=%s, participant=%s, provider=%s",
            session.id,
            room_id,
            participant_id,
            self._provider.name,
        )

        return session

    async def end_session(self, session: VoiceSession) -> None:
        """End a realtime voice session.

        Disconnects both provider and transport, fires framework event.

        Args:
            session: The session to end.
        """
        room_id = self._session_rooms.get(session.id, session.room_id)

        # Disconnect provider and transport
        try:
            await self._provider.disconnect(session)
        except Exception:
            logger.exception("Error disconnecting provider for session %s", session.id)

        try:
            await self._transport.disconnect(session)
        except Exception:
            logger.exception("Error disconnecting transport for session %s", session.id)

        session.state = VoiceSessionState.ENDED
        with self._state_lock:
            self._sessions.pop(session.id, None)
            self._session_rooms.pop(session.id, None)
            self._session_bindings.pop(session.id, None)
            self._audio_generation.pop(session.id, None)
        self._session_transport_rates.pop(session.id, None)
        self._audio_forward_count.pop(session.id, None)

        # End any active turn span, then the session span
        telemetry = self._telemetry_provider
        turn_span_id = self._turn_spans.pop(session.id, None)
        if turn_span_id:
            telemetry.end_span(turn_span_id)
        session_span_id = self._session_spans.pop(session.id, None)
        if session_span_id:
            telemetry.end_span(session_span_id)
            telemetry.flush()

        resamplers = self._session_resamplers.pop(session.id, None)
        if resamplers:
            resamplers[0].close()
            resamplers[1].close()

        # Fire framework event
        if self._framework:
            await self._framework._emit_framework_event(
                "voice_session_ended",
                room_id=room_id,
                channel_id=self.channel_id,
                data={
                    "session_id": session.id,
                    "participant_id": session.participant_id,
                    "channel_id": self.channel_id,
                },
            )

        logger.info("Realtime session %s ended", session.id)

    async def reconfigure_session(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        """Reconfigure an active session with new agent parameters.

        Used during agent handoff to switch the AI personality, voice,
        and tools.  Providers with session resumption (e.g. Gemini Live)
        preserve conversation history across the reconfiguration.

        Args:
            session: The active session to reconfigure.
            system_prompt: New system instructions for the AI.
            voice: New voice ID for audio output.
            tools: New tool/function definitions.
            temperature: New sampling temperature.
            provider_config: Provider-specific configuration overrides.
        """
        await self._provider.reconfigure(
            session,
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            provider_config=provider_config,
        )

        # Update channel defaults so new sessions use the current config
        if system_prompt is not None:
            self._system_prompt = system_prompt
        if voice is not None:
            self._voice = voice
        if tools is not None:
            self._tools = tools

        logger.info("Realtime session %s reconfigured", session.id)

    async def connect_session(
        self,
        session: Any,
        room_id: str,
        binding: ChannelBinding,
    ) -> None:
        """Accept a realtime voice session via process_inbound.

        Delegates to :meth:`start_session` which handles provider/transport
        connection, resampling, and framework events.
        """
        await self.start_session(
            room_id,
            session.participant_id,
            connection=session,
            metadata=getattr(session, "metadata", None),
        )

    async def disconnect_session(self, session: Any, room_id: str) -> None:
        """Clean up realtime sessions on remote disconnect."""
        for rt_session in self._get_room_sessions(room_id):
            await self.end_session(rt_session)

    def update_binding(self, room_id: str, binding: ChannelBinding) -> None:
        """Update cached bindings for all sessions in a room.

        Called by the framework after mute/unmute/set_access so the
        audio gate in ``_forward_client_audio`` sees the new state.
        """
        with self._state_lock:
            for sid, rid in self._session_rooms.items():
                if rid == room_id:
                    self._session_bindings[sid] = binding

    # -- Channel ABC --

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        """Not used directly — audio flows via start_session."""
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
                provider=self.provider_name,
            ),
            content=message.content,
        )

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """React to events from other channels — TEXT INJECTION.

        When a supervisor or other channel sends a message, extract the text
        and inject it into the provider session so the AI incorporates it.
        Skips events from this channel (self-loop prevention).
        """
        # Self-loop prevention: skip our own events
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()

        text = self.extract_text(event)
        if not text:
            return ChannelOutput.empty()

        # Determine injection role from event metadata
        inject_role = "system"
        if event.metadata and isinstance(event.metadata, dict):
            inject_role = event.metadata.get("inject_role", "system")

        room_id = event.room_id

        # Inject text into all active sessions for this room
        for session in self._get_room_sessions(room_id):
            try:
                await self._provider.inject_text(session, text, role=inject_role)

                # Fire ON_REALTIME_TEXT_INJECTED hook (async)
                if self._framework:
                    from roomkit.telemetry.context import reset_span

                    _, _tok = self._rt_span_ctx(session.id)
                    try:
                        await self._framework.hook_engine.run_async_hooks(
                            room_id,
                            HookTrigger.ON_REALTIME_TEXT_INJECTED,
                            event,
                            context,
                            skip_event_filter=True,
                        )
                    finally:
                        if _tok is not None:
                            reset_span(_tok)

                logger.info(
                    "Injected text into session %s from channel %s: %.50s",
                    session.id,
                    event.source.channel_id,
                    text,
                )
            except Exception:
                logger.exception(
                    "Error injecting text into session %s",
                    session.id,
                )

        return ChannelOutput.empty()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """No-op delivery — customer is on voice, can't see text."""
        return ChannelOutput.empty()

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.AUDIO, ChannelMediaType.TEXT],
            supports_audio=True,
            custom={"realtime": True, "server_vad": True},
        )

    async def close(self) -> None:
        """End all sessions and close provider + transport."""
        # End all active sessions
        for session in list(self._sessions.values()):
            try:
                await self.end_session(session)
            except Exception:
                logger.exception("Error ending session %s during close", session.id)

        # Cancel all outstanding scheduled tasks
        tasks = list(self._scheduled_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._scheduled_tasks.clear()

        await self._provider.close()
        await self._transport.close()

    # -- Client messaging --

    async def _send_client_message(self, session: VoiceSession, message: dict[str, Any]) -> None:
        """Send a JSON message to the client via the transport.

        Uses ``getattr`` because ``send_message`` is not part of the
        VoiceBackend ABC — it's a concrete method on transports that
        support it (WebSocket, FastRTC, Local, Mock).
        """
        send = getattr(self._transport, "send_message", None)
        if send is not None:
            await send(session, message)

    # -- Task tracking --

    def _track_task(
        self, loop: asyncio.AbstractEventLoop, coro: Any, *, name: str,
    ) -> asyncio.Task[Any]:
        """Create a tracked asyncio task with automatic cleanup and error logging."""
        task = loop.create_task(coro, name=name)
        self._scheduled_tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        """Done-callback: log exceptions and remove from tracked set."""
        self._scheduled_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Unhandled exception in task %s: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )

    # -- Internal callbacks --

    def _on_client_audio(self, session: VoiceSession, audio: AudioFrame | bytes) -> Any:
        """Forward client audio to provider."""
        if not isinstance(audio, bytes):
            audio = audio.data
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._forward_client_audio(session, audio),
            name=f"rt_client_audio:{session.id}",
        )

    async def _forward_client_audio(self, session: VoiceSession, audio: bytes) -> None:
        if session.state != VoiceSessionState.ACTIVE:
            return
        # Enforce ChannelBinding.access and muted per RFC §7.5
        binding = self._session_bindings.get(session.id)
        if binding is not None and (
            binding.access in (Access.READ_ONLY, Access.NONE) or binding.muted
        ):
            return
        try:
            resamplers = self._session_resamplers.get(session.id)
            transport_rate = self._session_transport_rates.get(session.id)
            if resamplers and transport_rate and transport_rate != self._input_sample_rate:
                from roomkit.voice.audio_frame import AudioFrame

                frame = AudioFrame(
                    data=audio,
                    sample_rate=transport_rate,
                    channels=1,
                    sample_width=2,
                )
                frame = resamplers[0].resample(frame, self._input_sample_rate, 1, 2)
                audio = frame.data

            self._fire_audio_level_task(
                session,
                rms_db(audio),
                HookTrigger.ON_INPUT_AUDIO_LEVEL,
            )
            await self._provider.send_audio(session, audio)
        except (ConnectionError, _ConnectionClosed):
            # WebSocket or TCP connection dropped — mark session so
            # subsequent audio chunks are silently discarded.
            if session.state == VoiceSessionState.ACTIVE:
                logger.warning(
                    "Connection lost for session %s, stopping audio forwarding",
                    session.id,
                )
                session.state = VoiceSessionState.ENDED
        except Exception:
            if session.state == VoiceSessionState.ACTIVE:
                logger.exception("Error forwarding client audio for session %s", session.id)
            # Don't log again — provider already marked session as ended

    def _on_provider_audio(self, session: VoiceSession, audio: bytes) -> None:
        """Resample + forward provider audio to transport.

        Returns ``None`` so the provider's ``_fire_audio_callbacks`` does not
        await anything — the receive loop is never blocked.

        Each task captures the current generation counter so that tasks
        created before an interrupt are silently discarded.
        """
        self._audio_forward_count[session.id] = self._audio_forward_count.get(session.id, 0) + 1
        audio = self._resample_outbound(session, audio)
        if not audio:
            return
        with self._state_lock:
            gen = self._audio_generation.get(session.id, 0)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._send_outbound_audio(session, audio, gen),
            name=f"rt_send_audio:{session.id}",
        )

    async def _send_outbound_audio(self, session: VoiceSession, audio: bytes, gen: int) -> None:
        """Send audio to transport, skipping if the generation is stale."""
        if self._audio_generation.get(session.id, 0) != gen:
            return
        await self._transport.send_audio(session, audio)
        # Fallback: fire output level at queue-insertion time for transports
        # without playback callbacks (WebSocket, WebRTC).  For transports
        # with playback callbacks (LocalAudioBackend), the level fires
        # at real playback pace from _on_transport_audio_played instead.
        if not self._transport.supports_playback_callback:
            self._fire_audio_level_task(
                session,
                rms_db(audio),
                HookTrigger.ON_OUTPUT_AUDIO_LEVEL,
            )

    def _on_transport_audio_played(self, session: VoiceSession, audio: AudioFrame | bytes) -> None:
        """Fire ON_OUTPUT_AUDIO_LEVEL at real playback pace (PortAudio callback).

        Called from the transport's speaker thread via ``on_audio_played``.
        This provides time-aligned output levels when the transport supports it.
        """
        raw = audio if isinstance(audio, bytes) else audio.data
        self._fire_audio_level_task(session, rms_db(raw), HookTrigger.ON_OUTPUT_AUDIO_LEVEL)

    def _resample_outbound(self, session: VoiceSession, audio: bytes) -> bytes:
        """Resample outbound audio from provider rate to transport rate."""
        resamplers = self._session_resamplers.get(session.id)
        transport_rate = self._session_transport_rates.get(session.id)
        if resamplers and transport_rate and transport_rate != self._output_sample_rate:
            from roomkit.voice.audio_frame import AudioFrame

            frame = AudioFrame(
                data=audio,
                sample_rate=self._output_sample_rate,
                channels=1,
                sample_width=2,
            )
            frame = resamplers[1].resample(frame, transport_rate, 1, 2)
            return bytes(frame.data)
        return audio

    def _on_provider_transcription(
        self, session: VoiceSession, text: str, role: str, is_final: bool
    ) -> Any:
        """Handle transcription from provider."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._process_transcription(session, text, role, is_final),
            name=f"rt_transcription:{session.id}",
        )

    async def _process_transcription(
        self, session: VoiceSession, text: str, role: str, is_final: bool
    ) -> None:
        """Process a transcription: fire hooks, emit event, send to client.

        Partial transcriptions skip hooks and telemetry spans — they are
        forwarded to the client UI only.  Final transcriptions go through
        the full pipeline: hooks, client UI, and RoomEvent emission.
        """
        if not self._framework:
            return

        room_id = self._session_rooms.get(session.id)
        if not room_id:
            return

        # Partial transcriptions: send to client UI only, no hooks/telemetry
        if not is_final:
            try:
                await self._send_client_message(
                    session,
                    {
                        "type": "transcription",
                        "text": text,
                        "role": role,
                        "is_final": False,
                    },
                )
            except Exception:
                logger.exception("Error sending partial transcription for session %s", session.id)
            return

        from roomkit.telemetry.context import reset_span

        _, _tok = self._rt_span_ctx(session.id)
        try:
            context = await self._framework._build_context(room_id)

            # Fire ON_TRANSCRIPTION hooks (sync, can modify/block)
            from roomkit.voice.realtime.events import RealtimeTranscriptionEvent

            transcription_event = RealtimeTranscriptionEvent(
                session=session,
                text=text,
                role=role,  # type: ignore[arg-type]
                is_final=is_final,
            )

            hook_result = await self._framework.hook_engine.run_sync_hooks(
                room_id,
                HookTrigger.ON_TRANSCRIPTION,
                transcription_event,
                context,
                skip_event_filter=True,
            )

            if not hook_result.allowed:
                logger.info("Transcription blocked by hook: %s", hook_result.reason)
                return

            # Use potentially modified text
            final_text = text
            if hook_result.event is not None and isinstance(
                hook_result.event, RealtimeTranscriptionEvent
            ):
                final_text = hook_result.event.text
            elif isinstance(hook_result.event, str):
                final_text = hook_result.event

            # Send transcription to client UI
            await self._send_client_message(
                session,
                {
                    "type": "transcription",
                    "text": final_text,
                    "role": role,
                    "is_final": is_final,
                },
            )

            # Emit final transcriptions as RoomEvents
            if self._emit_transcription_events and final_text.strip():
                participant_id = session.participant_id if role == "user" else None
                logger.info(
                    "Emitting transcription as RoomEvent: role=%s, text=%s",
                    role,
                    final_text,
                )
                await self._framework.send_event(
                    room_id,
                    self.channel_id,
                    TextContent(body=final_text),
                    participant_id=participant_id,
                    metadata={
                        "voice_session_id": session.id,
                        "source": "realtime_voice",
                        "role": role,
                    },
                    provider=self.provider_name,
                )

        except Exception:
            logger.exception(
                "Error processing transcription for session %s (room=%s, is_final=%s)",
                session.id,
                room_id,
                is_final,
            )
        finally:
            if _tok is not None:
                reset_span(_tok)

    def _on_provider_speech_start(self, session: VoiceSession) -> Any:
        """Handle speech start from provider's server-side VAD.

        Bumps the generation counter so pending send_audio tasks become
        stale, resets the outbound resampler to discard its buffered frame,
        and signals the transport to interrupt outbound audio.
        """
        # Bump generation — pending tasks with the old generation will skip
        with self._state_lock:
            self._audio_generation[session.id] = self._audio_generation.get(session.id, 0) + 1

        # Discard outbound resampler state so stale audio doesn't leak
        resamplers = self._session_resamplers.get(session.id)
        if resamplers:
            resamplers[1].reset()

        self._transport.interrupt(session)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._handle_speech_event(session, "start"),
            name=f"rt_speech_start:{session.id}",
        )

    def _on_provider_speech_end(self, session: VoiceSession) -> Any:
        """Handle speech end from provider's server-side VAD."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._handle_speech_event(session, "end"),
            name=f"rt_speech_end:{session.id}",
        )

    async def _handle_speech_event(self, session: VoiceSession, event_type: str) -> None:
        """Fire speech hooks and publish ephemeral indicator."""
        if not self._framework:
            return

        room_id = self._session_rooms.get(session.id)
        if not room_id:
            return

        from roomkit.telemetry.context import reset_span

        _, _tok = self._rt_span_ctx(session.id)
        try:
            context = await self._framework._build_context(room_id)
            trigger = (
                HookTrigger.ON_SPEECH_START if event_type == "start" else HookTrigger.ON_SPEECH_END
            )

            await self._framework.hook_engine.run_async_hooks(
                room_id,
                trigger,
                session,
                context,
                skip_event_filter=True,
            )

            # Send speaking indicator to client
            await self._send_client_message(
                session,
                {
                    "type": "speaking",
                    "speaking": event_type == "start",
                    "who": "user",
                },
            )

            # On speech start, tell client to flush audio queue (barge-in)
            if event_type == "start":
                await self._send_client_message(
                    session,
                    {
                        "type": "clear_audio",
                    },
                )

        except Exception:
            logger.exception("Error handling speech %s for session %s", event_type, session.id)
        finally:
            if _tok is not None:
                reset_span(_tok)

    def _on_provider_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Handle tool call from provider."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._handle_tool_call(session, call_id, name, arguments),
            name=f"rt_tool_call:{session.id}:{call_id}",
        )

    async def _handle_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Execute a tool call and submit the result to the provider.

        If a ``tool_handler`` was provided, it is called directly.
        Otherwise, the ``ON_REALTIME_TOOL_CALL`` hook is fired.
        """
        room_id = self._session_rooms.get(session.id)

        from roomkit.telemetry.context import reset_span, set_current_span

        # Set session span context so hooks inside are parented correctly
        _rt_parent = self._session_spans.get(session.id)
        _rt_tok = set_current_span(_rt_parent) if _rt_parent else None

        # Telemetry: tool call span (child of current turn span)
        telemetry = self._telemetry_provider
        parent = self._turn_spans.get(session.id) or self._session_spans.get(session.id)
        tool_span_id = telemetry.start_span(
            SpanKind.REALTIME_TOOL_CALL,
            f"realtime_tool:{name}",
            parent_id=parent,
            attributes={Attr.REALTIME_TOOL_NAME: name},
            room_id=room_id,
            session_id=session.id,
            channel_id=self.channel_id,
        )

        # Optionally mute mic during tool execution to prevent barge-in
        # that causes providers (e.g. Gemini) to silently drop the tool
        # result.  Off by default — use set_access() for fine-grained
        # control, or set mute_on_tool_call=True for the simple toggle.
        if self._mute_on_tool_call and self._transport is not None:
            self._transport.set_input_muted(session, True)

        try:
            result_str: str

            if self._tool_handler is not None:
                # Use the tool handler callback
                logger.info(
                    "Executing tool %s(%s) via handler for session %s",
                    name,
                    call_id,
                    session.id,
                )
                raw = await self._tool_handler(session, name, arguments)
                result_str = raw if isinstance(raw, str) else json.dumps(raw)
            elif self._framework and room_id:
                # Fall back to hooks
                context = await self._framework._build_context(room_id)

                from roomkit.voice.realtime.events import RealtimeToolCallEvent

                tool_event = RealtimeToolCallEvent(
                    session=session,
                    tool_call_id=call_id,
                    name=name,
                    arguments=arguments,
                )

                hook_result = await self._framework.hook_engine.run_sync_hooks(
                    room_id,
                    HookTrigger.ON_REALTIME_TOOL_CALL,
                    tool_event,
                    context,
                    skip_event_filter=True,
                )

                if hook_result.allowed:
                    result_str = json.dumps(
                        hook_result.event.metadata.get("result", {"status": "ok"})
                        if hasattr(hook_result.event, "metadata") and hook_result.event is not None
                        else {"status": "ok"}
                    )
                else:
                    result_str = json.dumps(
                        {"error": hook_result.reason or "Tool call blocked by hook"}
                    )
            else:
                result_str = json.dumps({"error": f"No handler for tool {name}"})

            if len(result_str) > self._tool_result_max_length:
                original_len = len(result_str)
                logger.warning(
                    "Tool result for %s(%s) truncated from %d to %d chars (session %s)",
                    name,
                    call_id,
                    original_len,
                    self._tool_result_max_length,
                    session.id,
                )
                # Reserve space for the truncation notice so the AI model
                # knows the output was cut and may have been rendered
                # elsewhere (e.g. via an MCP extension app).
                notice = (
                    f"\n... [truncated — original result was {original_len} chars. "
                    "The full content has been delivered to the client.]"
                )
                result_str = result_str[: self._tool_result_max_length - len(notice)] + notice

            await self._provider.submit_tool_result(session, call_id, result_str)

            telemetry.end_span(tool_span_id)
            logger.info(
                "Tool call %s(%s) handled for session %s",
                name,
                call_id,
                session.id,
            )

        except Exception:
            telemetry.end_span(tool_span_id, status="error", error_message=f"tool {name} failed")
            logger.exception("Error handling tool call %s for session %s", call_id, session.id)
            try:
                await self._provider.submit_tool_result(
                    session,
                    call_id,
                    json.dumps({"error": "Internal error handling tool call"}),
                )
            except Exception:
                logger.exception("Error submitting fallback tool result")
        finally:
            if self._mute_on_tool_call and self._transport is not None:
                self._transport.set_input_muted(session, False)
            if _rt_tok is not None:
                reset_span(_rt_tok)

    def _on_provider_response_start(self, session: VoiceSession) -> Any:
        """Handle AI response start — publish typing indicator."""
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
        """Handle AI response end — flush resampler, signal transport, clear indicator.

        IMPORTANT: end_of_response is scheduled as a task (not called
        synchronously) so it runs AFTER all pending ``_send_outbound_audio``
        tasks.  Those tasks were created via ``loop.create_task()`` in
        ``_on_provider_audio`` and asyncio processes ready tasks in FIFO
        creation order.  A synchronous call here would put RESPONSE_END on
        the pacer queue *before* the last audio chunks, causing the pacer
        to split one response into a truncated response + an orphan tail
        that never gets its own end marker.
        """
        # Flush the outbound resampler's pending frame (sinc one-frame delay)
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

        # Schedule end_of_response as a task so it runs AFTER all pending
        # audio tasks (flush above + any _send_outbound_audio still queued).
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

    async def _signal_end_of_response(self, session: VoiceSession) -> None:
        """Signal end-of-response to the transport.

        Runs as an asyncio task so it executes AFTER all pending
        ``_send_outbound_audio`` tasks, preserving audio→RESPONSE_END
        ordering on the pacer queue.
        """
        self._transport.end_of_response(session)

    async def _handle_response_indicator(
        self, session: VoiceSession, *, is_speaking: bool
    ) -> None:
        """Publish ephemeral speaking indicator for the AI."""
        telemetry = self._telemetry_provider
        if not is_speaking:
            forwarded = self._audio_forward_count.pop(session.id, 0)
            if forwarded:
                logger.info(
                    "Response ended: forwarded %d audio chunks for session %s",
                    forwarded,
                    session.id,
                )
            # End turn span
            turn_span_id = self._turn_spans.pop(session.id, None)
            if turn_span_id:
                telemetry.end_span(
                    turn_span_id,
                    attributes={"audio_chunks_forwarded": forwarded},
                )
        elif is_speaking:
            self._audio_forward_count[session.id] = 0
            # Start turn span (child of session span)
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

            # Publish to realtime backend for dashboard subscribers
            if self._framework:
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

    def _on_client_disconnected(self, session: VoiceSession) -> Any:
        """Handle client disconnection — end the session."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._handle_client_disconnect(session),
            name=f"rt_client_disconnect:{session.id}",
        )

    async def _handle_client_disconnect(self, session: VoiceSession) -> None:
        """Clean up after client disconnects."""
        if session.id in self._sessions:
            await self.end_session(session)

    # -- Audio level hooks --

    def _fire_audio_level_task(
        self, session: VoiceSession, level_db: float, trigger: HookTrigger
    ) -> None:
        """Schedule a task to fire an audio level hook, throttled to ~10/sec.

        Works from both the event-loop thread and foreign threads (e.g.
        PortAudio speaker callback).
        """
        now = time.monotonic()
        if trigger == HookTrigger.ON_INPUT_AUDIO_LEVEL:
            if now - self._last_input_level_at < 0.1:
                return
            self._last_input_level_at = now
        else:
            if now - self._last_output_level_at < 0.1:
                return
            self._last_output_level_at = now
        if not self._framework:
            return
        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
        if not room_id:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Foreign thread — dispatch via cached event loop.
            cached = self._event_loop
            if cached is not None and cached.is_running():
                cached.call_soon_threadsafe(
                    self._create_audio_level_task, session, level_db, room_id, trigger
                )
            return
        # Cache the loop for future cross-thread calls.
        self._event_loop = loop
        self._create_audio_level_task(session, level_db, room_id, trigger)

    def _create_audio_level_task(
        self,
        session: VoiceSession,
        level_db: float,
        room_id: str,
        trigger: HookTrigger,
    ) -> None:
        """Create the audio level hook task (must be called on the event loop thread)."""
        self._track_task(
            asyncio.get_running_loop(),
            self._fire_audio_level(session, level_db, room_id, trigger),
            name=f"rt_audio_level:{session.id}",
        )

    async def _fire_audio_level(
        self,
        session: VoiceSession,
        level_db: float,
        room_id: str,
        trigger: HookTrigger,
    ) -> None:
        """Fire an ON_INPUT_AUDIO_LEVEL or ON_OUTPUT_AUDIO_LEVEL hook."""
        if not self._framework:
            return
        from roomkit.telemetry.context import reset_span

        _, _tok = self._rt_span_ctx(session.id)
        try:
            from roomkit.voice.events import AudioLevelEvent

            context = await self._framework._build_context(room_id)
            event = AudioLevelEvent(session=session, level_db=level_db)
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                trigger,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing %s hook", trigger)
        finally:
            if _tok is not None:
                reset_span(_tok)

    # -- Helpers --

    def _get_room_sessions(self, room_id: str) -> list[VoiceSession]:
        """Get all active sessions for a room."""
        return [s for s in self._sessions.values() if self._session_rooms.get(s.id) == room_id]
