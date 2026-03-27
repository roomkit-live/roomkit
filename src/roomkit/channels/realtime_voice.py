"""RealtimeVoiceChannel — wraps speech-to-speech AI APIs as a RoomKit channel."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.channels._realtime_vad import RealtimeVADMixin
from roomkit.channels._skill_constants import TOOL_ACTIVATE_SKILL
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
    from roomkit.channels._realtime_skills import RealtimeSkillSupport
    from roomkit.core.framework import RoomKit
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

# Tool handler: async callable (name, arguments) -> result string
ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]

_current_voice_session: contextvars.ContextVar[VoiceSession | None] = contextvars.ContextVar(
    "_current_voice_session",
    default=None,
)


def get_current_voice_session() -> VoiceSession | None:
    """Get the voice session for the current tool call.

    Available inside tool handlers called by RealtimeVoiceChannel.
    Returns None outside of a tool call context.
    """
    return _current_voice_session.get()


logger = logging.getLogger("roomkit.channels.realtime_voice")


class RealtimeVoiceChannel(RealtimeVADMixin, Channel):
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
        tools: list[dict[str, Any] | Any] | None = None,
        temperature: float | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        transport_sample_rate: int | None = None,
        emit_transcription_events: bool = True,
        tool_handler: ToolHandler | None = None,
        mute_on_tool_call: bool = False,
        tool_result_max_length: int = 16384,
        recording: Any | None = None,
        skills: SkillRegistry | None = None,
        script_executor: ScriptExecutor | None = None,
    ) -> None:
        """Initialize realtime voice channel.

        Args:
            channel_id: Unique channel identifier.
            provider: The realtime voice provider (OpenAI, Gemini, etc.).
            transport: The audio transport (WebSocket, etc.).
            system_prompt: Default system prompt for the AI.
            voice: Default voice ID for audio output.
            tools: Tool definitions as dicts, or Tool objects with
                ``.definition`` and ``.handler``.  Tool objects have
                their handlers extracted and composed automatically.
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
                Signature: ``async (name, arguments) -> str``.
                If not set, falls back to handlers extracted from Tool
                objects, or ``ON_TOOL_CALL`` hooks.
            mute_on_tool_call: If True, mute the transport microphone during
                tool execution to prevent barge-in that causes providers
                (e.g. Gemini) to silently drop the tool result.  Defaults
                to False — use ``set_access()`` for fine-grained control.
            tool_result_max_length: Maximum character length of tool results
                before truncation.  Large results (e.g. SVG payloads) can
                overflow the provider's context window.  Defaults to 16384.
            recording: Optional ``ChannelRecordingConfig`` to enable
                room-level audio recording from this channel. Records
                both input (mic) and output (AI) audio tracks.
            skills: Optional ``SkillRegistry`` with discovered skills.
                When provided, skill infrastructure tools are injected
                and the skills preamble is appended to the system prompt.
            script_executor: Optional ``ScriptExecutor`` for running
                skill scripts.  Ignored when *skills* is ``None``.
        """
        super().__init__(channel_id)
        self._provider: RealtimeVoiceProvider = provider
        self._transport = transport
        self._recording = recording
        self._system_prompt = system_prompt
        self._voice = voice
        self._temperature = temperature
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._transport_sample_rate = transport_sample_rate
        self._emit_transcription_events = emit_transcription_events

        # Extract Tool objects: split into definition dicts + composed handler
        from roomkit.tools.base import Tool as _ToolProto

        tool_defs: list[dict[str, Any]] | None = None
        extracted_handler: ToolHandler | None = None
        if tools:
            has_tool_objects = any(isinstance(t, _ToolProto) for t in tools)
            if has_tool_objects:
                from roomkit.tools.compose import extract_tools

                ai_tools, extracted_handler = extract_tools(tools)
                tool_defs = [
                    {"name": t.name, "description": t.description, "parameters": t.parameters}
                    for t in ai_tools
                ]
            else:
                tool_defs = tools

        self._tools = tool_defs

        # Merge explicit tool_handler with handlers extracted from Tool objects
        effective_handler = tool_handler
        if extracted_handler and tool_handler:
            from roomkit.tools.compose import compose_tool_handlers

            effective_handler = compose_tool_handlers(tool_handler, extracted_handler)
        elif extracted_handler:
            effective_handler = extracted_handler
        self._tool_handler = effective_handler
        self._mute_on_tool_call = mute_on_tool_call
        self._tool_result_max_length = tool_result_max_length
        self._framework: RoomKit | None = None

        # Skills support — skill defs are composed into the tool list at
        # session-start / reconfigure time, NOT stored in self._tools, to
        # avoid doubling when reconfigure_session updates self._tools.
        self._skill_support: RealtimeSkillSupport | None = None
        if skills and skills.skill_count > 0:
            from roomkit.channels._realtime_skills import RealtimeSkillSupport

            self._skill_support = RealtimeSkillSupport(skills, script_executor)

        # Lock for shared state accessed from both asyncio and audio threads
        self._state_lock = threading.Lock()

        # Active sessions: session_id -> (session, room_id, binding)
        self._sessions: dict[str, VoiceSession] = {}
        self._session_rooms: dict[str, str] = {}  # session_id -> room_id
        # Cached bindings for audio gating (access/muted enforcement)
        self._session_bindings: dict[str, ChannelBinding] = {}

        # Per-session resolved tool list (channel defaults + metadata overrides),
        # cached so skill activation can reconfigure without losing tools.
        self._session_tools: dict[str, list[dict[str, Any]]] = {}

        # Per-session recording tracks: session_id -> (audio_track, room_id)
        self._recording_tracks: dict[str, tuple[Any, str]] = {}

        # Per-session resamplers: (inbound, outbound) pairs
        self._session_resamplers: dict[str, tuple[Any, Any]] = {}
        # Per-session transport sample rates (from transport metadata)
        self._session_transport_rates: dict[str, int] = {}
        # Audio forward counters (for diagnostics)
        self._audio_forward_count: dict[str, int] = {}
        # Per-session generation counter: bumped on interrupt so pending
        # send_audio tasks created before the interrupt become stale and skip.
        self._audio_generation: dict[str, int] = {}
        # Last assistant text per session (for barge-in event context)
        self._last_assistant_text: dict[str, str] = {}
        # Barge-in state: set when user interrupts AI, cleared on next final transcription
        self._barge_in_active: set[str] = set()
        # Throttle audio level hooks to ~10/sec per direction
        self._last_input_level_at: float = 0.0
        self._last_output_level_at: float = 0.0
        # Cached event loop for cross-thread scheduling (e.g. PortAudio callback)
        self._event_loop: asyncio.AbstractEventLoop | None = None
        # Manual VAD mode: local pipeline VAD drives speech events
        self._manual_vad: bool = False

        # Idle tracking: set when BOTH provider is done AND user is not speaking
        self._idle_events: dict[str, asyncio.Event] = {}
        self._user_speaking: dict[str, bool] = {}
        self._provider_idle: dict[str, bool] = {}

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

    @property
    def provider(self) -> RealtimeVoiceProvider:
        """The underlying realtime voice provider."""
        return self._provider

    @property
    def session_rooms(self) -> dict[str, str]:
        """Mapping of session_id to room_id."""
        with self._state_lock:
            return dict(self._session_rooms)

    def get_room_sessions(self, room_id: str) -> list[VoiceSession]:
        """Get all active sessions for a room."""
        with self._state_lock:
            return [s for s in self._sessions.values() if self._session_rooms.get(s.id) == room_id]

    async def wait_idle(self, room_id: str, timeout: float = 15.0) -> None:
        """Wait until all sessions in the room are idle (not speaking).

        An idle session has finished its last response and all audio
        has been forwarded to the transport.
        """
        for session in self.get_room_sessions(room_id):
            event = self._idle_events.get(session.id)
            if event is not None and not event.is_set():
                await asyncio.wait_for(event.wait(), timeout=timeout)

    async def _set_idle(self, session: VoiceSession) -> None:
        """Mark provider as idle (called after response end + audio drain)."""
        self._provider_idle[session.id] = True
        self._update_idle_event(session.id)

    def _update_idle_event(self, session_id: str) -> None:
        """Update the idle event based on combined provider + user state."""
        idle = self._idle_events.get(session_id)
        if idle is None:
            return
        provider_done = self._provider_idle.get(session_id, True)
        user_silent = not self._user_speaking.get(session_id, False)
        if provider_done and user_silent:
            idle.set()
        else:
            idle.clear()

    @property
    def tool_handler(self) -> ToolHandler | None:
        """The current tool handler for realtime tool calls."""
        return self._tool_handler

    @tool_handler.setter
    def tool_handler(self, value: ToolHandler | None) -> None:
        self._tool_handler = value

    def _rt_span_ctx(
        self, session_id: str
    ) -> tuple[str | None, contextvars.Token[str | None] | None]:
        """Set the realtime session span as current for child spans.

        Returns (parent_id, token) — caller must reset via ``reset_span(token)``
        in a finally block.
        """
        from roomkit.telemetry.context import set_current_span

        with self._state_lock:
            parent = self._session_spans.get(session_id)
        token = set_current_span(parent) if parent else None
        return parent, token

    def _propagate_telemetry(self) -> None:
        """Propagate telemetry to realtime provider."""
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            self._provider._telemetry = telemetry  # type: ignore[attr-defined]

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
        with self._state_lock:
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

    def configure(
        self,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update channel defaults for future sessions.

        Active sessions are not affected — use ``reconfigure_session``
        for those.
        """
        if system_prompt is not None:
            self._system_prompt = system_prompt
        if voice is not None:
            self._voice = voice
        if tools is not None:
            self._tools = tools

    # -- Public helpers --

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> None:
        """Inject a text turn into the provider session.

        Args:
            session: The active voice session.
            text: Text to inject.
            role: Role for the text ('user' or 'system').
            silent: If True, add to conversation context without
                requesting a response.  The agent sees the text on
                its next turn but does not react immediately.
        """
        await self._provider.inject_text(session, text, role=role, silent=silent)
        logger.info(
            "Injected text into session %s (role=%s, silent=%s, len=%d)",
            session.id,
            role,
            silent,
            len(text),
        )

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

        # Initialize idle tracking (starts idle — no response, no speech)
        idle = asyncio.Event()
        idle.set()
        self._idle_events[session.id] = idle
        self._user_speaking[session.id] = False
        self._provider_idle[session.id] = True

        # Initialize skill activation state for this session
        if self._skill_support:
            self._skill_support.init_session(session.id)

        # Per-room config overrides from metadata
        system_prompt = meta.get("system_prompt", self._system_prompt)
        voice = meta.get("voice", self._voice)
        tools = meta.get("tools", self._tools)
        temperature = meta.get("temperature", self._temperature)
        provider_config = meta.get("provider_config")

        # Cache the resolved base tool list (channel defaults + metadata
        # overrides) so skill activation can reconfigure without losing them.
        self._session_tools[session.id] = list(tools) if tools else []

        # Inject skills: prepend skill tool defs, apply gating, enrich prompt
        if self._skill_support:
            system_prompt = self._skill_support.inject_skills_prompt(system_prompt)
            skill_defs = self._skill_support.skill_tool_dicts()
            tools = skill_defs + (tools or [])
            tools = self._skill_support.get_visible_tools(tools, session.id)

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

        # Detect manual VAD mode: if the transport has a local pipeline
        # with VAD, use it for speech detection instead of server-side VAD.
        # Cache event loop early for cross-thread VAD callbacks.
        self._event_loop = asyncio.get_running_loop()
        if not self._manual_vad:
            self._manual_vad = self._detect_vad_mode()
            if self._manual_vad:
                self._wire_local_vad()

        # Connect to provider (with telemetry span).
        # If provider.connect fails, clean up the already-accepted transport
        # session to avoid leaking the connection.
        try:
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
                    server_vad=not self._manual_vad,
                    provider_config=provider_config,
                )
        except Exception:
            logger.exception(
                "provider.connect failed for session %s; cleaning up transport", session.id
            )
            self._session_resamplers.pop(session.id, None)
            self._session_transport_rates.pop(session.id, None)
            with contextlib.suppress(Exception):
                await self._transport.disconnect(session)
            raise

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

        # Wire room-level audio recording if configured
        self._wire_realtime_recording(room_id, session)

        logger.info(
            "Realtime session %s started: room=%s, participant=%s, provider=%s",
            session.id,
            room_id,
            participant_id,
            self._provider.name,
        )

        # Fire ON_SESSION_STARTED — session is fully active, provider
        # connected, transport ready.  No dual-signal needed for realtime
        # channels: the session is live as soon as provider.connect() succeeds.
        if self._framework:
            try:
                from roomkit.models.session_event import SessionStartedEvent

                context = await self._framework._build_context(room_id)
                ready_event = SessionStartedEvent(
                    room_id=room_id,
                    channel_id=self.channel_id,
                    channel_type=ChannelType.REALTIME_VOICE,
                    participant_id=session.participant_id,
                    session=session,
                )
                await self._framework.hook_engine.run_async_hooks(
                    room_id,
                    HookTrigger.ON_SESSION_STARTED,
                    ready_event,
                    context,
                    skip_event_filter=True,
                )
                await self._framework._emit_framework_event(
                    "session_started",
                    room_id=room_id,
                    data={
                        "session_id": session.id,
                        "channel_id": self.channel_id,
                    },
                )
            except Exception:
                logger.exception("Error firing ON_SESSION_STARTED hook")

        return session

    async def end_session(self, session: VoiceSession) -> None:
        """End a realtime voice session.

        Disconnects both provider and transport, fires framework event.

        Args:
            session: The session to end.
        """
        with self._state_lock:
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

        # Clean up skill activation state
        if self._skill_support:
            self._skill_support.cleanup_session(session.id)

        with self._state_lock:
            self._sessions.pop(session.id, None)
            self._session_rooms.pop(session.id, None)
            self._session_bindings.pop(session.id, None)
            self._session_tools.pop(session.id, None)
            self._audio_generation.pop(session.id, None)
            self._session_transport_rates.pop(session.id, None)
            self._audio_forward_count.pop(session.id, None)
            self._last_assistant_text.pop(session.id, None)
            self._barge_in_active.discard(session.id)
            idle = self._idle_events.pop(session.id, None)
            if idle is not None:
                idle.set()  # unblock any waiters
            self._user_speaking.pop(session.id, None)
            self._provider_idle.pop(session.id, None)
            turn_span_id = self._turn_spans.pop(session.id, None)
            session_span_id = self._session_spans.pop(session.id, None)
            resamplers = self._session_resamplers.pop(session.id, None)

        # End any active turn span, then the session span
        telemetry = self._telemetry_provider
        if turn_span_id:
            telemetry.end_span(turn_span_id)
        if session_span_id:
            telemetry.end_span(session_span_id)
            telemetry.flush()

        if resamplers:
            for r in resamplers:
                try:
                    r.close()
                except Exception:
                    logger.exception("Error closing resampler for session %s", session.id)

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
        # Save caller values before skills mutation — self._tools and
        # self._system_prompt must store the *user* values, not the
        # skill-enriched versions, to avoid doubling on the next session.
        caller_tools = tools
        caller_prompt = system_prompt

        # Inject skills into reconfigured prompt/tools
        if self._skill_support:
            if system_prompt is not None:
                system_prompt = self._skill_support.inject_skills_prompt(system_prompt)
            if tools is not None:
                skill_defs = self._skill_support.skill_tool_dicts()
                tools = skill_defs + tools
                tools = self._skill_support.get_visible_tools(tools, session.id)

        await self._provider.reconfigure(
            session,
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            provider_config=provider_config,
        )

        # Update channel defaults so new sessions use the current config.
        # Store the original caller values (without skill enrichment).
        if caller_prompt is not None:
            self._system_prompt = caller_prompt
        if voice is not None:
            self._voice = voice
        if caller_tools is not None:
            self._tools = caller_tools

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
        with self._state_lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            try:
                await self.end_session(session)
            except Exception:
                logger.exception("Error ending session %s during close", session.id)

        # Cancel all outstanding scheduled tasks with timeout
        tasks = list(self._scheduled_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning("Timed out waiting for %d tasks during close", len(tasks))
        self._scheduled_tasks.clear()

        try:
            await self._provider.close()
        except Exception:
            logger.exception("Error closing provider during channel close")
        try:
            await self._transport.close()
        except Exception:
            logger.exception("Error closing transport during channel close")

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
        self,
        loop: asyncio.AbstractEventLoop,
        coro: Any,
        *,
        name: str,
    ) -> asyncio.Task[Any]:
        """Create a tracked asyncio task with automatic cleanup and error logging."""
        task = loop.create_task(coro, name=name)
        task.add_done_callback(self._task_done)
        self._scheduled_tasks.add(task)
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

    # -------------------------------------------------------------------------
    # Room-level audio recording
    # -------------------------------------------------------------------------

    def _wire_realtime_recording(self, room_id: str, session: VoiceSession) -> None:
        """Wire room-level audio recording for a realtime voice session.

        Registers a single audio track that receives both mic input and
        AI output.  The recorder injects silence to fill gaps between
        speech segments, keeping the audio stream continuous and in sync
        with video.
        """
        if self._recording is None or not getattr(self._recording, "audio", False):
            return
        if self._framework is None:
            return
        mgr = self._framework._room_recorder_mgr
        if not mgr.has_recorders(room_id):
            return

        from roomkit.recorder.base import RecordingTrack

        # Single audio track — both mic and AI output feed into it.
        # The recorder fills silence gaps to keep continuous audio.
        audio_track = RecordingTrack(
            id=f"audio:{session.id}",
            kind="audio",
            channel_id=self.channel_id,
            participant_id=session.participant_id,
            codec="pcm_s16le",
            sample_rate=self._output_sample_rate,
        )
        mgr.on_track_added(room_id, audio_track)

        with self._state_lock:
            self._recording_tracks[session.id] = (audio_track, room_id)

        logger.info(
            "Realtime recording wired for session %s (rate=%dHz)",
            session.id,
            self._output_sample_rate,
        )

    def _on_client_audio(self, session: VoiceSession, audio: AudioFrame | bytes) -> Any:
        """Forward client audio to provider."""
        if not isinstance(audio, bytes):
            audio = audio.data

        # Recording tap: send mic audio to room recorder
        with self._state_lock:
            rec = self._recording_tracks.get(session.id)
        if rec is not None and self._framework is not None:
            audio_track, rec_room_id = rec
            self._framework._room_recorder_mgr.on_data(
                rec_room_id,
                audio_track,
                audio,
                time.monotonic() * 1000,
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._forward_client_audio(session, audio, time.monotonic()),
            name=f"rt_client_audio:{session.id}",
        )

    async def _forward_client_audio(
        self,
        session: VoiceSession,
        audio: bytes,
        enqueued_at: float = 0.0,
    ) -> None:
        if session.state != VoiceSessionState.ACTIVE:
            return
        # Enforce ChannelBinding.access and muted per RFC §7.5
        with self._state_lock:
            binding = self._session_bindings.get(session.id)
            resamplers = self._session_resamplers.get(session.id)
            transport_rate = self._session_transport_rates.get(session.id)
        if binding is not None and (
            binding.access in (Access.READ_ONLY, Access.NONE) or binding.muted
        ):
            return
        try:
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
        with self._state_lock:
            resamplers = self._session_resamplers.get(session.id)
            transport_rate = self._session_transport_rates.get(session.id)
            gen = self._audio_generation.get(session.id, 0)
            # Early gate: skip resampling + task creation if output is muted
            binding = self._session_bindings.get(session.id)
            if binding is not None and binding.output_muted:
                return
        audio = self._resample_outbound_with(audio, resamplers, transport_rate)
        if not audio:
            return

        # Fire outbound audio taps (e.g. avatar lip-sync)
        rate = transport_rate or self._output_sample_rate
        for cb in getattr(self, "_outbound_audio_taps", []):
            try:
                cb(session, audio, rate)
            except Exception:
                logger.debug("Outbound audio tap error", exc_info=True)

        # Recording tap: send AI audio to room recorder
        with self._state_lock:
            rec = self._recording_tracks.get(session.id)
        if rec is not None and self._framework is not None:
            audio_track, rec_room_id = rec
            self._framework._room_recorder_mgr.on_data(
                rec_room_id,
                audio_track,
                audio,
                time.monotonic() * 1000,
            )

        with self._state_lock:
            self._audio_forward_count[session.id] = (
                self._audio_forward_count.get(session.id, 0) + 1
            )
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
        """Send audio to transport, skipping if the generation is stale or output is muted."""
        with self._state_lock:
            current_gen = self._audio_generation.get(session.id, 0)
            user_speaking = self._user_speaking.get(session.id, False)
        if current_gen != gen:
            return
        # In manual VAD mode, suppress AI audio while the user is speaking.
        # The provider may keep sending chunks until it processes our
        # activityStart signal.
        if user_speaking:
            return

        # Enforce output_muted on the channel binding
        binding = self._session_bindings.get(session.id)
        if binding is not None and binding.output_muted:
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

    def _set_aec_active(self, active: bool) -> None:
        """Toggle AEC bypass on the transport's pipeline (if present)."""
        pipeline_cfg = getattr(self._transport, "_pipeline_config", None)
        if pipeline_cfg is not None and pipeline_cfg.aec is not None:
            pipeline_cfg.aec.set_active(active)

    def _reset_aec(self) -> None:
        """Reset AEC adaptive filter on barge-in to avoid transient artifacts."""
        pipeline_cfg = getattr(self._transport, "_pipeline_config", None)
        if pipeline_cfg is not None and pipeline_cfg.aec is not None:
            pipeline_cfg.aec.reset()

    def _resample_outbound(self, session: VoiceSession, audio: bytes) -> bytes:
        """Resample outbound audio from provider rate to transport rate."""
        with self._state_lock:
            resamplers = self._session_resamplers.get(session.id)
            transport_rate = self._session_transport_rates.get(session.id)
        return self._resample_outbound_with(audio, resamplers, transport_rate)

    def _resample_outbound_with(
        self,
        audio: bytes,
        resamplers: tuple[Any, Any] | None,
        transport_rate: int | None,
    ) -> bytes:
        """Resample outbound audio using pre-snapshotted resamplers/rate."""
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

        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
        if not room_id:
            return

        # Partial transcriptions: send to client UI and fire async hooks
        # (fire-and-forget — partials are informational, not blockable).
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

            from roomkit.voice.events import PartialTranscriptionEvent

            partial_event = PartialTranscriptionEvent(
                session=session,
                text=text,
                confidence=0.0,
                is_stable=False,
                role=role,
            )
            context = await self._framework._build_context(room_id)
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_PARTIAL_TRANSCRIPTION,
                partial_event,
                context,
                skip_event_filter=True,
            )
            return

        from roomkit.telemetry.context import reset_span

        _, _tok = self._rt_span_ctx(session.id)
        try:
            context = await self._framework._build_context(room_id)

            # Fire ON_TRANSCRIPTION hooks (sync, can modify/block)
            from roomkit.voice.realtime.events import RealtimeTranscriptionEvent

            # Check and clear barge-in state for user transcriptions.
            was_barge_in = role == "user" and session.id in self._barge_in_active
            if was_barge_in and is_final:
                self._barge_in_active.discard(session.id)

            transcription_event = RealtimeTranscriptionEvent(
                session=session,
                text=text,
                role=role,  # type: ignore[arg-type]
                is_final=is_final,
                was_barge_in=was_barge_in,
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

            # Track last assistant text for barge-in context
            if role == "assistant":
                self._last_assistant_text[session.id] = final_text

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

        In manual VAD mode, local VAD drives speech events — ignore
        provider callbacks to prevent duplicate events.

        Bumps the generation counter so pending send_audio tasks become
        stale, resets the outbound resampler to discard its buffered frame,
        and signals the transport to interrupt outbound audio.
        """
        if self._manual_vad:
            return
        self._user_speaking[session.id] = True
        self._update_idle_event(session.id)
        # Bump generation — pending tasks with the old generation will skip.
        # Read barge-in state in the same lock acquisition for consistency.
        with self._state_lock:
            self._audio_generation[session.id] = self._audio_generation.get(session.id, 0) + 1
            resamplers = self._session_resamplers.get(session.id)
            is_barge_in = self._audio_forward_count.get(session.id, 0) > 0

        if is_barge_in:
            self._barge_in_active.add(session.id)

        # Discard outbound resampler state so stale audio doesn't leak
        if resamplers:
            resamplers[1].reset()

        # Keep AEC active but do NOT reset the adaptive filter on barge-in.
        # Resetting clears the converged echo model, forcing the filter to
        # re-learn from scratch.  During convergence (~200-500 ms), echo
        # passes through uncancelled and triggers the provider's server-side
        # VAD again — creating an infinite echo→interrupt→reset loop.
        if is_barge_in:
            logger.debug(
                "Barge-in detected for session %s (forwarded %d chunks) — "
                "keeping AEC filter intact",
                session.id,
                self._audio_forward_count.get(session.id, 0),
            )

        self._transport.interrupt(session)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if is_barge_in:
            self._track_task(
                loop,
                self._fire_barge_in_hook(session),
                name=f"rt_barge_in:{session.id}",
            )

        self._track_task(
            loop,
            self._handle_speech_event(session, "start"),
            name=f"rt_speech_start:{session.id}",
        )

    def _on_provider_speech_end(self, session: VoiceSession) -> Any:
        """Handle speech end from provider's server-side VAD."""
        if self._manual_vad:
            return
        self._user_speaking[session.id] = False
        self._update_idle_event(session.id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._handle_speech_event(session, "end"),
            name=f"rt_speech_end:{session.id}",
        )

    async def _fire_barge_in_hook(self, session: VoiceSession) -> None:
        """Fire ON_BARGE_IN when user interrupts AI playback."""
        if not self._framework:
            return
        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
        if not room_id:
            return
        try:
            from roomkit.voice.events import BargeInEvent

            context = await self._framework._build_context(room_id)
            event = BargeInEvent(
                session=session,
                interrupted_text=self._last_assistant_text.get(session.id, ""),
                audio_position_ms=0,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_BARGE_IN,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_BARGE_IN hook")

    async def _handle_speech_event(self, session: VoiceSession, event_type: str) -> None:
        """Fire speech hooks and publish ephemeral indicator."""
        if not self._framework:
            return

        with self._state_lock:
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
        The ``ON_TOOL_CALL`` hook is then fired (handler result, if any,
        is passed as ``event.result`` so the hook can observe or override).
        """
        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
            _rt_parent = self._session_spans.get(session.id)
            parent = self._turn_spans.get(session.id) or _rt_parent

        from roomkit.telemetry.context import reset_span, set_current_span

        # Set session span context so hooks inside are parented correctly
        _rt_tok = set_current_span(_rt_parent) if _rt_parent else None

        # Telemetry: tool call span (child of current turn span)
        telemetry = self._telemetry_provider
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

            # Step 0: Skill infrastructure tools — handle internally
            if self._skill_support and self._skill_support.is_skill_tool(name):
                result_str = await self._skill_support.handle_tool_call(
                    name, arguments, session.id
                )
                # After activation, push newly-visible tools to the provider
                if name == TOOL_ACTIVATE_SKILL:
                    base = self._session_tools.get(session.id, self._tools or [])
                    all_tools = self._skill_support.skill_tool_dicts() + base
                    updated = self._skill_support.newly_visible_after_activation(
                        all_tools, session.id, arguments.get("name", "")
                    )
                    if updated is not None:
                        await self._provider.reconfigure(session, tools=updated)
                await self._provider.submit_tool_result(session, call_id, result_str)
                telemetry.end_span(tool_span_id)
                logger.info("Skill tool %s(%s) handled for session %s", name, call_id, session.id)
                return

            # Step 1: Run tool_handler (if exists)
            handler_result: str | None = None
            if self._tool_handler is not None:
                logger.info(
                    "Executing tool %s(%s) via handler for session %s",
                    name,
                    call_id,
                    session.id,
                )
                token = _current_voice_session.set(session)
                try:
                    raw = await self._tool_handler(name, arguments)
                finally:
                    _current_voice_session.reset(token)
                handler_result = raw if isinstance(raw, str) else json.dumps(raw)

            # Step 2: Run ON_TOOL_CALL hook (if framework + room)
            from roomkit.models.tool_call import ToolCallEvent

            tool_event = ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.REALTIME_VOICE,
                tool_call_id=call_id,
                name=name,
                arguments=arguments,
                result=handler_result,
                room_id=room_id,
                session=session,
            )

            if self._framework and room_id:
                context = await self._framework._build_context(room_id)
                hook_result = await self._framework.hook_engine.run_sync_hooks(
                    room_id,
                    HookTrigger.ON_TOOL_CALL,
                    tool_event,
                    context,
                    skip_event_filter=True,
                )

                if not hook_result.allowed:
                    result_str = json.dumps(
                        {"error": hook_result.reason or "Tool call blocked by hook"}
                    )
                elif "result" in hook_result.metadata:
                    # Hook provided/overrode the result
                    hook_val = hook_result.metadata["result"]
                    result_str = hook_val if isinstance(hook_val, str) else json.dumps(hook_val)
                elif handler_result is not None:
                    result_str = handler_result
                elif hook_result.hook_errors:
                    # Hook(s) raised exceptions — propagate the error so the
                    # AI model knows the tool call failed instead of assuming
                    # success and hallucinating a result.
                    errors = "; ".join(
                        f"{e['hook']}: {e['error']}" for e in hook_result.hook_errors
                    )
                    result_str = json.dumps(
                        {
                            "error": f"Tool call failed: {errors}. "
                            "Take a fresh screenshot and retry.",
                        }
                    )
                else:
                    result_str = json.dumps({"status": "ok"})

                # Emit framework event for observability
                await self._framework._emit_framework_event(
                    "tool_call",
                    room_id=room_id,
                    channel_id=self.channel_id,
                    data={
                        "tool_name": name,
                        "tool_call_id": call_id,
                        "channel_type": str(ChannelType.REALTIME_VOICE),
                    },
                )
            elif handler_result is not None:
                result_str = handler_result
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
        """Handle AI response start — activate AEC, publish typing indicator."""
        self._provider_idle[session.id] = False
        self._update_idle_event(session.id)
        self._set_aec_active(True)
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
        creation order.  A synchronous call here would put RESPONSE_END on
        the pacer queue *before* the last audio chunks, causing the pacer
        to split one response into a truncated response + an orphan tail
        that never gets its own end marker.
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

        # NOTE: do NOT deactivate AEC here — the speaker is still playing
        # buffered audio chunks.  AEC stays active for the session lifetime;
        # the algorithm handles silence correctly (passthrough when no echo).

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
        # Mark idle AFTER all audio tasks + end-of-response signal complete
        self._track_task(
            loop,
            self._set_idle(session),
            name=f"rt_idle:{session.id}",
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
            with self._state_lock:
                forwarded = self._audio_forward_count.pop(session.id, 0)
                turn_span_id = self._turn_spans.pop(session.id, None)
            if forwarded:
                logger.info(
                    "Response ended: forwarded %d audio chunks for session %s",
                    forwarded,
                    session.id,
                )
            # End turn span with usage from the provider (if available)
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

            # Publish to realtime backend for dashboard subscribers
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
        with self._state_lock:
            active = session.id in self._sessions
        if active:
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
        with self._state_lock:
            return [s for s in self._sessions.values() if self._session_rooms.get(s.id) == room_id]
