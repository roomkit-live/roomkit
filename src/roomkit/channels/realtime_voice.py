"""RealtimeVoiceChannel — wraps speech-to-speech AI APIs as a RoomKit channel."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.channels._realtime_audio import RealtimeAudioMixin
from roomkit.channels._realtime_context import (
    _current_voice_session as _current_voice_session,
)
from roomkit.channels._realtime_context import (
    get_current_voice_session as get_current_voice_session,
)
from roomkit.channels._realtime_response import RealtimeResponseMixin
from roomkit.channels._realtime_speech import RealtimeSpeechMixin
from roomkit.channels._realtime_tools import RealtimeToolsMixin
from roomkit.channels._realtime_transcription import RealtimeTranscriptionMixin
from roomkit.channels._voice_pipeline import VoicePipelineMixin
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.voice.backends.base import VoiceBackend
from roomkit.voice.base import VoiceSession, VoiceSessionState

try:
    from websockets.exceptions import ConnectionClosed as _ConnectionClosed
except ImportError:  # websockets not installed
    _ConnectionClosed = ConnectionError  # ty: ignore[invalid-assignment]

if TYPE_CHECKING:
    from roomkit.channels._realtime_skills import RealtimeSkillSupport
    from roomkit.core.framework import RoomKit
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.engine import AudioPipeline
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

# Tool handler: async callable (name, arguments) -> result string
ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]

logger = logging.getLogger("roomkit.channels.realtime_voice")


class RealtimeVoiceChannel(
    RealtimeToolsMixin,
    RealtimeTranscriptionMixin,
    RealtimeSpeechMixin,
    RealtimeAudioMixin,
    RealtimeResponseMixin,
    VoicePipelineMixin,
    Channel,
):
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
        pipeline: AudioPipelineConfig | None = None,
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
            pipeline: Optional ``AudioPipelineConfig`` for local audio
                processing (AEC, VAD, denoiser, etc.).  When set, mic
                audio is processed through the pipeline before being
                forwarded to the provider, and pipeline VAD drives
                speech detection instead of server-side VAD.
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
        self._pipeline_config = pipeline
        self._pipeline: AudioPipeline | None = None

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
        # True when the channel's pipeline has VAD — local VAD drives speech
        # events, and provider speech callbacks are ignored.
        self._has_pipeline_vad: bool = False

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

        # Direct audio path: only when no pipeline is configured.
        # When pipeline= is set, _create_pipeline() registers
        # _pipeline_on_audio_received instead (in start_session).
        if self._pipeline_config is None:
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
            self._provider._telemetry = telemetry  # ty: ignore[unresolved-attribute]

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

        # Set up audio pipeline BEFORE accept() so that the PortAudio
        # callback closure captures the pipeline's on_audio_received
        # callback (not the direct _on_client_audio path).
        self._event_loop = asyncio.get_running_loop()
        has_pipeline_vad = False
        if self._pipeline_config is not None:
            if self._pipeline_config.turn_detector is not None:
                logger.warning(
                    "turn_detector is ignored on RealtimeVoiceChannel — "
                    "the provider handles endpointing. Use VAD silence "
                    "timeout to control pause sensitivity instead.",
                )
            pl = self._create_pipeline(self._pipeline_config, self._transport)
            pl.on_vad_event(self._on_pipeline_vad_event)
            pl.on_processed_frame(self._on_pipeline_processed_frame)
            has_pipeline_vad = self._pipeline_config.vad is not None
            self._has_pipeline_vad = has_pipeline_vad

        # Accept client connection (with telemetry span)
        with telemetry.span(
            SpanKind.BACKEND_CONNECT,
            "transport.accept",
            parent_id=session_span_id,
            session_id=session.id,
            attributes={Attr.BACKEND_TYPE: self._transport.name},
        ):
            await self._transport.accept(session, connection)

        # Activate pipeline session after accept (session is now ready)
        if self._pipeline is not None:
            self._pipeline_session_active(session)

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
                    server_vad=not has_pipeline_vad,
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

        # Notify pipeline of session end
        self._pipeline_session_ended(session)

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
            self._recording_tracks.pop(session.id, None)
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
        audio gate in ``_pipeline_on_audio_received`` (pipeline path)
        or ``_forward_client_audio`` (direct path) sees the new state.
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

    # -- Helpers --

    def _get_room_sessions(self, room_id: str) -> list[VoiceSession]:
        """Get all active sessions for a room."""
        with self._state_lock:
            return [s for s in self._sessions.values() if self._session_rooms.get(s.id) == room_id]
