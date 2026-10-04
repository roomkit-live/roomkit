"""RealtimeVoiceChannel — wraps speech-to-speech AI APIs as a RoomKit channel."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.channels._realtime_audio import _MAX_QUEUED_AUDIO_CHUNKS, RealtimeAudioMixin
from roomkit.channels._realtime_context import (
    _current_voice_session as _current_voice_session,
)
from roomkit.channels._realtime_context import (
    get_current_voice_session as get_current_voice_session,
)
from roomkit.channels._realtime_delegation import RealtimeDelegationMixin
from roomkit.channels._realtime_response import RealtimeResponseMixin
from roomkit.channels._realtime_speech import RealtimeSpeechMixin
from roomkit.channels._realtime_text_injected import fire_text_injected
from roomkit.channels._realtime_tool_calls import ToolCallBook
from roomkit.channels._realtime_tool_executor import SESSION_ENDED, report_interrupted_calls
from roomkit.channels._realtime_tool_gate import RealtimeToolGateMixin
from roomkit.channels._realtime_tool_recovery import RealtimeToolRecoveryMixin
from roomkit.channels._realtime_tools import RealtimeToolsMixin
from roomkit.channels._realtime_transcription import RealtimeTranscriptionMixin
from roomkit.channels._served_tools import (
    CollisionLog,
    dict_tool_name,
    refuse_given_twice,
    refuse_served_names,
    refuse_unnamable,
    warn_tools_uncallable,
)
from roomkit.channels._skill_constants import (
    ACTIVATE_SKILL_SCHEMA,
    READ_REFERENCE_SCHEMA,
    RUN_SCRIPT_SCHEMA,
    TOOL_RUN_SCRIPT,
)
from roomkit.channels._tool_registry import (
    ChannelRegistry,
    SessionConfig,
    channel_tool,
    schema_tool,
)
from roomkit.channels._voice_pipeline import VoicePipelineMixin
from roomkit.channels._voice_recording_hooks import VoiceRecordingHooksMixin
from roomkit.channels.ai import ToolResult
from roomkit.channels.base import (
    Channel,
    FrameworkAwareChannel,
    RealtimeModelHost,
    _check_room_scope,
)
from roomkit.core.task_utils import _finish_cleanup
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
from roomkit.telemetry.context import reset_span
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.tools._human_input_channel import ChannelHumanInput, warn_plain_handler
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.tools.timeout import ToolTimeouts
from roomkit.voice.backends.base import VoiceBackend
from roomkit.voice.base import VoiceSession, VoiceSessionState

try:
    from websockets.exceptions import ConnectionClosed as _ConnectionClosed
except ImportError:  # websockets not installed
    _ConnectionClosed = ConnectionError

if TYPE_CHECKING:
    from roomkit.channels._realtime_skills import RealtimeSkillSupport
    from roomkit.channels._realtime_tool_search import RealtimeToolSearchSupport
    from roomkit.core.framework import RoomKit
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.models import Skill
    from roomkit.skills.registry import SkillRegistry
    from roomkit.tools.policy import ToolPolicy
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.engine import AudioPipeline
    from roomkit.voice.realtime.injection import VoiceInjectionResult
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider
    from roomkit.voice.realtime.reasoning import ReasoningBackend

# Tool handler: async callable (name, arguments) -> result. Same contract as
# roomkit.channels.ai.ToolHandler — a handler shared with an AIChannel may
# answer with a content-part list; the realtime paths flatten it to text
# before it reaches the voice provider (see tools.result.result_text).
ToolHandler = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]

logger = logging.getLogger("roomkit.channels.realtime_voice")


@dataclass
class _SessionTeardown:
    """One teardown per session: the task running it, and what joiners wait on."""

    owner: asyncio.Task[Any] | None
    done: asyncio.Future[None]


@dataclass
class _ConnectingSession:
    """Own a handshake separately from the application task awaiting it."""

    task: asyncio.Task[VoiceSession]
    disconnected: bool = False
    deferred: bool = False
    callbacks: list[tuple[Callable[..., Any], tuple[Any, ...]]] = field(default_factory=list)
    # The provider's tool calls the journal holds, and the ones it issued once
    # the start failed: each is reported, cancelled, if the start fails.
    tool_calls: list[tuple[Any, ...]] = field(default_factory=list)
    audio_bytes: int = 0
    failure: Exception | None = None

    def fail(self, error: Exception) -> None:
        self.failure = error
        if not self.task.done() and not self.task.cancelling():
            self.task.cancel()


class RealtimeVoiceChannel(
    RealtimeToolRecoveryMixin,
    RealtimeToolsMixin,
    RealtimeToolGateMixin,
    RealtimeDelegationMixin,
    RealtimeTranscriptionMixin,
    RealtimeSpeechMixin,
    RealtimeAudioMixin,
    RealtimeResponseMixin,
    VoiceRecordingHooksMixin,
    VoicePipelineMixin,
    RealtimeModelHost,
    FrameworkAwareChannel,
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
        owns_transport: bool = True,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any] | Any] | None = None,
        temperature: float | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        transport_sample_rate: int | None = None,
        emit_transcription_events: bool = True,
        tool_handler: ToolHandler | None = None,
        human_input_handler: HumanInputToolHandler | None = None,
        mute_on_tool_call: bool = False,
        tool_result_max_length: int = 16384,
        tool_timeout_seconds: float | None = 10.0,
        tool_timeouts: Mapping[str, float | None] | None = None,
        pipeline: AudioPipelineConfig | None = None,
        recording: Any | None = None,
        skills: SkillRegistry | None = None,
        script_executor: ScriptExecutor | None = None,
        skill_delivery_mode: str | None = None,
        tool_recovery: bool = True,
        tool_search: bool | None = None,
        tool_search_pinned: list[str] | None = None,
        tool_search_threshold: int = 20,
        reasoning_backend: ReasoningBackend | None = None,
        reasoning_timeout_s: float = 120.0,
        tool_policy: ToolPolicy | None = None,
    ) -> None:
        """Initialize realtime voice channel.

        Args:
            channel_id: Unique channel identifier.
            provider: The realtime voice provider (OpenAI, Gemini, etc.).
            transport: The audio transport (WebSocket, etc.).
            owns_transport: Close the transport with this channel (default True).
                Set False for a shared transport with unsubscribe support, such
                as FastRTC. Only this channel's sessions and callbacks close.
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
                resampling. Transports can override it in session metadata:
                ``transport_sample_rate`` for capture and, when different,
                ``transport_output_sample_rate`` for playback. FastRTC and SIP
                declare their rates automatically. Without either metadata or
                a configured rate, audio passes through unchanged.
            emit_transcription_events: If True, emit final transcriptions
                as RoomEvents so other channels see them.
            tool_handler: Async callable to execute tool calls.
                Signature: ``async (name, arguments) -> str``.
                If not set, falls back to handlers extracted from Tool
                objects, or ``ON_TOOL_CALL`` hooks.
                Raise ``roomkit.ToolRefusedError`` to decline a call, or
                ``roomkit.ToolFailedError`` to say it ran and failed: the
                message reaches the model verbatim and the call is marked
                failed, where a returned body would read as work that was done.
            human_input_handler: The tools that ask a person, as on an
                ``AIChannel``: the channel declares their
                ``tool_definitions`` in every session and serves them before
                *tool_handler*, on every door (the provider's call, a call
                recovered from speech, a reasoning backend's), under the
                handler's own ``timeout`` rather than the default call bound.
                Each request fires ``ON_USER_INPUT_REQUIRED``, whose BLOCK
                rejects it, and the requests still open are settled when the
                channel closes (RFC §9.3, §21.6).
            mute_on_tool_call: If True, mute the transport microphone during
                tool execution to prevent barge-in that causes providers
                (e.g. Gemini) to silently drop the tool result.  Defaults
                to False — use ``set_access()`` for fine-grained control.
            tool_result_max_length: Maximum character length of tool results
                before truncation.  Large results (e.g. SVG payloads) can
                overflow the provider's context window.  Defaults to 16384.
            tool_timeout_seconds: How long one tool call may take before its
                handler is cancelled and the call fails (RFC §21.6), so a
                handler that never answers cannot leave the caller in silence.
                Defaults to 10 s; ``None`` leaves calls unbounded. A tool that
                waits on another agent (orchestration's) keeps its own bound.
            tool_timeouts: A bound per tool name, above the default (``None``
                for a tool that may take as long as it needs).
            tool_policy: Allow/deny rules for the session's tools, with the
                meaning they have on an ``AIChannel`` (RFC §21.1). A tool the
                policy denies is not declared to the session, is never named
                by Tool Search, and is refused if called anyway, whether the
                call comes from the provider, from spoken text the channel
                recovered, or from a reasoning backend. Role overrides apply
                to the session's participant, read when the session starts.
            pipeline: Optional ``AudioPipelineConfig`` for local audio
                processing (AEC, VAD, denoiser, etc.).  When set, mic
                audio is processed through the pipeline before being
                forwarded to the provider, and pipeline VAD drives
                speech detection instead of server-side VAD. With a
                full-duplex provider the VAD is observation only: its
                events reach hooks and metrics but never interrupt the
                model (RFC §12.4.1).
            recording: Optional ``ChannelRecordingConfig`` to enable
                room-level audio recording from this channel. Records
                both input (mic) and output (AI) audio tracks.
            skills: Optional ``SkillRegistry`` with discovered skills.
                When provided, skill infrastructure tools are injected
                and the skills preamble is appended to the system prompt.
            script_executor: Optional ``ScriptExecutor`` for running
                skill scripts.  Ignored when *skills* is ``None``.
            skill_delivery_mode: How skill bodies reach the model.
                ``"inline_full"`` bakes every skill's full instructions
                into the initial ``system_instruction`` at session start;
                ``activate_skill`` becomes a declarative ACK and no
                ``provider.reconfigure`` is needed. ``"on_demand"``
                puts only metadata in the prompt; ``activate_skill``
                loads the body via ``provider.reconfigure``.
                Defaults to ``"inline_full"``
                when the provider reports
                ``supports_mid_session_reconfigure=False`` (e.g.
                Gemini 3.x Flash Live), ``"on_demand"`` otherwise.
            tool_recovery: If True, detect tool calls that the model speaks
                as text (e.g. ``call:name{args}``) instead of issuing through
                the function calling API, and run them: a call said as a
                sentence of its own that ends the utterance, never one a
                sentence mentions. Defaults to True.
                Recovered calls pass the same pre-execution gate as any other
                — declared catalogue, tool policy, skill gating, argument
                schema, ``BEFORE_TOOL_USE`` —
                and their outcome returns as injected context rather than a
                tool result, because the model has no pending call to answer.
                Set to False to let a spoken call stay speech.
            tool_search: Auto-enable Tool Search when the catalogue is
                large enough to exceed the realtime model's reliable
                tool-selection window (Google: 10–20 active tools on
                Gemini Live). Pass ``True`` / ``False`` to force,
                ``None`` (default) for automatic activation when
                ``len(tools) > tool_search_threshold``.
            tool_search_pinned: Tool names that should ALWAYS be
                visible (never hidden behind search). Use for tools
                the agent calls reflexively — e.g. ``hangup_call``,
                ``get_datetime``, in-flight session-control tools.
            tool_search_threshold: Auto-activation threshold and the
                cap on how many tools may be live at once. Defaults
                to 20 to match Google's published recommendation.
            reasoning_backend: Serves the integrator-side reasoning
                delegations of a full-duplex provider (RFC §12.4.1): when
                the model hands work over, the backend receives the
                transcript recorded since the previous delegation and its
                outputs return to the model as spoken or silent context.
                Its tool calls pass the same pre-execution gate as any
                realtime tool call. Only a full-duplex provider delegates;
                on any other provider the backend never runs.
            reasoning_timeout_s: Bound on one delegation's run. A backend
                that exceeds it is abandoned and the model is told so.
                Defaults to 120 seconds.
        """
        super().__init__(channel_id)
        for name, rate in (
            ("input_sample_rate", input_sample_rate),
            ("output_sample_rate", output_sample_rate),
        ):
            if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if transport_sample_rate is not None and (
            not isinstance(transport_sample_rate, int)
            or isinstance(transport_sample_rate, bool)
            or transport_sample_rate <= 0
        ):
            raise ValueError("transport_sample_rate must be a positive integer")
        if (
            not isinstance(tool_result_max_length, int)
            or isinstance(tool_result_max_length, bool)
            or tool_result_max_length <= 0
        ):
            raise ValueError("tool_result_max_length must be a positive integer")
        if (
            not isinstance(tool_search_threshold, int)
            or isinstance(tool_search_threshold, bool)
            or tool_search_threshold <= 0
        ):
            raise ValueError("tool_search_threshold must be a positive integer")
        if (
            isinstance(reasoning_timeout_s, bool)
            or not isinstance(reasoning_timeout_s, int | float)
            or reasoning_timeout_s <= 0
        ):
            raise ValueError("reasoning_timeout_s must be a positive number")
        self._provider: RealtimeVoiceProvider = provider
        self._transport = transport
        self._owns_transport = owns_transport
        self._transport_unsubscribers: list[Callable[[], None] | None] = []
        self._recording = recording
        self._system_prompt = system_prompt
        self._voice = voice
        self._temperature = temperature
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._transport_sample_rate = transport_sample_rate
        self._emit_transcription_events = emit_transcription_events
        self._tool_recovery_enabled = tool_recovery

        # Reasoning delegation (RFC §12.4.1): the integrator-side backend a
        # full-duplex model's delegations are served through, the transcript
        # ledger it reads, and the delegations still running per session.
        self._reasoning_backend = reasoning_backend
        self._reasoning_timeout_s = float(reasoning_timeout_s)
        self._transcript_ledger: dict[str, list[list[str]]] = {}
        self._delegated_before: set[str] = set()
        self._pending_delegations: dict[str, set[str]] = {}
        if reasoning_backend is not None and not provider.full_duplex:
            logger.warning(
                "reasoning_backend configured on channel %s but provider %s is not "
                "full-duplex: it delegates nothing, so the backend will never run",
                channel_id,
                provider.name,
            )

        self._init_human_input(human_input_handler, tool_handler)
        self._init_host_tools(tools, tool_handler)
        self._mute_on_tool_call = mute_on_tool_call
        self._tool_result_max_length = tool_result_max_length
        self._tool_timeouts = ToolTimeouts(tool_timeout_seconds, dict(tool_timeouts or {}))
        self._tool_policy = tool_policy
        # session_id -> the participant's role, for the policy's role overrides.
        self._session_roles: dict[str, str | None] = {}
        # session_id -> the tool policy of the pipeline agent the session
        # speaks as, which holds beside the channel's (RFC §19.5).
        self._session_agent_policies: dict[str, ToolPolicy] = {}
        self._framework: RoomKit | None = None
        self._pipeline_config = pipeline
        self._pipeline: AudioPipeline | None = None
        interruption = pipeline.interruption if pipeline is not None else None
        self._barge_in_guard_ms = max(
            0,
            interruption.allow_during_first_ms if interruption is not None else 0,
        )

        self._init_channel_tools(
            skills,
            script_executor,
            skill_delivery_mode,
            tool_search=tool_search,
            tool_search_threshold=tool_search_threshold,
            tool_search_pinned=tool_search_pinned,
        )

        # Lock for shared state accessed from both asyncio and audio threads
        self._state_lock = threading.Lock()

        # Active sessions: session_id -> (session, room_id, binding)
        self._sessions: dict[str, VoiceSession] = {}
        self._connecting_sessions: dict[str, _ConnectingSession] = {}
        # One teardown per session, owned by its first caller; the mirror of
        # ``_connecting_sessions`` for the other end of a session's life.
        self._session_teardowns: dict[str, _SessionTeardown] = {}
        self._closing = False
        self._session_rooms: dict[str, str] = {}  # session_id -> room_id
        # Cached bindings for audio gating (access/muted enforcement)
        self._session_bindings: dict[str, ChannelBinding] = {}

        # Per-session resolved tool list (channel defaults + metadata overrides),
        # cached so skill activation can reconfigure without losing tools.
        self._session_tools: dict[str, list[dict[str, Any]]] = {}
        self._session_config_locks: dict[str, asyncio.Lock] = {}

        # Per-session recording tracks: session_id -> (audio_track, room_id)
        self._recording_tracks: dict[str, tuple[Any, str]] = {}

        # Per-session resamplers: (inbound, outbound) pairs
        self._session_resamplers: dict[str, tuple[Any, Any]] = {}
        # Single-thread executor owning every resampler state mutation
        # (resample/flush/reset/close) — created lazily by the audio mixin,
        # shut down in close(). FIFO keeps frames ordered and state safe.
        self._resample_executor: ThreadPoolExecutor | None = None
        # Per-session transport sample rates (from transport metadata)
        self._session_transport_rates: dict[str, int] = {}
        self._session_transport_output_rates: dict[str, int] = {}
        # Audio forward counters (for diagnostics)
        self._audio_forward_count: dict[str, int] = {}
        # Per-session generation counter: bumped on interrupt so pending
        # send_audio tasks created before the interrupt become stale and skip.
        self._audio_generation: dict[str, int] = {}
        # Outbound send queue + resident worker per session — one consumer
        # task per session instead of one task per 20 ms provider chunk.
        self._audio_send_queues: dict[str, asyncio.Queue[Any]] = {}
        # Outbound chunks dropped per session because the transport fell behind.
        self._audio_dropped: dict[str, int] = {}
        self._audio_send_workers: dict[str, asyncio.Task[Any]] = {}
        # Inbound frames received after transport.accept() but before the AI
        # provider handshake completes.  The audio mixin owns the bounded
        # buffering/flush mechanics; these maps mark sessions still connecting.
        self._preconnect_audio: dict[str, list[tuple[bool, bytes, float]]] = {}
        self._preconnect_audio_bytes: dict[str, int] = {}
        self._preconnect_audio_dropped: set[str] = set()
        self._init_transcription_state()
        # Barge-in state: set when user interrupts AI, cleared on next final transcription
        self._barge_in_active: set[str] = set()
        # Playback onset reported by the transport: physical for local audio,
        # estimated from RTP transmission for SIP.
        self._playback_started_at: dict[str, float] = {}
        # Assistant duration reported by playback callbacks, excluding filler
        # silence. Known transport buffering is subtracted on interruption.
        self._playback_position_ms: dict[str, float] = {}
        self._playback_buffer: dict[str, tuple[float, float]] = {}
        self._response_generation: dict[str, int] = {}
        self._audio_drained: set[str] = set()
        # Throttle audio level hooks to ~10/sec per direction
        self._last_input_level_at: float = 0.0
        self._last_output_level_at: float = 0.0
        # Cached event loop for cross-thread scheduling (e.g. PortAudio callback)
        self._event_loop: asyncio.AbstractEventLoop | None = None
        # Per-session flag: True when session's pipeline has VAD — local VAD
        # drives speech events, and provider speech callbacks are ignored.
        self._has_pipeline_vad: dict[str, bool] = {}

        # Idle tracking: set when BOTH provider is done AND user is not speaking
        self._idle_events: dict[str, asyncio.Event] = {}
        self._user_speaking: dict[str, bool] = {}
        self._provider_idle: dict[str, bool] = {}
        # The tool calls in flight, per session, each with its one delivery
        # and its one report (RFC §12.4).
        self._tool_calls = ToolCallBook()
        self._awaiting_tool_response: set[str] = set()
        # Wall-clock of the last user-turn start (VAD SPEECH_START). Consumed by
        # _realtime_transcription when emitting the final user turn as a
        # RoomEvent, so tool_calls fired mid-turn sort after the user message.
        from datetime import datetime as _dt  # local alias; cross-platform

        self._user_turn_start_at: dict[str, _dt] = {}

        # Track fire-and-forget tasks for clean shutdown
        self._scheduled_tasks: set[asyncio.Task[Any]] = set()
        self._tool_reports: set[asyncio.Task[Any]] = set()

        # Telemetry span tracking: session_id -> span_id
        self._session_spans: dict[str, str] = {}
        self._turn_spans: dict[str, str] = {}

        # Wire internal callbacks
        provider.on_audio(self._gate_provider_callback(self._on_provider_audio))
        provider.on_transcription(self._gate_provider_callback(self._on_provider_transcription))
        provider.on_transcription(self._gate_provider_callback(self._on_transcript_fragment))
        provider.on_speech_start(self._gate_provider_callback(self._on_provider_speech_start))
        provider.on_speech_end(self._gate_provider_callback(self._on_provider_speech_end))
        provider.on_tool_call(
            self._gate_provider_callback(self._on_provider_tool_call, tool_call=True)
        )
        provider.on_tool_call_cancelled(
            self._gate_provider_callback(self._on_provider_tool_call_cancelled)
        )
        provider.on_delegation(self._gate_provider_callback(self._on_provider_delegation))
        provider.on_response_start(self._gate_provider_callback(self._on_provider_response_start))
        provider.on_response_end(self._gate_provider_callback(self._on_provider_response_end))
        provider.on_error(self._on_startup_provider_error)

        # Direct audio path: only when no pipeline is configured.
        # When pipeline= is set, _create_pipeline() registers
        # _pipeline_on_audio_received instead (in start_session).
        if self._pipeline_config is None:
            self._transport_unsubscribers.append(
                transport.on_audio_received(self._on_client_audio)
            )
        self._transport_unsubscribers.append(
            transport.on_client_disconnected(self._on_client_disconnected)
        )
        if transport.supports_playback_callback:
            self._transport_unsubscribers.append(
                transport.on_audio_played(self._on_transport_audio_played)
            )

    def _init_human_input(
        self, human_input_handler: HumanInputToolHandler | None, tool_handler: ToolHandler | None
    ) -> None:
        """The person's tools, which the channel serves itself (RFC §9.3)."""
        self._human_input = ChannelHumanInput(human_input_handler, self.channel_type)
        definitions = self._human_input.definitions
        warn_tools_uncallable(definitions, "human-input tool(s)", self._provider, self.channel_id)
        warn_plain_handler(tool_handler, self.channel_id)

    def _init_host_tools(
        self, tools: list[dict[str, Any] | Any] | None, tool_handler: ToolHandler | None
    ) -> None:
        """The host's tool definitions and the handler that serves them."""
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
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "tags": getattr(t, "tags", []) or [],
                    }
                    for t in ai_tools
                ]
            else:
                tool_defs = tools

        # Host tools that collide with the channel's own (RFC §21.1), each
        # reported once; a name given twice, or one no vendor accepts, is refused.
        self._collisions = CollisionLog(self.channel_id)
        names = [dict_tool_name(tool) for tool in tool_defs or []]
        refuse_unnamable(names, self.channel_id)
        refuse_given_twice(names, self.channel_id)
        self._tools = tool_defs
        warn_tools_uncallable(tool_defs, "tool(s)", self._provider, self.channel_id)
        # What the channel serves itself and what orchestration sets up on it,
        # each tool with its traits, for every room or one (RFC §19.7, §21.1).
        self._registry = ChannelRegistry(self.channel_id, self._host_tool_names)

        # Merge explicit tool_handler with handlers extracted from Tool objects
        effective_handler = tool_handler
        if extracted_handler and tool_handler:
            from roomkit.tools.compose import compose_tool_handlers

            effective_handler = compose_tool_handlers(tool_handler, extracted_handler)
        elif extracted_handler:
            effective_handler = extracted_handler
        self._tool_handler = effective_handler

    def _init_channel_tools(
        self,
        skills: SkillRegistry | None,
        script_executor: ScriptExecutor | None,
        skill_delivery_mode: str | None,
        *,
        tool_search: bool | None,
        tool_search_threshold: int,
        tool_search_pinned: list[str] | None,
    ) -> None:
        """The tools the channel serves itself: the skills' and Tool Search's."""
        names = skills.skill_names if skills is not None else None
        warn_tools_uncallable(names, "skill(s)", self._provider, self.channel_id)
        self._skill_support = self._skill_support_for(skills, script_executor, skill_delivery_mode)
        self._tool_search_support = self._tool_search_for(
            self._tools,
            skills,
            tool_search=tool_search,
            threshold=tool_search_threshold,
            pinned=tool_search_pinned,
        )
        self._register_channel_tools()
        refuse_served_names(
            (dict_tool_name(tool) for tool in self._tools or []),
            self._channel_tool_names() | self._human_input_names(),
            self.channel_id,
        )
        self._human_input.refuse_collisions(self._channel_tool_names(), self.channel_id)

    def _skill_support_for(
        self,
        skills: SkillRegistry | None,
        script_executor: ScriptExecutor | None,
        skill_delivery_mode: str | None,
    ) -> RealtimeSkillSupport | None:
        """This channel's skills support, when it has skills to deliver.

        Skill defs are composed into the tool list at session-start and
        reconfigure time, NOT stored in self._tools, so a session's own
        reconfiguration does not double them.

        Delivery mode is resolved from the explicit kwarg if given,
        otherwise from the provider's reconfigure capability: providers
        that cannot safely reconfigure mid-session (Gemini 3.x) must
        default to ``inline_full`` so every skill body is in the
        initial system_instruction. Others default to ``on_demand`` so
        the prompt stays short until a skill is activated.
        """
        if not (skills and skills.has_entries):
            return None
        from roomkit.channels._realtime_skills import RealtimeSkillSupport

        provider = self._provider
        if skill_delivery_mode is None:
            resolved_mode = (
                "on_demand" if provider.supports_mid_session_reconfigure else "inline_full"
            )
        else:
            resolved_mode = skill_delivery_mode
        support = RealtimeSkillSupport(
            skills,
            script_executor,
            delivery_mode=resolved_mode,
            reconfigure_capable=provider.supports_mid_session_reconfigure,
            exempt_tools=self._exempt_tool_names,
        )
        if support.uses_tool_result and not provider.supports_context_preservation:
            raise ValueError("on_demand skills require provider context preservation")
        return support

    def _register_channel_tools(self) -> None:
        """Register the tools the channel serves itself, each with its traits.

        The skills' and Tool Search's run on paths of their own, with the
        session they act on: their entries carry no server.
        """
        schemas: list[dict[str, Any]] = []
        if self._skill_support is not None:
            schemas += [ACTIVATE_SKILL_SCHEMA, READ_REFERENCE_SCHEMA, RUN_SCRIPT_SCHEMA]
        if self._tool_search_support is not None:
            schemas += self._tool_search_support.search_tool_dicts()
        for schema in schemas:
            self._registry.register(channel_tool(schema_tool(schema), None), owner=self)

    def _host_tool_names(self) -> list[str]:
        """The names the host's own tools carry: its definitions and its
        human-input tools, served by the handlers it gave."""
        names = [name for tool in self._tools or [] if (name := dict_tool_name(tool))]
        return names + sorted(self._human_input_names())

    @property
    def _telemetry_provider(self) -> NoopTelemetryProvider:
        """Access telemetry provider (set by register_channel)."""
        return getattr(self, "_telemetry", None) or NoopTelemetryProvider()

    @property
    def provider(self) -> RealtimeVoiceProvider:
        """The underlying realtime voice provider."""
        return self._provider

    @property
    def transport(self) -> VoiceBackend:
        """The transport backend carrying this channel's audio."""
        return self._transport

    @property
    def session_rooms(self) -> dict[str, str]:
        """Mapping of session_id to room_id."""
        with self._state_lock:
            return dict(self._session_rooms)

    def get_room_sessions(self, room_id: str) -> list[VoiceSession]:
        """Get all active sessions for a room."""
        with self._state_lock:
            return [s for s in self._sessions.values() if self._session_rooms.get(s.id) == room_id]

    async def wait_idle(
        self,
        room_id: str,
        timeout: float = 15.0,
        *,
        session_ids: list[str] | None = None,
    ) -> None:
        """Wait until the selected sessions in the room are idle (not speaking).

        ``session_ids=None`` includes every session. A proactive delivery can
        restrict the wait to its pinned destinations without waiting on others.

        An idle session has submitted its tool results, finished the provider
        response that follows them, and all audio
        has been forwarded to the transport. A queued transport such as SIP
        may still be playing that audio.
        """
        for session in self.get_room_sessions(room_id):
            if session_ids is not None and session.id not in session_ids:
                continue
            event = self._idle_events.get(session.id)
            if event is not None and not event.is_set():
                await asyncio.wait_for(event.wait(), timeout=timeout)

    async def _set_idle(self, session: VoiceSession, response_generation: int) -> None:
        """Settle only the response whose audio just reached the transport."""
        if (
            session.id in self._sessions
            and self._response_generation.get(session.id, 0) == response_generation
        ):
            self._audio_drained.add(session.id)
            self._update_idle_event(session.id)

    def _update_idle_event(self, session_id: str) -> None:
        """Update the idle event based on combined provider + user state."""
        idle = self._idle_events.get(session_id)
        if idle is None:
            return
        provider_done = self._provider_idle.get(session_id, True)
        user_silent = not self._user_speaking.get(session_id, False)
        drained = session_id in self._audio_drained or session_id not in self._response_generation
        tools_done = (
            not self._tool_calls.busy(session_id)
            and session_id not in self._awaiting_tool_response
        )
        delegations_done = not self._pending_delegations.get(session_id)
        if provider_done and user_silent and drained and tools_done and delegations_done:
            idle.set()
        else:
            idle.clear()

    def _expect_provider_output(self, session_id: str) -> None:
        """Keep idle closed until a submitted result has an assistant continuation."""
        self._awaiting_tool_response.add(session_id)
        self._provider_idle[session_id] = False
        self._update_idle_event(session_id)

    def _note_provider_output(self, session_id: str) -> None:
        """A new provider response can continue submitted tool results."""
        if session_id in self._awaiting_tool_response:
            self._awaiting_tool_response.discard(session_id)
            self._provider_idle[session_id] = False
            self._update_idle_event(session_id)

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
        """Propagate telemetry to the realtime provider and the reasoning backend."""
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            self._provider._telemetry = telemetry  # ty: ignore[unresolved-attribute]
            if self._reasoning_backend is not None:
                self._reasoning_backend._adopt_telemetry(telemetry)

    def set_framework(self, framework: RoomKit) -> None:
        """Set the framework reference for event routing.

        Called automatically when the channel is registered with RoomKit.
        """
        self._framework = framework
        self._sync_trace_emitter()
        if self._human_input.given:
            hook = framework._build_on_user_input_required_hook(self.channel_id)
            self._human_input.register(self.channel_id, hook)

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
        for those. A tool under a name the channel or orchestration serves,
        or given twice, is refused (RFC §21.1), and so is a name no vendor
        accepts (RFC §6.7).
        """
        if tools is not None:
            names = [dict_tool_name(tool) for tool in tools]
            refuse_unnamable(names, self.channel_id)
            served = self._channel_tool_names() | self._human_input_names()
            refuse_served_names(names, served, self.channel_id)
            refuse_given_twice(names, self.channel_id)
            self._registry.refuse_host_names(names)
        if system_prompt is not None:
            self._system_prompt = system_prompt
        if voice is not None:
            self._voice = voice
        if tools is not None:
            self._tools = tools

    # -- Public helpers --

    async def start_audio_stream(self, session: VoiceSession) -> None:
        """Open the realtime audio path on the provider.

        Low-level escape hatch for opening the audio stream without
        injecting any text.  Most callers should use
        ``inject_text(..., start_audio_stream=True)`` instead — that
        composes the open + inject in a single call.  No-op on providers
        that don't need it.
        """
        await self._provider.start_audio_stream(session)

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
        start_audio_stream: bool = False,
        chain_depth: int = 0,
    ) -> VoiceInjectionResult | None:
        """Inject a text turn into the provider session.

        Args:
            session: The active voice session.
            text: Text to inject.
            role: The intent (RFC §12.4) — ``"system"`` for an
                instruction, ``"user"`` for content. Anything that directs
                the model, an opening greeting included, is ``"system"``:
                a full-duplex provider voices ``"user"`` text as its own.
            silent: If True, add to conversation context without
                requesting a response.  The agent sees the text on
                its next turn but does not react immediately.
            start_audio_stream: If True, open the realtime audio path
                on the provider before sending the text.  Set this on
                the first inject in outbound flows where the app speaks
                first (e.g. SIP dial greetings); no-op on providers
                that don't need priming (OpenAI, xAI).
            chain_depth: The chain depth of what the text stands for; the
                model's answer to it is one deeper (RFC §12.4). 0, the
                default, for text that opens a chain.
        """
        if start_audio_stream:
            await self._provider.start_audio_stream(session)
        result = await self._provider.inject_text(session, text, role=role, silent=silent)
        if result is not None and result.status == "sent":
            if not silent:
                self._session_answer_depth(session.id).injected(chain_depth)
            await self._fire_text_injected(session, text, role=role)
        logger.info(
            "Text injection into session %s: %s (role=%s, silent=%s, len=%d)",
            session.id,
            result.status if result is not None else "unknown",
            role,
            silent,
            len(text),
        )
        return result

    async def _fire_text_injected(self, session: VoiceSession, text: str, *, role: str) -> None:
        """Announce a text injection to ON_REALTIME_TEXT_INJECTED (RFC §12.5),
        under the session's span. A caller reaching for ``inject_text``
        directly is exactly the case that audit exists for."""
        source = EventSource(
            channel_id=self.channel_id,
            channel_type=self.channel_type,
            participant_id=session.participant_id,
            provider=self.provider_name,
        )
        _, _tok = self._rt_span_ctx(session.id)
        try:
            await fire_text_injected(self._framework, source, session, text, role=role)
        finally:
            if _tok is not None:
                reset_span(_tok)

    async def inject_image(
        self,
        session: VoiceSession,
        image_data: bytes,
        mime_type: str = "image/png",
        *,
        prompt: str = "",
        silent: bool = False,
    ) -> None:
        """Inject an image into the provider session for multimodal analysis.

        Args:
            session: The active voice session.
            image_data: Raw image bytes.
            mime_type: MIME type of the image.
            prompt: Optional text prompt accompanying the image.
            silent: If True, add to context without requesting a response.
        """
        try:
            await self._provider.inject_image(
                session, image_data, mime_type, prompt=prompt, silent=silent
            )
        except NotImplementedError:
            logger.warning(
                "Provider %s does not support image injection (session %s)",
                self._provider.name,
                session.id,
            )
            return
        logger.info(
            "Injected image into session %s (mime=%s, size=%d, prompt=%s)",
            session.id,
            mime_type,
            len(image_data),
            bool(prompt),
        )

    # -- Session lifecycle --

    def _prepare_session_audio(self, session: VoiceSession) -> None:
        """Validate the negotiated rate and build per-session resamplers."""
        # Prefer the per-session rate set by transports such as SIP after
        # codec negotiation. A malformed transport value must fail before it
        # reaches AudioFrame/resampler arithmetic; the caller's start-session
        # rollback then releases the already accepted transport.
        transport_rate = session.metadata.get("transport_sample_rate", self._transport_sample_rate)
        output_rate = session.metadata.get("transport_output_sample_rate", transport_rate)
        if transport_rate is None and output_rate is None:
            return
        transport_rate = self._input_sample_rate if transport_rate is None else transport_rate
        for key, rate in (
            ("transport_sample_rate", transport_rate),
            ("transport_output_sample_rate", output_rate),
        ):
            if not isinstance(rate, int) or isinstance(rate, bool) or not 1 <= rate <= 192_000:
                raise ValueError(f"negotiated {key} must be an integer between 1 and 192000")

        self._session_transport_rates[session.id] = transport_rate
        self._session_transport_output_rates[session.id] = output_rate
        needs_inbound = transport_rate != self._input_sample_rate
        needs_outbound = output_rate != self._output_sample_rate
        if not (needs_inbound or needs_outbound):
            return

        # NumPy first — vectorised interpolation releases the GIL. The sinc
        # fallback preserves correctness but can starve realtime pacing.
        try:
            from roomkit.voice.pipeline.resampler.numpy import (
                NumpyResamplerProvider as _Resampler,
            )
        except ImportError:
            from roomkit.voice.pipeline.resampler.sinc import (
                SincResamplerProvider as _Resampler,
            )

            logger.warning(
                "numpy unavailable — falling back to the pure-Python "
                "sinc resampler for session %s. Realtime pacing may "
                "underrun under load; install numpy for GIL-releasing "
                "resampling.",
                session.id,
            )

        self._session_resamplers[session.id] = (_Resampler(), _Resampler())
        logger.info(
            "Realtime resampler for session %s: %s (transport_in=%d, transport_out=%d, "
            "provider_in=%d, provider_out=%d)",
            session.id,
            _Resampler.__name__,
            transport_rate,
            output_rate,
            self._input_sample_rate,
            self._output_sample_rate,
        )

    async def _finish_session_start(
        self,
        session: VoiceSession,
        room_id: str,
        participant_id: str,
    ) -> None:
        """Publish an active session and flush audio captured during handshake."""
        if self._framework:
            stored_binding = await self._framework._store.get_binding(room_id, self.channel_id)
            if stored_binding is not None:
                with self._state_lock:
                    self._session_bindings[session.id] = stored_binding

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

        self._wire_realtime_recording(room_id, session)

        # The transport was live throughout the provider handshake. Flush
        # caller speech before exposing the session-start notification; new
        # frames continue joining the bounded buffer until this drain ends.
        await self._flush_preconnect_audio(session)

        logger.info(
            "Realtime session %s started: room=%s, participant=%s, provider=%s",
            session.id,
            room_id,
            participant_id,
            self._provider.name,
        )

        # Hook failures are observational and do not roll back a live session.
        if self._framework:
            try:
                from roomkit.models.session_event import SessionStartedEvent

                context = await self._framework._build_context(room_id)
                ready_event = SessionStartedEvent(
                    room_id=room_id,
                    channel_id=self.channel_id,
                    channel_type=self.channel_type,
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

        await self._send_client_message(session, {"type": "session_started"})

    async def start_session(
        self,
        room_id: str,
        participant_id: str,
        connection: Any,
        *,
        metadata: dict[str, Any] | None = None,
        organization_id: str | None = None,
    ) -> VoiceSession:
        """Start a new realtime voice session.

        Connects both the transport (client audio) and the provider
        (AI service), then fires a framework event.

        Args:
            room_id: The room to join.
            participant_id: The participant's ID.
            connection: Protocol-specific connection (e.g. WebSocket), or an
                awaitable resolving to it. With an awaitable, provider setup
                starts while the connection is pending (e.g. SIP ringing).
                Cancel the start task if the connection will never arrive.
                Both branches belong to this session and are rolled back on
                failure. The channel publishes the session only when both are ready.
                Provider callbacks wait for publication in a bounded startup
                queue; exceeding 2 MiB of audio or 500 events aborts startup.
                Monotonic timestamps are written to metadata's
                ``connection_timing`` for measuring setup and pre-answer cost.
            metadata: Optional session metadata. May include overrides
                for system_prompt, voice, tools, temperature.
            organization_id: The organization the caller acts for (RFC §17.2).
                The room is read scoped to it before the session exists:
                another organization's room is not found, and none of its
                recordings is told of the session. Left unset, the room is
                not read.

        Returns:
            The created VoiceSession.

        Raises:
            RoomNotFoundError: *organization_id* is set and the room is
                missing, another organization's, or unreadable because the
                channel is not registered with a framework.
        """
        if self._closing:
            raise RuntimeError("Realtime voice channel is closing")
        await _check_room_scope(self._framework, room_id, organization_id)
        session = VoiceSession(
            id=uuid4().hex,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=self.channel_id,
            state=VoiceSessionState.CONNECTING,
            metadata=metadata or {},
        )
        task = asyncio.create_task(
            self._start_session(session, connection), name=f"rt_connect:{session.id}"
        )
        pending = _ConnectingSession(task, deferred=inspect.isawaitable(connection))
        self._connecting_sessions[session.id] = pending
        try:
            return await task
        finally:
            self._connecting_sessions.pop(session.id, None)

    async def _start_session(self, session: VoiceSession, connection: Any) -> VoiceSession:
        try:
            return await self._connect_session(session, connection)
        except (Exception, asyncio.CancelledError):
            # Run to its end whatever cancels the start again: close() waits
            # for this task, so nothing of the rollback is left to its sweep.
            await _finish_cleanup(self._roll_back_start(session))
            pending = self._connecting_sessions.get(session.id)
            if pending is not None and pending.failure is not None:
                raise pending.failure from None
            raise

    async def _roll_back_start(self, session: VoiceSession) -> None:
        """End a start that failed: the session's end once it was filed, else
        the handshake's rollback; then the reports of the calls the provider
        issued while it was pending (RFC §12.4)."""
        if session.id in self._sessions:
            await self.end_session(session)
        else:
            await self._cleanup_failed_start(session)
        pending = self._connecting_sessions.get(session.id)
        if pending is not None and pending.tool_calls:
            await self._report_start_calls(session, pending.tool_calls)

    async def _cleanup_failed_start(self, session: VoiceSession) -> None:
        """Roll back every partially initialized handshake through one path."""
        session.state = VoiceSessionState.ENDED
        # Its in-flight calls end as at a session's end: stopped and reported.
        await self._stop_session_tools(session)
        with contextlib.suppress(Exception):
            self._pipeline_session_ended(session)
        with self._state_lock:
            resamplers = self._session_resamplers.pop(session.id, None)
            self._session_transport_rates.pop(session.id, None)
            self._session_transport_output_rates.pop(session.id, None)
            self._preconnect_audio.pop(session.id, None)
            self._preconnect_audio_bytes.pop(session.id, None)
            self._preconnect_audio_dropped.discard(session.id)
            idle = self._idle_events.pop(session.id, None)
            if idle is not None:
                idle.set()
            self._user_speaking.pop(session.id, None)
            self._provider_idle.pop(session.id, None)
            self._awaiting_tool_response.discard(session.id)
            self._session_tools.pop(session.id, None)
            self._session_roles.pop(session.id, None)
            self._session_agent_policies.pop(session.id, None)
            self._session_config_locks.pop(session.id, None)
            self._response_generation.pop(session.id, None)
            self._audio_drained.discard(session.id)
            self._has_pipeline_vad.pop(session.id, None)
            span_id = self._session_spans.pop(session.id, None)
        if self._skill_support:
            self._skill_support.cleanup_session(session.id)
        if self._tool_search_support:
            self._tool_search_support.cleanup_session(session.id)
        for owner in (self._provider, self._transport):
            with contextlib.suppress(Exception):
                await owner.disconnect(session)
        if resamplers:
            for resampler in resamplers:
                with contextlib.suppress(Exception):
                    resampler.close()
        if span_id:
            self._telemetry_provider.end_span(span_id)
            self._telemetry_provider.flush()

    def _check_connecting_session(self, session: VoiceSession) -> None:
        pending = self._connecting_sessions.get(session.id)
        if self._closing or (pending is not None and pending.disconnected):
            raise asyncio.CancelledError("Voice transport disconnected during connection")

    async def _session_config(
        self, session: VoiceSession
    ) -> tuple[str | None, str | None, Any, float | None, dict[str, Any] | None]:
        """The prompt, voice, tools, temperature and provider settings a new
        session starts with (RFC §12.4): what the session was opened with, else
        what was set for its room (a pipeline's active agent), else the
        channel's."""
        meta = session.metadata
        room = await self._room_session_config(session.room_id)
        if room is not None and room.tool_policy is not None:
            # The agent the session speaks as answers to its own policy too.
            self._session_agent_policies[session.id] = room.tool_policy
        # Field by field: what the room's agent leaves unset is the channel's.
        room_prompt, room_voice, room_tools = (
            (room.system_prompt, room.voice, room.tools) if room is not None else (None,) * 3
        )
        system_prompt = meta.get(
            "system_prompt", room_prompt if room_prompt is not None else self._system_prompt
        )
        meta["system_prompt"] = system_prompt
        voice = meta.get("voice", room_voice if room_voice is not None else self._voice)
        tools = meta.get("tools", room_tools if room_tools is not None else self._tools)
        if "tools" in meta:
            # Given with the session, so refused here as at construction (§6.7).
            given = meta["tools"] or []
            refuse_unnamable((dict_tool_name(tool) for tool in given), self.channel_id)
        temperature = meta.get("temperature", self._temperature)
        provider_config = meta.get("provider_config")
        if self._skill_support and self._skill_support.uses_tool_result:
            provider_config = {**(provider_config or {}), "preserve_context": True}
        return system_prompt, voice, tools, temperature, provider_config

    async def _room_session_config(self, room_id: str) -> SessionConfig | None:
        """What orchestration set for *room_id*'s sessions, if it set anything."""
        source = self._registry.session_source
        return await source(room_id) if source is not None else None

    async def _connect_session(self, session: VoiceSession, connection: Any) -> VoiceSession:
        room_id, participant_id = session.room_id, session.participant_id
        self._session_config_locks[session.id] = asyncio.Lock()
        meta = session.metadata
        timing: dict[str, float] = {}
        meta["connection_timing"] = timing
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
        with self._state_lock:
            self._preconnect_audio[session.id] = []
            self._preconnect_audio_bytes[session.id] = 0

        # Initialize skill activation state for this session
        if self._skill_support:
            self._skill_support.init_session(session.id)

        system_prompt, voice, tools, temperature, provider_config = await self._session_config(
            session
        )
        # The participant's role, for the role overrides of the policies the
        # session answers to (a pipeline agent's set just above); read before
        # any tool list is composed for the session.
        self._session_roles[session.id] = await self._resolve_session_role(
            room_id, participant_id, session.id
        )

        # Cache the resolved base tool list (channel defaults + metadata
        # overrides) so skill activation can reconfigure without losing them.
        # Under the lock its readers take: a recovered tool call reaches them
        # from a background task.
        with self._state_lock:
            self._session_tools[session.id] = self._declared_once(deepcopy(tools or []), room_id)
        offered = {dict_tool_name(tool) for tool in self._session_declared_tools(session.id)}
        self._human_input.warn_unoffered(offered, self.channel_id)

        if self._tool_search_support:
            self._tool_search_support.init_session(session.id, self._session_tools[session.id])
        system_prompt = self._compose_session_prompt(session, system_prompt)

        tools = self._compose_session_tools(session, tools)

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
            has_pipeline_vad = self._pipeline_config.vad is not None
            if has_pipeline_vad and self._provider.full_duplex:
                logger.info(
                    "Pipeline VAD runs in observation role for session %s: provider %s "
                    "is full-duplex, so its speech events feed hooks and metrics but "
                    "never interrupt the model (RFC §12.4.1)",
                    session.id,
                    self._provider.name,
                )
            # Set BEFORE creating pipeline so that provider callbacks
            # arriving early see the correct flag and don't double-fire.
            with self._state_lock:
                self._has_pipeline_vad[session.id] = has_pipeline_vad
            # The pipeline belongs to the channel, not to a session. Creating
            # it again would register another transport callback and forward
            # every later microphone frame once per session ever started.
            if self._pipeline is None:
                pl = self._create_pipeline(self._pipeline_config, self._transport)
                pl.on_vad_event(self._on_pipeline_vad_event)
                pl.on_processed_frame(self._on_pipeline_processed_frame)
                if self._pipeline_config.recorder is not None:
                    self._wire_recording_hooks(pl)

        # Some transports start their microphone inside accept(). Activate
        # per-session pipeline state first so the earliest callback cannot run
        # through an uninitialized recorder/AEC/debug stream.
        if self._pipeline is not None:
            self._pipeline_session_active(session, parent_span=session_span_id)

        async def accept_transport() -> None:
            resolved = await connection if inspect.isawaitable(connection) else connection
            self._check_connecting_session(session)
            with telemetry.span(
                SpanKind.BACKEND_CONNECT,
                "transport.accept",
                parent_id=session_span_id,
                session_id=session.id,
                attributes={Attr.BACKEND_TYPE: self._transport.name},
            ):
                await self._transport.accept(session, resolved)
            self._check_connecting_session(session)
            self._prepare_session_audio(session)
            timing["transport_ready_at"] = time.monotonic()

        async def connect_provider() -> None:
            # Connect to provider (with telemetry span).
            # If provider.connect fails, clean up the already-accepted transport
            # session to avoid leaking the connection.
            try:
                timing["provider_connect_started_at"] = time.monotonic()
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
                        server_vad=self._provider.full_duplex or not has_pipeline_vad,
                        provider_config=provider_config,
                    )
                timing["provider_ready_at"] = time.monotonic()
                logger.info(
                    "Realtime provider ready: room=%s session=%s connect_ms=%.1f",
                    room_id,
                    session.id,
                    (timing["provider_ready_at"] - timing["provider_connect_started_at"]) * 1000,
                )
            except (Exception, asyncio.CancelledError) as exc:
                # CancelledError here is the orchestrator deliberately aborting
                # a still-handshaking session (e.g. carrier hung up before the
                # provider WS connected). It's expected control flow, not a
                # bug — log it as info without a traceback so dashboards stay
                # quiet. Real failures keep their full stack trace.
                if isinstance(exc, asyncio.CancelledError):
                    logger.info(
                        "provider.connect cancelled for session %s — transport cleaned up",
                        session.id,
                    )
                else:
                    logger.exception(
                        "provider.connect failed for session %s; cleaned up transport",
                        session.id,
                    )
                raise

        if inspect.isawaitable(connection):
            # One handshake owns both branches. gather alone would leave its
            # sibling running after an error; drain both before rollback.
            branches = [
                asyncio.create_task(accept_transport(), name=f"rt_transport:{session.id}"),
                asyncio.create_task(connect_provider(), name=f"rt_provider:{session.id}"),
            ]
            try:
                await asyncio.gather(*branches)
            finally:
                for branch in branches:
                    if not branch.done():
                        branch.cancel()
                await asyncio.gather(*branches, return_exceptions=True)
        else:
            # Preserve ordinary startup ordering, including codec validation
            # before any provider connection is opened.
            await accept_transport()
            await connect_provider()

        self._check_connecting_session(session)
        session.state = VoiceSessionState.ACTIVE
        with self._state_lock:
            self._sessions[session.id] = session
            self._session_rooms[session.id] = room_id

        await self._finish_session_start(session, room_id, participant_id)
        self._check_connecting_session(session)
        pending = self._connecting_sessions[session.id]
        pending.deferred = False
        # The journal's calls go to the executor now: none is left to report.
        pending.tool_calls.clear()
        for callback, args in pending.callbacks:
            callback(*args)
        pending.callbacks.clear()

        return session

    async def end_session(self, session: VoiceSession) -> None:
        """End a realtime voice session.

        Disconnects both provider and transport, fires framework event.

        Concurrent callers share one teardown: the first owns it, and a caller
        that arrives while it runs (``close()``, the transport's disconnect
        callback, a hangup tool) waits for it instead of tearing the session
        down a second time. A joiner learns that the teardown ended, not how:
        the owner's exception is the owner's. The work stays in the owner's
        task, so a tool that is the first to end its own session is still the
        current task the tool sweep spares; a joiner cancelled while waiting
        leaves the owner's teardown untouched; and a call re-entered from the
        teardown itself (a handler of the ended event ending the session it is
        told about) returns at once instead of waiting on its own task. A
        subclass's own teardown rides ``_before_session_teardown`` and is
        covered by the same arbitration.

        Args:
            session: The session to end.
        """
        current = asyncio.current_task()
        with self._state_lock:
            teardown = self._session_teardowns.get(session.id)
            owned = teardown is None
            if teardown is None:
                teardown = _SessionTeardown(
                    owner=current, done=asyncio.get_running_loop().create_future()
                )
                self._session_teardowns[session.id] = teardown
        if not owned:
            if teardown.owner is not current:
                await asyncio.shield(teardown.done)
            return
        try:
            await self._before_session_teardown(session)
            await self._end_session_owned(session)
        finally:
            with self._state_lock:
                self._session_teardowns.pop(session.id, None)
            teardown.done.set_result(None)

    async def _before_session_teardown(self, session: VoiceSession) -> None:
        """A subclass's own teardown, run by the owner before the base one."""

    async def _stop_session_tools(self, session: VoiceSession) -> None:
        """Stop the session's in-flight tools and report each call they leave
        unanswered, once, as cancelled (RFC §12.4).

        Calls owned by other sessions sharing this channel are left alone. A
        call whose own handler ends the session (a hang-up tool) is not
        interrupted: it runs on and reports its own outcome.
        """
        current = asyncio.current_task()
        interrupted = [c for c in self._tool_calls.take(session.id) if c.task is not current]
        prefixes = (
            f"rt_tool_call:{session.id}:",
            f"rt_tool_recovery:{session.id}:",
            f"rt_delegation:{session.id}:",
        )
        tool_tasks = [
            task
            for task in self._scheduled_tasks
            if task is not current and task.get_name().startswith(prefixes)
        ]
        for task in tool_tasks:
            task.cancel()
        if tool_tasks:
            _, pending = await asyncio.wait(tool_tasks, timeout=5.0)
            if pending:
                logger.warning(
                    "Timed out cancelling %d tools for session %s", len(pending), session.id
                )
        await report_interrupted_calls(self, interrupted, SESSION_ENDED)

    async def _end_session_owned(self, session: VoiceSession) -> None:
        """The teardown itself, run once per session by ``end_session``."""
        # Stop admitting calls and audio before this teardown's asynchronous
        # steps. A session hangup must also stop its in-flight tools without
        # touching calls owned by other sessions sharing this channel.
        session.state = VoiceSessionState.ENDED
        self._pipeline_session_ending(session)
        await self._stop_session_tools(session)

        with self._state_lock:
            room_id = self._session_rooms.get(session.id, session.room_id)

        # Notify client before tearing down the connection
        try:
            await self._send_client_message(session, {"type": "session_ended"})
        except Exception:
            # A dead client is the common reason a session ends. Notification
            # failure must never skip provider, transport, pipeline and span
            # cleanup below.
            logger.debug(
                "Could not notify client that session %s ended",
                session.id,
                exc_info=True,
            )

        # Disconnect provider and transport
        try:
            await self._provider.disconnect(session)
        except Exception:
            logger.exception("Error disconnecting provider for session %s", session.id)

        try:
            await self._transport.disconnect(session)
        except Exception:
            logger.exception("Error disconnecting transport for session %s", session.id)

        self._pipeline_session_ended(session)

        # Clean up skill activation state
        if self._skill_support:
            self._skill_support.cleanup_session(session.id)
        # Clean up tool-search exposure window
        if self._tool_search_support:
            self._tool_search_support.cleanup_session(session.id)

        with self._state_lock:
            self._sessions.pop(session.id, None)
            self._session_rooms.pop(session.id, None)
            self._session_bindings.pop(session.id, None)
            self._session_tools.pop(session.id, None)
            self._session_roles.pop(session.id, None)
            self._session_agent_policies.pop(session.id, None)
            self._session_config_locks.pop(session.id, None)
            self._audio_generation.pop(session.id, None)
            self._session_transport_rates.pop(session.id, None)
            self._session_transport_output_rates.pop(session.id, None)
            self._audio_forward_count.pop(session.id, None)
            self._forget_transcription_state(session.id)
            self._end_recording_track(session.id)
            self._barge_in_active.discard(session.id)
            self._playback_started_at.pop(session.id, None)
            self._playback_position_ms.pop(session.id, None)
            self._playback_buffer.pop(session.id, None)
            self._response_generation.pop(session.id, None)
            self._audio_drained.discard(session.id)
            self._has_pipeline_vad.pop(session.id, None)
            idle = self._idle_events.pop(session.id, None)
            if idle is not None:
                idle.set()  # unblock any waiters
            self._user_speaking.pop(session.id, None)
            self._user_turn_start_at.pop(session.id, None)
            self._provider_idle.pop(session.id, None)
            self._tool_calls.take(session.id)
            self._awaiting_tool_response.discard(session.id)
            self._transcript_ledger.pop(session.id, None)
            self._delegated_before.discard(session.id)
            self._pending_delegations.pop(session.id, None)
            turn_span_id = self._turn_spans.pop(session.id, None)
            session_span_id = self._session_spans.pop(session.id, None)
            resamplers = self._session_resamplers.pop(session.id, None)
            send_queue = self._audio_send_queues.pop(session.id, None)
            self._audio_send_workers.pop(session.id, None)
            self._audio_dropped.pop(session.id, None)
            self._preconnect_audio.pop(session.id, None)
            self._preconnect_audio_bytes.pop(session.id, None)
            self._preconnect_audio_dropped.discard(session.id)

        if self._reasoning_backend is not None:
            try:
                await self._reasoning_backend.session_ended(session.id)
            except Exception:
                logger.exception(
                    "Error releasing reasoning backend state for session %s", session.id
                )

        # Release the send worker — it exits on the sentinel; anything still
        # queued belongs to the closed session and is dropped with it.
        if send_queue is not None:
            send_queue.put_nowait(None)

        # End any active turn span, then the session span
        telemetry = self._telemetry_provider
        if turn_span_id:
            telemetry.end_span(turn_span_id)
        if session_span_id:
            telemetry.end_span(session_span_id)
            telemetry.flush()

        if resamplers:

            def _close_resamplers() -> None:
                for r in resamplers:
                    try:
                        r.close()
                    except Exception:
                        logger.exception("Error closing resampler for session %s", session.id)

            # Through the resample executor: FIFO behind any in-flight
            # resample, so close never races state mutation.
            ex = self._resample_executor
            if ex is not None:
                ex.submit(_close_resamplers)
            else:
                _close_resamplers()

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

    def _compose_session_prompt(
        self,
        session: VoiceSession,
        prompt: str | None,
        *,
        pending_skill: Skill | None = None,
        search_active: bool | None = None,
    ) -> str | None:
        """One composition path for connection, discovery, activation and handoff."""
        if not self._provider.supports_tools:
            return prompt  # no tool to call, so no skill or search to advertise
        if self._skill_support:
            scripts_allowed = self._session_admits(session.id, TOOL_RUN_SCRIPT)
            prompt = self._skill_support.inject_skills_prompt(
                prompt, scripts_allowed=scripts_allowed
            )
            addendum = self._skill_support.activated_skills_prompt(session.id, pending_skill)
            if addendum:
                prompt += "\n\n" + addendum
        support = self._tool_search_support
        if support and (support.active(session.id) if search_active is None else search_active):
            prompt = (prompt or "") + "\n\n" + support.preamble
        return prompt

    def _tool_search_for(
        self,
        tool_defs: list[dict[str, Any]] | None,
        skills: SkillRegistry | None,
        *,
        tool_search: bool | None,
        threshold: int,
        pinned: list[str] | None,
    ) -> RealtimeToolSearchSupport | None:
        """This channel's Tool Search, unless ``tool_search=False``.

        It hides a session's catalogue when that session declares enough tools
        to overflow the realtime model's reliable tool-selection window (Google
        Gemini Live: 10–20 active tools), decided per session on what it
        declares, or in every session when a fixed-declaration provider needs
        it to open a skill's gated tools. Auto-detect by default;
        ``tool_search=True`` or ``False`` forces. Reconfigurable providers receive native
        declarations on discovery; fixed-declaration providers receive schemas
        through ``list_tools`` and carry execution through ``call_tool`` into
        the same channel dispatch.
        """
        provider = self._provider
        fixed_skill_gates = bool(
            self._skill_support
            and skills is not None
            and not provider.supports_mid_session_reconfigure
            and any(meta.gated_tool_names for meta in skills.all_metadata())
        )
        if fixed_skill_gates and tool_search is False:
            raise ValueError("Fixed-provider skill gates require Tool Search; tool_search=False")
        if tool_search is False:
            return None
        from roomkit.channels._realtime_tool_search import RealtimeToolSearchSupport

        return RealtimeToolSearchSupport(
            tool_defs or [],
            pinned=pinned,
            threshold=threshold,
            reconfigure_capable=provider.supports_mid_session_reconfigure,
            reachable=self._tool_reachable,
            never_deferred=self._session_never_deferred,
            listed=self._session_declared_tools,
            auto=tool_search is None and not fixed_skill_gates,
        )

    def _session_never_deferred(self, session_id: str) -> set[str]:
        """The tools Tool Search never hides in the session's room (RFC §21.1)."""
        with self._state_lock:
            room_id = self._session_rooms.get(session_id)
        return self._registry.names(room_id, lambda traits: not traits.deferrable)

    def _compose_session_tools(
        self,
        session: VoiceSession,
        tools: list[dict[str, Any]] | None,
        *,
        reset_exposure: bool = False,
        pending_skill: Skill | None = None,
        search_active: bool | None = None,
    ) -> list[dict[str, Any]] | None:
        """Compose the same infrastructure, orchestration and skill gates on
        connect, discovery, activation and handoff; *search_active* decides Tool
        Search for tools the session is about to declare."""
        if not self._provider.supports_tools:
            return None  # the model calls no tool (RFC §12.4)
        session_id, room_id = session.id, session.room_id
        # What orchestration set up for the room and the person's tools are
        # declared whatever Tool Search hides (RFC §9.3, §21.1).
        always = self._orchestration_dicts(room_id) + self._human_input_dicts()
        if (
            tools is None
            and not always
            and not self._tool_search_support
            and not self._skill_support
        ):
            return None
        visible = self._declared_once(deepcopy(tools or []), room_id)
        if self._tool_search_support:
            visible = self._tool_search_support.visible_tools(
                session_id,
                visible,
                reset_exposure=reset_exposure,
                keep=self._registry.names(room_id, lambda traits: not traits.deferrable),
                active=search_active,
            )
        visible += always
        if self._skill_support:
            visible = self._skill_support.skill_tool_dicts() + visible
            visible = self._skill_support.get_visible_tools(visible, session_id, pending_skill)
        return self._policy_filter(session_id, visible)

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
            tools: New tool/function definitions; one under a name no vendor
                accepts raises ``ValueError``, the session left as it was.
            temperature: New sampling temperature.
            provider_config: Provider-specific configuration overrides.
        """
        if tools is not None:
            refuse_unnamable((dict_tool_name(tool) for tool in tools), self.channel_id)
        lock = self._session_config_locks.get(session.id)
        if lock is None:
            return
        async with lock:
            if session.state == VoiceSessionState.ENDED:
                return
            # Save caller values before skills mutation: the session keeps the
            # *user* values, not the skill-enriched versions, to avoid doubling
            # on its next reconfiguration.
            caller_tools = deepcopy(tools)
            caller_prompt = system_prompt
            # Whether Tool Search hides the new catalogue decides both the
            # prompt's preamble and the declaration; it is kept only once the
            # provider took them.
            search_active = self._search_activates(session, caller_tools)

            system_prompt = self._compose_session_prompt(
                session,
                system_prompt
                if system_prompt is not None
                else session.metadata.get("system_prompt", self._system_prompt),
                search_active=search_active,
            )
            if tools is not None:
                tools = self._compose_session_tools(
                    session, tools, reset_exposure=True, search_active=search_active
                )

            await self._provider.reconfigure(
                session,
                system_prompt=system_prompt,
                voice=voice,
                tools=tools,
                temperature=temperature,
                provider_config=provider_config,
            )

            if session.state == VoiceSessionState.ENDED:
                return
            # The session's own configuration: the channel's, for future
            # sessions and other rooms, is not this session's to change (RFC
            # §12.4). Stored as the caller gave it (without skill enrichment).
            if caller_prompt is not None:
                session.metadata["system_prompt"] = caller_prompt
            if caller_tools is not None:
                self._store_session_tools(session, caller_tools)

            logger.info("Realtime session %s reconfigured", session.id)

    def _search_activates(
        self, session: VoiceSession, tools: list[dict[str, Any]] | None
    ) -> bool | None:
        """Whether Tool Search would hide *tools* in *session*; ``None`` when
        they change nothing it decides on."""
        if tools is None or self._tool_search_support is None:
            return None
        candidate = self._declared_once(deepcopy(tools), session.room_id)
        return self._tool_search_support.activates(session.id, candidate)

    def _store_session_tools(self, session: VoiceSession, tools: list[dict[str, Any]]) -> None:
        """Keep *tools* as a live session's base catalogue, and let Tool Search
        decide on it (outside the state lock, which its measure takes)."""
        stored = self._declared_once(deepcopy(tools), session.room_id)
        with self._state_lock:
            if session.id not in self._sessions:
                return
            self._session_tools[session.id] = stored
        if self._tool_search_support:
            self._tool_search_support.init_session(session.id, stored)

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
        for rt_session in self.get_room_sessions(room_id):
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

        silent = binding.muted or binding.output_muted or not binding.can_write
        # Inject text into all active sessions for this room
        for session in self.get_room_sessions(room_id):
            try:
                result = await self._provider.inject_text(
                    session, text, role=inject_role, silent=silent
                )
                if result is None or result.status != "sent":
                    logger.debug("Text injection unconfirmed for session %s", session.id)
                    continue
                if not silent:
                    self._session_answer_depth(session.id).injected(event.chain_depth)

                # Fire ON_REALTIME_TEXT_INJECTED hook (async)
                if self._framework:
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
        """No-op for realtime voice — content is injected via kit.deliver()'s
        ``_deliver_to_realtime_voice`` path, which calls ``inject_text``
        directly. Events broadcast through channel.deliver() would double-
        feed the session."""
        return ChannelOutput.empty()

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.AUDIO, ChannelMediaType.TEXT],
            supports_audio=True,
            custom={"realtime": True, "server_vad": True},
        )

    async def close(self) -> None:
        """End owned sessions, unsubscribe callbacks, and close owned resources."""
        for unsubscribe in self._transport_unsubscribers:
            if unsubscribe is not None:
                unsubscribe()
        self._transport_unsubscribers.clear()
        # The frames in flight finish while their sessions are live: nothing
        # they fire reaches a provider or a client after the session ended.
        await self._pipeline_quiesce()
        self._closing = True
        connecting = list(self._connecting_sessions.values())
        current = asyncio.current_task()
        tasks = []
        for pending in connecting:
            pending.disconnected = True
            if pending.task is not current and not pending.task.done():
                if not pending.task.cancelling():
                    pending.task.cancel()
                tasks.append(pending.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # End all active sessions
        with self._state_lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            try:
                await self.end_session(session)
            except Exception:
                logger.exception("Error ending session %s during close", session.id)

        # A teardown under way whose session already left ``_sessions`` is in
        # neither list above and not done: wait for it before the sweep below
        # cancels the task running it, mid-emit.
        with self._state_lock:
            outstanding = [teardown.done for teardown in self._session_teardowns.values()]
        if outstanding:
            await asyncio.gather(
                *(asyncio.shield(done) for done in outstanding), return_exceptions=True
            )

        # What the ended sessions left is settled first: the sweep below would
        # cut it.
        await self._settle_ended_sessions()

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

        # Queued resampler close jobs still run; sessions are already ended
        # so nothing new can be queued.
        if self._resample_executor is not None:
            self._resample_executor.shutdown(wait=False)
            self._resample_executor = None

        if self._reasoning_backend is not None:
            try:
                await self._reasoning_backend.close()
            except Exception:
                logger.exception("Error closing reasoning backend during channel close")
        try:
            await self._provider.close()
        except Exception:
            logger.exception("Error closing provider during channel close")
        try:
            if self._owns_transport:
                await self._transport.close()
        except Exception:
            logger.exception("Error closing transport during channel close")

    # -- Client messaging --

    async def _settle_ended_sessions(self) -> None:
        """Settle what the ended sessions left: the reports of their abandoned
        calls, then the person's requests still open, the channel taking no
        more (RFC §9.3)."""
        await self._settle_tool_reports()
        await self._human_input.close(self.channel_id)

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

    def _recording_room(self, session: VoiceSession) -> str | None:
        """The room a session's recording reports to. The pipeline starts the
        recording before the session is filed under its room, so the session's
        own room answers until then."""
        with self._state_lock:
            return self._session_rooms.get(session.id, session.room_id)

    def _recording_span(self, session: VoiceSession) -> str | None:
        with self._state_lock:
            return self._session_spans.get(session.id)

    def _schedule_recording_hook(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None:
        """Run a recording hook as a tracked task, from the loop or a foreign thread."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            cached = self._event_loop
            if cached is not None and cached.is_running():
                cached.call_soon_threadsafe(
                    functools.partial(self._track_task, cached, coro, name=name)
                )
            else:
                coro.close()
            return
        self._track_task(loop, coro, name=name)

    # -- Internal callbacks --

    def _gate_provider_callback(
        self, callback: Callable[..., Any], *, tool_call: bool = False
    ) -> Callable[..., Any]:
        """Keep startup events ordered until transport and authorization are ready.

        No application task runs while SIP is ringing. Audio enters the normal
        send FIFO only after the negotiated codec and binding are available.
        Rollback discards the journal with its connecting-session owner; a
        *tool_call* is kept apart as well, even once the start failed (the
        provider is connected until the rollback disconnects it), and
        reported, cancelled, if the start fails (RFC §12.4).
        """

        def dispatch(session: VoiceSession, *args: Any) -> Any:
            pending = self._connecting_sessions.get(session.id)
            if pending is None or not pending.deferred:
                return callback(session, *args)
            if tool_call:
                pending.tool_calls.append(args)
            if pending.failure is not None or pending.task.cancelling():
                return None
            size = sum(len(arg) for arg in args if isinstance(arg, bytes))
            if (
                len(pending.callbacks) >= _MAX_QUEUED_AUDIO_CHUNKS
                or pending.audio_bytes + size > 2 * 1024 * 1024
            ):
                pending.fail(RuntimeError("Provider startup event buffer exceeded"))
                return None
            pending.audio_bytes += size
            pending.callbacks.append((callback, (session, *args)))
            return None

        return dispatch

    def _on_startup_provider_error(self, session: VoiceSession, code: str, message: str) -> Any:
        pending = self._connecting_sessions.get(session.id)
        if pending is not None and session.state == VoiceSessionState.ENDED:
            pending.fail(RuntimeError(f"Provider connection failed [{code}]: {message}"))
            self._announce_provider_error(session, code, message)
            return None
        return self._on_provider_error(session, code, message)

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
        pending = self._connecting_sessions.get(session.id)
        if pending is not None:
            pending.disconnected = True
            if not pending.task.done() and not pending.task.cancelling():
                pending.task.cancel()
            return
        with self._state_lock:
            active = session.id in self._sessions
        if active:
            await self.end_session(session)
