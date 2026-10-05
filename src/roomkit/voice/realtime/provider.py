"""RealtimeVoiceProvider abstract base class."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Iterable
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.telemetry.base import Attr
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.injection import VoiceInjectionResult

# Re-exported: the realtime providers and callers import it from here.
from roomkit.voice.voices import VoiceInfo as VoiceInfo

logger = logging.getLogger("roomkit.voice.realtime.provider")

if TYPE_CHECKING:
    from roomkit.providers.ai.base import ModelInfo
    from roomkit.video.video_frame import VideoFrame

# Callback type aliases
RealtimeAudioCallback = Callable[[VoiceSession, bytes], Any]
RealtimeTranscriptionCallback = Callable[[VoiceSession, str, str, bool], Any]
"""(session, text, role, is_final)"""
RealtimeSpeechStartCallback = Callable[[VoiceSession], Any]
RealtimeSpeechEndCallback = Callable[[VoiceSession], Any]
RealtimeToolCallCallback = Callable[[VoiceSession, str, str, dict[str, Any] | str], Any]
"""(session, call_id, name, arguments): the arguments as a mapping, or the text
the model wrote when they do not read as one, a call the channel refuses (RFC §12.4)"""
RealtimeToolCallCancelledCallback = Callable[[VoiceSession, list[str]], Any]
"""(session, call_ids) — the model will not read these calls' results"""
RealtimeResponseStartCallback = Callable[[VoiceSession], Any]
RealtimeResponseEndCallback = Callable[[VoiceSession], Any]
RealtimeErrorCallback = Callable[[VoiceSession, str, str], Any]
"""(session, code, message)"""
RealtimeDelegationCallback = Callable[[VoiceSession, str, str], Any]
"""(session, delegation_id, target) — ``target`` is ``"hosted"`` or ``"integrator"``"""
RealtimeUsageCallback = Callable[[VoiceSession, dict[str, Any]], Any]
"""(session, usage) — what the provider has just recorded for the session"""


class RealtimeVoiceProvider(ABC):
    """Abstract base class for speech-to-speech AI providers.

    Wraps APIs like OpenAI Realtime and Gemini Live that handle
    audio-in → audio-out with built-in AI, VAD, and transcription.

    The provider manages a bidirectional audio/event stream with the
    AI service. Callbacks are registered for events the provider emits.

    Example:
        provider = OpenAIRealtimeProvider(api_key="sk-...", model="gpt-realtime-2.1")

        provider.on_audio(handle_audio)
        provider.on_transcription(handle_transcription)
        provider.on_tool_call(handle_tool_call)

        await provider.connect(session, system_prompt="You are a helpful agent.")
        await provider.send_audio(session, audio_bytes)
        await provider.disconnect(session)

    Subclasses **must** call ``super().__init__()`` to initialise
    the callback lists.
    """

    def __init__(self) -> None:
        self._audio_callbacks: list[RealtimeAudioCallback] = []
        self._transcription_callbacks: list[RealtimeTranscriptionCallback] = []
        self._speech_start_callbacks: list[RealtimeSpeechStartCallback] = []
        self._speech_end_callbacks: list[RealtimeSpeechEndCallback] = []
        self._tool_call_callbacks: list[RealtimeToolCallCallback] = []
        self._tool_call_cancelled_callbacks: list[RealtimeToolCallCancelledCallback] = []
        self._response_start_callbacks: list[RealtimeResponseStartCallback] = []
        self._response_end_callbacks: list[RealtimeResponseEndCallback] = []
        self._error_callbacks: list[RealtimeErrorCallback] = []
        self._delegation_callbacks: list[RealtimeDelegationCallback] = []
        self._usage_callbacks: list[RealtimeUsageCallback] = []
        self._usage_tasks: set[asyncio.Task[None]] = set()
        # The calls each session issued that still owe a result, by session
        # and call id, each with what its provider needs to answer it (a
        # name, a pending future): the one book of RFC §12.4's ids.
        self._open_tool_calls: dict[str, dict[str, Any]] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'openai_realtime', 'gemini_live')."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier of the model behind this session, for logs and traces.

        Deliberately **not** abstract, unlike
        :attr:`~roomkit.providers.ai.base.AIProvider.model_name`: every
        conversational provider runs one named model, but a speech-to-speech
        service need not expose one — ElevenLabs binds an agent configured in
        its dashboard, PersonaPlex serves a single self-hosted model. Those
        keep this default, which returns :attr:`name`.

        So a caller must read the value as *"the best identifier this provider
        can give"*, not as a guaranteed model id: it may be a provider name.
        Compare it against :meth:`available_models` when the distinction
        matters.

        For a composed stack the question has several answers, and the
        override says which stage it names — Deepgram's is its *think* model,
        not its speech-to-text or its voice.
        """
        return self.name

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        """Whether ``reconfigure(...)`` can safely run mid-session.

        Some realtime models (notably the ``gemini-3.x`` Live family)
        reject ``send_client_content`` after the first model turn with
        a WebSocket 1007 close and offer no documented alternative for
        dynamic system_instruction updates. The base reconfigure also
        tears down the live WebSocket and reconnects via session
        resumption, which on those models is fragile when the system
        prompt is non-trivial and silently drops in-flight tool calls
        (their ``call_id`` is connection-scoped).

        Channels that orchestrate dynamic tool / skill exposure must
        check this flag before calling ``reconfigure`` and fall back to
        delivering the same information through a different surface
        (e.g. baking it into ``system_instruction`` at session start,
        or returning it through the tool result that triggered the
        change). Providers default to ``True`` for backwards
        compatibility; subclasses override to ``False`` when their
        upstream model cannot safely reconfigure.
        """
        return True

    @staticmethod
    def _session_task(coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task[Any]:
        """A task a session lives on (its receive loop, its keepalive), in a
        context of its own (RFC §12.4).

        A connection can open inside whatever called it, a tool handler for a
        handoff that reconfigures the session, and a task created there would
        carry that call's context (its voice session, its AI loop, the call it
        serves) into every event of the new connection.
        """
        return asyncio.create_task(coro, name=name, context=contextvars.Context())

    @property
    def supports_tools(self) -> bool:
        """Whether the model can call tools (RFC §12.4).

        ``False`` for a provider whose service never issues a function call:
        the channel then declares none to it and warns once, rather than leave
        a catalogue nobody can call.
        """
        return True

    @property
    def supports_context_preservation(self) -> bool:
        """Whether ``provider_config={"preserve_context": True}`` is supported.

        In this mode the provider must retain tool-delivered instructions
        verbatim for the session: no compression, eviction, or reconnection
        with uncertain history. It must end the session and emit an error
        before continuing without that context. Provider duration limits
        still apply; this does not promise an indefinitely long session.
        """
        return False

    @property
    def full_duplex(self) -> bool:
        """Whether the model listens and speaks at the same time (RFC §12.4.1).

        A full-duplex provider (OpenAI GPT-Live) handles being talked over by
        itself and exposes no response or speech boundaries on its wire: it
        synthesizes ``on_response_start``/``on_response_end`` from its own
        output, and its ``interrupt()`` and ``truncate_audio()`` are no-ops.
        The channel reads this flag to leave interruption to the model — no
        playback flush, no gating of provider audio on user speech, and a
        pipeline VAD kept to the observation role. Such a provider also holds
        no tools of its own: reasoning and tool use are delegated to a backend
        model, announced through :meth:`on_delegation`.
        """
        return False

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """Curated, offline catalog of voices this provider offers.

        No API key or network required — call it on the class to discover the
        ``voice`` ids that :meth:`connect` accepts. The base returns an empty
        list; each provider overrides it with its catalog.
        """
        return []

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of speech-to-speech models this provider runs.

        The realtime counterpart of :meth:`available_voices` — call it on the
        class to discover the ``model`` ids the constructor accepts.
        Deliberately *not* folded into
        :meth:`~roomkit.providers.ai.base.AIProvider.available_models` for the
        reason the image catalog gives (RFC §25.6): the sets are disjoint — no
        realtime id answers a chat completion, and no chat id opens a realtime
        session — so merging them would oblige every consumer of the
        conversational catalog to filter out models it can never use.

        The base returns an empty list, and two providers keep it on purpose:
        Deepgram composes its agent from stages that have catalogs of their own
        (``speak`` is the voice catalog, ``think`` reads the vendors' *chat*
        catalogs), and ElevenLabs binds a dashboard-configured agent, so
        neither has an end-to-end model id to list.
        """
        return []

    async def list_voices(self) -> list[VoiceInfo]:
        """Voices reported live by the provider's API.

        The base implementation returns the curated :meth:`available_voices`.
        Providers whose API exposes a voices endpoint (e.g. ElevenLabs) override
        this to query it, backfilling metadata from the catalog via
        :meth:`_merge_curated`. Fixed-voice providers (OpenAI Realtime, Gemini
        Live) keep the curated list.
        """
        return self.available_voices()

    @classmethod
    def _merge_curated(cls, live: list[VoiceInfo]) -> list[VoiceInfo]:
        """Backfill metadata absent from live results using the curated catalog.

        For each live voice that also appears in :meth:`available_voices`, fill
        any missing ``name``/``language``/``gender``/``description`` from the
        curated entry, keeping whatever the API reported.
        """
        curated = {v.id: v for v in cls.available_voices()}
        merged: list[VoiceInfo] = []
        for voice in live:
            match = curated.get(voice.id)
            if match is None:
                merged.append(voice)
                continue
            merged.append(
                voice.model_copy(
                    update={
                        "name": voice.name or match.name,
                        "language": voice.language or match.language,
                        "gender": voice.gender or match.gender,
                        "description": voice.description or match.description,
                    }
                )
            )
        return merged

    @abstractmethod
    async def connect(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        server_vad: bool = True,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        """Connect a session to the provider's AI service.

        Args:
            session: The realtime session to connect.
            system_prompt: System instructions for the AI.
            voice: Voice ID for audio output.
            tools: Tool/function definitions the AI can call.
            temperature: Sampling temperature.
            input_sample_rate: Sample rate of input audio (Hz).
            output_sample_rate: Desired sample rate for output audio (Hz).
            server_vad: Whether to use server-side voice activity detection.
            provider_config: Provider-specific configuration options.
                Each provider documents which keys it accepts.
        """
        ...

    @abstractmethod
    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        """Send audio data to the provider for processing.

        Args:
            session: The active session.
            audio: Raw PCM audio bytes.
        """
        ...

    @abstractmethod
    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> VoiceInjectionResult | None:
        """Inject text and report the provider's submission boundary.

        Return ``VoiceInjectionResult`` to distinguish a completed send from
        a guaranteed non-submission or uncertain acceptance. Returning ``None``
        is supported, but proactive delivery reports its outcome as unknown.

        The role is an intent, mapped by each provider onto what its wire
        offers (RFC §12.4): ``"system"`` is an instruction from the
        application — how to behave, or what to do now — and ``"user"`` is
        content for the conversation: a user turn the model answers on a
        turn-based provider, words the model says aloud on a full-duplex one.
        Text that directs the model, an opening greeting included, is always
        ``"system"``; sent as ``"user"`` it would be voiced by a full-duplex
        model instead of followed.

        Args:
            session: The active session.
            text: Text to inject.
            role: The intent — ``"system"`` (instruction) or ``"user"``
                (content). A provider may document further ones.
            silent: If True, add to conversation context without
                requesting a response.  The agent sees the text on
                its next turn but does not react immediately.
        """
        ...

    async def inject_image(
        self,
        session: VoiceSession,
        image_data: bytes,
        mime_type: str = "image/png",
        *,
        prompt: str = "",
        silent: bool = False,
    ) -> None:
        """Inject an image into the conversation for multimodal analysis.

        Not all providers support vision. The default implementation
        raises ``NotImplementedError``. Providers with multimodal input
        (e.g. Gemini Live) should override this.

        Args:
            session: The active session.
            image_data: Raw image bytes.
            mime_type: MIME type of the image.
            prompt: Optional text prompt accompanying the image.
            silent: If True, add to context without requesting a response.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support image injection")

    @abstractmethod
    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        """Submit a tool call result back to the provider.

        Args:
            session: The active session.
            call_id: The tool call ID from the on_tool_call callback.
            result: JSON-serialized result string.
        """
        ...

    async def submit_tool_error(self, session: VoiceSession, call_id: str, result: str) -> None:
        """Submit the result of a call that failed: refused, failed, blocked,
        served by nothing (RFC §12.4).

        A provider whose protocol marks a result as an error overrides this so
        the model does not read a refusal as a success. The default submits
        it as any result: the body says it failed.
        """
        await self.submit_tool_result(session, call_id, result)

    async def submit_delegation_output(
        self,
        session: VoiceSession,
        delegation_id: str,
        text: str,
        *,
        spoken: bool,
    ) -> None:
        """Return a reasoning backend's output to the model (RFC §12.4.1).

        Answers a delegation the provider announced through
        :meth:`on_delegation` with ``target="integrator"``. ``spoken=True``
        asks the model to relay the text to the user in its own words;
        ``spoken=False`` adds it as silent context the model may draw on.
        A provider that bounds one append splits the text rather than refuse
        it. Only full-duplex providers implement this; the default raises,
        since a provider without delegation has nothing to answer.

        Args:
            session: The active session.
            delegation_id: Identifier from the ``on_delegation`` callback,
                returned unchanged.
            text: The backend's output.
            spoken: Whether the model should voice it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support reasoning delegation")

    @abstractmethod
    async def interrupt(self, session: VoiceSession) -> None:
        """Interrupt the current AI response.

        Args:
            session: The active session.
        """
        ...

    async def truncate_audio(self, session: VoiceSession, audio_end_ms: int) -> None:  # noqa: B027
        """Synchronize provider context with the audio the user actually heard.

        Realtime providers that keep conversation state may generate audio
        faster than a transport can play it. When playback is interrupted,
        implementations can override this hook to remove the unheard tail
        from their server-side context. Providers that manage playback
        themselves, or whose protocol has no equivalent operation, keep the
        default no-op.

        Args:
            session: The active session.
            audio_end_ms: Played duration of the interrupted response in
                milliseconds, measured from physical playback onset.
        """

    @abstractmethod
    async def disconnect(self, session: VoiceSession) -> None:
        """Disconnect a session from the provider.

        Args:
            session: The session to disconnect.
        """
        ...

    async def reconfigure(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        """Reconfigure a session with new parameters.

        Used during agent handoff to switch the AI personality, voice,
        and tools.  The default implementation disconnects and reconnects;
        providers with session resumption (e.g. Gemini Live) should
        override to preserve conversation context.

        Args:
            session: The active session to reconfigure.
            system_prompt: New system instructions.
            voice: New voice ID.
            tools: New tool/function definitions.
            temperature: New sampling temperature.
            provider_config: Provider-specific configuration overrides.
        """
        await self.disconnect(session)
        # The participant's session did not end — only the upstream connection
        # did. Without this the reconnect below would be a transition out of
        # ENDED, which RFC §12.1 forbids.
        session.renegotiate()
        await self.connect(
            session,
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            provider_config=provider_config,
        )

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        """Send a raw provider-specific event to the underlying service.

        This is an escape hatch for sending protocol-level messages that
        are not covered by the standard provider API (e.g. OpenAI's
        ``session.update`` or ``input_audio_buffer.commit``).

        The default implementation raises :exc:`NotImplementedError`.
        Providers that support raw events should override this.

        Args:
            session: The active session.
            event: A JSON-serializable dict that will be sent verbatim
                to the provider's underlying connection.
        """
        raise NotImplementedError(f"{self.name} does not support send_event()")

    def is_responding(self, session_id: str) -> bool:
        """Check if the provider is actively generating a response.

        Returns ``True`` between ``response.created`` and ``response.done``.
        """
        return False

    async def close(self) -> None:
        """Release all provider resources."""

    # -- Usage recording --

    def _record_usage(
        self,
        session: VoiceSession,
        input_tokens: int,
        output_tokens: int,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Store token usage on session and record telemetry metrics.

        Called by provider implementations after parsing usage from
        their API response. Centralises session._last_usage storage
        and telemetry metric recording.

        Args:
            session: The active session.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens produced.
            details: Optional provider-specific detail dict merged
                into _last_usage (e.g. token breakdowns).
        """
        usage: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if details:
            usage.update(details)
        session._last_usage = usage
        self._publish_usage(session)

        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            attrs = {"session_id": session.id, Attr.MODEL: self.model_name}
            telemetry.record_metric(
                "roomkit.realtime.input_tokens",
                float(input_tokens),
                unit="tokens",
                attributes=attrs,
            )
            telemetry.record_metric(
                "roomkit.realtime.output_tokens",
                float(output_tokens),
                unit="tokens",
                attributes=attrs,
            )

    def _publish_usage(self, session: VoiceSession) -> None:
        """Hand the session's recorded usage to the ``on_usage`` callbacks.

        Called by every path that records usage, tokens or seconds, once the
        session holds the new reading. The callbacks run on a task of their
        own: recording happens inside a provider's event handler, which must
        not wait on an integrator's bookkeeping, and :meth:`_fire` already
        keeps one failing callback from reaching the others.

        Without a running loop — a provider driven from synchronous test code
        — there is nothing to schedule on, and the snapshot on the session is
        the only surface left.
        """
        if not self._usage_callbacks:
            return
        snapshot = dict(session._last_usage)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No running loop: on_usage callbacks skipped for session %s", session.id)
            return
        task = loop.create_task(
            self._fire(self._usage_callbacks, session, snapshot, label="usage"),
            name=f"realtime_usage:{session.id}",
        )
        self._usage_tasks.add(task)
        task.add_done_callback(self._usage_tasks.discard)
        task.add_done_callback(log_task_exception)

    # -- Callback registration --

    def on_audio(self, callback: RealtimeAudioCallback) -> None:
        """Register callback for audio output from the provider."""
        self._audio_callbacks.append(callback)

    def on_transcription(self, callback: RealtimeTranscriptionCallback) -> None:
        """Register callback for transcription events."""
        self._transcription_callbacks.append(callback)

    def on_speech_start(self, callback: RealtimeSpeechStartCallback) -> None:
        """Register callback for speech start detection."""
        self._speech_start_callbacks.append(callback)

    def on_speech_end(self, callback: RealtimeSpeechEndCallback) -> None:
        """Register callback for speech end detection."""
        self._speech_end_callbacks.append(callback)

    def on_tool_call(self, callback: RealtimeToolCallCallback) -> None:
        """Register callback for tool/function calls from the AI.

        Called as ``(session, call_id, name, arguments)``. ``arguments`` is the
        call's arguments as a mapping, or the text the model wrote when they
        do not read as one (invalid JSON, an array, a fragment): read the
        wire's arguments with
        :func:`~roomkit.providers.ai.tool_calls.readable_arguments`, or, on a
        wire that says whether the response cut the call, with
        :func:`~roomkit.providers.ai.tool_calls.realtime_call_arguments`, which
        marks a cut call :class:`~roomkit.providers.ai.tool_calls.CutArguments`.
        The channel refuses a call that arrives as text (one marked cut as cut
        off), so it never runs on a mapping that only passes for arguments
        (RFC §6.4, §12.4).
        """
        self._tool_call_callbacks.append(callback)

    def on_tool_call_cancelled(self, callback: RealtimeToolCallCancelledCallback) -> None:
        """Register callback for tool calls the model will not read (RFC §12.4).

        Called as ``(session, call_ids)`` for every call the provider
        abandons, whatever the cause: the model discarding it (Gemini Live's
        ``tool_call_cancellation`` when the user interrupts), a reconnect
        orphaning it (call ids are connection-scoped), the provider's own
        wait on the result timing out, the connection or the session ending.
        The channel cancels the handler still running for such a call and
        reports it to ON_TOOL_CALL's observers as cancelled.
        """
        self._tool_call_cancelled_callbacks.append(callback)

    def on_delegation(self, callback: RealtimeDelegationCallback) -> None:
        """Register callback for reasoning delegations (RFC §12.4.1).

        Called as ``(session, delegation_id, target)`` when a full-duplex
        model hands reasoning or tool use to a backend. ``target`` is
        ``"hosted"`` when the provider's service runs the backend — its
        function calls then arrive through :meth:`on_tool_call` — and
        ``"integrator"`` when the application must answer through
        :meth:`submit_delegation_output`. The request carries no task text:
        the backend works out what was asked from the conversation.
        """
        self._delegation_callbacks.append(callback)

    def on_response_start(self, callback: RealtimeResponseStartCallback) -> None:
        """Register callback for when the AI starts generating a response."""
        self._response_start_callbacks.append(callback)

    def on_response_end(self, callback: RealtimeResponseEndCallback) -> None:
        """Register callback for when the AI finishes a response."""
        self._response_end_callbacks.append(callback)

    def on_usage(self, callback: RealtimeUsageCallback) -> None:
        """Register callback for the usage this provider reports (RFC §12.4.2).

        Called as ``(session, usage)`` every time the provider records what a
        session consumed: the two token totals and whatever breakdown its API
        sends beside them, the cumulative duration of a provider billed by
        session seconds, a hosted backend's own tokens.

        This is the surface to bill a call from. :attr:`VoiceSession.last_usage`
        holds the same map, but the next report replaces it and the channel
        clears it at the end of each turn, so a reader that polls can miss a
        turn while a callback sees every one.
        """
        self._usage_callbacks.append(callback)

    def on_error(self, callback: RealtimeErrorCallback) -> None:
        """Register callback for provider errors."""
        self._error_callbacks.append(callback)

    # -- Callback dispatch --

    async def _abandon_tool_calls(self, session: VoiceSession, call_ids: Iterable[str]) -> None:
        """Report calls whose results the model will not read (RFC §12.4).

        Every call a provider abandons is reported here, whatever the cause;
        the call ids come from the provider's own book of open calls.
        """
        abandoned = list(call_ids)
        if abandoned:
            await self._fire(
                self._tool_call_cancelled_callbacks,
                session,
                abandoned,
                label="tool_call_cancelled",
            )

    # -- The book of open calls (RFC §12.4) --

    def _book_tool_call(self, session: VoiceSession, call_id: str, payload: Any = None) -> bool:
        """Book *call_id* as issued on *session* and owing a result, with
        *payload*, what answering it takes. ``False``, nothing booked, for a
        call without an id or under an id still in flight: the channel
        refuses both and nothing is sent for them."""
        calls = self._open_tool_calls.setdefault(session.id, {})
        if not call_id or call_id in calls:
            return False
        calls[call_id] = payload
        return True

    def _holds_tool_call(self, session: VoiceSession, call_id: str) -> bool:
        """Whether *call_id* is booked on *session*, its result still owed."""
        return call_id in self._open_tool_calls.get(session.id, {})

    def _open_tool_call_count(self, session: VoiceSession) -> int:
        """How many of *session*'s calls still owe a result."""
        return len(self._open_tool_calls.get(session.id, {}))

    def _has_open_tool_calls(self, session: VoiceSession) -> bool:
        """Whether *session* has a call whose result is still owed."""
        return self._open_tool_call_count(session) > 0

    def _forget_tool_calls(self, session_id: str) -> None:
        """Drop a session's book, its calls already abandoned and reported."""
        self._open_tool_calls.pop(session_id, None)

    def _answerable_tool_call(self, session: VoiceSession, call_id: str) -> tuple[bool, Any]:
        """Take *call_id* off the book to send its result: ``(True, payload)``
        while it is booked, before the send yields, so a call issued under the
        id meanwhile is a new call; ``(False, None)``, logged, for one the
        provider abandoned or never issued: nothing goes out for it, on every
        provider (RFC §12.4)."""
        calls = self._open_tool_calls.get(session.id, {})
        if call_id not in calls:
            self._log_dropped_result(session, call_id)
            return False, None
        return True, calls.pop(call_id)

    def _log_dropped_result(self, session: VoiceSession, call_id: str) -> None:
        """Say that a result for *call_id* goes out nowhere: the call was
        abandoned, never issued, or belongs to a connection that is gone."""
        logger.info(
            "[%s] result for tool call %r dropped: abandoned or never issued (session %s)",
            self.name,
            call_id,
            session.id,
        )

    def _drop_tool_call(self, session: VoiceSession, call_id: str, payload: Any) -> None:
        """Take *call_id* off the book while it is still booked with
        *payload*: once its result went out, the id may name a newer call."""
        calls = self._open_tool_calls.get(session.id)
        if calls is not None and call_id in calls and calls[call_id] is payload:
            del calls[call_id]

    def _take_tool_calls(
        self,
        session: VoiceSession,
        call_ids: Iterable[str] | None = None,
        *,
        where: Callable[[Any], bool] | None = None,
    ) -> dict[str, Any]:
        """Take off the book *session*'s calls among *call_ids* (all of them
        when ``None``) whose payload passes *where*, each with its payload."""
        calls = self._open_tool_calls.get(session.id, {})
        # Each id once: a server may name a call twice in one cancellation.
        wanted = list(calls) if call_ids is None else list(dict.fromkeys(call_ids))
        wanted = [cid for cid in wanted if cid in calls]
        taken = {cid: calls.pop(cid) for cid in wanted if where is None or where(calls[cid])}
        if not calls:
            self._open_tool_calls.pop(session.id, None)
        return taken

    async def _abandon_open_tool_calls(
        self,
        session: VoiceSession,
        call_ids: Iterable[str] | None = None,
        *,
        where: Callable[[Any], bool] | None = None,
    ) -> None:
        """Abandon the booked calls among *call_ids* (all of *session*'s
        when ``None``, or those whose payload passes *where*) and report
        them, each once: an id freed already is not reported again."""
        taken = self._take_tool_calls(session, call_ids, where=where)
        await self._abandon_tool_calls(session, list(taken))

    async def _fire(
        self,
        callbacks: list[Any],
        *args: Any,
        label: str = "callback",
    ) -> None:
        """Fire all registered callbacks with the given arguments.

        Supports both sync and async callbacks. Exceptions are logged
        but never propagate — one failing callback must not break the
        provider's event loop.
        """
        for cb in callbacks:
            try:
                result = cb(*args)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in %s callback", label)

    async def start_audio_stream(self, session: VoiceSession) -> None:  # noqa: B027
        """Open the audio input path on the provider.

        Some providers need to be nudged into "audio is flowing" mode before
        other operations (e.g. text injection) are safe — for instance, when
        the application wants to trigger a greeting before the remote side
        has spoken.  Providers that care override this to send an initial
        silent audio frame (or flip their protocol state); providers that
        don't inherit the no-op.

        Safe to call unconditionally.  No-op if the session is not active or
        the stream has already been opened.
        """

    # -- Manual VAD activity signals --

    async def send_activity_start(self, session: VoiceSession) -> None:  # noqa: B027
        """Signal that user speech activity has started.

        Used in manual VAD mode: local VAD detects speech and the channel
        calls this to inform the provider.  The provider translates this
        into its protocol's activity signal (e.g. Gemini ``ActivityStart``).

        Default: no-op.  Override when the provider supports manual mode.
        """

    async def send_activity_end(self, session: VoiceSession) -> None:  # noqa: B027
        """Signal that user speech activity has ended.

        Used in manual VAD mode: local VAD detects silence and the channel
        calls this to inform the provider.  The provider translates this
        into its protocol's activity signal (e.g. Gemini ``ActivityEnd``).

        Default: no-op.  Override when the provider supports manual mode.
        """


# Video callback: (session, VideoFrame) — used by RealtimeAudioVideoProvider
RealtimeVideoCallback = Callable[[VoiceSession, "VideoFrame"], Any]
"""(session, video_frame)"""


class RealtimeAudioVideoProvider(RealtimeVoiceProvider):
    """Realtime provider that produces both audio and video output.

    Extends :class:`RealtimeVoiceProvider` with an ``on_video`` callback
    for providers (e.g. Anam AI) that deliver synchronized audio+video
    from a cloud avatar pipeline.

    Subclasses implement the same abstract methods as
    :class:`RealtimeVoiceProvider`; the only addition is the video
    callback registration.
    """

    def on_video(self, callback: RealtimeVideoCallback) -> None:
        """Register callback for video frames from the provider.

        Args:
            callback: Called with (session, video_frame) when the provider
                produces a video frame.
        """
