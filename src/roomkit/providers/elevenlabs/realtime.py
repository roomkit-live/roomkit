"""ElevenLabs Conversational AI realtime provider.

Uses the official ElevenLabs Python SDK ``AsyncConversation`` class with
a custom ``AsyncAudioInterface`` that bridges audio between the SDK and
RoomKit's callback system.

Tools take a different route than on every other realtime provider. The SDK
runs **client tools** through a :class:`~elevenlabs.conversational_ai.conversation.ClientTools`
registry keyed by tool name instead of forwarding JSON schemas over the wire,
so the schemas RoomKit passes to :meth:`ElevenLabsRealtimeProvider.connect`
only register handlers here: the matching tools must also exist on the agent
itself (dashboard or Agents API) as **client** tools, under the same names.
Anything the agent calls that was not declared to ``connect`` comes back to it
as an error, and anything declared here that the agent does not know about is
never called.

Requires the ``elevenlabs`` package (v2.40+)::

    pip install 'roomkit[realtime-elevenlabs]'
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from functools import partial
from typing import Any, NoReturn, cast

from roomkit.providers.elevenlabs import sdk_patch
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.voices import VOICES as _VOICES
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.injection import VoiceInjectionResult, say_line_instruction
from roomkit.voice.realtime.provider import RealtimeVoiceProvider, VoiceInfo

logger = logging.getLogger("roomkit.providers.elevenlabs.realtime")

_CONNECT_TIMEOUT = 30.0
# The SDK consumes 16 kHz, mono, 16-bit PCM. This covers the entire connection
# timeout at realtime pace while keeping a stalled/malicious transport bounded.
_PENDING_AUDIO_LIMIT = int(16_000 * 2 * _CONNECT_TIMEOUT)


class _ClientToolError(Exception):
    """A client tool's failed result: the SDK sends its text as an error."""


class ElevenLabsRealtimeProvider(RealtimeVoiceProvider):
    """Realtime voice provider using the ElevenLabs Conversational AI SDK.

    Uses the SDK's ``AsyncConversation`` with a custom ``AsyncAudioInterface``
    that bridges audio between the SDK and RoomKit's async callback system.

    Example::

        from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
        from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider

        config = ElevenLabsRealtimeConfig(api_key="xi-...", agent_id="agent_abc123")
        provider = ElevenLabsRealtimeProvider(config)
        provider.on_audio(handle_audio)

        await provider.connect(session, system_prompt="You are helpful.")
        await provider.send_audio(session, audio_bytes)
    """

    def __init__(self, config: ElevenLabsRealtimeConfig) -> None:
        super().__init__()
        self._config = config

        # Per-session state
        self._sessions: dict[str, VoiceSession] = {}
        self._conversations: dict[str, Any] = {}  # AsyncConversation objects
        self._input_callbacks: dict[str, Any] = {}  # async audio input callbacks
        self._client_tools: dict[str, Any] = {}  # ClientTools objects
        self._supervisors: dict[str, asyncio.Task[None]] = {}
        self._readiness: dict[str, asyncio.Future[None]] = {}
        self._pending_audio: dict[str, bytearray] = {}
        self._audio_locks: dict[str, asyncio.Lock] = {}
        self._closing: set[str] = set()

        # Track active responses
        self._responding: set[str] = set()
        self._last_audio_at: dict[str, float] = {}
        self._response_watchdogs: dict[str, asyncio.Task[None]] = {}

    def is_responding(self, session_id: str) -> bool:
        return session_id in self._responding

    @property
    def name(self) -> str:
        return "ElevenLabsRealtimeProvider"

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        """ElevenLabs conversations cannot be reconfigured in place.

        The ConvAI protocol takes its overrides once, in the initiation
        message; there is no in-band equivalent of ``session.update``. The
        base ``reconfigure`` would therefore disconnect and reconnect, and
        on this provider that ends the conversation server-side and starts a
        different one: the transcript, the agent's memory of the turn and
        every pending ``tool_call_id`` go with it. Callers that add tools or
        skills mid-session must deliver them another way (see the channel's
        ``skill_delivery_mode``).
        """
        return False

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """Curated, offline catalog of ElevenLabs default voices."""
        return list(_VOICES)

    async def list_voices(self) -> list[VoiceInfo]:
        """List voices the account exposes via the ElevenLabs voices API."""
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=self._config.api_key.get_secret_value())
        resp = await asyncio.to_thread(client.voices.get_all)
        live = [
            VoiceInfo(
                id=v.voice_id,
                name=(v.name or "").split(" - ")[0] or None,
                gender=(v.labels or {}).get("gender"),
                language=(v.labels or {}).get("language"),
                description=v.description or None,
            )
            for v in (resp.voices or [])
            if getattr(v, "voice_id", None)
        ]
        return self._merge_curated(live)

    # -- Connection lifecycle --

    async def connect(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 16000,
        server_vad: bool = True,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        if input_sample_rate != 16_000 or output_sample_rate != 16_000:
            raise ValueError(
                "ElevenLabs Conversational AI requires 16000 Hz input and output; "
                "configure RealtimeVoiceChannel with input_sample_rate=16000 and "
                "output_sample_rate=16000"
            )
        try:
            from elevenlabs import ElevenLabs
            from elevenlabs.conversational_ai.conversation import (
                AsyncConversation,
                ClientTools,
                ConversationInitiationData,
            )
        except ImportError as exc:
            raise ImportError(
                "elevenlabs>=2.40 is required for ElevenLabsRealtimeProvider. "
                "Install with: pip install 'roomkit[realtime-elevenlabs]'"
            ) from exc

        pc = provider_config or {}

        # Build config overrides
        config_override: dict[str, Any] = {}
        agent_override: dict[str, Any] = {}
        if system_prompt:
            agent_override["prompt"] = {"prompt": system_prompt}
        if pc.get("language"):
            agent_override["language"] = pc["language"]
        if pc.get("first_message") is not None:
            agent_override["first_message"] = pc["first_message"]
        if agent_override:
            config_override["agent"] = agent_override

        tts_override: dict[str, Any] = {}
        if voice:
            tts_override["voice_id"] = voice
        if pc.get("speed") is not None:
            # ElevenLabs accepts 0.7-1.2; the agent must whitelist the speed
            # override in its security settings or the session is rejected.
            tts_override["speed"] = min(1.2, max(0.7, float(pc["speed"])))
        if tts_override:
            config_override["tts"] = tts_override

        extra_body: dict[str, Any] = {}
        if temperature is not None:
            extra_body["temperature"] = temperature

        init_config = ConversationInitiationData(
            extra_body=extra_body or None,
            conversation_config_override=config_override or None,
            dynamic_variables=pc.get("dynamic_variables"),
        )

        # Create async bridge AudioInterface
        bridge = _AsyncBridgeAudioInterface(self, session)

        # One ClientTools per session, bound to the loop RoomKit runs on.
        # Both halves of that sentence are load-bearing. Left to itself the
        # SDK spins up its own event loop in a private thread, and the
        # callback that ships a tool result does ``asyncio.create_task`` on
        # whatever loop is current — which would be that thread's, while the
        # WebSocket belongs to ours. And the instance is not reusable: the
        # SDK's ``end_session`` stops it, after which any further tool call
        # raises. Registration is also refused for a name already present,
        # so a shared instance would break on the second connect.
        # A name with no handler goes to the channel too, whose gate refuses
        # and reports it (RFC §12.4), rather than being answered by the SDK.
        client_tools = sdk_patch.client_tools(
            ClientTools,
            loop=asyncio.get_running_loop(),
            route=partial(self._route_unregistered, session),
        )
        self._register_client_tools(client_tools, session, tools)

        # Create SDK client (pass base_url for regional endpoints)
        base_url = self._config.base_url.replace("wss://", "https://").replace("ws://", "http://")
        client = ElevenLabs(api_key=self._config.api_key.get_secret_value(), base_url=base_url)

        conversation = sdk_patch.conversation(AsyncConversation)(
            client,
            self._config.agent_id,
            requires_auth=self._config.requires_auth,
            # The SDK types this parameter nominally; the bridge implements
            # the full AsyncAudioInterface contract structurally (see its
            # docstring for why it cannot subclass the optional SDK's ABC).
            # Handed over as Any rather than under a `ty: ignore`: the
            # nominal mismatch only exists where the optional SDK is
            # installed, so a suppression is unused — and rejected — in any
            # environment without it.
            audio_interface=cast(Any, bridge),
            config=init_config,
            client_tools=client_tools,
            callback_agent_response=self._make_agent_response_cb(session),
            callback_agent_response_correction=self._make_correction_cb(session),
            callback_user_transcript=self._make_user_transcript_cb(session),
            callback_latency_measurement=self._make_latency_cb(session),
            callback_end_session=self._make_end_session_cb(session),
        )

        await self._abandon_previous_connection(session)
        self._sessions[session.id] = session
        self._conversations[session.id] = conversation
        self._client_tools[session.id] = client_tools
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._readiness[session.id] = ready
        self._pending_audio[session.id] = bytearray()
        self._audio_locks[session.id] = asyncio.Lock()
        self._closing.discard(session.id)

        # The SDK creates the session's receive task inside start_session:
        # started from a fresh context, it inherits nothing from the caller,
        # which may be a tool handler reconnecting the session (RFC §12.4).
        try:
            await self._session_task(
                conversation.start_session(), name=f"elevenlabs_start:{session.id}"
            )
        except Exception:
            self._forget_session(session.id)
            raise

        # ``start_session`` only spawns the task that opens the WebSocket, so
        # a rejected key, an unknown agent id or a dead network surfaces
        # inside that task rather than here. Without a supervisor the session
        # would sit in ACTIVE forever, silent, with nothing raised anywhere.
        # An unusually fast SDK failure can run callback_end_session while
        # start_session() is yielding.  In that case _fail_session() has
        # already removed this session and completed ``ready`` with the real
        # error; installing a supervisor now would resurrect orphan state.
        if self._sessions.get(session.id) is session:
            self._supervisors[session.id] = self._session_task(
                self._supervise_session(session, conversation),
                name=f"elevenlabs_session:{session.id}",
            )

        # The SDK task calls ``audio_interface.start`` only after its WebSocket
        # is open and the initiation message has been sent.  Until then there is
        # no input callback and accepting audio would silently discard it.
        try:
            await asyncio.wait_for(asyncio.shield(ready), timeout=_CONNECT_TIMEOUT)
            if self._sessions.get(session.id) is not session:
                raise RuntimeError(f"ElevenLabs session {session.id} ended while connecting")
        except TimeoutError as exc:
            if session.id in self._sessions:
                with contextlib.suppress(Exception):
                    await self.disconnect(session)
            raise TimeoutError(
                f"ElevenLabs session {session.id} was not ready within {_CONNECT_TIMEOUT:g}s"
            ) from exc
        except BaseException:
            if session.id in self._sessions:
                with contextlib.suppress(Exception):
                    await self.disconnect(session)
            raise
        finally:
            if self._readiness.get(session.id) is ready:
                self._readiness.pop(session.id, None)

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        logger.info("ElevenLabs Realtime session connected: %s", session.id)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        lock = self._audio_locks.get(session.id)
        if lock is None:
            # Backwards-compatible path for an integrator that installed the
            # callback directly rather than through connect()/the SDK bridge.
            cb = self._input_callbacks.get(session.id)
            if cb is not None:
                await cb(audio)
            return
        async with lock:
            cb = self._input_callbacks.get(session.id)
            if cb is not None:
                await cb(audio)
                return

            ready = self._readiness.get(session.id)
            if ready is None or ready.done():
                return
            pending = self._pending_audio.setdefault(session.id, bytearray())
            if len(pending) + len(audio) > _PENDING_AUDIO_LIMIT:
                error = RuntimeError(
                    f"ElevenLabs pre-connect audio exceeded {_PENDING_AUDIO_LIMIT} bytes"
                )
                ready.set_exception(error)
                raise error
            pending.extend(audio)

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> VoiceInjectionResult:
        """Send a user message, or a contextual update when ``silent``.

        The agent answers a user message, so ``system`` and ``user`` both
        travel as one. No client event makes the agent speak a given text, so
        an ``assistant`` line is sent as an instruction to say it (RFC §12.4).
        """
        conversation = self._conversations.get(session.id)
        if conversation is None:
            return VoiceInjectionResult(
                status="not_sent", reason="voice_not_connected", retryable=True
            )
        if role == "assistant" and not silent:
            text = say_line_instruction(text)
        if silent:
            logger.debug("[ElevenLabs →] contextual_update (silent inject)")
            await conversation.send_contextual_update(text)
        else:
            logger.debug("[ElevenLabs →] user_message")
            await conversation.send_user_message(text)
        return VoiceInjectionResult(status="sent")

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        """Complete the SDK handler waiting on ``call_id``.

        The SDK sends the value the registered handler returns, so a result
        reaches the agent by resolving the future that handler is awaiting.
        """
        future = self._take_pending_tool(session, call_id)
        if future is not None and not future.done():
            future.set_result(result)

    async def submit_tool_error(self, session: VoiceSession, call_id: str, result: str) -> None:
        """Complete the SDK handler waiting on ``call_id`` with an error.

        The SDK marks a client tool's result as an error only when its
        handler raises, sending the exception's text: the failed call's
        result travels as that text, ``is_error`` set (RFC §12.4).
        """
        future = self._take_pending_tool(session, call_id)
        if future is not None and not future.done():
            future.set_exception(_ClientToolError(result))

    def _take_pending_tool(
        self, session: VoiceSession, call_id: str
    ) -> asyncio.Future[str] | None:
        """The future the SDK handler for ``call_id`` awaits, off the book;
        ``None`` for a call that timed out, whose session ended, or that was
        never issued (RFC §12.4)."""
        held, future = self._answerable_tool_call(session, call_id)
        return future if held else None

    async def interrupt(self, session: VoiceSession) -> None:
        # ElevenLabs decides interruption server-side from its own VAD; the
        # protocol has no "stop talking" client event. ``user_activity`` is
        # the one lever it offers — it tells the agent the user is active,
        # which holds off its next turn.
        conversation = self._conversations.get(session.id)
        if conversation is not None:
            await conversation.register_user_activity()

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        raise NotImplementedError(
            "ElevenLabsRealtimeProvider uses the SDK; raw events are not supported"
        )

    async def disconnect(self, session: VoiceSession) -> None:
        # Mark first: the teardown below trips the SDK's end-session callback
        # and completes the supervisor, and neither is an error when we are
        # the ones closing.
        self._closing.add(session.id)

        await self._end_response(session)
        # A handoff reconnects through here too (the base reconfigure): the
        # new conversation never issued these calls (RFC §12.4).
        abandoned = self._reject_pending_tools(session, "the voice session ended")
        await self._abandon_tool_calls(session, abandoned)

        conversation = self._conversations.pop(session.id, None)
        try:
            if conversation is not None:
                await conversation.end_session()
                with contextlib.suppress(asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(conversation.wait_for_session_end(), timeout=5.0)
        finally:
            self._forget_session(session.id)
            session.state = VoiceSessionState.ENDED
        logger.info("ElevenLabs session disconnected: %s", session.id)

    async def close(self) -> None:
        errors: list[BaseException] = []
        for session_id in list(self._sessions):
            session = self._sessions.get(session_id)
            if session:
                try:
                    await self.disconnect(session)
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise BaseExceptionGroup("Failed to close ElevenLabs sessions", errors)

    # -- Async callback factories for SDK --

    def _make_agent_response_cb(self, session: VoiceSession) -> Any:
        """Agent text for the turn — which arrives *before* its audio.

        ConvAI produces the LLM text first and streams the synthesis after
        it, so this is the start of a turn, not its end. Ending the response
        here left the turn open for good: ``response_end`` fired before the
        first chunk, then the chunk reopened the response and nothing ever
        closed it, so the speaking indicator stayed lit and the session
        never went idle. The end of the turn is inferred from the audio
        going quiet (see :meth:`_watch_response_end`).
        """

        async def cb(text: str) -> None:
            await self._fire(
                self._transcription_callbacks,
                session,
                text,
                "assistant",
                True,
                label="transcription",
            )

        return cb

    def _make_correction_cb(self, session: VoiceSession) -> Any:
        async def cb(original: str, corrected: str) -> None:
            await self._fire(
                self._transcription_callbacks,
                session,
                corrected,
                "assistant",
                True,
                label="transcription",
            )

        return cb

    def _make_user_transcript_cb(self, session: VoiceSession) -> Any:
        async def cb(text: str) -> None:
            await self._fire(
                self._transcription_callbacks,
                session,
                text,
                "user",
                True,
                label="transcription",
            )
            # Transcript arrival signals the user finished speaking
            await self._fire(
                self._speech_end_callbacks,
                session,
                label="speech_end",
            )

        return cb

    def _make_latency_cb(self, session: VoiceSession) -> Any:
        async def cb(latency: int) -> None:
            logger.debug("ElevenLabs latency: %dms (session %s)", latency, session.id)

        return cb

    def _make_end_session_cb(self, session: VoiceSession) -> Any:
        """The SDK ended the conversation — surface it unless we asked for it."""

        async def cb() -> None:
            if session.id in self._closing:
                return
            await self._fail_session(
                session,
                "session_ended",
                "The ElevenLabs conversation was closed by the service",
            )

        return cb

    # -- Client tools --

    def _register_client_tools(
        self,
        client_tools: Any,
        session: VoiceSession,
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """Register one SDK handler per declared tool name."""
        registered: set[str] = set()
        for tool in tools or []:
            name = tool.get("name") if isinstance(tool, dict) else None
            if not name:
                logger.warning("ElevenLabs: skipping tool definition without a name: %r", tool)
                continue
            if name in registered:
                continue
            client_tools.register(name, self._make_tool_handler(session, name), is_async=True)
            registered.add(name)

        if registered:
            logger.info(
                "ElevenLabs session %s: registered %d client tool handler(s): %s. "
                "The agent must declare the same names as client tools.",
                session.id,
                len(registered),
                ", ".join(sorted(registered)),
            )

    async def _route_unregistered(
        self, session: VoiceSession, name: str, parameters: dict[str, Any]
    ) -> str:
        """A call to a name RoomKit did not declare, or to none, bridged as a
        declared one: the channel refuses and reports it."""
        return await self._make_tool_handler(session, name or "")(parameters)

    def _make_tool_handler(self, session: VoiceSession, name: str) -> Any:
        """Build the SDK handler that hands a call to RoomKit and waits.

        Whatever the handler returns is what the SDK sends back as the tool
        result, so the call is bridged by parking on a future that
        :meth:`submit_tool_result` completes once the channel has run its
        handler, hooks and gates.
        """

        async def handler(parameters: dict[str, Any]) -> str:
            # The service's id, and the model's arguments whole: a
            # ``tool_call_id`` the model wrote is one of them (sdk_patch).
            call_id, arguments = sdk_patch.split_call(parameters)

            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            if not self._book_tool_call(session, call_id, future):
                await self._hand_on_unanswerable(session, call_id, name, arguments)

            await self._fire(
                self._tool_call_callbacks,
                session,
                call_id,
                name,
                arguments,
                label="tool_call",
            )

            try:
                # ``None``: the channel bounds the call and answers it either way.
                return await asyncio.wait_for(future, timeout=self._config.tool_timeout_s)
            except TimeoutError:
                # The agent reads an error now and never the result: the call
                # is abandoned, its id freed before the channel is told, as
                # every provider frees it (RFC §12.4).
                await self._abandon_open_tool_calls(
                    session, [call_id], where=lambda booked: booked is future
                )
                # Raising is how the SDK is told this is an error result;
                # returning a string would read as a successful call.
                raise RuntimeError(
                    f"Tool '{name}' did not return within {self._config.tool_timeout_s:g}s"
                ) from None
            finally:
                # Once its result went out, the id may already name a newer
                # call, whose future this must not take.
                self._drop_tool_call(session, call_id, future)

        return handler

    async def _hand_on_unanswerable(
        self, session: VoiceSession, call_id: str, name: str, arguments: dict[str, Any] | str
    ) -> NoReturn:
        """Hand on a call no result can be sent for (no id, or an id still in
        flight): the channel refuses and reports it, and nothing goes out, the
        id's one result being the first call's (RFC §12.4). The SDK answers
        every outcome but a cancellation on the wire (``sdk_patch``)."""
        await self._fire(
            self._tool_call_callbacks, session, call_id, name, arguments, label="tool_call"
        )
        raise asyncio.CancelledError

    async def _abandon_previous_connection(self, session: VoiceSession) -> None:
        """The base step, each pending call's SDK handler failed first: the
        replaced conversation's handler is not left waiting for a result
        that will never come."""
        abandoned = self._reject_pending_tools(session, "the voice session connected again")
        await self._abandon_tool_calls(session, abandoned)

    def _reject_pending_tools(self, session: VoiceSession, reason: str) -> list[str]:
        """Fail every in-flight call so no SDK handler is left hanging; the
        ids of the calls it abandoned."""
        pending = self._take_tool_calls(session)
        for call_id, future in pending.items():
            if not future.done():
                future.set_exception(RuntimeError(f"Tool call {call_id} abandoned: {reason}"))
        return list(pending)

    # -- Response lifecycle --

    async def _start_response(self, session: VoiceSession) -> None:
        """Open a response on the first audio chunk of a turn."""
        if session.id in self._responding:
            return
        self._responding.add(session.id)
        await self._fire(self._response_start_callbacks, session, label="response_start")
        self._response_watchdogs[session.id] = asyncio.create_task(
            self._watch_response_end(session),
            name=f"elevenlabs_response:{session.id}",
        )

    async def _end_response(self, session: VoiceSession) -> None:
        """Close an open response, cancelling its watchdog."""
        watchdog = self._response_watchdogs.pop(session.id, None)
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
        if session.id not in self._responding:
            return
        self._responding.discard(session.id)
        self._last_audio_at.pop(session.id, None)
        await self._fire(self._response_end_callbacks, session, label="response_end")

    async def _watch_response_end(self, session: VoiceSession) -> None:
        """Declare the turn over once the audio stream has gone quiet.

        ConvAI has no end-of-audio marker: ``agent_response`` lands before
        the synthesis and ``agent_response_complete`` is opt-in per agent
        and not surfaced by the SDK's callbacks. Silence on the stream is
        what is left. A tool call in flight suspends the count — the agent
        resumes speaking on the same turn once it has the result.
        """
        idle_s = self._config.response_idle_ms / 1000
        while session.id in self._responding:
            await asyncio.sleep(idle_s / 2)
            if self._has_open_tool_calls(session):
                continue
            last = self._last_audio_at.get(session.id)
            if last is None or (time.monotonic() - last) >= idle_s:
                await self._end_response(session)
                return

    # -- Session supervision --

    async def _supervise_session(self, session: VoiceSession, conversation: Any) -> None:
        """Turn a session that dies on its own into an error callback."""
        try:
            await conversation.wait_for_session_end()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if session.id in self._closing:
                return
            await self._fail_session(session, "connection_failed", str(exc))
        else:
            if session.id not in self._closing and session.id in self._sessions:
                await self._fail_session(
                    session,
                    "session_ended",
                    "The ElevenLabs conversation ended before RoomKit disconnected it",
                )

    async def _fail_session(self, session: VoiceSession, code: str, message: str) -> None:
        """Report a session lost from under us, exactly once."""
        if session.id not in self._sessions:
            return

        logger.error("ElevenLabs session %s failed (%s): %s", session.id, code, message)
        ready = self._readiness.get(session.id)
        if ready is not None and not ready.done():
            ready.set_exception(RuntimeError(message))
        await self._end_response(session)
        abandoned = self._reject_pending_tools(session, message)
        self._forget_session(session.id)
        await self._abandon_tool_calls(session, abandoned)

        session.state = VoiceSessionState.ENDED
        await self._fire(self._error_callbacks, session, code, message, label="error")

    def _forget_session(self, session_id: str) -> None:
        """Drop every per-session structure, cancelling the supervisor."""
        supervisor = self._supervisors.pop(session_id, None)
        if supervisor is not None and supervisor is not asyncio.current_task():
            supervisor.cancel()
        watchdog = self._response_watchdogs.pop(session_id, None)
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()

        self._closing.discard(session_id)
        self._sessions.pop(session_id, None)
        self._conversations.pop(session_id, None)
        self._input_callbacks.pop(session_id, None)
        self._client_tools.pop(session_id, None)
        ready = self._readiness.pop(session_id, None)
        if ready is not None and not ready.done():
            ready.cancel()
        self._pending_audio.pop(session_id, None)
        self._audio_locks.pop(session_id, None)
        self._forget_tool_calls(session_id)
        self._last_audio_at.pop(session_id, None)
        self._responding.discard(session_id)


class _AsyncBridgeAudioInterface:
    """Bridges the ElevenLabs SDK's AsyncAudioInterface to RoomKit callbacks.

    Implements the SDK's ``AsyncAudioInterface`` contract structurally —
    ``start``, ``stop``, ``output``, ``interrupt``, all async — without
    subclassing it: the SDK is an optional dependency this module must stay
    importable without, and importing it at module scope trips the
    deprecated-websockets warning its own import raises. Runs in the same
    event loop as the rest of RoomKit — no thread bridging.
    """

    def __init__(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        self._provider = provider
        self._session = session

    async def start(self, input_callback: Any) -> None:
        """Install the SDK callback and flush audio captured during its handshake."""
        session_id = self._session.id
        lock = self._provider._audio_locks.get(session_id)
        if lock is None:
            return
        async with lock:
            ready = self._provider._readiness.get(session_id)
            if ready is None or ready.done():
                return
            self._provider._input_callbacks[session_id] = input_callback
            pending = self._provider._pending_audio.pop(session_id, None)
            if pending:
                await input_callback(bytes(pending))
            ready.set_result(None)
        logger.debug("ElevenLabs audio bridge started (session %s)", self._session.id)

    async def stop(self) -> None:
        """Clean up when SDK conversation ends."""
        self._provider._input_callbacks.pop(self._session.id, None)
        logger.debug("ElevenLabs audio bridge stopped (session %s)", self._session.id)

    async def output(self, audio: bytes) -> None:
        """Called by SDK with agent audio — forward to RoomKit callbacks."""
        session = self._session
        provider = self._provider

        # The SDK can have an already-queued output callback when a remote
        # close or RoomKit disconnect tears the session down.  Do not reopen
        # response state or emit audio for an ended session.
        if provider._sessions.get(session.id) is not session or session.id in provider._closing:
            return

        provider._last_audio_at[session.id] = time.monotonic()
        await provider._start_response(session)

        await provider._fire(provider._audio_callbacks, session, audio, label="audio")

    async def interrupt(self) -> None:
        """Called by SDK when user interrupted agent playback.

        A cut turn is still a finished turn: closing the response here is
        what clears the speaking indicator and lets the session go idle.
        """
        await self._provider._end_response(self._session)
