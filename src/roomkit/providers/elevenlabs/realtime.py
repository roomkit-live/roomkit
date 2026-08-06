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
import uuid
from typing import Any, cast

from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.voices import VOICES as _VOICES
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider, VoiceInfo

logger = logging.getLogger("roomkit.providers.elevenlabs.realtime")


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
        self._pending_tools: dict[str, dict[str, asyncio.Future[str]]] = {}
        self._supervisors: dict[str, asyncio.Task[None]] = {}
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
        client_tools = ClientTools(loop=asyncio.get_running_loop())
        self._register_client_tools(client_tools, session, tools)

        # Create SDK client (pass base_url for regional endpoints)
        base_url = self._config.base_url.replace("wss://", "https://").replace("ws://", "http://")
        client = ElevenLabs(api_key=self._config.api_key.get_secret_value(), base_url=base_url)

        conversation = AsyncConversation(
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

        self._sessions[session.id] = session
        self._conversations[session.id] = conversation
        self._client_tools[session.id] = client_tools
        self._closing.discard(session.id)

        # Start the SDK session (creates async task internally)
        try:
            await conversation.start_session()
        except Exception:
            self._forget_session(session.id)
            raise

        # ``start_session`` only spawns the task that opens the WebSocket, so
        # a rejected key, an unknown agent id or a dead network surfaces
        # inside that task rather than here. Without a supervisor the session
        # would sit in ACTIVE forever, silent, with nothing raised anywhere.
        self._supervisors[session.id] = asyncio.create_task(
            self._supervise_session(session, conversation),
            name=f"elevenlabs_session:{session.id}",
        )

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        logger.info("ElevenLabs Realtime session connected: %s", session.id)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        cb = self._input_callbacks.get(session.id)
        if cb is None:
            return
        await cb(audio)

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> None:
        conversation = self._conversations.get(session.id)
        if conversation is None:
            return
        if silent:
            logger.debug("[ElevenLabs →] contextual_update (silent inject)")
            await conversation.send_contextual_update(text)
        else:
            logger.debug("[ElevenLabs →] user_message")
            await conversation.send_user_message(text)

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        """Complete the SDK handler waiting on ``call_id``.

        The SDK sends the value the registered handler returns, so a result
        reaches the agent by resolving the future that handler is awaiting.
        """
        pending = self._pending_tools.get(session.id)
        future = pending.pop(call_id, None) if pending else None
        if future is None:
            logger.warning(
                "[ElevenLabs] tool result for unknown call %s (session %s) — "
                "the call timed out or the session ended",
                call_id,
                session.id,
            )
            return
        if not future.done():
            future.set_result(result)

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
        self._reject_pending_tools(session.id, "the voice session ended")

        conversation = self._conversations.pop(session.id, None)
        if conversation is not None:
            await conversation.end_session()
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(conversation.wait_for_session_end(), timeout=5.0)

        self._forget_session(session.id)

        session.state = VoiceSessionState.ENDED
        logger.info("ElevenLabs session disconnected: %s", session.id)

    async def close(self) -> None:
        for session_id in list(self._sessions):
            session = self._sessions.get(session_id)
            if session:
                await self.disconnect(session)

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

    def _make_tool_handler(self, session: VoiceSession, name: str) -> Any:
        """Build the SDK handler that hands a call to RoomKit and waits.

        Whatever the handler returns is what the SDK sends back as the tool
        result, so the call is bridged by parking on a future that
        :meth:`submit_tool_result` completes once the channel has run its
        handler, hooks and gates.
        """

        async def handler(parameters: dict[str, Any]) -> str:
            arguments = dict(parameters)
            # The SDK folds the call id into the arguments; the tool itself
            # never declared it, so it must not travel to the handler.
            call_id = str(arguments.pop("tool_call_id", "") or f"el-{uuid.uuid4().hex}")

            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            self._pending_tools.setdefault(session.id, {})[call_id] = future

            await self._fire(
                self._tool_call_callbacks,
                session,
                call_id,
                name,
                arguments,
                label="tool_call",
            )

            try:
                return await asyncio.wait_for(future, timeout=self._config.tool_timeout_s)
            except TimeoutError:
                # Raising is how the SDK is told this is an error result;
                # returning a string would read as a successful call.
                raise RuntimeError(
                    f"Tool '{name}' did not return within {self._config.tool_timeout_s:g}s"
                ) from None
            finally:
                pending = self._pending_tools.get(session.id)
                if pending is not None:
                    pending.pop(call_id, None)

        return handler

    def _reject_pending_tools(self, session_id: str, reason: str) -> None:
        """Fail every in-flight call so no SDK handler is left hanging."""
        pending = self._pending_tools.pop(session_id, {})
        for call_id, future in pending.items():
            if not future.done():
                future.set_exception(RuntimeError(f"Tool call {call_id} abandoned: {reason}"))

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
            if self._pending_tools.get(session.id):
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

    async def _fail_session(self, session: VoiceSession, code: str, message: str) -> None:
        """Report a session lost from under us, exactly once."""
        if session.id not in self._sessions:
            return

        logger.error("ElevenLabs session %s failed (%s): %s", session.id, code, message)
        await self._end_response(session)
        self._reject_pending_tools(session.id, message)
        self._forget_session(session.id)

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
        self._pending_tools.pop(session_id, None)
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
        """Store the SDK's async audio input callback for send_audio()."""
        self._provider._input_callbacks[self._session.id] = input_callback
        logger.debug("ElevenLabs audio bridge started (session %s)", self._session.id)

    async def stop(self) -> None:
        """Clean up when SDK conversation ends."""
        self._provider._input_callbacks.pop(self._session.id, None)
        logger.debug("ElevenLabs audio bridge stopped (session %s)", self._session.id)

    async def output(self, audio: bytes) -> None:
        """Called by SDK with agent audio — forward to RoomKit callbacks."""
        session = self._session
        provider = self._provider

        provider._last_audio_at[session.id] = time.monotonic()
        await provider._start_response(session)

        await provider._fire(provider._audio_callbacks, session, audio, label="audio")

    async def interrupt(self) -> None:
        """Called by SDK when user interrupted agent playback.

        A cut turn is still a finished turn: closing the response here is
        what clears the speaking indicator and lets the session go idle.
        """
        await self._provider._end_response(self._session)
