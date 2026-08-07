"""Deepgram Voice Agent provider for speech-to-speech conversations.

Connects to Deepgram's Voice Agent API over WebSocket
(``wss://agent.deepgram.com/v1/agent/converse``). Unlike single-model
speech-to-speech APIs, Deepgram composes an agent from three independently
chosen stages — ``listen`` (Nova/Flux STT), ``think`` (an LLM, managed or
pointed at your own endpoint) and ``speak`` (Aura TTS) — configured in one
``Settings`` message at session start and changeable mid-session.

Deepgram owns turn detection, barge-in signalling and transcription; RoomKit
streams raw audio in and receives raw audio out.

Requires the ``websockets`` package::

    pip install 'roomkit[realtime-deepgram]'
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import SecretStr

from roomkit.providers.deepgram.config import DeepgramAgentConfig
from roomkit.providers.deepgram.settings import build_settings, patch_speak, patch_think
from roomkit.providers.deepgram.voices import VOICES as _VOICES
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider, VoiceInfo

logger = logging.getLogger("roomkit.providers.deepgram.realtime")

_CONNECT_TIMEOUT = 30.0
_SETTINGS_TIMEOUT = 15.0
_CLOSE_TIMEOUT = 2.0
_HANDSHAKE_BUFFER_LIMIT = 4 * 1024 * 1024

_KEEPALIVE = json.dumps({"type": "KeepAlive"})

_THINK_RECONFIGURE_KEYS = frozenset(
    {"think_provider", "think_model", "think_endpoint", "context_length"}
)
_SPEAK_RECONFIGURE_KEYS = frozenset(
    {"speak_model", "speak_language", "speak_provider", "speak_endpoint"}
)


@dataclass
class _PendingCall:
    """Wire fields that must survive until FunctionCallResponse."""

    name: str
    thought_signature: str | None = None


@dataclass
class _SessionState:
    """Per-session connection state."""

    session: VoiceSession
    ws: Any
    think: dict[str, Any]
    speak: dict[str, Any]
    # Managed-LLM prompt-size warning threshold; None disables the check.
    max_prompt_chars: int | None
    receive_task: asyncio.Task[None] | None = None
    keepalive_task: asyncio.Task[None] | None = None
    responding: bool = False
    # Whether response_start has already fired for the turn being spoken.
    # Reset by AgentAudioDone, which closes the turn.
    audio_started: bool = False
    # FunctionCallResponse requires fields that the RealtimeVoiceProvider contract
    # does not hand back to submit_tool_result(), so preserve them by call id.
    pending_calls: dict[str, _PendingCall] = field(default_factory=dict)
    # Prompt/model/voice updates are read-modify-write operations. Serialise them
    # so concurrent handoffs and supervisor injections cannot lose each other.
    update_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # A configured greeting can arrive before SettingsApplied. The channel does
    # not publish the session until connect() returns, so defer callbacks until
    # that boundary or the first greeting frames would be silently discarded.
    callbacks_ready: bool = False
    deferred_messages: list[Any] = field(default_factory=list)
    deferred_bytes: int = 0
    # The agent's transcript for the turn in progress.  Deepgram delivers
    # ConversationText sentence by sentence; sentences accumulate here (fired
    # as delta partials) and one full final goes out when the turn closes.
    assistant_text: str = ""


class DeepgramAgentProvider(RealtimeVoiceProvider):
    """Realtime voice provider using the Deepgram Voice Agent API.

    Requires the ``websockets`` package.

    Example::

        from roomkit.providers.deepgram import DeepgramAgentConfig, DeepgramAgentProvider

        provider = DeepgramAgentProvider(DeepgramAgentConfig(api_key="..."))
        provider.on_audio(handle_output_audio)
        provider.on_transcription(handle_transcription)

        await provider.connect(session, system_prompt="You are a helpful assistant.")
        await provider.send_audio(session, audio_bytes)

    Provider config keys (via the ``provider_config`` dict):
        listen_model, listen_version, listen_language (str): override the STT stage.
        keyterms (list[str]), smart_format (bool): further STT tuning.
        think_provider, think_model (str), think_endpoint (dict), context_length:
            override the LLM stage — ``think_endpoint`` points at a self-hosted,
            OpenAI-compatible server.
        speak_model, speak_language (str): override the TTS stage.
        speak_provider (dict): full ``agent.speak.provider`` replacement, sent
            verbatim — names a non-Deepgram TTS vendor (``eleven_labs``,
            ``cartesia``…) with that vendor's own fields. Takes precedence over
            ``voice``/``speak_model``.
        speak_endpoint (dict): ``agent.speak.endpoint`` (URL + auth headers) —
            required for BYO-key TTS vendors; Deepgram-managed ones need none.
        max_prompt_chars (int | None): per-session override of the managed-LLM
            prompt-size warning threshold; ``None`` disables the warning.
        greeting (str): line the agent speaks when the session opens.
        input_encoding, output_encoding (str): ``linear16`` (default), ``mulaw``,
            ``alaw``, ``opus``… Use ``mulaw`` at 8000 Hz for a telephony transport
            and no resampling is needed on either leg.
        output_container (str), output_bitrate (int): container/bitrate for
            encodings that need them.
        tags (list[str]): labels attached to the session in Deepgram's dashboard.
        settings (dict): deep-merged into the final ``Settings`` payload, last.
            Escape hatch for fields this provider does not model explicitly.

    Known limitations:
        - Turn detection is always Deepgram's; ``connect(server_vad=False)`` is
          not honoured and warns.
        - ``ConversationText`` is final-only, so no interim transcriptions.
        - The protocol has no client-side interrupt message. Barge-in is signalled
          the other way, by ``UserStartedSpeaking``, which this provider surfaces
          as ``on_speech_start`` — that is what drives RoomKit's playback flush.
        - Deepgram caps a session at two hours, warning at 1 h 55
          (``MAXIMUM_SESSION_LENGTH_APPROACHING``).
    """

    def __init__(
        self,
        config: DeepgramAgentConfig | None = None,
        *,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            self._config = config
        else:
            if api_key is None:
                raise ValueError("Either config or api_key must be provided")
            key = SecretStr(api_key) if isinstance(api_key, str) else api_key
            self._config = DeepgramAgentConfig(
                api_key=key,
                **({"base_url": base_url} if base_url else {}),
            )

        self._model = self._config.think_model
        self._states: dict[str, _SessionState] = {}

    @property
    def name(self) -> str:
        return "DeepgramAgentProvider"

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """Curated, offline catalog of Deepgram Aura voices."""
        return list(_VOICES)

    def is_responding(self, session_id: str) -> bool:
        state = self._states.get(session_id)
        return state is not None and state.responding

    # -- Connection lifecycle -----------------------------------------------

    def _import_websockets(self) -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                f"websockets is required for {self.name}. "
                "Install with: pip install 'roomkit[realtime-deepgram]'"
            ) from exc
        return websockets

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._config.api_key.get_secret_value()}"}

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
        websockets = self._import_websockets()
        pc = provider_config or {}

        if not server_vad:
            logger.warning(
                "Deepgram Voice Agent always performs its own turn detection — "
                "server_vad=False is not supported and was ignored (session %s)",
                session.id,
            )

        # Built before opening the socket so validation errors fail fast.
        settings = build_settings(
            self._config,
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            pc=pc,
        )

        # The raw settings escape hatch may replace nested objects completely.
        # Resolve the pieces used by live state before opening a socket so a bad
        # override cannot leak a connected WebSocket.
        try:
            agent = settings["agent"]
            listen = agent["listen"]
            think = agent["think"]
            speak = agent["speak"]
            listen_provider = listen["provider"]
            think_provider = think["provider"]
            speak_provider = speak["provider"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Deepgram settings must contain agent listen, think, and speak providers"
            ) from exc
        required_objects = (
            agent,
            listen,
            think,
            speak,
            listen_provider,
            think_provider,
            speak_provider,
        )
        if not all(isinstance(value, dict) for value in required_objects):
            raise ValueError("Deepgram agent stages and providers must be objects")

        max_prompt_chars = pc.get("max_prompt_chars", self._config.max_prompt_chars)
        self._warn_prompt_over_limit(session.id, think, max_prompt_chars)
        self._warn_voice_ignored(session.id, voice, speak_provider)

        ws = await asyncio.wait_for(
            websockets.connect(self._config.base_url, additional_headers=self._auth_headers()),
            timeout=_CONNECT_TIMEOUT,
        )

        # Mirror what was actually sent (escape hatch included) so mid-session
        # updates patch the live configuration rather than a stale copy.
        state = _SessionState(
            session=session,
            ws=ws,
            think=think,
            speak=speak,
            max_prompt_chars=max_prompt_chars,
        )
        self._states[session.id] = state

        try:
            # Deepgram's opening handshake is server-first. Sending Settings
            # before Welcome is a protocol violation and can race with auth.
            await asyncio.wait_for(self._await_welcome(state), timeout=_SETTINGS_TIMEOUT)
            await ws.send(json.dumps(settings))
            await asyncio.wait_for(self._await_settings_applied(state), timeout=_SETTINGS_TIMEOUT)
        except BaseException:
            await self._finalize_session(session.id, state)
            raise

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id
        state.callbacks_ready = True

        state.receive_task = asyncio.create_task(
            self._receive_loop(session.id), name=f"deepgram_agent_recv:{session.id}"
        )
        state.keepalive_task = asyncio.create_task(
            self._keepalive_loop(session.id), name=f"deepgram_agent_ka:{session.id}"
        )

        speak_label = (
            speak_provider.get("model")
            or speak_provider.get("model_id")
            or speak_provider.get("type")
        )
        logger.info(
            "Deepgram Agent session connected: %s (listen=%s, think=%s, speak=%s)",
            session.id,
            listen_provider.get("model"),
            think_provider.get("model"),
            speak_label,
        )

    async def _await_welcome(self, state: _SessionState) -> None:
        """Wait for the server-first Welcome before sending client messages."""
        while True:
            event = await self._handle_message(state, await state.ws.recv())
            if event is None:
                continue
            etype = event.get("type")
            if etype == "Welcome":
                return
            if etype == "Error":
                raise ConnectionError(
                    f"Deepgram rejected connection [{event.get('code')}]: "
                    f"{event.get('description')}"
                )

    async def _await_settings_applied(self, state: _SessionState) -> None:
        """Consume frames until Deepgram acknowledges the Settings message.

        A ``greeting`` makes the agent speak immediately, so audio can arrive
        before the acknowledgement; those frames go through the normal dispatch
        rather than being dropped.
        """
        while True:
            event = await self._handle_message(state, await state.ws.recv())
            if event is None:
                continue
            etype = event.get("type")
            if etype == "SettingsApplied":
                return
            if etype == "Error":
                raise ConnectionError(
                    f"Deepgram rejected Settings [{event.get('code')}]: {event.get('description')}"
                )

    async def _receive_loop(self, session_id: str) -> None:
        state = self._states.get(session_id)
        if state is None:
            return
        try:
            # Let RealtimeVoiceChannel publish its session maps after connect()
            # returns before callbacks for handshake-time greeting audio fire.
            await asyncio.sleep(0)
            deferred = state.deferred_messages
            state.deferred_messages = []
            state.deferred_bytes = 0
            for message in deferred:
                await self._handle_message(state, message)
                if self._states.get(session_id) is not state:
                    return
            async for message in state.ws:
                await self._handle_message(state, message)
                if self._states.get(session_id) is not state:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Deepgram receive loop failed (session %s): %s", session_id, exc)
            state.session.state = VoiceSessionState.ENDED
            await self._fire(
                self._error_callbacks, state.session, "connection_error", str(exc), label="error"
            )
        else:
            logger.info("Deepgram closed the connection (session %s)", session_id)
            if self._states.get(session_id) is state:
                state.session.state = VoiceSessionState.ENDED
                await self._fire(
                    self._error_callbacks,
                    state.session,
                    "connection_closed",
                    "Deepgram closed the connection unexpectedly",
                    label="error",
                )
        finally:
            await self._finalize_session(session_id, state)

    async def _keepalive_loop(self, session_id: str) -> None:
        """Hold the socket open through silences — Deepgram closes quiet ones."""
        state = self._states.get(session_id)
        interval = self._config.keepalive_interval
        if state is None or interval <= 0:
            return
        try:
            while True:
                await asyncio.sleep(interval)
                await state.ws.send(_KEEPALIVE)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Deepgram keepalive failed (session %s): %s", session_id, exc)
            state.session.state = VoiceSessionState.ENDED
            await self._fire(
                self._error_callbacks, state.session, "connection_error", str(exc), label="error"
            )
            await self._finalize_session(session_id, state)

    # -- Inbound dispatch ----------------------------------------------------

    async def _handle_message(self, state: _SessionState, message: Any) -> dict[str, Any] | None:
        """Dispatch one frame; returns the decoded event, or None for audio."""
        if isinstance(message, bytes | bytearray):
            if not state.callbacks_ready:
                self._defer_handshake_message(state, bytes(message))
                return None
            await self._begin_response(state)
            await self._fire(self._audio_callbacks, state.session, bytes(message), label="audio")
            return None
        try:
            event = json.loads(message)
        except (TypeError, ValueError):
            logger.warning(
                "Deepgram sent an undecodable text frame (session %s)", state.session.id
            )
            return None
        if not isinstance(event, dict):
            return None
        if not state.callbacks_ready and event.get("type") not in {
            "Welcome",
            "SettingsApplied",
            "Error",
            "Warning",
        }:
            self._defer_handshake_message(state, message)
            return event
        await self._dispatch(state, event)
        return event

    @staticmethod
    def _defer_handshake_message(state: _SessionState, message: Any) -> None:
        """Bound greeting data retained until the public session is visible."""
        size = len(message) if isinstance(message, bytes | bytearray | str) else 0
        if state.deferred_bytes + size > _HANDSHAKE_BUFFER_LIMIT:
            raise ConnectionError(
                f"Deepgram sent more than {_HANDSHAKE_BUFFER_LIMIT} bytes before SettingsApplied"
            )
        state.deferred_messages.append(message)
        state.deferred_bytes += size

    # Maps each server event type to the handler method that processes it.
    # Acknowledgements (Welcome, SettingsApplied, *Updated) and telemetry
    # (LatencyReport, History) have no handler and fall through to a debug line.
    _EVENT_HANDLERS: dict[str, str] = {
        "ConversationText": "_on_conversation_text",
        "UserStartedSpeaking": "_on_user_started_speaking",
        "AgentStartedSpeaking": "_on_agent_started_speaking",
        "AgentAudioDone": "_on_agent_audio_done",
        "FunctionCallRequest": "_on_function_call_request",
        "Error": "_on_diagnostic",
        "Warning": "_on_diagnostic",
        "InjectionRefused": "_on_injection_refused",
    }

    async def _dispatch(self, state: _SessionState, event: dict[str, Any]) -> None:
        """Route a server event to its handler via the dispatch table."""
        etype = str(event.get("type") or "")
        handler_name = self._EVENT_HANDLERS.get(etype)
        if handler_name is None:
            logger.debug("[Deepgram ←] %s (session %s)", etype, state.session.id)
            return
        await getattr(self, handler_name)(state, event)

    async def _on_user_started_speaking(self, state: _SessionState, event: dict[str, Any]) -> None:
        # The agent may still be generating, but from here on RoomKit treats
        # the turn as the user's — the channel flushes playback on this.
        # Close the agent's transcript first so its entry lands complete
        # before the user's turn opens.
        await self._flush_assistant_transcript(state)
        state.responding = False
        await self._fire(self._speech_start_callbacks, state.session, label="speech_start")

    async def _begin_response(self, state: _SessionState) -> None:
        """Open the agent's turn, at the latest on its first audio frame.

        The channel arms echo cancellation on ``on_response_start``
        (``channels/_realtime_response.py``); the AEC sits in bypass until then,
        so any speaker output played before it is never cancelled and comes
        straight back through an open mic — which Deepgram's transcription then
        reports as user speech. Deepgram does not guarantee that
        ``AgentStartedSpeaking`` reaches us ahead of the audio it announces, so
        the first frame opens the turn if that event has not already done it.
        Idempotent for the rest of the turn; ``AgentAudioDone`` closes it.
        """
        if state.audio_started:
            return
        state.audio_started = True
        state.responding = True
        await self._fire(self._response_start_callbacks, state.session, label="response_start")

    async def _on_agent_started_speaking(
        self, state: _SessionState, event: dict[str, Any]
    ) -> None:
        await self._begin_response(state)

    async def _on_agent_audio_done(self, state: _SessionState, event: dict[str, Any]) -> None:
        await self._flush_assistant_transcript(state)
        state.responding = False
        state.audio_started = False
        await self._fire(self._response_end_callbacks, state.session, label="response_end")

    async def _on_injection_refused(self, state: _SessionState, event: dict[str, Any]) -> None:
        logger.warning("Deepgram refused a text injection (session %s)", state.session.id)

    async def _on_diagnostic(self, state: _SessionState, event: dict[str, Any]) -> None:
        """Surface diagnostics and retire sessions after fatal errors."""
        etype = str(event.get("type") or "Error")
        code = str(event.get("code") or etype.lower())
        description = str(event.get("description") or "")
        log = logger.error if etype == "Error" else logger.warning
        log("Deepgram %s [%s] (session %s): %s", etype, code, state.session.id, description)
        if etype == "Error":
            state.session.state = VoiceSessionState.ENDED
        await self._fire(self._error_callbacks, state.session, code, description, label="error")
        # Deepgram defines Error as fatal and Warning as non-fatal. Handshake
        # errors are finalized by connect(); live errors must stop both worker
        # tasks here instead of leaving an apparently active dead session.
        if etype == "Error" and state.receive_task is not None:
            await self._finalize_session(state.session.id, state)

    async def _on_conversation_text(self, state: _SessionState, event: dict[str, Any]) -> None:
        """Emit transcripts, translating Deepgram's cadence to the contract.

        The user's transcript arrives once per utterance and *is* the end of
        their turn (Deepgram has no "user stopped speaking" event; without
        firing speech_end here the channel would hold ``_user_speaking``
        forever).  The agent's transcript, though, arrives *sentence by
        sentence* — final-only on the wire.  Fired as-is, every sentence
        became its own final entry downstream (RoomKit UI rendered a
        paragraph gap per sentence).  Sentences therefore accumulate as
        delta partials, and one full final closes the turn — at
        ``AgentAudioDone``, on a barge-in, or when the session ends.
        """
        role = str(event.get("role") or "assistant")
        content = str(event.get("content") or "")
        if role == "user":
            await self._fire(self._speech_end_callbacks, state.session, label="speech_end")
            # A user transcript proves the agent's turn is over even if
            # AgentAudioDone was lost — close the pending transcript first.
            await self._flush_assistant_transcript(state)
            if content:
                await self._fire(
                    self._transcription_callbacks,
                    state.session,
                    content,
                    role,
                    True,
                    label="transcription",
                )
            return
        if not content:
            return
        delta = (" " if state.assistant_text else "") + content
        state.assistant_text += delta
        await self._fire(
            self._transcription_callbacks,
            state.session,
            delta,
            role,
            False,
            label="transcription",
        )

    async def _flush_assistant_transcript(self, state: _SessionState) -> None:
        """Fire the accumulated agent transcript as the turn's one final."""
        if not state.assistant_text:
            return
        text = state.assistant_text
        state.assistant_text = ""
        await self._fire(
            self._transcription_callbacks,
            state.session,
            text,
            "assistant",
            True,
            label="transcription",
        )

    async def _on_function_call_request(self, state: _SessionState, event: dict[str, Any]) -> None:
        for function in event.get("functions") or []:
            if not isinstance(function, dict):
                continue
            if not function.get("client_side", True):
                # Deepgram runs this one itself and reports the outcome.
                continue
            call_id = str(function.get("id") or "")
            fname = str(function.get("name") or "")
            if not call_id or not fname:
                logger.warning(
                    "Deepgram sent a function call without a non-empty id and name (session %s)",
                    state.session.id,
                )
                continue
            if call_id in state.pending_calls:
                logger.warning(
                    "Deepgram reused pending function call id %s (session %s); ignoring duplicate",
                    call_id,
                    state.session.id,
                )
                continue
            raw_args = function.get("arguments")
            try:
                # Deepgram sends arguments as a JSON *string*.
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except ValueError:
                logger.warning(
                    "Deepgram sent unparseable arguments for %s (session %s): %r",
                    fname,
                    state.session.id,
                    raw_args,
                )
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            raw_signature = function.get("thought_signature")
            signature = raw_signature if isinstance(raw_signature, str) and raw_signature else None
            state.pending_calls[call_id] = _PendingCall(
                name=fname,
                thought_signature=signature,
            )
            await self._fire(
                self._tool_call_callbacks,
                state.session,
                call_id,
                fname,
                arguments,
                label="tool_call",
            )

    # -- Outbound client API -------------------------------------------------

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        state = self._states.get(session.id)
        if state is None:
            return
        await state.ws.send(audio)

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> None:
        """Inject text into the conversation.

        ``role="assistant"`` puts the words in the agent's mouth
        (``InjectAgentMessage``); ``role="user"`` makes the agent hear them and
        answer (``InjectUserMessage``).

        Deepgram has no message that adds to the conversation silently, so a
        silent injection — and any system-role text — is appended to the system
        prompt via ``UpdatePrompt`` instead: the agent takes it into account on
        its next turn without reacting to it now.
        """
        state = self._states.get(session.id)
        if state is None:
            return

        if silent or role == "system":
            await self._append_to_prompt(state, text)
            return

        is_agent_message = role == "assistant"
        mtype = "InjectAgentMessage" if is_agent_message else "InjectUserMessage"
        content_key = "message" if is_agent_message else "content"
        logger.debug("[Deepgram →] %s (session %s)", mtype, session.id)
        await state.ws.send(json.dumps({"type": mtype, content_key: text}))

    def _warn_prompt_over_limit(
        self, session_id: str, think: dict[str, Any], limit: int | None
    ) -> None:
        """Early client-side signal for Deepgram's managed-LLM prompt cap.

        Deepgram truncates an over-limit prompt and keeps the session alive
        (a non-fatal ``PROMPT_TOO_LONG`` warning); it does not refuse the
        update. A bring-your-own ``endpoint`` has no documented cap, so no
        warning applies there.
        """
        if limit is None or think.get("endpoint"):
            return
        prompt = str(think.get("prompt") or "")
        if len(prompt) > limit:
            logger.warning(
                "Deepgram prompt for session %s is %d chars, over the %d-char "
                "managed-LLM cap — Deepgram will truncate it (PROMPT_TOO_LONG)",
                session_id,
                len(prompt),
                limit,
            )

    @staticmethod
    def _warn_voice_ignored(
        session_id: str, voice: str | None, speak_provider: dict[str, Any]
    ) -> None:
        """Flag a ``voice`` that cannot apply because the TTS vendor is not Deepgram.

        ``voice`` names an Aura model; when a ``speak_provider`` (or the raw
        settings escape hatch) selects another vendor, the voice belongs in
        that vendor's own fields instead and the argument is dropped.
        """
        if voice and speak_provider.get("type", "deepgram") != "deepgram":
            logger.warning(
                "voice=%r names a Deepgram Aura model but the speak provider is %r "
                "— ignored (session %s)",
                voice,
                speak_provider.get("type"),
                session_id,
            )

    async def _append_to_prompt(self, state: _SessionState, text: str) -> None:
        """Append to the live system prompt without dropping what was there."""
        async with state.update_lock:
            current = str(state.think.get("prompt") or "")
            prompt = f"{current}\n\n{text}".strip() if current else text
            self._warn_prompt_over_limit(
                state.session.id, {**state.think, "prompt": prompt}, state.max_prompt_chars
            )
            logger.debug("[Deepgram →] UpdatePrompt (session %s)", state.session.id)
            await state.ws.send(json.dumps({"type": "UpdatePrompt", "prompt": prompt}))
            state.think["prompt"] = prompt

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        state = self._states.get(session.id)
        if state is None:
            return
        pending = state.pending_calls.get(call_id)
        fname = pending.name if pending is not None else ""
        if pending is None:
            logger.warning(
                "No pending Deepgram function call for id %s (session %s) — "
                "responding without a name",
                call_id,
                session.id,
            )
        response = {
            "type": "FunctionCallResponse",
            "id": call_id,
            "name": fname,
            "content": result,
        }
        if pending is not None and pending.thought_signature is not None:
            response["thought_signature"] = pending.thought_signature
        await state.ws.send(json.dumps(response))
        # Do not lose the call metadata when the send fails. Also avoid removing
        # a newer call if Deepgram reused the id while this send was in flight.
        if pending is not None and state.pending_calls.get(call_id) is pending:
            state.pending_calls.pop(call_id, None)

    async def interrupt(self, session: VoiceSession) -> None:
        """Mark the agent's turn as over locally.

        The Voice Agent protocol has no client-side interrupt: Deepgram signals
        barge-in the other way, with ``UserStartedSpeaking``, and the channel
        flushes playback on the resulting ``on_speech_start``. All that is left
        here is to stop reporting the session as responding.
        """
        state = self._states.get(session.id)
        if state is None:
            return
        state.responding = False
        logger.debug(
            "interrupt() is local-only for Deepgram — no wire message (session %s)", session.id
        )

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
        """Update the live session in place, without reconnecting.

        ``UpdateThink`` carries prompt, model and functions in a single message
        and ``UpdateSpeak`` swaps the voice, so — unlike the base implementation —
        an agent handoff keeps the WebSocket and the conversation context.
        """
        state = self._states.get(session.id)
        if state is None:
            await super().reconfigure(
                session,
                system_prompt=system_prompt,
                voice=voice,
                tools=tools,
                temperature=temperature,
                provider_config=provider_config,
            )
            return

        pc = provider_config or {}

        async with state.update_lock:
            if "max_prompt_chars" in pc:
                state.max_prompt_chars = pc["max_prompt_chars"]
            think_changed = (
                system_prompt is not None
                or tools is not None
                or temperature is not None
                or any(key in pc for key in _THINK_RECONFIGURE_KEYS)
            )
            if think_changed:
                think = patch_think(
                    state.think,
                    system_prompt=system_prompt,
                    tools=tools,
                    temperature=temperature,
                    pc=pc,
                )
                self._warn_prompt_over_limit(session.id, think, state.max_prompt_chars)
                logger.debug("[Deepgram →] UpdateThink (session %s)", session.id)
                await state.ws.send(json.dumps({"type": "UpdateThink", "think": think}))
                state.think = think

            speak_changed = voice is not None or any(key in pc for key in _SPEAK_RECONFIGURE_KEYS)
            if speak_changed:
                speak = patch_speak(state.speak, voice=voice, pc=pc)
                self._warn_voice_ignored(session.id, voice, speak.get("provider") or {})
                logger.debug("[Deepgram →] UpdateSpeak (session %s)", session.id)
                await state.ws.send(json.dumps({"type": "UpdateSpeak", "speak": speak}))
                state.speak = speak

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        state = self._states.get(session.id)
        if state is None:
            return
        await state.ws.send(json.dumps(event))

    async def _finalize_session(self, session_id: str, state: _SessionState) -> None:
        """Atomically retire one live state and all resources it owns."""
        if self._states.get(session_id) is not state:
            return
        self._states.pop(session_id, None)
        with contextlib.suppress(Exception):
            # A session dying mid-response still owes its consumers the
            # final for whatever the agent managed to say.
            await self._flush_assistant_transcript(state)
        state.responding = False
        state.audio_started = False
        state.pending_calls.clear()
        state.deferred_messages.clear()
        state.deferred_bytes = 0
        state.session.state = VoiceSessionState.ENDED

        current = asyncio.current_task()
        for task in (state.receive_task, state.keepalive_task):
            if task is None or task is current:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        with contextlib.suppress(Exception):
            await asyncio.wait_for(state.ws.close(), timeout=_CLOSE_TIMEOUT)

    async def disconnect(self, session: VoiceSession) -> None:
        state = self._states.get(session.id)
        if state is not None:
            await self._finalize_session(session.id, state)
        else:
            session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for session_id in list(self._states.keys()):
            state = self._states.get(session_id)
            if state is not None:
                await self.disconnect(state.session)
