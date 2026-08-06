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
from roomkit.providers.deepgram.settings import build_settings, build_speak, build_think
from roomkit.providers.deepgram.voices import VOICES as _VOICES
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider, VoiceInfo

logger = logging.getLogger("roomkit.providers.deepgram.realtime")

_CONNECT_TIMEOUT = 30.0
_SETTINGS_TIMEOUT = 15.0
_CLOSE_TIMEOUT = 2.0

_KEEPALIVE = json.dumps({"type": "KeepAlive"})

# Deepgram rejects managed-LLM prompts longer than this.
_MAX_PROMPT_CHARS = 25_000


@dataclass
class _SessionState:
    """Per-session connection state."""

    session: VoiceSession
    ws: Any
    think: dict[str, Any]
    speak: dict[str, Any]
    receive_task: asyncio.Task[None] | None = None
    keepalive_task: asyncio.Task[None] | None = None
    responding: bool = False
    # Whether response_start has already fired for the turn being spoken.
    # Reset by AgentAudioDone, which closes the turn.
    audio_started: bool = False
    # FunctionCallResponse requires the function name, but the RealtimeVoiceProvider
    # contract only hands submit_tool_result() the call id — so remember the pairing.
    pending_calls: dict[str, str] = field(default_factory=dict)


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

        ws = await asyncio.wait_for(
            websockets.connect(self._config.base_url, additional_headers=self._auth_headers()),
            timeout=_CONNECT_TIMEOUT,
        )

        # Mirror what was actually sent (escape hatch included) so mid-session
        # updates patch the live configuration rather than a stale copy.
        state = _SessionState(
            session=session,
            ws=ws,
            think=settings["agent"]["think"],
            speak=settings["agent"]["speak"],
        )
        self._states[session.id] = state

        try:
            await ws.send(json.dumps(settings))
            await asyncio.wait_for(self._await_settings_applied(state), timeout=_SETTINGS_TIMEOUT)
        except BaseException:
            self._states.pop(session.id, None)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.close(), timeout=_CLOSE_TIMEOUT)
            raise

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        state.receive_task = asyncio.create_task(
            self._receive_loop(session.id), name=f"deepgram_agent_recv:{session.id}"
        )
        state.keepalive_task = asyncio.create_task(
            self._keepalive_loop(session.id), name=f"deepgram_agent_ka:{session.id}"
        )

        logger.info(
            "Deepgram Agent session connected: %s (listen=%s, think=%s, speak=%s)",
            session.id,
            settings["agent"]["listen"]["provider"].get("model"),
            state.think["provider"].get("model"),
            state.speak["provider"].get("model"),
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
            async for message in state.ws:
                await self._handle_message(state, message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Deepgram receive loop failed (session %s): %s", session_id, exc)
            await self._fire(
                self._error_callbacks, state.session, "connection_error", str(exc), label="error"
            )
        else:
            logger.info("Deepgram closed the connection (session %s)", session_id)

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
            logger.debug("Deepgram keepalive stopped (session %s): %s", session_id, exc)

    # -- Inbound dispatch ----------------------------------------------------

    async def _handle_message(self, state: _SessionState, message: Any) -> dict[str, Any] | None:
        """Dispatch one frame; returns the decoded event, or None for audio."""
        if isinstance(message, bytes | bytearray):
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
        await self._dispatch(state, event)
        return event

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
        state.responding = False
        state.audio_started = False
        await self._fire(self._response_end_callbacks, state.session, label="response_end")

    async def _on_injection_refused(self, state: _SessionState, event: dict[str, Any]) -> None:
        logger.warning("Deepgram refused a text injection (session %s)", state.session.id)

    async def _on_diagnostic(self, state: _SessionState, event: dict[str, Any]) -> None:
        """Surface an Error or a Warning — both share one payload shape."""
        etype = str(event.get("type") or "Error")
        code = str(event.get("code") or etype.lower())
        description = str(event.get("description") or "")
        log = logger.error if etype == "Error" else logger.warning
        log("Deepgram %s [%s] (session %s): %s", etype, code, state.session.id, description)
        await self._fire(self._error_callbacks, state.session, code, description, label="error")

    async def _on_conversation_text(self, state: _SessionState, event: dict[str, Any]) -> None:
        """Emit a final transcript, and close the user's turn when it is theirs.

        Deepgram has no "user stopped speaking" event: the user's transcript *is*
        the end of their turn. Without firing speech_end here the channel would
        hold ``_user_speaking`` forever and never go idle again.
        """
        role = str(event.get("role") or "assistant")
        content = str(event.get("content") or "")
        if role == "user":
            await self._fire(self._speech_end_callbacks, state.session, label="speech_end")
        if content:
            await self._fire(
                self._transcription_callbacks,
                state.session,
                content,
                role,
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
            state.pending_calls[call_id] = fname
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

        mtype = "InjectAgentMessage" if role == "assistant" else "InjectUserMessage"
        logger.debug("[Deepgram →] %s (session %s)", mtype, session.id)
        await state.ws.send(json.dumps({"type": mtype, "content": text}))

    async def _append_to_prompt(self, state: _SessionState, text: str) -> None:
        """Append to the live system prompt without dropping what was there."""
        current = str(state.think.get("prompt") or "")
        prompt = f"{current}\n\n{text}".strip() if current else text
        if len(prompt) > _MAX_PROMPT_CHARS:
            logger.warning(
                "Deepgram prompt for session %s is %d chars, over the %d-char limit "
                "for managed LLMs — the update will likely be refused",
                state.session.id,
                len(prompt),
                _MAX_PROMPT_CHARS,
            )
        state.think["prompt"] = prompt
        logger.debug("[Deepgram →] UpdatePrompt (session %s)", state.session.id)
        await state.ws.send(json.dumps({"type": "UpdatePrompt", "prompt": prompt}))

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        state = self._states.get(session.id)
        if state is None:
            return
        fname = state.pending_calls.pop(call_id, "")
        if not fname:
            logger.warning(
                "No pending Deepgram function call for id %s (session %s) — "
                "responding without a name",
                call_id,
                session.id,
            )
        await state.ws.send(
            json.dumps(
                {
                    "type": "FunctionCallResponse",
                    "id": call_id,
                    "name": fname,
                    "content": result,
                }
            )
        )

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

        if system_prompt is not None or tools is not None or temperature is not None:
            prompt: str | None = (
                system_prompt if system_prompt is not None else state.think.get("prompt")
            )
            think = build_think(
                self._config,
                system_prompt=prompt,
                tools=tools,
                temperature=temperature,
                pc=pc,
            )
            # Keep the previous functions when the caller did not restate them.
            if tools is None and state.think.get("functions"):
                think["functions"] = state.think["functions"]
            state.think = think
            logger.debug("[Deepgram →] UpdateThink (session %s)", session.id)
            await state.ws.send(json.dumps({"type": "UpdateThink", "think": think}))

        if voice is not None:
            speak = build_speak(self._config, voice=voice, pc=pc)
            state.speak = speak
            logger.debug("[Deepgram →] UpdateSpeak (session %s)", session.id)
            await state.ws.send(json.dumps({"type": "UpdateSpeak", "speak": speak}))

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        state = self._states.get(session.id)
        if state is None:
            return
        await state.ws.send(json.dumps(event))

    async def disconnect(self, session: VoiceSession) -> None:
        state = self._states.pop(session.id, None)
        if state is not None:
            for task in (state.receive_task, state.keepalive_task):
                if task is None:
                    continue
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            with contextlib.suppress(Exception):
                await asyncio.wait_for(state.ws.close(), timeout=_CLOSE_TIMEOUT)

        session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for session_id in list(self._states.keys()):
            state = self._states.get(session_id)
            if state is not None:
                await self.disconnect(state.session)
