"""ElevenLabs Conversational AI realtime provider.

Connects via WebSocket to ElevenLabs' server-orchestrated agent platform.
ElevenLabs handles STT, LLM, TTS, VAD, and turn-taking server-side —
the provider sends/receives audio and handles client-side tool calls.

Requires the ``websockets`` and ``elevenlabs`` packages::

    pip install 'roomkit[realtime-elevenlabs]'
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import Any

from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import (
    RealtimeAudioCallback,
    RealtimeErrorCallback,
    RealtimeResponseEndCallback,
    RealtimeResponseStartCallback,
    RealtimeSpeechEndCallback,
    RealtimeSpeechStartCallback,
    RealtimeToolCallCallback,
    RealtimeTranscriptionCallback,
    RealtimeVoiceProvider,
)

logger = logging.getLogger("roomkit.providers.elevenlabs.realtime")

# All client event types we want to receive from the server.
_CLIENT_EVENTS = [
    "audio",
    "agent_response",
    "agent_response_correction",
    "user_transcript",
    "tentative_user_transcript",
    "interruption",
    "ping",
    "client_tool_call",
    "vad_score",
    "conversation_initiation_metadata",
]


class ElevenLabsRealtimeProvider(RealtimeVoiceProvider):
    """Realtime voice provider using ElevenLabs Conversational AI.

    ElevenLabs agents are pre-configured on the dashboard with an LLM,
    voice, and system prompt.  Runtime overrides (system prompt, voice,
    temperature) are applied via ``conversation_config_override`` at
    connection time.

    Example::

        from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
        from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider

        config = ElevenLabsRealtimeConfig(
            api_key="xi-...",
            agent_id="agent_abc123",
        )
        provider = ElevenLabsRealtimeProvider(config)
        provider.on_audio(handle_audio)
        provider.on_transcription(handle_transcription)

        await provider.connect(session, system_prompt="You are helpful.")
        await provider.send_audio(session, audio_bytes)
    """

    def __init__(self, config: ElevenLabsRealtimeConfig) -> None:
        self._config = config

        # Per-session state
        self._connections: dict[str, Any] = {}
        self._receive_tasks: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, VoiceSession] = {}
        self._last_interrupt_id: dict[str, int] = {}
        self._conversation_ids: dict[str, str] = {}

        # Callbacks
        self._audio_callbacks: list[RealtimeAudioCallback] = []
        self._transcription_callbacks: list[RealtimeTranscriptionCallback] = []
        self._speech_start_callbacks: list[RealtimeSpeechStartCallback] = []
        self._speech_end_callbacks: list[RealtimeSpeechEndCallback] = []
        self._tool_call_callbacks: list[RealtimeToolCallCallback] = []
        self._response_start_callbacks: list[RealtimeResponseStartCallback] = []
        self._response_end_callbacks: list[RealtimeResponseEndCallback] = []
        self._error_callbacks: list[RealtimeErrorCallback] = []

        # Track active responses
        self._responding: set[str] = set()

    def is_responding(self, session_id: str) -> bool:
        return session_id in self._responding

    @property
    def name(self) -> str:
        return "ElevenLabsRealtimeProvider"

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
            import websockets
        except ImportError as exc:
            raise ImportError(
                "websockets is required for ElevenLabsRealtimeProvider. "
                "Install with: pip install 'roomkit[realtime-elevenlabs]'"
            ) from exc

        if tools:
            logger.warning(
                "ElevenLabs agents have tools configured on the dashboard; "
                "the 'tools' parameter passed to connect() is ignored. "
                "Configure client tools via the ElevenLabs agent settings."
            )

        pc = provider_config or {}
        url = await self._get_ws_url()
        headers = self._build_headers()

        ws = await asyncio.wait_for(
            websockets.connect(url, additional_headers=headers, max_size=16 * 1024 * 1024),
            timeout=30.0,
        )

        self._connections[session.id] = ws
        self._sessions[session.id] = session
        self._last_interrupt_id[session.id] = 0

        # Build conversation_initiation_client_data
        init_data = self._build_init_data(
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            provider_config=pc,
        )

        await ws.send(json.dumps(init_data))

        # Wait for conversation_initiation_metadata
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
            meta = json.loads(raw)
            if meta.get("type") == "conversation_initiation_metadata":
                event = meta["conversation_initiation_metadata_event"]
                self._conversation_ids[session.id] = event.get("conversation_id", "")
                logger.info(
                    "ElevenLabs conversation started: conversation_id=%s, "
                    "input_format=%s, output_format=%s (session %s)",
                    event.get("conversation_id"),
                    event.get("user_input_audio_format"),
                    event.get("agent_output_audio_format"),
                    session.id,
                )
            else:
                # Not the expected init message — handle it normally later
                logger.warning(
                    "Expected conversation_initiation_metadata, got %s", meta.get("type")
                )
        except TimeoutError:
            logger.warning("Timeout waiting for conversation metadata (session %s)", session.id)

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = self._conversation_ids.get(session.id, session.id)

        # Start receive loop
        self._receive_tasks[session.id] = asyncio.create_task(
            self._receive_loop(session),
            name=f"elevenlabs_rt_recv:{session.id}",
        )

        logger.info("ElevenLabs Realtime session connected: %s", session.id)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        # ElevenLabs audio messages have no "type" field — identified by
        # the presence of "user_audio_chunk".
        await ws.send(json.dumps({"user_audio_chunk": base64.b64encode(audio).decode("ascii")}))

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return

        if silent:
            # Contextual update: non-interrupting context injection
            logger.debug("[ElevenLabs →] contextual_update (silent inject)")
            await ws.send(json.dumps({"type": "contextual_update", "text": text}))
        else:
            logger.debug("[ElevenLabs →] user_message (role=%s)", role)
            await ws.send(json.dumps({"type": "user_message", "user_message": {"text": text}}))

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(
            json.dumps(
                {
                    "type": "client_tool_result",
                    "tool_call_id": call_id,
                    "result": result,
                    "is_error": False,
                }
            )
        )
        logger.debug("[ElevenLabs →] client_tool_result (call_id=%s)", call_id)

    async def interrupt(self, session: VoiceSession) -> None:
        # ElevenLabs handles interruption server-side via VAD.
        # The server sends an "interruption" event when it detects user speech
        # during agent playback.  We filter stale audio via _last_interrupt_id.
        # Sending user_activity signals the server that the user is active,
        # which delays agent speech by ~2s.
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(json.dumps({"type": "user_activity"}))
        logger.debug("[ElevenLabs →] user_activity (interrupt hint, session %s)", session.id)

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(json.dumps(event))

    async def disconnect(self, session: VoiceSession) -> None:
        task = self._receive_tasks.pop(session.id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        ws = self._connections.pop(session.id, None)
        self._sessions.pop(session.id, None)
        self._last_interrupt_id.pop(session.id, None)
        self._conversation_ids.pop(session.id, None)
        self._responding.discard(session.id)
        if ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.close(), timeout=2.0)

        session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for session_id in list(self._sessions):
            session = self._sessions.get(session_id)
            if session:
                await self.disconnect(session)

    # -- Callback registration --

    def on_audio(self, callback: RealtimeAudioCallback) -> None:
        self._audio_callbacks.append(callback)

    def on_transcription(self, callback: RealtimeTranscriptionCallback) -> None:
        self._transcription_callbacks.append(callback)

    def on_speech_start(self, callback: RealtimeSpeechStartCallback) -> None:
        self._speech_start_callbacks.append(callback)

    def on_speech_end(self, callback: RealtimeSpeechEndCallback) -> None:
        self._speech_end_callbacks.append(callback)

    def on_tool_call(self, callback: RealtimeToolCallCallback) -> None:
        self._tool_call_callbacks.append(callback)

    def on_response_start(self, callback: RealtimeResponseStartCallback) -> None:
        self._response_start_callbacks.append(callback)

    def on_response_end(self, callback: RealtimeResponseEndCallback) -> None:
        self._response_end_callbacks.append(callback)

    def on_error(self, callback: RealtimeErrorCallback) -> None:
        self._error_callbacks.append(callback)

    # -- Internal helpers --

    async def _get_ws_url(self) -> str:
        """Build or fetch the WebSocket URL for the conversation."""
        if self._config.requires_auth:
            from elevenlabs.client import AsyncElevenLabs

            client = AsyncElevenLabs(api_key=self._config.api_key)
            response = await client.conversational_ai.get_signed_url(
                agent_id=self._config.agent_id
            )
            return str(response.signed_url)
        return f"{self._config.base_url}/v1/convai/conversation?agent_id={self._config.agent_id}"

    def _build_headers(self) -> dict[str, str]:
        """Build WebSocket connection headers."""
        if self._config.requires_auth:
            # Signed URL already contains auth — no header needed.
            return {}
        return {"xi-api-key": self._config.api_key}

    def _build_init_data(
        self,
        *,
        system_prompt: str | None,
        voice: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        provider_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the conversation_initiation_client_data message."""
        agent_override: dict[str, Any] = {}
        if system_prompt:
            agent_override["prompt"] = {"prompt": system_prompt}
        if provider_config.get("language"):
            agent_override["language"] = provider_config["language"]
        if provider_config.get("first_message") is not None:
            agent_override["first_message"] = provider_config["first_message"]

        tts_override: dict[str, Any] = {}
        if voice:
            tts_override["voice_id"] = voice

        conversation_override: dict[str, Any] = {
            "client_events": _CLIENT_EVENTS,
        }

        config_override: dict[str, Any] = {}
        if agent_override:
            config_override["agent"] = agent_override
        if tts_override:
            config_override["tts"] = tts_override
        config_override["conversation"] = conversation_override

        extra_body: dict[str, Any] = {}
        if temperature is not None:
            extra_body["temperature"] = temperature

        init_data: dict[str, Any] = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": config_override,
            "custom_llm_extra_body": extra_body,
        }

        if provider_config.get("dynamic_variables"):
            init_data["dynamic_variables"] = provider_config["dynamic_variables"]

        return init_data

    # -- Receive loop --

    async def _receive_loop(self, session: VoiceSession) -> None:
        """Process server events from ElevenLabs Conversational AI."""
        ws = self._connections.get(session.id)
        if ws is None:
            return

        try:
            async for raw_message in ws:
                try:
                    event = json.loads(raw_message)
                    await self._handle_server_event(session, event)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from ElevenLabs for session %s", session.id)
                except Exception:
                    logger.exception("Error handling ElevenLabs event for session %s", session.id)
        except asyncio.CancelledError:
            raise
        except Exception:
            if session.state == VoiceSessionState.ACTIVE:
                logger.warning(
                    "ElevenLabs WebSocket closed unexpectedly for session %s", session.id
                )
                session.state = VoiceSessionState.ENDED
                await self._fire_error_callbacks(
                    session,
                    "connection_closed",
                    f"WebSocket closed unexpectedly for session {session.id}",
                )
            else:
                logger.debug("ElevenLabs WebSocket closed for session %s", session.id)

    # Event types that carry bulk audio data — suppress in protocol log.
    _NOISY_EVENTS = frozenset({"audio", "vad_score", "tentative_user_transcript"})

    async def _handle_server_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        """Map ElevenLabs server events to callbacks."""
        event_type = event.get("type", "")

        if event_type not in self._NOISY_EVENTS:
            logger.debug("[ElevenLabs ←] %s (session %s)", event_type, session.id)

        if event_type == "audio":
            audio_event = event.get("audio_event", {})
            event_id = int(audio_event.get("event_id", 0))
            last_interrupt = self._last_interrupt_id.get(session.id, 0)
            if event_id <= last_interrupt:
                return  # Stale audio — already interrupted
            audio_b64 = audio_event.get("audio_base_64", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                # ElevenLabs has no explicit "response started" event.
                # The first audio chunk after an interruption (or session
                # start) signals that the agent has begun speaking.
                if session.id not in self._responding:
                    self._responding.add(session.id)
                    await self._fire_callbacks(self._response_start_callbacks, session)
                await self._fire_audio_callbacks(session, audio_bytes)

        elif event_type == "user_transcript":
            text = event.get("user_transcription_event", {}).get("user_transcript", "").strip()
            if text:
                await self._fire_transcription_callbacks(session, text, "user", True)

        elif event_type == "tentative_user_transcript":
            text = (
                event.get("tentative_user_transcript", {})
                .get("tentative_user_transcript", "")
                .strip()
            )
            if not text:
                text = event.get("text", "").strip()
            if text:
                await self._fire_transcription_callbacks(session, text, "user", False)

        elif event_type == "agent_response":
            text = event.get("agent_response_event", {}).get("agent_response", "").strip()
            if text:
                self._responding.discard(session.id)
                await self._fire_transcription_callbacks(session, text, "assistant", True)
                await self._fire_callbacks(self._response_end_callbacks, session)

        elif event_type == "agent_response_correction":
            corrected = (
                event.get("agent_response_correction_event", {})
                .get("corrected_agent_response", "")
                .strip()
            )
            if corrected:
                await self._fire_transcription_callbacks(session, corrected, "assistant", True)

        elif event_type == "client_tool_call":
            tool_call = event.get("client_tool_call", {})
            tool_name = tool_call.get("tool_name", "")
            tool_call_id = tool_call.get("tool_call_id", "")
            parameters = tool_call.get("parameters", {})
            logger.info(
                "[ElevenLabs] tool_call: %s (call_id=%s, session %s)",
                tool_name,
                tool_call_id,
                session.id,
            )
            await self._fire_tool_call_callbacks(session, tool_call_id, tool_name, parameters)

        elif event_type == "interruption":
            interruption_event = event.get("interruption_event", {})
            event_id = int(interruption_event.get("event_id", 0))
            self._last_interrupt_id[session.id] = event_id
            self._responding.discard(session.id)
            logger.info("[ElevenLabs] interruption event_id=%d (session %s)", event_id, session.id)
            await self._fire_callbacks(self._speech_start_callbacks, session)

        elif event_type == "ping":
            ping_event = event.get("ping_event", {})
            event_id = ping_event.get("event_id")
            ws = self._connections.get(session.id)
            if ws is not None and event_id is not None:
                await ws.send(json.dumps({"type": "pong", "event_id": event_id}))

        elif event_type == "vad_score":
            pass  # Logged at trace level only; no callback mapping.

        elif event_type == "conversation_initiation_metadata":
            # Already handled during connect(); ignore duplicates.
            pass

        elif event_type == "error":
            error = event.get("error", event)
            code = str(error.get("code", error.get("type", "unknown")))
            message = str(error.get("message", "Unknown error"))
            logger.error("[ElevenLabs] error [%s] %s (session %s)", code, message, session.id)
            await self._fire_error_callbacks(session, code, message)

    # -- Callback helpers --

    async def _fire_callbacks(self, callbacks: list[Any], session: VoiceSession) -> None:
        for cb in callbacks:
            try:
                result = cb(session)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in callback for session %s", session.id)

    async def _fire_audio_callbacks(self, session: VoiceSession, audio: bytes) -> None:
        for cb in self._audio_callbacks:
            try:
                result = cb(session, audio)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in audio callback for session %s", session.id)

    async def _fire_transcription_callbacks(
        self, session: VoiceSession, text: str, role: str, is_final: bool
    ) -> None:
        for cb in self._transcription_callbacks:
            try:
                result = cb(session, text, role, is_final)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in transcription callback for session %s", session.id)

    async def _fire_tool_call_callbacks(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        for cb in self._tool_call_callbacks:
            try:
                result = cb(session, call_id, name, arguments)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in tool call callback for session %s", session.id)

    async def _fire_error_callbacks(self, session: VoiceSession, code: str, message: str) -> None:
        for cb in self._error_callbacks:
            try:
                result = cb(session, code, message)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in error callback for session %s", session.id)
