"""ElevenLabs Conversational AI realtime provider.

Uses the official ElevenLabs Python SDK ``AsyncConversation`` class with
a custom ``AsyncAudioInterface`` that bridges audio between the SDK and
RoomKit's callback system.

Requires the ``elevenlabs`` package (v2.40+)::

    pip install 'roomkit[realtime-elevenlabs]'
"""

from __future__ import annotations

import asyncio
import contextlib
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
        self._config = config

        # Per-session state
        self._sessions: dict[str, VoiceSession] = {}
        self._conversations: dict[str, Any] = {}  # AsyncConversation objects
        self._input_callbacks: dict[str, Any] = {}  # async audio input callbacks

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
            from elevenlabs import ElevenLabs
            from elevenlabs.conversational_ai.conversation import (
                AsyncConversation,
                ConversationInitiationData,
            )
        except ImportError as exc:
            raise ImportError(
                "elevenlabs>=2.40 is required for ElevenLabsRealtimeProvider. "
                "Install with: pip install 'roomkit[realtime-elevenlabs]'"
            ) from exc

        if tools:
            logger.warning(
                "ElevenLabs agents have tools configured on the dashboard; "
                "the 'tools' parameter passed to connect() is ignored. "
                "Configure client tools via the ElevenLabs agent settings."
            )

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

        # Create SDK client (pass base_url for regional endpoints)
        base_url = self._config.base_url.replace("wss://", "https://").replace("ws://", "http://")
        client = ElevenLabs(api_key=self._config.api_key, base_url=base_url)

        conversation = AsyncConversation(
            client,
            self._config.agent_id,
            requires_auth=self._config.requires_auth,
            audio_interface=bridge,
            config=init_config,
            callback_agent_response=self._make_agent_response_cb(session),
            callback_agent_response_correction=self._make_correction_cb(session),
            callback_user_transcript=self._make_user_transcript_cb(session),
            callback_latency_measurement=self._make_latency_cb(session),
        )

        self._sessions[session.id] = session
        self._conversations[session.id] = conversation

        # Start the SDK session (creates async task internally)
        try:
            await conversation.start_session()
        except Exception:
            self._sessions.pop(session.id, None)
            self._conversations.pop(session.id, None)
            raise

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
        # Tool results are handled by the SDK's ClientTools mechanism
        logger.debug("[ElevenLabs] submit_tool_result — SDK handles tools internally")

    async def interrupt(self, session: VoiceSession) -> None:
        # ElevenLabs handles interruption server-side via VAD
        conversation = self._conversations.get(session.id)
        if conversation is not None:
            await conversation.register_user_activity()

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        raise NotImplementedError(
            "ElevenLabsRealtimeProvider uses the SDK; raw events are not supported"
        )

    async def disconnect(self, session: VoiceSession) -> None:
        conversation = self._conversations.pop(session.id, None)
        if conversation is not None:
            await conversation.end_session()
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(conversation.wait_for_session_end(), timeout=5.0)

        self._sessions.pop(session.id, None)
        self._input_callbacks.pop(session.id, None)
        self._responding.discard(session.id)

        session.state = VoiceSessionState.ENDED
        logger.info("ElevenLabs session disconnected: %s", session.id)

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

    # -- Async callback factories for SDK --

    def _make_agent_response_cb(self, session: VoiceSession) -> Any:
        async def cb(text: str) -> None:
            self._responding.discard(session.id)
            await self._fire_transcription_callbacks(session, text, "assistant", True)
            await self._fire_callbacks(self._response_end_callbacks, session)

        return cb

    def _make_correction_cb(self, session: VoiceSession) -> Any:
        async def cb(original: str, corrected: str) -> None:
            await self._fire_transcription_callbacks(session, corrected, "assistant", True)

        return cb

    def _make_user_transcript_cb(self, session: VoiceSession) -> Any:
        async def cb(text: str) -> None:
            await self._fire_transcription_callbacks(session, text, "user", True)
            # Transcript arrival signals the user finished speaking
            await self._fire_callbacks(self._speech_end_callbacks, session)

        return cb

    def _make_latency_cb(self, session: VoiceSession) -> Any:
        async def cb(latency: int) -> None:
            logger.debug("ElevenLabs latency: %dms (session %s)", latency, session.id)

        return cb

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


class _AsyncBridgeAudioInterface:
    """Bridges the ElevenLabs SDK's AsyncAudioInterface to RoomKit callbacks.

    All methods are async, matching the SDK's ``AsyncAudioInterface`` contract.
    Runs in the same event loop as the rest of RoomKit — no thread bridging.
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

        # Fire response_start on first audio chunk
        if session.id not in provider._responding:
            provider._responding.add(session.id)
            await provider._fire_callbacks(provider._response_start_callbacks, session)

        await provider._fire_audio_callbacks(session, audio)

    async def interrupt(self) -> None:
        """Called by SDK when user interrupted agent playback."""
        self._provider._responding.discard(self._session.id)
