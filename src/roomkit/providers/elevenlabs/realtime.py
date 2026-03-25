"""ElevenLabs Conversational AI realtime provider.

Wraps the official ElevenLabs Python SDK ``Conversation`` class, which
runs in a background thread with its own sync WebSocket.  A custom
``AudioInterface`` bridges audio between the SDK and RoomKit's async
callback system.

Requires the ``elevenlabs`` package::

    pip install 'roomkit[realtime-elevenlabs]'
"""

from __future__ import annotations

import asyncio
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

    ElevenLabs agents are pre-configured on the dashboard with an LLM,
    voice, and system prompt.  Runtime overrides (system prompt, voice,
    temperature) are applied via ``conversation_config_override`` at
    connection time.

    The provider wraps the SDK's synchronous ``Conversation`` class,
    running it in a background thread.  Audio and events are bridged
    to RoomKit's async callback system via the event loop.

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
        self._sessions: dict[str, VoiceSession] = {}
        self._conversations: dict[str, Any] = {}  # SDK Conversation objects
        self._input_callbacks: dict[str, Any] = {}  # audio input_callback from SDK
        self._loops: dict[str, asyncio.AbstractEventLoop] = {}

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
                Conversation,
                ConversationInitiationData,
            )
        except ImportError as exc:
            raise ImportError(
                "elevenlabs is required for ElevenLabsRealtimeProvider. "
                "Install with: pip install 'roomkit[realtime-elevenlabs]'"
            ) from exc

        if tools:
            logger.warning(
                "ElevenLabs agents have tools configured on the dashboard; "
                "the 'tools' parameter passed to connect() is ignored. "
                "Configure client tools via the ElevenLabs agent settings."
            )

        pc = provider_config or {}
        loop = asyncio.get_running_loop()
        self._loops[session.id] = loop

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

        dynamic_variables = pc.get("dynamic_variables")

        init_config = ConversationInitiationData(
            extra_body=extra_body or None,
            conversation_config_override=config_override or None,
            dynamic_variables=dynamic_variables,
        )

        # Create bridge AudioInterface
        bridge = _BridgeAudioInterface(self, session, loop)

        # Create SDK client and Conversation
        client = ElevenLabs(api_key=self._config.api_key)

        conversation = Conversation(
            client,
            self._config.agent_id,
            requires_auth=self._config.requires_auth,
            audio_interface=bridge,
            config=init_config,
            callback_agent_response=lambda text: self._on_agent_response(session, text),
            callback_agent_response_correction=lambda orig, corrected: (
                self._on_agent_response_correction(session, orig, corrected)
            ),
            callback_user_transcript=lambda text: self._on_user_transcript(session, text),
            callback_latency_measurement=lambda ms: logger.debug(
                "ElevenLabs latency: %dms (session %s)", ms, session.id
            ),
        )

        self._sessions[session.id] = session
        self._conversations[session.id] = conversation

        # Start the SDK session (runs in background thread)
        conversation.start_session()

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        logger.info("ElevenLabs Realtime session connected: %s", session.id)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        cb = self._input_callbacks.get(session.id)
        if cb is None:
            return
        # The SDK's input_callback is called from the SDK's thread context,
        # but it's thread-safe (just sends over the WebSocket).
        cb(audio)

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> None:
        # The SDK doesn't expose text injection directly.
        # Log a warning for now.
        logger.warning(
            "inject_text() is not supported by ElevenLabs SDK Conversation; text=%r (session %s)",
            text[:100],
            session.id,
        )

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        # Tool results are handled by the SDK's ClientTools mechanism.
        # The RealtimeVoiceChannel's tool_handler is wired via on_tool_call.
        logger.debug("[ElevenLabs] submit_tool_result not needed — SDK handles tools internally")

    async def interrupt(self, session: VoiceSession) -> None:
        # ElevenLabs handles interruption server-side via VAD.
        logger.debug("[ElevenLabs] interrupt — server-side VAD handles this")

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        raise NotImplementedError(
            "ElevenLabsRealtimeProvider uses the SDK; raw events are not supported"
        )

    async def disconnect(self, session: VoiceSession) -> None:
        conversation = self._conversations.pop(session.id, None)
        if conversation is not None:
            conversation.end_session()
            # Wait for the SDK thread to finish (with timeout)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: conversation.wait_for_session_end()
            )

        self._sessions.pop(session.id, None)
        self._input_callbacks.pop(session.id, None)
        self._loops.pop(session.id, None)
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

    # -- SDK callback handlers (called from SDK thread) --

    def _on_agent_response(self, session: VoiceSession, text: str) -> None:
        """Called by SDK when agent finishes a response."""
        loop = self._loops.get(session.id)
        if loop is None:
            return
        self._responding.discard(session.id)
        asyncio.run_coroutine_threadsafe(
            self._fire_transcription_callbacks(session, text, "assistant", True), loop
        )
        asyncio.run_coroutine_threadsafe(
            self._fire_callbacks(self._response_end_callbacks, session), loop
        )

    def _on_agent_response_correction(
        self, session: VoiceSession, original: str, corrected: str
    ) -> None:
        """Called by SDK when agent corrects a previous response."""
        loop = self._loops.get(session.id)
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._fire_transcription_callbacks(session, corrected, "assistant", True),
            loop,
        )

    def _on_user_transcript(self, session: VoiceSession, text: str) -> None:
        """Called by SDK when user speech is transcribed."""
        loop = self._loops.get(session.id)
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._fire_transcription_callbacks(session, text, "user", True), loop
        )

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


class _BridgeAudioInterface:
    """Bridges the ElevenLabs SDK's sync AudioInterface to RoomKit's async callbacks.

    Called from the SDK's background thread.  Audio output and interruption
    events are dispatched to the async event loop via ``call_soon_threadsafe``.
    """

    def __init__(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._provider = provider
        self._session = session
        self._loop = loop

    def start(self, input_callback: Any) -> None:
        """Store the SDK's audio input callback for send_audio()."""
        self._provider._input_callbacks[self._session.id] = input_callback
        logger.debug("ElevenLabs audio bridge started (session %s)", self._session.id)

    def stop(self) -> None:
        """Clean up when SDK conversation ends."""
        self._provider._input_callbacks.pop(self._session.id, None)
        logger.debug("ElevenLabs audio bridge stopped (session %s)", self._session.id)

    def output(self, audio: bytes) -> None:
        """Called by SDK with agent audio — forward to RoomKit callbacks."""
        session = self._session
        provider = self._provider

        # Fire response_start on first audio chunk
        if session.id not in provider._responding:
            provider._responding.add(session.id)
            asyncio.run_coroutine_threadsafe(
                provider._fire_callbacks(provider._response_start_callbacks, session),
                self._loop,
            )

        asyncio.run_coroutine_threadsafe(
            provider._fire_audio_callbacks(session, audio),
            self._loop,
        )

    def interrupt(self) -> None:
        """Called by SDK on interruption — forward to RoomKit callbacks."""
        session = self._session
        provider = self._provider
        provider._responding.discard(session.id)

        asyncio.run_coroutine_threadsafe(
            provider._fire_callbacks(provider._speech_start_callbacks, session),
            self._loop,
        )
