"""RealtimeVoiceProvider abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from roomkit.voice.realtime.base import RealtimeSession

# Callback type aliases
RealtimeAudioCallback = Callable[[RealtimeSession, bytes], Any]
RealtimeTranscriptionCallback = Callable[[RealtimeSession, str, str, bool], Any]
"""(session, text, role, is_final)"""
RealtimeSpeechStartCallback = Callable[[RealtimeSession], Any]
RealtimeSpeechEndCallback = Callable[[RealtimeSession], Any]
RealtimeToolCallCallback = Callable[[RealtimeSession, str, str, dict[str, Any]], Any]
"""(session, call_id, name, arguments)"""
RealtimeResponseStartCallback = Callable[[RealtimeSession], Any]
RealtimeResponseEndCallback = Callable[[RealtimeSession], Any]
RealtimeErrorCallback = Callable[[RealtimeSession, str, str], Any]
"""(session, code, message)"""


class RealtimeVoiceProvider(ABC):
    """Abstract base class for speech-to-speech AI providers.

    Wraps APIs like OpenAI Realtime and Gemini Live that handle
    audio-in → audio-out with built-in AI, VAD, and transcription.

    The provider manages a bidirectional audio/event stream with the
    AI service. Callbacks are registered for events the provider emits.

    Example:
        provider = OpenAIRealtimeProvider(api_key="sk-...", model="gpt-4o-realtime")

        provider.on_audio(handle_audio)
        provider.on_transcription(handle_transcription)
        provider.on_tool_call(handle_tool_call)

        await provider.connect(session, system_prompt="You are a helpful agent.")
        await provider.send_audio(session, audio_bytes)
        await provider.disconnect(session)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'openai_realtime', 'gemini_live')."""
        ...

    @abstractmethod
    async def connect(
        self,
        session: RealtimeSession,
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
    async def send_audio(self, session: RealtimeSession, audio: bytes) -> None:
        """Send audio data to the provider for processing.

        Args:
            session: The active session.
            audio: Raw PCM audio bytes.
        """
        ...

    @abstractmethod
    async def inject_text(
        self, session: RealtimeSession, text: str, *, role: str = "user"
    ) -> None:
        """Inject text into the conversation (e.g. supervisor guidance).

        Args:
            session: The active session.
            text: Text to inject.
            role: Role for the injected text ('user' or 'system').
        """
        ...

    @abstractmethod
    async def submit_tool_result(
        self, session: RealtimeSession, call_id: str, result: str
    ) -> None:
        """Submit a tool call result back to the provider.

        Args:
            session: The active session.
            call_id: The tool call ID from the on_tool_call callback.
            result: JSON-serialized result string.
        """
        ...

    @abstractmethod
    async def interrupt(self, session: RealtimeSession) -> None:
        """Interrupt the current AI response.

        Args:
            session: The active session.
        """
        ...

    @abstractmethod
    async def disconnect(self, session: RealtimeSession) -> None:
        """Disconnect a session from the provider.

        Args:
            session: The session to disconnect.
        """
        ...

    async def send_event(self, session: RealtimeSession, event: dict[str, Any]) -> None:
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

    async def close(self) -> None:
        """Release all provider resources."""

    # -- Callback registration --

    def on_audio(self, callback: RealtimeAudioCallback) -> None:
        """Register callback for audio output from the provider.

        Args:
            callback: Called with (session, audio_bytes) when the provider
                produces audio output.
        """

    def on_transcription(self, callback: RealtimeTranscriptionCallback) -> None:
        """Register callback for transcription events.

        Args:
            callback: Called with (session, text, role, is_final).
        """

    def on_speech_start(self, callback: RealtimeSpeechStartCallback) -> None:
        """Register callback for speech start detection.

        Args:
            callback: Called with (session) when user speech is detected.
        """

    def on_speech_end(self, callback: RealtimeSpeechEndCallback) -> None:
        """Register callback for speech end detection.

        Args:
            callback: Called with (session) when user speech ends.
        """

    def on_tool_call(self, callback: RealtimeToolCallCallback) -> None:
        """Register callback for tool/function calls from the AI.

        Args:
            callback: Called with (session, call_id, name, arguments).
        """

    def on_response_start(self, callback: RealtimeResponseStartCallback) -> None:
        """Register callback for when the AI starts generating a response.

        Args:
            callback: Called with (session).
        """

    def on_response_end(self, callback: RealtimeResponseEndCallback) -> None:
        """Register callback for when the AI finishes a response.

        Args:
            callback: Called with (session).
        """

    def on_error(self, callback: RealtimeErrorCallback) -> None:
        """Register callback for provider errors.

        Args:
            callback: Called with (session, code, message).
        """
