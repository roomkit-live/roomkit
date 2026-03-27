"""RealtimeVoiceProvider abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import VoiceSession

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

# Callback type aliases
RealtimeAudioCallback = Callable[[VoiceSession, bytes], Any]
RealtimeTranscriptionCallback = Callable[[VoiceSession, str, str, bool], Any]
"""(session, text, role, is_final)"""
RealtimeSpeechStartCallback = Callable[[VoiceSession], Any]
RealtimeSpeechEndCallback = Callable[[VoiceSession], Any]
RealtimeToolCallCallback = Callable[[VoiceSession, str, str, dict[str, Any]], Any]
"""(session, call_id, name, arguments)"""
RealtimeResponseStartCallback = Callable[[VoiceSession], Any]
RealtimeResponseEndCallback = Callable[[VoiceSession], Any]
RealtimeErrorCallback = Callable[[VoiceSession, str, str], Any]
"""(session, code, message)"""


class RealtimeVoiceProvider(ABC):
    """Abstract base class for speech-to-speech AI providers.

    Wraps APIs like OpenAI Realtime and Gemini Live that handle
    audio-in → audio-out with built-in AI, VAD, and transcription.

    The provider manages a bidirectional audio/event stream with the
    AI service. Callbacks are registered for events the provider emits.

    Example:
        provider = OpenAIRealtimeProvider(api_key="sk-...", model="gpt-realtime-1.5")

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
    ) -> None:
        """Inject text into the conversation (e.g. supervisor guidance).

        Args:
            session: The active session.
            text: Text to inject.
            role: Role for the injected text ('user' or 'system').
            silent: If True, add to conversation context without
                requesting a response.  The agent sees the text on
                its next turn but does not react immediately.
        """
        ...

    @abstractmethod
    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        """Submit a tool call result back to the provider.

        Args:
            session: The active session.
            call_id: The tool call ID from the on_tool_call callback.
            result: JSON-serialized result string.
        """
        ...

    @abstractmethod
    async def interrupt(self, session: VoiceSession) -> None:
        """Interrupt the current AI response.

        Args:
            session: The active session.
        """
        ...

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

        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            attrs = {"session_id": session.id, "model": getattr(self, "_model", self.name)}
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

    # -- Manual VAD activity signals --

    async def send_activity_start(self, session: VoiceSession) -> None:
        """Signal that user speech activity has started.

        Used in manual VAD mode: local VAD detects speech and the channel
        calls this to inform the provider.  The provider translates this
        into its protocol's activity signal (e.g. Gemini ``ActivityStart``).

        Default: no-op.  Override when the provider supports manual mode.
        """

    async def send_activity_end(self, session: VoiceSession) -> None:
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
