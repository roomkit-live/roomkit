"""Mock realtime voice provider and transport for testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from roomkit.voice.realtime.base import RealtimeSession, RealtimeSessionState
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
from roomkit.voice.realtime.transport import (
    RealtimeAudioTransport,
    TransportAudioCallback,
    TransportDisconnectCallback,
)


@dataclass
class MockCall:
    """Record of a method call for test assertions."""

    method: str
    args: dict[str, Any] = field(default_factory=dict)


class MockRealtimeProvider(RealtimeVoiceProvider):
    """Mock realtime voice provider for testing.

    Tracks all method calls and provides helpers to simulate
    provider events (transcriptions, audio, tool calls, etc.).

    Example:
        provider = MockRealtimeProvider()

        # Track calls
        await provider.connect(session, system_prompt="Hello")
        assert provider.calls[-1].method == "connect"

        # Simulate events
        await provider.simulate_transcription(session, "Hi there", "user", True)
        await provider.simulate_audio(session, b"audio-data")
        await provider.simulate_tool_call(session, "call-1", "get_weather", {"city": "NYC"})
    """

    def __init__(self) -> None:
        self.calls: list[MockCall] = []
        self.sent_audio: list[tuple[str, bytes]] = []
        self.injected_texts: list[tuple[str, str, str]] = []  # (session_id, text, role)
        self.tool_results: list[tuple[str, str, str]] = []  # (session_id, call_id, result)
        self._sessions: dict[str, RealtimeSession] = {}
        # Callbacks
        self._audio_callbacks: list[RealtimeAudioCallback] = []
        self._transcription_callbacks: list[RealtimeTranscriptionCallback] = []
        self._speech_start_callbacks: list[RealtimeSpeechStartCallback] = []
        self._speech_end_callbacks: list[RealtimeSpeechEndCallback] = []
        self._tool_call_callbacks: list[RealtimeToolCallCallback] = []
        self._response_start_callbacks: list[RealtimeResponseStartCallback] = []
        self._response_end_callbacks: list[RealtimeResponseEndCallback] = []
        self._error_callbacks: list[RealtimeErrorCallback] = []

    @property
    def name(self) -> str:
        return "MockRealtimeProvider"

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
        session.state = RealtimeSessionState.ACTIVE
        session.provider_session_id = f"mock-{session.id}"
        self._sessions[session.id] = session
        self.calls.append(
            MockCall(
                method="connect",
                args={
                    "session_id": session.id,
                    "system_prompt": system_prompt,
                    "voice": voice,
                    "tools": tools,
                    "temperature": temperature,
                    "input_sample_rate": input_sample_rate,
                    "output_sample_rate": output_sample_rate,
                    "server_vad": server_vad,
                },
            )
        )

    async def send_audio(self, session: RealtimeSession, audio: bytes) -> None:
        self.sent_audio.append((session.id, audio))
        self.calls.append(
            MockCall(method="send_audio", args={"session_id": session.id, "size": len(audio)})
        )

    async def inject_text(
        self, session: RealtimeSession, text: str, *, role: str = "user"
    ) -> None:
        self.injected_texts.append((session.id, text, role))
        self.calls.append(
            MockCall(
                method="inject_text",
                args={"session_id": session.id, "text": text, "role": role},
            )
        )

    async def submit_tool_result(
        self, session: RealtimeSession, call_id: str, result: str
    ) -> None:
        self.tool_results.append((session.id, call_id, result))
        self.calls.append(
            MockCall(
                method="submit_tool_result",
                args={"session_id": session.id, "call_id": call_id, "result": result},
            )
        )

    async def interrupt(self, session: RealtimeSession) -> None:
        self.calls.append(MockCall(method="interrupt", args={"session_id": session.id}))

    async def disconnect(self, session: RealtimeSession) -> None:
        session.state = RealtimeSessionState.ENDED
        self._sessions.pop(session.id, None)
        self.calls.append(MockCall(method="disconnect", args={"session_id": session.id}))

    async def close(self) -> None:
        self._sessions.clear()
        self.calls.append(MockCall(method="close"))

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

    # -- Test helpers: simulate provider events --

    async def simulate_audio(self, session: RealtimeSession, audio: bytes) -> None:
        """Simulate audio output from the provider."""
        for cb in self._audio_callbacks:
            result = cb(session, audio)
            if hasattr(result, "__await__"):
                await result

    async def simulate_transcription(
        self,
        session: RealtimeSession,
        text: str,
        role: str = "assistant",
        is_final: bool = True,
    ) -> None:
        """Simulate a transcription event from the provider."""
        for cb in self._transcription_callbacks:
            result = cb(session, text, role, is_final)
            if hasattr(result, "__await__"):
                await result

    async def simulate_speech_start(self, session: RealtimeSession) -> None:
        """Simulate speech start detection."""
        for cb in self._speech_start_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result

    async def simulate_speech_end(self, session: RealtimeSession) -> None:
        """Simulate speech end detection."""
        for cb in self._speech_end_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result

    async def simulate_tool_call(
        self,
        session: RealtimeSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Simulate a tool call from the provider."""
        args = arguments or {}
        for cb in self._tool_call_callbacks:
            result = cb(session, call_id, name, args)
            if hasattr(result, "__await__"):
                await result

    async def simulate_response_start(self, session: RealtimeSession) -> None:
        """Simulate response generation starting."""
        for cb in self._response_start_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result

    async def simulate_response_end(self, session: RealtimeSession) -> None:
        """Simulate response generation ending."""
        for cb in self._response_end_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result

    async def simulate_error(self, session: RealtimeSession, code: str, message: str) -> None:
        """Simulate a provider error."""
        for cb in self._error_callbacks:
            result = cb(session, code, message)
            if hasattr(result, "__await__"):
                await result


class MockRealtimeTransport(RealtimeAudioTransport):
    """Mock realtime audio transport for testing.

    Tracks all method calls and provides helpers to simulate
    client events (audio, disconnection).

    Example:
        transport = MockRealtimeTransport()

        await transport.accept(session, "fake-ws")
        assert transport.calls[-1].method == "accept"

        # Simulate client audio
        await transport.simulate_client_audio(session, b"audio-data")
    """

    def __init__(self) -> None:
        self.calls: list[MockCall] = []
        self.sent_audio: list[tuple[str, bytes]] = []
        self.sent_messages: list[tuple[str, dict[str, Any]]] = []
        self._connections: dict[str, Any] = {}
        # Callbacks
        self._audio_callbacks: list[TransportAudioCallback] = []
        self._disconnect_callbacks: list[TransportDisconnectCallback] = []

    @property
    def name(self) -> str:
        return "MockRealtimeTransport"

    async def accept(self, session: RealtimeSession, connection: Any) -> None:
        self._connections[session.id] = connection
        self.calls.append(MockCall(method="accept", args={"session_id": session.id}))

    async def send_audio(self, session: RealtimeSession, audio: bytes) -> None:
        self.sent_audio.append((session.id, audio))
        self.calls.append(
            MockCall(method="send_audio", args={"session_id": session.id, "size": len(audio)})
        )

    async def send_message(self, session: RealtimeSession, message: dict[str, Any]) -> None:
        self.sent_messages.append((session.id, message))
        self.calls.append(
            MockCall(
                method="send_message",
                args={"session_id": session.id, "type": message.get("type")},
            )
        )

    async def disconnect(self, session: RealtimeSession) -> None:
        self._connections.pop(session.id, None)
        self.calls.append(MockCall(method="disconnect", args={"session_id": session.id}))

    async def close(self) -> None:
        self._connections.clear()
        self.calls.append(MockCall(method="close"))

    def set_input_muted(self, session: RealtimeSession, muted: bool) -> None:
        self.calls.append(
            MockCall(
                method="set_input_muted",
                args={"session_id": session.id, "muted": muted},
            )
        )

    # -- Callback registration --

    def on_audio_received(self, callback: TransportAudioCallback) -> None:
        self._audio_callbacks.append(callback)

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)

    # -- Test helpers --

    async def simulate_client_audio(self, session: RealtimeSession, audio: bytes) -> None:
        """Simulate audio received from the client browser."""
        for cb in self._audio_callbacks:
            result = cb(session, audio)
            if hasattr(result, "__await__"):
                await result

    async def simulate_client_disconnect(self, session: RealtimeSession) -> None:
        """Simulate the client disconnecting."""
        self._connections.pop(session.id, None)
        for cb in self._disconnect_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result
