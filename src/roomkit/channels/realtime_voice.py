"""RealtimeVoiceChannel — wraps speech-to-speech AI APIs as a RoomKit channel."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.voice.realtime.base import RealtimeSession, RealtimeSessionState

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider
    from roomkit.voice.realtime.transport import RealtimeAudioTransport

# Tool handler: async callable (session, name, arguments) -> result dict or str
ToolHandler = Callable[
    ["RealtimeSession", str, dict[str, Any]],
    Awaitable[dict[str, Any] | str],
]

logger = logging.getLogger("roomkit.channels.realtime_voice")


class RealtimeVoiceChannel(Channel):
    """Real-time voice channel using speech-to-speech AI providers.

    Wraps APIs like OpenAI Realtime and Gemini Live as a first-class
    RoomKit channel. Audio flows directly between the user's browser
    and the provider; transcriptions are emitted into the Room so
    other channels (supervisor dashboards, logging) see the conversation.

    Category is TRANSPORT so that:
    - ``on_event()`` receives broadcasts (for text injection from supervisors)
    - ``deliver()`` is called but returns empty (customer is on voice)

    Example:
        from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()

        channel = RealtimeVoiceChannel(
            "realtime-1",
            provider=provider,
            transport=transport,
            system_prompt="You are a helpful agent.",
        )
        kit.register_channel(channel)
    """

    channel_type = ChannelType.REALTIME_VOICE
    category = ChannelCategory.TRANSPORT
    direction = ChannelDirection.BIDIRECTIONAL

    def __init__(
        self,
        channel_id: str,
        *,
        provider: RealtimeVoiceProvider,
        transport: RealtimeAudioTransport,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        emit_transcription_events: bool = True,
        tool_handler: ToolHandler | None = None,
    ) -> None:
        """Initialize realtime voice channel.

        Args:
            channel_id: Unique channel identifier.
            provider: The realtime voice provider (OpenAI, Gemini, etc.).
            transport: The audio transport (WebSocket, etc.).
            system_prompt: Default system prompt for the AI.
            voice: Default voice ID for audio output.
            tools: Default tool/function definitions.
            temperature: Default sampling temperature.
            input_sample_rate: Default input audio sample rate (Hz).
            output_sample_rate: Default output audio sample rate (Hz).
            emit_transcription_events: If True, emit final transcriptions
                as RoomEvents so other channels see them.
            tool_handler: Async callable to execute tool calls.
                Signature: ``async (session, name, arguments) -> result``.
                Return a dict or JSON string.  If not set, falls back to
                ``ON_REALTIME_TOOL_CALL`` hooks.
        """
        super().__init__(channel_id)
        self._provider = provider
        self._transport = transport
        self._system_prompt = system_prompt
        self._voice = voice
        self._tools = tools
        self._temperature = temperature
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._emit_transcription_events = emit_transcription_events
        self._tool_handler = tool_handler
        self._framework: RoomKit | None = None

        # Active sessions: session_id -> (session, room_id, binding)
        self._sessions: dict[str, RealtimeSession] = {}
        self._session_rooms: dict[str, str] = {}  # session_id -> room_id

        # Wire internal callbacks
        provider.on_audio(self._on_provider_audio)
        provider.on_transcription(self._on_provider_transcription)
        provider.on_speech_start(self._on_provider_speech_start)
        provider.on_speech_end(self._on_provider_speech_end)
        provider.on_tool_call(self._on_provider_tool_call)
        provider.on_response_start(self._on_provider_response_start)
        provider.on_response_end(self._on_provider_response_end)
        provider.on_error(self._on_provider_error)

        transport.on_audio_received(self._on_client_audio)
        transport.on_client_disconnected(self._on_client_disconnected)

    def set_framework(self, framework: RoomKit) -> None:
        """Set the framework reference for event routing.

        Called automatically when the channel is registered with RoomKit.
        """
        self._framework = framework

    @property
    def info(self) -> dict[str, Any]:
        return {
            "provider": self._provider.name,
            "transport": self._transport.name,
            "system_prompt": self._system_prompt is not None,
            "voice": self._voice,
        }

    # -- Session lifecycle --

    async def start_session(
        self,
        room_id: str,
        participant_id: str,
        connection: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RealtimeSession:
        """Start a new realtime voice session.

        Connects both the transport (client audio) and the provider
        (AI service), then fires a framework event.

        Args:
            room_id: The room to join.
            participant_id: The participant's ID.
            connection: Protocol-specific connection (e.g. WebSocket).
            metadata: Optional session metadata. May include overrides
                for system_prompt, voice, tools, temperature.

        Returns:
            The created RealtimeSession.
        """
        meta = metadata or {}

        session = RealtimeSession(
            id=uuid4().hex,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=self.channel_id,
            state=RealtimeSessionState.CONNECTING,
            metadata=meta,
        )

        # Per-room config overrides from metadata
        system_prompt = meta.get("system_prompt", self._system_prompt)
        voice = meta.get("voice", self._voice)
        tools = meta.get("tools", self._tools)
        temperature = meta.get("temperature", self._temperature)
        provider_config = meta.get("provider_config")

        # Accept client connection
        await self._transport.accept(session, connection)

        # Connect to provider
        await self._provider.connect(
            session,
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            input_sample_rate=self._input_sample_rate,
            output_sample_rate=self._output_sample_rate,
            provider_config=provider_config,
        )

        session.state = RealtimeSessionState.ACTIVE
        self._sessions[session.id] = session
        self._session_rooms[session.id] = room_id

        # Fire framework event
        if self._framework:
            await self._framework._emit_framework_event(
                "voice_session_started",
                room_id=room_id,
                channel_id=self.channel_id,
                data={
                    "session_id": session.id,
                    "participant_id": participant_id,
                    "channel_id": self.channel_id,
                    "provider": self._provider.name,
                },
            )

        logger.info(
            "Realtime session %s started: room=%s, participant=%s, provider=%s",
            session.id,
            room_id,
            participant_id,
            self._provider.name,
        )

        return session

    async def end_session(self, session: RealtimeSession) -> None:
        """End a realtime voice session.

        Disconnects both provider and transport, fires framework event.

        Args:
            session: The session to end.
        """
        room_id = self._session_rooms.get(session.id, session.room_id)

        # Disconnect provider and transport
        try:
            await self._provider.disconnect(session)
        except Exception:
            logger.exception("Error disconnecting provider for session %s", session.id)

        try:
            await self._transport.disconnect(session)
        except Exception:
            logger.exception("Error disconnecting transport for session %s", session.id)

        session.state = RealtimeSessionState.ENDED
        self._sessions.pop(session.id, None)
        self._session_rooms.pop(session.id, None)

        # Fire framework event
        if self._framework:
            await self._framework._emit_framework_event(
                "voice_session_ended",
                room_id=room_id,
                channel_id=self.channel_id,
                data={
                    "session_id": session.id,
                    "participant_id": session.participant_id,
                    "channel_id": self.channel_id,
                },
            )

        logger.info("Realtime session %s ended", session.id)

    # -- Channel ABC --

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        """Not used directly — audio flows via start_session."""
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
            ),
            content=message.content,
        )

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """React to events from other channels — TEXT INJECTION.

        When a supervisor or other channel sends a message, extract the text
        and inject it into the provider session so the AI incorporates it.
        Skips events from this channel (self-loop prevention).
        """
        # Self-loop prevention: skip our own events
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()

        text = self.extract_text(event)
        if not text:
            return ChannelOutput.empty()

        # Determine injection role from event metadata
        inject_role = "system"
        if event.metadata and isinstance(event.metadata, dict):
            inject_role = event.metadata.get("inject_role", "system")

        room_id = event.room_id

        # Inject text into all active sessions for this room
        for session in self._get_room_sessions(room_id):
            try:
                await self._provider.inject_text(session, text, role=inject_role)

                # Fire ON_REALTIME_TEXT_INJECTED hook (async)
                if self._framework:
                    await self._framework.hook_engine.run_async_hooks(
                        room_id,
                        HookTrigger.ON_REALTIME_TEXT_INJECTED,
                        event,
                        context,
                        skip_event_filter=True,
                    )

                logger.info(
                    "Injected text into session %s from channel %s: %.50s",
                    session.id,
                    event.source.channel_id,
                    text,
                )
            except Exception:
                logger.exception(
                    "Error injecting text into session %s",
                    session.id,
                )

        return ChannelOutput.empty()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """No-op delivery — customer is on voice, can't see text."""
        return ChannelOutput.empty()

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.AUDIO, ChannelMediaType.TEXT],
            supports_audio=True,
            custom={"realtime": True, "server_vad": True},
        )

    async def close(self) -> None:
        """End all sessions and close provider + transport."""
        # End all active sessions
        for session in list(self._sessions.values()):
            try:
                await self.end_session(session)
            except Exception:
                logger.exception("Error ending session %s during close", session.id)

        await self._provider.close()
        await self._transport.close()

    # -- Internal callbacks --

    def _on_client_audio(self, session: RealtimeSession, audio: bytes) -> Any:
        """Forward client audio to provider."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._forward_client_audio(session, audio),
            name=f"rt_client_audio:{session.id}",
        )

    async def _forward_client_audio(self, session: RealtimeSession, audio: bytes) -> None:
        if session.state != RealtimeSessionState.ACTIVE:
            return
        try:
            await self._provider.send_audio(session, audio)
        except Exception:
            if session.state == RealtimeSessionState.ACTIVE:
                logger.exception("Error forwarding client audio for session %s", session.id)
            # Don't log again — provider already marked session as ended

    def _on_provider_audio(self, session: RealtimeSession, audio: bytes) -> Any:
        """Forward provider audio to client."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._forward_provider_audio(session, audio),
            name=f"rt_provider_audio:{session.id}",
        )

    async def _forward_provider_audio(self, session: RealtimeSession, audio: bytes) -> None:
        try:
            await self._transport.send_audio(session, audio)
        except Exception:
            logger.exception("Error forwarding provider audio for session %s", session.id)

    def _on_provider_transcription(
        self, session: RealtimeSession, text: str, role: str, is_final: bool
    ) -> Any:
        """Handle transcription from provider."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._process_transcription(session, text, role, is_final),
            name=f"rt_transcription:{session.id}",
        )

    async def _process_transcription(
        self, session: RealtimeSession, text: str, role: str, is_final: bool
    ) -> None:
        """Process a transcription: fire hooks, emit event, send to client."""
        if not self._framework:
            return

        room_id = self._session_rooms.get(session.id)
        if not room_id:
            return

        try:
            context = await self._framework._build_context(room_id)

            # Fire ON_TRANSCRIPTION hooks (sync, can modify/block)
            from roomkit.voice.realtime.events import RealtimeTranscriptionEvent

            transcription_event = RealtimeTranscriptionEvent(
                session=session,
                text=text,
                role=role,  # type: ignore[arg-type]
                is_final=is_final,
            )

            hook_result = await self._framework.hook_engine.run_sync_hooks(
                room_id,
                HookTrigger.ON_TRANSCRIPTION,
                transcription_event,
                context,
                skip_event_filter=True,
            )

            if not hook_result.allowed:
                logger.info("Transcription blocked by hook: %s", hook_result.reason)
                return

            # Use potentially modified text
            final_text = text
            if hook_result.event is not None and isinstance(
                hook_result.event, RealtimeTranscriptionEvent
            ):
                final_text = hook_result.event.text
            elif isinstance(hook_result.event, str):
                final_text = hook_result.event

            # Send transcription to client UI
            await self._transport.send_message(
                session,
                {
                    "type": "transcription",
                    "text": final_text,
                    "role": role,
                    "is_final": is_final,
                },
            )

            # Emit final transcriptions as RoomEvents
            if is_final and self._emit_transcription_events and final_text.strip():
                participant_id = session.participant_id if role == "user" else None
                logger.info(
                    "Emitting transcription as RoomEvent: role=%s, text=%.80s",
                    role,
                    final_text,
                )
                await self._framework.send_event(
                    room_id,
                    self.channel_id,
                    TextContent(body=final_text),
                    participant_id=participant_id,
                    metadata={
                        "voice_session_id": session.id,
                        "source": "realtime_voice",
                        "role": role,
                    },
                )

        except Exception:
            logger.exception(
                "Error processing transcription for session %s (room=%s, is_final=%s)",
                session.id,
                room_id,
                is_final,
            )

    def _on_provider_speech_start(self, session: RealtimeSession) -> Any:
        """Handle speech start from provider's server-side VAD."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._handle_speech_event(session, "start"),
            name=f"rt_speech_start:{session.id}",
        )

    def _on_provider_speech_end(self, session: RealtimeSession) -> Any:
        """Handle speech end from provider's server-side VAD."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._handle_speech_event(session, "end"),
            name=f"rt_speech_end:{session.id}",
        )

    async def _handle_speech_event(self, session: RealtimeSession, event_type: str) -> None:
        """Fire speech hooks and publish ephemeral indicator."""
        if not self._framework:
            return

        room_id = self._session_rooms.get(session.id)
        if not room_id:
            return

        try:
            context = await self._framework._build_context(room_id)
            trigger = (
                HookTrigger.ON_SPEECH_START if event_type == "start" else HookTrigger.ON_SPEECH_END
            )

            await self._framework.hook_engine.run_async_hooks(
                room_id,
                trigger,
                session,
                context,
                skip_event_filter=True,
            )

            # Send speaking indicator to client
            await self._transport.send_message(
                session,
                {
                    "type": "speaking",
                    "speaking": event_type == "start",
                    "who": "user",
                },
            )

            # On speech start, tell client to flush audio queue (barge-in)
            if event_type == "start":
                await self._transport.send_message(
                    session,
                    {
                        "type": "clear_audio",
                    },
                )

        except Exception:
            logger.exception("Error handling speech %s for session %s", event_type, session.id)

    def _on_provider_tool_call(
        self,
        session: RealtimeSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Handle tool call from provider."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._handle_tool_call(session, call_id, name, arguments),
            name=f"rt_tool_call:{session.id}:{call_id}",
        )

    async def _handle_tool_call(
        self,
        session: RealtimeSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Execute a tool call and submit the result to the provider.

        If a ``tool_handler`` was provided, it is called directly.
        Otherwise, the ``ON_REALTIME_TOOL_CALL`` hook is fired.
        """
        room_id = self._session_rooms.get(session.id)

        try:
            result_str: str

            if self._tool_handler is not None:
                # Use the tool handler callback
                logger.info(
                    "Executing tool %s(%s) via handler for session %s",
                    name,
                    call_id,
                    session.id,
                )
                raw = await self._tool_handler(session, name, arguments)
                result_str = raw if isinstance(raw, str) else json.dumps(raw)
            elif self._framework and room_id:
                # Fall back to hooks
                context = await self._framework._build_context(room_id)

                from roomkit.voice.realtime.events import RealtimeToolCallEvent

                tool_event = RealtimeToolCallEvent(
                    session=session,
                    tool_call_id=call_id,
                    name=name,
                    arguments=arguments,
                )

                hook_result = await self._framework.hook_engine.run_sync_hooks(
                    room_id,
                    HookTrigger.ON_REALTIME_TOOL_CALL,
                    tool_event,
                    context,
                    skip_event_filter=True,
                )

                if hook_result.allowed:
                    result_str = json.dumps(
                        hook_result.event.metadata.get("result", {"status": "ok"})
                        if hasattr(hook_result.event, "metadata") and hook_result.event is not None
                        else {"status": "ok"}
                    )
                else:
                    result_str = json.dumps(
                        {"error": hook_result.reason or "Tool call blocked by hook"}
                    )
            else:
                result_str = json.dumps({"error": f"No handler for tool {name}"})

            await self._provider.submit_tool_result(session, call_id, result_str)

            logger.info(
                "Tool call %s(%s) handled for session %s",
                name,
                call_id,
                session.id,
            )

        except Exception:
            logger.exception("Error handling tool call %s for session %s", call_id, session.id)
            try:
                await self._provider.submit_tool_result(
                    session,
                    call_id,
                    json.dumps({"error": "Internal error handling tool call"}),
                )
            except Exception:
                logger.exception("Error submitting fallback tool result")

    def _on_provider_response_start(self, session: RealtimeSession) -> Any:
        """Handle AI response start — publish typing indicator."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._handle_response_indicator(session, is_speaking=True),
            name=f"rt_response_start:{session.id}",
        )

    def _on_provider_response_end(self, session: RealtimeSession) -> Any:
        """Handle AI response end — clear typing indicator."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._handle_response_indicator(session, is_speaking=False),
            name=f"rt_response_end:{session.id}",
        )

    async def _handle_response_indicator(
        self, session: RealtimeSession, *, is_speaking: bool
    ) -> None:
        """Publish ephemeral speaking indicator for the AI."""
        try:
            await self._transport.send_message(
                session,
                {
                    "type": "speaking",
                    "speaking": is_speaking,
                    "who": "assistant",
                },
            )

            # Publish to realtime backend for dashboard subscribers
            if self._framework:
                room_id = self._session_rooms.get(session.id)
                if room_id:
                    await self._framework.publish_typing(
                        room_id,
                        user_id="assistant",
                        is_typing=is_speaking,
                        data={"session_id": session.id, "source": "realtime_voice"},
                    )

        except Exception:
            logger.exception("Error publishing response indicator for session %s", session.id)

    def _on_provider_error(self, session: RealtimeSession, code: str, message: str) -> Any:
        """Handle provider error."""
        logger.error(
            "Realtime provider error for session %s: [%s] %s",
            session.id,
            code,
            message,
        )

    def _on_client_disconnected(self, session: RealtimeSession) -> Any:
        """Handle client disconnection — end the session."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._handle_client_disconnect(session),
            name=f"rt_client_disconnect:{session.id}",
        )

    async def _handle_client_disconnect(self, session: RealtimeSession) -> None:
        """Clean up after client disconnects."""
        if session.id in self._sessions:
            await self.end_session(session)

    # -- Helpers --

    def _get_room_sessions(self, room_id: str) -> list[RealtimeSession]:
        """Get all active sessions for a room."""
        return [s for s in self._sessions.values() if self._session_rooms.get(s.id) == room_id]
