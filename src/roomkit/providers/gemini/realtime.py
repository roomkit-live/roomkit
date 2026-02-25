"""Google Gemini Live API provider for speech-to-speech conversations."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from pydantic import SecretStr

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

logger = logging.getLogger("roomkit.providers.gemini.realtime")


@dataclass
class _GeminiSessionState:
    """Consolidated per-session state for Gemini Live provider."""

    session: VoiceSession
    live_session: Any = None
    ctxmgr: Any = None
    live_config: Any = None
    receive_task: asyncio.Task[None] | None = None
    resumption_handle: str | None = None
    audio_chunk_count: int = 0
    response_started: bool = False
    audio_buffer: deque[bytes] = field(default_factory=lambda: deque(maxlen=100))
    send_audio_count: int = 0
    error_suppressed: bool = False
    started_at: float = 0.0
    turn_count: int = 0
    tool_result_bytes: int = 0


class _GoAwayError(Exception):
    """Raised when the server sends a GoAway signal to trigger proactive reconnection."""


class GeminiLiveProvider(RealtimeVoiceProvider):
    """Realtime voice provider using the Google Gemini Live API.

    Connects to Gemini's live streaming API for bidirectional
    audio conversations with built-in AI.

    Requires the ``google-genai`` package.

    Example:
        provider = GeminiLiveProvider(api_key="...", model="gemini-2.0-flash-live")
        provider.on_audio(handle_output_audio)
        provider.on_transcription(handle_transcription)

        await provider.connect(session, system_prompt="You are a helpful assistant.")
        await provider.send_audio(session, audio_bytes)
    """

    def __init__(
        self,
        *,
        api_key: str | SecretStr,
        model: str = "gemini-2.5-flash-native-audio-preview-12-2025",
    ) -> None:
        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiLiveProvider. "
                "Install with: pip install 'roomkit[realtime-gemini]'"
            ) from exc

        self._api_key = SecretStr(api_key) if isinstance(api_key, str) else api_key

        # Tighter WebSocket keepalive to detect dead connections faster
        # (defaults are 20s interval / 20s timeout — too slow for realtime audio)
        self._client = _genai.Client(
            api_key=self._api_key.get_secret_value(),
            http_options=_types.HttpOptions(
                async_client_args={
                    "ping_interval": 10,
                    "ping_timeout": 10,
                }
            ),
        )
        self._model = model

        # Consolidated per-session state: session_id -> _GeminiSessionState
        self._sessions: dict[str, _GeminiSessionState] = {}

        # Transcription buffers: accumulate chunks until finished=True
        # Key: (session_id, role) -> list of text chunks
        self._transcription_buffers: dict[tuple[str, str], list[str]] = {}

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
        return "GeminiLiveProvider"

    def _build_config(
        self,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> Any:
        """Build a LiveConnectConfig from parameters.

        Shared by :meth:`connect` and :meth:`reconfigure`.
        """
        from google.genai import types

        pc = provider_config or {}

        config: dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "input_audio_transcription": types.AudioTranscriptionConfig(),
            "output_audio_transcription": types.AudioTranscriptionConfig(),
        }

        # --- Voice / language ---
        speech_kwargs: dict[str, Any] = {}
        if voice:
            speech_kwargs["voice_config"] = types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        language = pc.get("language")
        if language:
            speech_kwargs["language_code"] = language
        if speech_kwargs:
            config["speech_config"] = types.SpeechConfig(**speech_kwargs)

        if system_prompt:
            config["system_instruction"] = system_prompt

        # --- Generation parameters ---
        if temperature is not None:
            config["temperature"] = temperature

        top_p = pc.get("top_p")
        if top_p is not None:
            config["top_p"] = float(top_p)

        top_k = pc.get("top_k")
        if top_k is not None:
            config["top_k"] = float(top_k)

        max_output_tokens = pc.get("max_output_tokens")
        if max_output_tokens is not None:
            config["max_output_tokens"] = int(max_output_tokens)

        seed = pc.get("seed")
        if seed is not None:
            config["seed"] = int(seed)

        # --- Affective dialog (expressive/emotional responses) ---
        enable_affective_dialog = pc.get("enable_affective_dialog")
        if enable_affective_dialog is not None:
            config["enable_affective_dialog"] = bool(enable_affective_dialog)

        # --- Thinking ---
        thinking_budget = pc.get("thinking_budget")
        if thinking_budget is not None:
            config["thinking_config"] = types.ThinkingConfig(
                thinking_budget=int(thinking_budget),
            )

        # --- Proactivity (AI can speak without being prompted) ---
        proactive_audio = pc.get("proactive_audio")
        if proactive_audio is not None:
            config["proactivity"] = types.ProactivityConfig(
                proactive_audio=bool(proactive_audio),
            )

        # --- VAD / realtime input config ---
        vad_kwargs: dict[str, Any] = {}
        start_sensitivity = pc.get("start_of_speech_sensitivity")
        if start_sensitivity:
            vad_kwargs["start_of_speech_sensitivity"] = start_sensitivity.upper()
        end_sensitivity = pc.get("end_of_speech_sensitivity")
        if end_sensitivity:
            vad_kwargs["end_of_speech_sensitivity"] = end_sensitivity.upper()
        silence_duration_ms = pc.get("silence_duration_ms")
        if silence_duration_ms is not None:
            vad_kwargs["silence_duration_ms"] = int(silence_duration_ms)
        prefix_padding_ms = pc.get("prefix_padding_ms")
        if prefix_padding_ms is not None:
            vad_kwargs["prefix_padding_ms"] = int(prefix_padding_ms)

        realtime_input_kwargs: dict[str, Any] = {}
        if vad_kwargs:
            realtime_input_kwargs["automatic_activity_detection"] = (
                types.AutomaticActivityDetection(**vad_kwargs)
            )
        no_interruption = pc.get("no_interruption")
        if no_interruption:
            realtime_input_kwargs["activity_handling"] = "NO_INTERRUPTION"
        if realtime_input_kwargs:
            config["realtime_input_config"] = types.RealtimeInputConfig(**realtime_input_kwargs)

        # --- Tools ---
        if tools:
            genai_tools = []
            for tool in tools:
                genai_tools.append(
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=tool.get("name", ""),
                                description=tool.get("description", ""),
                                parameters=tool.get("parameters"),
                            )
                        ]
                    )
                )
            config["tools"] = genai_tools

        # --- Session resilience ---
        config["session_resumption"] = types.SessionResumptionConfig(handle=None)
        config["context_window_compression"] = types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        )

        return types.LiveConnectConfig(**config)

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
        live_config = self._build_config(
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            provider_config=provider_config,
        )

        ctxmgr = self._client.aio.live.connect(
            model=self._model,
            config=live_config,
        )
        live_session = await ctxmgr.__aenter__()

        state = _GeminiSessionState(
            session=session,
            live_session=live_session,
            ctxmgr=ctxmgr,
            live_config=live_config,
            started_at=time.monotonic(),
        )
        self._sessions[session.id] = state

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        # Start receive loop
        state.receive_task = asyncio.create_task(
            self._receive_loop(session),
            name=f"gemini_live_recv:{session.id}",
        )

        logger.info("Gemini Live session connected: %s", session.id)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        from google.genai import types

        state = self._sessions.get(session.id)
        if state is None or state.live_session is None:
            logger.debug("[Gemini] send_audio: no live session for %s", session.id)
            return

        # Buffer audio while reconnecting instead of dropping it
        if session.state == VoiceSessionState.CONNECTING:
            state.audio_buffer.append(audio)
            return

        # Skip if connection is already closed
        if session.state != VoiceSessionState.ACTIVE:
            logger.debug(
                "[Gemini] send_audio: skipping, state=%s for %s",
                session.state,
                session.id,
            )
            return

        # Log first audio send and then periodically
        state.send_audio_count += 1
        if state.send_audio_count == 1:
            logger.info(
                "[Gemini] send_audio: first chunk (%d bytes) for %s",
                len(audio),
                session.id,
            )
        elif state.send_audio_count % 100 == 0:
            logger.debug(
                "[Gemini] send_audio: %d chunks sent for %s",
                state.send_audio_count,
                session.id,
            )

        try:
            await state.live_session.send_realtime_input(
                audio=types.Blob(data=audio, mime_type="audio/pcm"),
            )
            # Successful send — reset suppression so next failure fires callback
            state.error_suppressed = False
        except Exception as exc:
            # Connection lost — the receive loop will handle reconnection.
            # Don't mark ENDED here; just suppress further sends.
            if session.state == VoiceSessionState.ACTIVE:
                session.state = VoiceSessionState.CONNECTING
            # Fire error callback only once per reconnection cycle
            if not state.error_suppressed:
                state.error_suppressed = True
                await self._fire_error_callbacks(session, "send_audio_failed", str(exc))
            return

    async def inject_text(self, session: VoiceSession, text: str, *, role: str = "user") -> None:
        from google.genai import types

        state = self._sessions.get(session.id)
        if state is None or state.live_session is None:
            return

        await state.live_session.send_client_content(
            turns=types.Content(
                role=role if role in ("user", "model") else "user",
                parts=[types.Part(text=text)],
            ),
            turn_complete=True,
        )

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        import json

        from google.genai import types

        state = self._sessions.get(session.id)
        if state is None or state.live_session is None:
            return

        # Track tool result bytes for debugging
        state.tool_result_bytes += len(result)

        if len(result) > 16384:
            logger.warning(
                "Large tool result (%d chars) for call %s may cause Gemini to "
                "disconnect or silently fail (session %s)",
                len(result),
                call_id,
                session.id,
            )

        try:
            result_dict = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            result_dict = {"result": result}

        await state.live_session.send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    id=call_id,
                    name="",  # Gemini uses ID-based matching
                    response=result_dict,
                )
            ],
        )

    async def interrupt(self, session: VoiceSession) -> None:
        # Gemini doesn't have a direct cancel; send empty to reset
        state = self._sessions.get(session.id)
        if state is None or state.live_session is None:
            return
        logger.debug("Interrupt requested for Gemini session %s (no-op)", session.id)

    async def disconnect(self, session: VoiceSession) -> None:
        import contextlib

        state = self._sessions.pop(session.id, None)
        if state is None:
            session.state = VoiceSessionState.ENDED
            return

        # Cancel receive task
        if state.receive_task is not None:
            state.receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await state.receive_task

        # Clean up transcription buffers
        self._clear_transcription_buffers(session.id)

        # Record session metrics before cleanup
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        if state.started_at:
            uptime_s = time.monotonic() - state.started_at
            telemetry.record_metric(
                "roomkit.realtime.uptime_s",
                uptime_s,
                unit="s",
                attributes={"provider": "gemini", "session_id": session.id},
            )
        telemetry.record_metric(
            "roomkit.realtime.turn_count",
            float(state.turn_count),
            attributes={"provider": "gemini", "session_id": session.id},
        )
        if state.tool_result_bytes:
            telemetry.record_metric(
                "roomkit.realtime.tool_result_bytes",
                float(state.tool_result_bytes),
                attributes={"provider": "gemini", "session_id": session.id},
            )

        # Close live session via context manager exit
        if state.ctxmgr is not None:
            with contextlib.suppress(Exception):
                await state.ctxmgr.__aexit__(None, None, None)
        elif state.live_session is not None:
            with contextlib.suppress(Exception):
                await state.live_session.close()

        logger.info(
            "Gemini session %s disconnected: sent=%d audio chunks, received=%d audio chunks",
            session.id,
            state.send_audio_count,
            state.audio_chunk_count,
        )
        session.state = VoiceSessionState.ENDED

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
        """Reconfigure a session by rebuilding config and reconnecting.

        Uses Gemini's session resumption to preserve conversation
        history while switching system prompt, voice, and tools.
        """
        import contextlib

        state = self._sessions.get(session.id)
        if state is None:
            return

        new_config = self._build_config(
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            provider_config=provider_config,
        )
        state.live_config = new_config
        logger.info(
            "Reconfiguring Gemini session %s (voice=%s)",
            session.id,
            voice,
        )

        # Cancel the old receive task BEFORE reconnecting to prevent it
        # from detecting the disconnection and triggering a second
        # auto-reconnect (double-reconnect bug).
        if state.receive_task is not None:
            state.receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await state.receive_task
            state.receive_task = None

        await self._reconnect(session)

        # Start a fresh receive loop for the new connection.
        state.receive_task = asyncio.create_task(
            self._receive_loop(session),
            name=f"gemini_live_recv:{session.id}",
        )

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            state = self._sessions.get(session_id)
            if state:
                await self.disconnect(state.session)

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

    # -- Receive loop --

    _MAX_RECONNECTS = 5

    async def _receive_loop(self, session: VoiceSession) -> None:
        """Process server events from Gemini Live API.

        If the connection drops mid-session, the loop will attempt to
        reconnect up to ``_MAX_RECONNECTS`` times with exponential back-off.
        """
        reconnect_count = 0

        while True:
            state = self._sessions.get(session.id)
            if state is None:
                return

            # If the session was closed by the user, stop the loop
            if session.state == VoiceSessionState.ENDED:
                return

            # Handle reconnection if needed
            if state.live_session is None:
                reconnect_count += 1
                if reconnect_count > self._MAX_RECONNECTS:
                    logger.error(
                        "Gemini Live session %s: connection lost, max reconnects (%d) reached",
                        session.id,
                        self._MAX_RECONNECTS,
                    )
                    state.audio_buffer.clear()
                    session.state = VoiceSessionState.ENDED
                    await self._fire_error_callbacks(
                        session,
                        "max_reconnects",
                        f"Connection lost after {self._MAX_RECONNECTS} reconnect attempts",
                    )
                    return

                delay = min(0.5 * (2 ** (reconnect_count - 1)), 4.0)
                logger.warning(
                    "Gemini Live connection lost for session %s (attempt %d/%d), "
                    "reconnecting in %.1fs…",
                    session.id,
                    reconnect_count,
                    self._MAX_RECONNECTS,
                    delay,
                )
                await asyncio.sleep(delay)

                # If backoff exceeded buffer duration (~2s), the buffered
                # audio is too stale to be useful — flush it so the AI
                # doesn't process outdated speech.
                if delay > 2.0 and state.audio_buffer:
                    logger.info(
                        "Flushing %d stale audio chunks (%.1fs backoff) for session %s",
                        len(state.audio_buffer),
                        delay,
                        session.id,
                    )
                    state.audio_buffer.clear()

                try:
                    await self._reconnect(session)
                except Exception:
                    logger.exception("Reconnect failed for session %s", session.id)
                    continue

            # Process messages from the current session
            try:
                # local ref to live_session
                live_session = state.live_session
                if live_session is None:
                    continue

                logger.debug(
                    "[Gemini] Waiting for next turn on session %s…",
                    session.id,
                )
                async for response in live_session.receive():
                    # Successful message — reset count
                    reconnect_count = 0
                    await self._handle_server_response(session, response)

                # receive() generator exhausts after each turn_complete —
                # that's normal, just loop back to call receive() again.
                logger.debug(
                    "[Gemini] Turn generator exhausted for session %s, looping for next turn",
                    session.id,
                )

            except asyncio.CancelledError:
                raise
            except _GoAwayError:
                # Server warned it's about to disconnect — proactive reconnect.
                # This does NOT count against the reconnect limit.
                logger.info("Proactive reconnect (GoAway) for session %s", session.id)
                state.live_session = None
                reconnect_count = 0
            except Exception as exc:
                if session.state == VoiceSessionState.ENDED:  # type: ignore[comparison-overlap]
                    return  # state may be mutated by close() during await

                uptime = time.monotonic() - state.started_at if state.started_at else 0.0
                close_code = getattr(exc, "code", None)
                logger.warning(
                    "Gemini session %s disconnected — "
                    "uptime=%.1fs, turns=%d, tool_result_bytes=%d, "
                    "audio_chunks=%d, close_code=%s, error=%s: %s",
                    session.id,
                    uptime,
                    state.turn_count,
                    state.tool_result_bytes,
                    state.audio_chunk_count,
                    close_code,
                    type(exc).__name__,
                    exc,
                )
                state.live_session = None
                # Suppress duplicate send_audio_failed errors during reconnect
                state.error_suppressed = True

    def _clear_transcription_buffers(self, session_id: str) -> None:
        """Remove all transcription buffer entries for a session."""
        for key in list(self._transcription_buffers):
            if key[0] == session_id:
                del self._transcription_buffers[key]

    async def _reconnect(self, session: VoiceSession) -> None:
        """Reconnect to Gemini Live using the stored config."""
        import contextlib

        from google.genai import types

        state = self._sessions.get(session.id)
        if state is None:
            raise RuntimeError("No session state for reconnection")

        # Suppress audio sends during reconnection
        session.state = VoiceSessionState.CONNECTING

        # Tear down old connection
        old_ctxmgr = state.ctxmgr
        state.ctxmgr = None
        state.live_session = None
        if old_ctxmgr:
            with contextlib.suppress(Exception):
                await old_ctxmgr.__aexit__(None, None, None)

        # Clear stale transcription buffers
        self._clear_transcription_buffers(session.id)

        live_config = state.live_config
        if not live_config:
            raise RuntimeError("No stored config for reconnection")

        # Use stored resumption handle to preserve conversation context
        resumption_handle = state.resumption_handle
        if resumption_handle and live_config.session_resumption is not None:
            live_config.session_resumption.handle = resumption_handle
            logger.info("Reconnecting session %s with resumption handle", session.id)
        elif live_config.session_resumption is not None:
            # No handle available — reset to None for a fresh session
            live_config.session_resumption.handle = None

        try:
            ctxmgr = self._client.aio.live.connect(
                model=self._model,
                config=live_config,
            )
            live_session = await ctxmgr.__aenter__()
        except Exception as exc:
            # Fallback: if reconnection with handle failed, try one fresh connect
            if resumption_handle and live_config.session_resumption is not None:
                logger.warning(
                    "Gemini reconnection with handle failed for %s, trying fresh: %s",
                    session.id,
                    exc,
                )
                state.resumption_handle = None
                live_config.session_resumption.handle = None
                ctxmgr = self._client.aio.live.connect(
                    model=self._model,
                    config=live_config,
                )
                live_session = await ctxmgr.__aenter__()
            else:
                raise

        state.ctxmgr = ctxmgr
        state.live_session = live_session
        state.response_started = False

        # Replay buffered audio BEFORE marking ACTIVE — new audio arriving
        # via send_audio() must keep buffering until replay is done,
        # otherwise it bypasses the buffer and arrives out of order.
        if state.audio_buffer:
            logger.info(
                "Replaying %d buffered audio chunks for session %s",
                len(state.audio_buffer),
                session.id,
            )
            try:
                for chunk in state.audio_buffer:
                    await live_session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm"),
                    )
            finally:
                state.audio_buffer.clear()

        # Now safe to accept new audio directly
        session.state = VoiceSessionState.ACTIVE

        # Re-enable error callbacks for the next reconnection cycle
        state.error_suppressed = False

        logger.info("Gemini Live session %s reconnected", session.id)

    async def _handle_server_response(self, session: VoiceSession, response: Any) -> None:
        """Map Gemini Live responses to callbacks."""
        state = self._sessions.get(session.id)
        if state is None:
            return

        # Log every server message for debugging
        parts = []
        if hasattr(response, "data") and response.data:
            parts.append(f"audio={len(response.data)}B")
        if hasattr(response, "server_content") and response.server_content:
            sc = response.server_content
            if hasattr(sc, "model_turn") and sc.model_turn:
                parts.append("model_turn")
            if hasattr(sc, "turn_complete") and sc.turn_complete:
                parts.append("turn_complete")
            if hasattr(sc, "interrupted") and sc.interrupted:
                parts.append("interrupted")
            if hasattr(sc, "input_transcription") and sc.input_transcription:
                parts.append(f"input_tx={sc.input_transcription.text!r}")
            if hasattr(sc, "output_transcription") and sc.output_transcription:
                parts.append(f"output_tx={sc.output_transcription.text!r}")
        if hasattr(response, "tool_call") and response.tool_call:
            parts.append("tool_call")
        if hasattr(response, "voice_activity") and response.voice_activity:
            va = response.voice_activity
            vtype = getattr(va, "voice_activity_type", "?")
            parts.append(f"vad={vtype}")
        if hasattr(response, "go_away") and response.go_away:
            parts.append("go_away")
        if hasattr(response, "session_resumption_update") and response.session_resumption_update:
            parts.append("resumption_update")
        if not parts:
            parts.append(f"unknown_keys={[k for k in dir(response) if not k.startswith('_')]}")
        logger.debug("[Gemini] recv: %s (session %s)", ", ".join(parts), session.id)

        # Handle session resumption updates — store the latest handle
        if hasattr(response, "session_resumption_update") and response.session_resumption_update:
            update = response.session_resumption_update
            if update.resumable and update.new_handle:
                state.resumption_handle = update.new_handle
                logger.debug(
                    "Session resumption handle updated for %s (resumable=%s)",
                    session.id,
                    update.resumable,
                )

        # Handle audio data
        if hasattr(response, "data") and response.data:
            state.audio_chunk_count += 1
            if state.audio_chunk_count % 50 == 1:
                logger.debug(
                    "[Gemini] audio chunk #%d (%d bytes) for session %s",
                    state.audio_chunk_count,
                    len(response.data),
                    session.id,
                )
            await self._fire_audio_callbacks(session, response.data)

        # Handle tool calls
        if hasattr(response, "tool_call") and response.tool_call:
            for fc in response.tool_call.function_calls:
                await self._fire_tool_call_callbacks(
                    session,
                    fc.id,
                    fc.name,
                    dict(fc.args) if fc.args else {},
                )

        # Handle voice activity detection (speech start/end)
        if hasattr(response, "voice_activity") and response.voice_activity:
            va = response.voice_activity
            if hasattr(va, "voice_activity_type") and va.voice_activity_type:
                if va.voice_activity_type == "ACTIVITY_START":
                    logger.info("[VAD] speech_start (session %s)", session.id)
                    await self._fire_callbacks(self._speech_start_callbacks, session)
                elif va.voice_activity_type == "ACTIVITY_END":
                    logger.info("[VAD] speech_end (session %s)", session.id)
                    # Flush user transcription buffer — speech ended
                    await self._flush_transcription_buffer(session, "user")
                    await self._fire_callbacks(self._speech_end_callbacks, session)

        # Handle server content events
        if hasattr(response, "server_content") and response.server_content:
            content = response.server_content

            # Input transcription (user speech-to-text)
            if hasattr(content, "input_transcription") and content.input_transcription:
                tr = content.input_transcription
                if tr.text:
                    await self._handle_transcription_chunk(
                        session, tr.text, "user", bool(tr.finished)
                    )

            # Output transcription (model speech-to-text)
            if hasattr(content, "output_transcription") and content.output_transcription:
                tr = content.output_transcription
                if tr.text:
                    await self._handle_transcription_chunk(
                        session, tr.text, "assistant", bool(tr.finished)
                    )

            # Model started generating (has model_turn with parts)
            if (
                hasattr(content, "model_turn")
                and content.model_turn
                and not state.response_started
            ):
                # Flush user transcription — model responding means user
                # speech is done and transcription should be complete.
                await self._flush_transcription_buffer(session, "user")
                state.response_started = True
                state.audio_chunk_count = 0
                logger.info("[Gemini] response_start (session %s)", session.id)
                await self._fire_callbacks(self._response_start_callbacks, session)

            # Interrupted — user barged in while model was speaking
            if hasattr(content, "interrupted") and content.interrupted:
                logger.info(
                    "[Gemini] INTERRUPTED — AI cut off by barge-in (session %s)",
                    session.id,
                )
                await self._flush_transcription_buffer(session, "assistant")
                state.response_started = False
                await self._fire_callbacks(self._speech_start_callbacks, session)
                await self._fire_callbacks(self._response_end_callbacks, session)

            # Turn complete
            if hasattr(content, "turn_complete") and content.turn_complete:
                # Track turn count
                state.turn_count += 1
                # Flush both transcription buffers — turn ended.
                # User buffer may not have been flushed by ACTIVITY_END
                # (some models don't send voice_activity events, or
                # transcription chunks arrive after ACTIVITY_END).
                logger.info(
                    "[Gemini] turn_complete (session %s, %d audio chunks)",
                    session.id,
                    state.audio_chunk_count,
                )
                await self._flush_transcription_buffer(session, "user")
                await self._flush_transcription_buffer(session, "assistant")
                state.response_started = False
                await self._fire_callbacks(self._response_end_callbacks, session)

        # Handle GoAway LAST — after processing all other data in this message.
        # Raising here breaks out of the receive loop and triggers proactive
        # reconnection with the session resumption handle.
        if hasattr(response, "go_away") and response.go_away:
            time_left = getattr(response.go_away, "time_left", "unknown")
            logger.warning(
                "Gemini GoAway received for session %s (time_left=%s)",
                session.id,
                time_left,
            )
            raise _GoAwayError()

    async def _handle_transcription_chunk(
        self, session: VoiceSession, text: str, role: str, finished: bool
    ) -> None:
        """Accumulate transcription chunks and fire callback when complete.

        Gemini sends transcription text in incremental chunks.  We buffer
        them and fire a single ``is_final=True`` callback with the full
        text when ``finished`` is True.
        """
        key = (session.id, role)
        self._transcription_buffers.setdefault(key, []).append(text)

        if finished:
            full_text = "".join(self._transcription_buffers.pop(key, []))
            if full_text.strip():
                await self._fire_transcription_callbacks(session, full_text, role, True)
        else:
            # Send non-final for real-time display in the voice modal
            await self._fire_transcription_callbacks(session, text, role, False)

    async def _flush_transcription_buffer(self, session: VoiceSession, role: str) -> None:
        """Flush accumulated transcription buffer as a final transcription.

        Called at lifecycle boundaries (turn_complete, speech end) to ensure
        buffered text is emitted even if Gemini never sets ``finished=True``.
        """
        key = (session.id, role)
        chunks = self._transcription_buffers.pop(key, [])
        if chunks:
            full_text = "".join(chunks)
            if full_text.strip():
                logger.debug(
                    "Flushing %s transcription buffer (%d chars) for session %s",
                    role,
                    len(full_text),
                    session.id,
                )
                await self._fire_transcription_callbacks(session, full_text, role, True)

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

    async def _fire_error_callbacks(self, session: VoiceSession, code: str, message: str) -> None:
        for cb in self._error_callbacks:
            try:
                result = cb(session, code, message)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in error callback for session %s", session.id)

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
