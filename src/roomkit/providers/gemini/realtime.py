"""Google Gemini Live API provider for speech-to-speech conversations."""

from __future__ import annotations

import asyncio
import logging
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

logger = logging.getLogger("roomkit.providers.gemini.realtime")


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
        api_key: str,
        model: str = "gemini-2.5-flash-native-audio-preview-12-2025",
    ) -> None:
        try:
            from google import genai as _genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiLiveProvider. "
                "Install with: pip install 'roomkit[realtime-gemini]'"
            ) from exc

        self._client = _genai.Client(api_key=api_key)
        self._model = model

        # Active sessions: session_id -> genai live session
        self._live_sessions: dict[str, Any] = {}
        self._live_ctxmgrs: dict[str, Any] = {}  # session_id -> context manager
        self._live_configs: dict[str, Any] = {}  # session_id -> LiveConnectConfig
        self._receive_tasks: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, RealtimeSession] = {}

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

        live_config = types.LiveConnectConfig(**config)

        # Store config for potential reconnection
        self._live_configs[session.id] = live_config

        ctxmgr = self._client.aio.live.connect(
            model=self._model,
            config=live_config,
        )
        live_session = await ctxmgr.__aenter__()

        self._live_ctxmgrs[session.id] = ctxmgr
        self._live_sessions[session.id] = live_session
        self._sessions[session.id] = session

        session.state = RealtimeSessionState.ACTIVE
        session.provider_session_id = session.id

        # Start receive loop
        self._receive_tasks[session.id] = asyncio.create_task(
            self._receive_loop(session),
            name=f"gemini_live_recv:{session.id}",
        )

        logger.info("Gemini Live session connected: %s", session.id)

    async def send_audio(self, session: RealtimeSession, audio: bytes) -> None:
        from google.genai import types

        live = self._live_sessions.get(session.id)
        if live is None:
            return

        # Skip if connection is already closed
        if session.state != RealtimeSessionState.ACTIVE:
            return

        try:
            await live.send_realtime_input(
                audio=types.Blob(data=audio, mime_type="audio/pcm"),
            )
        except Exception:
            # Connection lost — the receive loop will handle reconnection.
            # Don't mark ENDED here; just suppress further sends.
            if session.state == RealtimeSessionState.ACTIVE:
                session.state = RealtimeSessionState.CONNECTING
            return

    async def inject_text(
        self, session: RealtimeSession, text: str, *, role: str = "user"
    ) -> None:
        from google.genai import types

        live = self._live_sessions.get(session.id)
        if live is None:
            return

        await live.send_client_content(
            turns=types.Content(
                role=role if role in ("user", "model") else "user",
                parts=[types.Part(text=text)],
            ),
            turn_complete=True,
        )

    async def submit_tool_result(
        self, session: RealtimeSession, call_id: str, result: str
    ) -> None:
        import json

        from google.genai import types

        live = self._live_sessions.get(session.id)
        if live is None:
            return

        try:
            result_dict = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            result_dict = {"result": result}

        await live.send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    id=call_id,
                    name="",  # Gemini uses ID-based matching
                    response=result_dict,
                )
            ],
        )

    async def interrupt(self, session: RealtimeSession) -> None:
        # Gemini doesn't have a direct cancel; send empty to reset
        live = self._live_sessions.get(session.id)
        if live is None:
            return
        logger.debug("Interrupt requested for Gemini session %s (no-op)", session.id)

    async def disconnect(self, session: RealtimeSession) -> None:
        import contextlib

        # Cancel receive task
        task = self._receive_tasks.pop(session.id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        # Close live session via context manager exit
        live = self._live_sessions.pop(session.id, None)
        ctxmgr = self._live_ctxmgrs.pop(session.id, None)
        self._sessions.pop(session.id, None)
        # Clean up transcription buffers and stored config
        for key in list(self._transcription_buffers):
            if key[0] == session.id:
                del self._transcription_buffers[key]
        self._live_configs.pop(session.id, None)
        if ctxmgr is not None:
            with contextlib.suppress(Exception):
                await ctxmgr.__aexit__(None, None, None)
        elif live is not None:
            with contextlib.suppress(Exception):
                await live.close()

        session.state = RealtimeSessionState.ENDED

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
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

    # -- Receive loop --

    _MAX_RECONNECTS = 3

    async def _receive_loop(self, session: RealtimeSession) -> None:
        """Process server events from Gemini Live API.

        ``live.receive()`` is an async generator that yields messages for
        a single model turn (stops after ``turn_complete``).  We call it
        in an outer loop so we keep listening across turns.

        If the connection drops mid-session (common with preview models),
        the loop will attempt to reconnect up to ``_MAX_RECONNECTS`` times
        with exponential back-off.
        """
        reconnect_count = 0

        while True:
            live = self._live_sessions.get(session.id)
            if live is None:
                return

            try:
                while True:
                    async for response in live.receive():
                        reconnect_count = 0  # reset on successful data
                        await self._handle_server_response(session, response)

            except asyncio.CancelledError:
                raise
            except Exception:
                if session.state == RealtimeSessionState.ENDED:
                    return

                reconnect_count += 1
                if reconnect_count > self._MAX_RECONNECTS:
                    logger.error(
                        "Gemini Live session %s: connection lost, max reconnects (%d) reached",
                        session.id,
                        self._MAX_RECONNECTS,
                    )
                    session.state = RealtimeSessionState.ENDED
                    return

                delay = min(0.5 * (2 ** (reconnect_count - 1)), 4.0)
                logger.warning(
                    "Gemini Live connection lost for session %s "
                    "(attempt %d/%d), reconnecting in %.1fs…",
                    session.id,
                    reconnect_count,
                    self._MAX_RECONNECTS,
                    delay,
                )

                await asyncio.sleep(delay)

                try:
                    await self._reconnect(session)
                except Exception:
                    logger.exception("Reconnect failed for session %s", session.id)
                    session.state = RealtimeSessionState.ENDED
                    return

    async def _reconnect(self, session: RealtimeSession) -> None:
        """Reconnect to Gemini Live using the stored config."""
        import contextlib

        # Suppress audio sends during reconnection
        session.state = RealtimeSessionState.CONNECTING

        # Tear down old connection
        old_ctxmgr = self._live_ctxmgrs.pop(session.id, None)
        self._live_sessions.pop(session.id, None)
        if old_ctxmgr:
            with contextlib.suppress(Exception):
                await old_ctxmgr.__aexit__(None, None, None)

        # Clear stale transcription buffers
        for key in list(self._transcription_buffers):
            if key[0] == session.id:
                del self._transcription_buffers[key]

        live_config = self._live_configs.get(session.id)
        if not live_config:
            raise RuntimeError("No stored config for reconnection")

        ctxmgr = self._client.aio.live.connect(
            model=self._model,
            config=live_config,
        )
        live_session = await ctxmgr.__aenter__()

        self._live_ctxmgrs[session.id] = ctxmgr
        self._live_sessions[session.id] = live_session
        session.state = RealtimeSessionState.ACTIVE
        session._response_started = False  # type: ignore[attr-defined]

        logger.info("Gemini Live session %s reconnected", session.id)

    async def _handle_server_response(self, session: RealtimeSession, response: Any) -> None:
        """Map Gemini Live responses to callbacks."""
        # Handle audio data
        if hasattr(response, "data") and response.data:
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
                    await self._fire_callbacks(self._speech_start_callbacks, session)
                elif va.voice_activity_type == "ACTIVITY_END":
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
                and (not hasattr(session, "_response_started") or not session._response_started)
            ):
                # Flush user transcription — model responding means user
                # speech is done and transcription should be complete.
                await self._flush_transcription_buffer(session, "user")
                session._response_started = True  # type: ignore[attr-defined]
                await self._fire_callbacks(self._response_start_callbacks, session)

            # Interrupted — user barged in while model was speaking
            if hasattr(content, "interrupted") and content.interrupted:
                logger.debug("Model interrupted for session %s", session.id)
                await self._flush_transcription_buffer(session, "assistant")
                session._response_started = False  # type: ignore[attr-defined]
                await self._fire_callbacks(self._speech_start_callbacks, session)
                await self._fire_callbacks(self._response_end_callbacks, session)

            # Turn complete
            if hasattr(content, "turn_complete") and content.turn_complete:
                # Flush both transcription buffers — turn ended.
                # User buffer may not have been flushed by ACTIVITY_END
                # (some models don't send voice_activity events, or
                # transcription chunks arrive after ACTIVITY_END).
                await self._flush_transcription_buffer(session, "user")
                await self._flush_transcription_buffer(session, "assistant")
                session._response_started = False  # type: ignore[attr-defined]
                await self._fire_callbacks(self._response_end_callbacks, session)

    async def _handle_transcription_chunk(
        self, session: RealtimeSession, text: str, role: str, finished: bool
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

    async def _flush_transcription_buffer(self, session: RealtimeSession, role: str) -> None:
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

    async def _fire_callbacks(self, callbacks: list[Any], session: RealtimeSession) -> None:
        for cb in callbacks:
            try:
                result = cb(session)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in callback for session %s", session.id)

    async def _fire_audio_callbacks(self, session: RealtimeSession, audio: bytes) -> None:
        for cb in self._audio_callbacks:
            try:
                result = cb(session, audio)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error in audio callback for session %s", session.id)

    async def _fire_transcription_callbacks(
        self, session: RealtimeSession, text: str, role: str, is_final: bool
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
        session: RealtimeSession,
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
