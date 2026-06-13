"""OpenAI Realtime API provider for speech-to-speech conversations."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from pydantic import SecretStr

from roomkit.providers.openai.voices import VOICES as _VOICES
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider, VoiceInfo

logger = logging.getLogger("roomkit.providers.openai.realtime")

# Default OpenAI Realtime API endpoint
_DEFAULT_BASE_URL = "wss://api.openai.com/v1/realtime"


class OpenAIRealtimeProvider(RealtimeVoiceProvider):
    """Realtime voice provider using the OpenAI Realtime API.

    Connects via WebSocket to OpenAI's Realtime API (GA), handling
    bidirectional audio streaming with built-in VAD, transcription,
    and AI responses.

    **Audio format constraints (GA API):**

    - ``audio/pcm`` is only accepted at ``24000`` Hz (``rate`` is fixed).
    - ``audio/pcmu`` (G.711 μ-law) and ``audio/pcma`` (G.711 A-law) are
      accepted for 8 kHz telephony; they have no ``rate`` field.
    - Other sample rates are rejected by the API.

    ``input_sample_rate`` / ``output_sample_rate`` must therefore be
    ``24000`` or ``8000``. For 8 kHz, pass ``provider_config["codec"]``
    as ``"pcmu"`` (default) or ``"pcma"``.

    **Note:** the GA API does not accept ``temperature``; passing it
    logs a warning and is ignored.

    Requires the ``websockets`` package.

    Example:
        provider = OpenAIRealtimeProvider(api_key="sk-...", model="gpt-realtime-1.5")
        provider.on_audio(handle_output_audio)
        provider.on_transcription(handle_transcription)

        await provider.connect(session, system_prompt="You are a helpful assistant.")
        await provider.send_audio(session, audio_bytes)
    """

    def __init__(
        self,
        *,
        api_key: str | SecretStr,
        model: str = "gpt-realtime-1.5",
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self._api_key = SecretStr(api_key) if isinstance(api_key, str) else api_key
        self._model = model
        self._base_url = base_url or _DEFAULT_BASE_URL

        # Active WebSocket connections: session_id -> ws
        self._connections: dict[str, Any] = {}
        self._receive_tasks: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, VoiceSession] = {}

        # Track active responses per session to avoid inject_text conflicts
        self._responding: set[str] = set()

    def is_responding(self, session_id: str) -> bool:
        return session_id in self._responding

    @property
    def name(self) -> str:
        return "OpenAIRealtimeProvider"

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """Curated, offline catalog of OpenAI Realtime voices (fixed set)."""
        return list(_VOICES)

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
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "websockets is required for OpenAIRealtimeProvider. "
                "Install with: pip install 'roomkit[realtime-openai]'"
            ) from exc

        if temperature is not None:
            logger.warning(
                "OpenAI Realtime GA API no longer supports the temperature parameter; ignoring"
            )

        # Validate audio rates up-front — building the format objects will
        # raise ValueError for unsupported rates, which is easier to debug
        # before any WebSocket is opened.
        pc = provider_config or {}
        codec = pc.get("codec", "pcmu")
        input_format = self._build_audio_format(input_sample_rate, codec)
        output_format = self._build_audio_format(output_sample_rate, codec)

        url = f"{self._base_url}?model={self._model}"
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
        }

        ws = await asyncio.wait_for(
            websockets.connect(url, additional_headers=headers),
            timeout=30.0,
        )

        self._connections[session.id] = ws
        self._sessions[session.id] = session

        # Build GA session config — audio settings nest under audio.input / audio.output.
        transcription: dict[str, Any] = {
            "model": pc.get("stt_model", "gpt-4o-transcribe"),
        }
        if pc.get("language"):
            transcription["language"] = pc["language"]
        if pc.get("transcription_prompt"):
            transcription["prompt"] = pc["transcription_prompt"]

        # noise_reduction: "near_field" for headphones/close mic,
        # "far_field" for laptop/conference room speakers.
        nr_type = pc.get("noise_reduction", "far_field")
        audio_input: dict[str, Any] = {
            "format": input_format,
            "transcription": transcription,
            "noise_reduction": {"type": nr_type},
        }
        audio_output: dict[str, Any] = {"format": output_format}
        if voice:
            audio_output["voice"] = voice
        if pc.get("speed") is not None:
            audio_output["speed"] = float(pc["speed"])

        # --- Turn detection / VAD (nested under audio.input in GA) ---
        # Default to semantic_vad — it uses a turn detection model that
        # distinguishes real speech from echo/noise residuals, which is
        # critical for laptop mic+speaker setups where AEC can't suppress
        # 100% of the echo.  server_vad (energy-based) is too sensitive.
        td_type = pc.get("turn_detection_type", "semantic_vad" if server_vad else None)
        turn_detection = self._build_turn_detection(td_type, pc)
        if turn_detection is not None:
            audio_input["turn_detection"] = turn_detection

        session_config: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {"input": audio_input, "output": audio_output},
        }

        if system_prompt:
            session_config["instructions"] = system_prompt
        if tools:
            # OpenAI Realtime requires "type": "function" on each tool.
            # Normalize tools that omit it (e.g. from DescribeScreenTool).
            session_config["tools"] = [{**t, "type": t.get("type", "function")} for t in tools]

        logger.info(
            "Sending session.update: turn_detection=%s, voice=%s",
            turn_detection,
            voice,
        )

        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": session_config,
                }
            )
        )

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        # Start receive loop
        self._receive_tasks[session.id] = asyncio.create_task(
            self._receive_loop(session),
            name=f"openai_rt_recv:{session.id}",
        )

        logger.info("OpenAI Realtime session connected: %s", session.id)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio).decode("ascii"),
                }
            )
        )

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

        logger.debug(
            "[OpenAI →] conversation.item.create (input_text, role=%s, silent=%s)",
            role,
            silent,
        )
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": role if role in ("user", "system") else "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )

        # Silent: add to context without requesting a response.
        # The agent sees it on the next user turn.
        if silent:
            logger.debug("[OpenAI] Silent inject — no response.create")
            return

        # Only request a new response if none is in progress.
        if session.id in self._responding:
            logger.debug(
                "[OpenAI] Skipping response.create — response already active (session %s)",
                session.id,
            )
            return

        logger.debug("[OpenAI →] response.create")
        await ws.send(json.dumps({"type": "response.create"}))

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return

        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    },
                }
            )
        )

        logger.debug("[OpenAI →] response.create (after tool result)")
        await ws.send(json.dumps({"type": "response.create"}))

    async def interrupt(self, session: VoiceSession) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        logger.debug("[OpenAI →] response.cancel")
        await ws.send(json.dumps({"type": "response.cancel"}))

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        ws = self._connections.get(session.id)
        if ws is None:
            return
        await ws.send(json.dumps(event))

    async def send_activity_start(self, session: VoiceSession) -> None:
        """No-op — audio flows continuously via input_audio_buffer.append."""
        logger.debug("[OpenAI] activity_start (no-op, session %s)", session.id)

    async def send_activity_end(self, session: VoiceSession) -> None:
        """Commit audio buffer and request a response (manual VAD mode)."""
        ws = self._connections.get(session.id)
        if ws is None:
            return
        logger.debug("[OpenAI →] input_audio_buffer.commit (session %s)", session.id)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        if session.id in self._responding:
            logger.debug("[OpenAI] skip response.create — responding (session %s)", session.id)
            return
        logger.debug("[OpenAI →] response.create (session %s)", session.id)
        await ws.send(json.dumps({"type": "response.create"}))

    async def disconnect(self, session: VoiceSession) -> None:
        import contextlib

        # Cancel receive task
        task = self._receive_tasks.pop(session.id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        # Close WebSocket (short timeout to avoid blocking on close handshake)
        ws = self._connections.pop(session.id, None)
        self._sessions.pop(session.id, None)
        self._responding.discard(session.id)
        if ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.close(), timeout=2.0)

        session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            session = self._sessions.get(session_id)
            if session:
                await self.disconnect(session)

    # -- Turn detection builder --

    @staticmethod
    def _build_turn_detection(td_type: str | None, pc: dict[str, Any]) -> dict[str, Any] | None:
        """Build the turn_detection dict for the GA session config."""
        td: dict[str, Any]
        if td_type == "semantic_vad":
            td = {"type": "semantic_vad"}
            if pc.get("eagerness"):
                td["eagerness"] = pc["eagerness"]
            if pc.get("interrupt_response") is not None:
                td["interrupt_response"] = bool(pc["interrupt_response"])
            if pc.get("create_response") is not None:
                td["create_response"] = bool(pc["create_response"])
            return td
        if td_type == "server_vad":
            td = {"type": "server_vad"}
            if pc.get("threshold") is not None:
                td["threshold"] = float(pc["threshold"])
            if pc.get("silence_duration_ms") is not None:
                td["silence_duration_ms"] = int(pc["silence_duration_ms"])
            if pc.get("prefix_padding_ms") is not None:
                td["prefix_padding_ms"] = int(pc["prefix_padding_ms"])
            if pc.get("idle_timeout_ms") is not None:
                td["idle_timeout_ms"] = int(pc["idle_timeout_ms"])
            if pc.get("interrupt_response") is not None:
                td["interrupt_response"] = bool(pc["interrupt_response"])
            if pc.get("create_response") is not None:
                td["create_response"] = bool(pc["create_response"])
            return td
        return None

    @staticmethod
    def _build_audio_format(rate: int, codec: str) -> dict[str, Any]:
        """Map a PCM sample rate to the GA API's audio format object.

        The GA API only accepts:
          * ``audio/pcm`` at 24000 Hz
          * ``audio/pcmu`` (G.711 μ-law) — 8 kHz implied, no ``rate`` field
          * ``audio/pcma`` (G.711 A-law) — 8 kHz implied, no ``rate`` field
        """
        if rate == 24000:
            return {"type": "audio/pcm", "rate": 24000}
        if rate == 8000:
            if codec not in ("pcmu", "pcma"):
                raise ValueError(
                    f"OpenAI Realtime 8 kHz requires codec='pcmu' or 'pcma', got {codec!r}"
                )
            return {"type": f"audio/{codec}"}
        raise ValueError(
            f"OpenAI Realtime API only accepts 24000 Hz (PCM) or 8000 Hz (G.711), got {rate}"
        )

    # -- Receive loop --

    async def _receive_loop(self, session: VoiceSession) -> None:
        """Process server events from OpenAI Realtime API."""
        ws = self._connections.get(session.id)
        if ws is None:
            return

        try:
            async for raw_message in ws:
                try:
                    event = json.loads(raw_message)
                    await self._handle_server_event(session, event)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from OpenAI for session %s", session.id)
                except Exception:
                    logger.exception("Error handling OpenAI event for session %s", session.id)
        except asyncio.CancelledError:
            raise
        except Exception:
            if session.state == VoiceSessionState.ACTIVE:
                logger.warning("OpenAI WebSocket closed unexpectedly for session %s", session.id)
                session.state = VoiceSessionState.ENDED
                await self._fire(
                    self._error_callbacks,
                    session,
                    "connection_closed",
                    f"WebSocket closed unexpectedly for session {session.id}",
                    label="error",
                )
            else:
                logger.debug("OpenAI WebSocket closed for session %s", session.id)

    # Server event types that carry bulk audio data — skip in protocol log.
    _NOISY_EVENTS = frozenset(
        {
            "response.output_audio.delta",
            "response.output_audio_transcript.delta",
        }
    )

    async def _handle_server_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        """Map OpenAI server events to callbacks."""
        event_type = event.get("type", "")

        # Protocol-level log: show every event except high-frequency audio deltas.
        if event_type not in self._NOISY_EVENTS:
            logger.debug(
                "[OpenAI ←] %s %s",
                event_type,
                {k: v for k, v in event.items() if k not in ("type", "delta", "audio")},
            )

        if event_type == "input_audio_buffer.speech_started":
            logger.info("[VAD] speech_start (session %s)", session.id)
            await self._fire(self._speech_start_callbacks, session, label="speech_start")

        elif event_type == "input_audio_buffer.speech_stopped":
            logger.info("[VAD] speech_end (session %s)", session.id)
            await self._fire(self._speech_end_callbacks, session, label="speech_end")

        elif event_type == "response.output_audio.delta":
            audio_b64 = event.get("delta", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                await self._fire(self._audio_callbacks, session, audio_bytes, label="audio")

        elif event_type == "response.output_audio_transcript.delta":
            text = event.get("delta", "")
            if text:
                await self._fire(
                    self._transcription_callbacks,
                    session,
                    text,
                    "assistant",
                    False,
                    label="transcription",
                )

        elif event_type == "conversation.item.input_audio_transcription.completed":
            text = event.get("transcript", "")
            if text:
                await self._fire(
                    self._transcription_callbacks,
                    session,
                    text,
                    "user",
                    True,
                    label="transcription",
                )

        elif event_type == "response.output_audio_transcript.done":
            text = event.get("transcript", "")
            if text:
                await self._fire(
                    self._transcription_callbacks,
                    session,
                    text,
                    "assistant",
                    True,
                    label="transcription",
                )

        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id", "")
            name = event.get("name", "")
            args_str = event.get("arguments", "{}")
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {"raw": args_str}
            await self._fire(
                self._tool_call_callbacks,
                session,
                call_id,
                name,
                arguments,
                label="tool_call",
            )

        elif event_type == "response.created":
            self._responding.add(session.id)
            logger.info("[OpenAI] response_start (session %s)", session.id)
            await self._fire(self._response_start_callbacks, session, label="response_start")

        elif event_type == "response.done":
            response = event.get("response", {})
            status = response.get("status", "")
            if status == "failed":
                details = response.get("status_details", {})
                err = details.get("error", {})
                err_type = err.get("type", "unknown")
                err_code = err.get("code", "")
                err_message = err.get("message", "Unknown error")
                logger.error(
                    "[OpenAI] response FAILED: type=%s code=%s message=%s (session %s)",
                    err_type,
                    err_code,
                    err_message,
                    session.id,
                )
                await self._fire(
                    self._error_callbacks,
                    session,
                    err_code or err_type,
                    err_message,
                    label="error",
                )
            else:
                logger.info("[OpenAI] response_done status=%s (session %s)", status, session.id)

            # Extract token usage from response.done and record via telemetry
            usage = response.get("usage", {})
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                input_token_details = usage.get("input_token_details", {})
                output_token_details = usage.get("output_token_details", {})
                logger.info(
                    "[OpenAI] usage: input=%d output=%d "
                    "(cached_input=%d, text_input=%d, audio_input=%d, "
                    "text_output=%d, audio_output=%d) (session %s)",
                    input_tokens,
                    output_tokens,
                    input_token_details.get("cached_tokens", 0),
                    input_token_details.get("text_tokens", 0),
                    input_token_details.get("audio_tokens", 0),
                    output_token_details.get("text_tokens", 0),
                    output_token_details.get("audio_tokens", 0),
                    session.id,
                )
                self._record_usage(
                    session,
                    input_tokens,
                    output_tokens,
                    details={
                        "input_token_details": input_token_details,
                        "output_token_details": output_token_details,
                    },
                )

            self._responding.discard(session.id)
            await self._fire(self._response_end_callbacks, session, label="response_end")

        elif event_type == "session.created":
            td_type = (
                event.get("session", {})
                .get("audio", {})
                .get("input", {})
                .get("turn_detection", {})
                .get("type")
            )
            logger.info(
                "[OpenAI] session.created: turn_detection=%s (session %s)",
                td_type,
                session.id,
            )

        elif event_type == "session.updated":
            td_type = (
                event.get("session", {})
                .get("audio", {})
                .get("input", {})
                .get("turn_detection", {})
                .get("type")
            )
            logger.info(
                "[OpenAI] session.updated: turn_detection=%s (session %s)",
                td_type,
                session.id,
            )

        elif event_type == "input_audio_buffer.committed":
            logger.debug("[OpenAI] audio_buffer committed (session %s)", session.id)

        elif event_type == "error":
            error = event.get("error", {})
            code = error.get("code", "unknown")
            message = error.get("message", "Unknown error")
            logger.error("[OpenAI] error [%s] %s (session %s)", code, message, session.id)
            await self._fire(self._error_callbacks, session, code, message, label="error")
