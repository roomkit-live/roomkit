"""Google Gemini Live API provider for speech-to-speech conversations."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import SecretStr

from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.gemini.realtime")

_MAX_INJECT_TEXT_LENGTH = 32_000
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitize_gemini_text(text: str) -> str:
    """Sanitize text for safe injection into the Gemini Live API.

    Strips null bytes, control characters (except whitespace),
    unpaired surrogates, and truncates to max length.
    """
    text = _CONTROL_CHAR_RE.sub("", text)
    text = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    if len(text) > _MAX_INJECT_TEXT_LENGTH:
        text = text[:_MAX_INJECT_TEXT_LENGTH] + "... [truncated]"
    return text


class _TranscriptionBuffer:
    """Accumulates Gemini transcription chunks until finished=True.

    Gemini sends transcription text in incremental chunks.  This buffer
    collects them per (session_id, role) and emits the concatenated
    result when ``flush`` is called or when ``append`` receives a
    ``finished=True`` chunk.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[str, str], list[str]] = {}

    def append(self, session_id: str, role: str, text: str, finished: bool) -> str | None:
        """Add a chunk. Returns the full text if ``finished``, else None."""
        key = (session_id, role)
        self._buffers.setdefault(key, []).append(text)
        if finished:
            full = "".join(self._buffers.pop(key, []))
            return full if full.strip() else None
        return None

    def flush(self, session_id: str, role: str) -> str | None:
        """Flush the buffer for a (session, role) pair. Returns text or None."""
        chunks = self._buffers.pop((session_id, role), [])
        if chunks:
            full = "".join(chunks)
            return full if full.strip() else None
        return None

    def clear_session(self, session_id: str) -> None:
        """Remove all buffers for a session."""
        for key in [k for k in self._buffers if k[0] == session_id]:
            del self._buffers[key]


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
    user_speech_active: bool = False
    audio_buffer: deque[bytes] = field(default_factory=lambda: deque(maxlen=100))
    error_suppressed: bool = False
    started_at: float = 0.0
    turn_count: int = 0
    tool_result_bytes: int = 0
    input_sample_rate: int = 16000
    pending_tool_calls: int = 0
    queued_injections: list[tuple[bytes, str, str, bool]] = field(default_factory=list)
    realtime_input_sent: bool = False
    queued_text_injections: list[tuple[str, str, bool]] = field(default_factory=list)
    # Effective config values, kept in sync across connect + reconfigure
    # so partial reconfigures (e.g. system_prompt-only) preserve the
    # other fields. Without these, ``_build_config`` (which treats
    # ``None`` as "absent") would silently wipe the unspecified fields
    # — most notably the tools list, leaving the model with no
    # functions to call after a skill activation.
    system_prompt: str | None = None
    voice: str | None = None
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None


class _GoAwayError(Exception):
    """Raised when the server sends a GoAway signal to trigger proactive reconnection."""


class GeminiLiveProvider(RealtimeVoiceProvider):
    """Realtime voice provider using the Google Gemini Live API.

    Connects to Gemini's live streaming API for bidirectional
    audio conversations with built-in AI.

    Requires the ``google-genai`` package.

    Example:
        provider = GeminiLiveProvider(api_key="...")
        provider.on_audio(handle_output_audio)
        provider.on_transcription(handle_transcription)

        await provider.connect(session, system_prompt="You are a helpful assistant.")
        await provider.send_audio(session, audio_bytes)
    """

    def __init__(
        self,
        *,
        api_key: str | SecretStr,
        model: str = "gemini-3.1-flash-live-preview",
    ) -> None:
        super().__init__()

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

        self._transcription_buffer = _TranscriptionBuffer()

        # Hot-path caches (instance-level to avoid shared mutable class state)
        self._blob_cls: Any = None
        self._mime_cache: dict[int, str] = {}

    @property
    def name(self) -> str:
        return "GeminiLiveProvider"

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        # gemini-3.x live models reject send_client_content with WS 1007
        # after the first model turn and offer no documented dynamic
        # system_instruction update. Their session_resumption is also
        # fragile with non-trivial system prompts. Disable mid-session
        # reconfigure for the whole 3.x family so callers route changes
        # through session-start delivery instead. 2.5-era models keep
        # the old behavior.
        return not (
            self._model.startswith("gemini-3.") or self._model.startswith("gemini-3-")
        )

    def _get_active_state(self, session: VoiceSession) -> _GeminiSessionState | None:
        """Return session state if the session is connected, else None."""
        state = self._sessions.get(session.id)
        if state is None or state.live_session is None:
            return None
        return state

    def _build_config(
        self,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        provider_config: dict[str, Any] | None = None,
        server_vad: bool = True,
    ) -> Any:
        """Build a LiveConnectConfig from parameters.

        Shared by :meth:`connect` and :meth:`reconfigure`.
        """
        from google.genai import types

        pc = provider_config or {}

        # Response modalities: ["AUDIO"], ["TEXT"], or ["AUDIO", "TEXT"]
        # Future: ["VIDEO"] when supported by the API.
        response_modalities = pc.get("response_modalities", ["AUDIO"])

        config: dict[str, Any] = {
            "response_modalities": response_modalities,
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
            val = start_sensitivity.upper()
            # Accept short form "LOW"/"HIGH" → expand to full enum name
            if val in ("LOW", "HIGH"):
                val = f"START_SENSITIVITY_{val}"
            vad_kwargs["start_of_speech_sensitivity"] = val
        end_sensitivity = pc.get("end_of_speech_sensitivity")
        if end_sensitivity:
            val = end_sensitivity.upper()
            if val in ("LOW", "HIGH"):
                val = f"END_SENSITIVITY_{val}"
            vad_kwargs["end_of_speech_sensitivity"] = val
        silence_duration_ms = pc.get("silence_duration_ms")
        if silence_duration_ms is not None:
            vad_kwargs["silence_duration_ms"] = int(silence_duration_ms)
        prefix_padding_ms = pc.get("prefix_padding_ms")
        if prefix_padding_ms is not None:
            vad_kwargs["prefix_padding_ms"] = int(prefix_padding_ms)

        # When server_vad=True (default), enable automatic activity detection
        # so the provider's server-side VAD handles speech boundaries.
        # When server_vad=False (manual mode), disable it — the channel sends
        # activityStart/activityEnd from local VAD instead.
        if server_vad:
            aad = types.AutomaticActivityDetection(**vad_kwargs)
        else:
            aad = types.AutomaticActivityDetection(disabled=True)
            logger.info("Server-side VAD disabled — using manual mode (local VAD)")

        realtime_input_kwargs: dict[str, Any] = {
            "automatic_activity_detection": aad,
        }
        no_interruption = pc.get("no_interruption")
        if no_interruption:
            realtime_input_kwargs["activity_handling"] = "NO_INTERRUPTION"
        config["realtime_input_config"] = types.RealtimeInputConfig(**realtime_input_kwargs)

        # --- Tools ---
        if tools:
            from roomkit.providers.gemini.schema import clean_gemini_schema

            genai_tools = []
            for tool in tools:
                genai_tools.append(
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=tool.get("name", ""),
                                description=tool.get("description", ""),
                                parameters=cast(Any, clean_gemini_schema(tool.get("parameters"))),
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

        # Debug dump of what we're handing to Gemini Live. Gated on
        # ``ROOMKIT_GEMINI_DEBUG=1`` so prod logs stay clean. Useful
        # for diagnosing why one session invokes tools and another
        # doesn't — compares cleanly across sessions when copy/pasted.
        import os as _os

        if _os.environ.get("ROOMKIT_GEMINI_DEBUG", "").lower() in {"1", "true", "yes"}:
            self._log_config_dump(config, system_prompt, tools)

        return types.LiveConnectConfig(**config)

    def _log_config_dump(
        self,
        config: dict[str, Any],
        system_prompt: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """Dump the LiveConnectConfig parameters for diagnostics.

        Called from ``_build_config`` when ``ROOMKIT_GEMINI_DEBUG`` is on.
        Logs at INFO so it's visible without raising the root level.
        Sister method ``_log_event`` dumps every server event coming the
        other way (text deltas, tool calls, transcription, errors) so
        you can see the full request/response cycle in the same log.
        """
        import json as _json

        from roomkit.providers.gemini.schema import clean_gemini_schema

        # ── System prompt ────────────────────────────────────────────
        # Full body, line-prefixed, with a length header so it's easy
        # to see at a glance. Capped at ~12 KB (~3 K tokens) — beyond
        # that we lose the per-line layout in container logs.
        prompt_len = len(system_prompt) if system_prompt else 0
        logger.info("ROOMKIT_GEMINI_DEBUG: ===== system_prompt (len=%d) =====", prompt_len)
        if system_prompt:
            shown = system_prompt[:12000]
            for line in shown.splitlines():
                logger.info("ROOMKIT_GEMINI_DEBUG: | %s", line)
            if prompt_len > 12000:
                logger.info(
                    "ROOMKIT_GEMINI_DEBUG: | … [truncated %d chars] …",
                    prompt_len - 12000,
                )
        logger.info("ROOMKIT_GEMINI_DEBUG: ===== /system_prompt =====")

        # ── Tools ────────────────────────────────────────────────────
        # All tool names + one-line descriptions. With 30+ tools this
        # is the single most useful piece of context when diagnosing
        # "model didn't pick the right tool" — you can see at a glance
        # what the model was actually shown.
        tool_count = len(tools or [])
        logger.info("ROOMKIT_GEMINI_DEBUG: ===== tools (count=%d) =====", tool_count)
        properties_without_type: list[str] = []
        for tool in tools or []:
            name = tool.get("name", "?")
            desc = (tool.get("description") or "").splitlines()[0][:140]
            cleaned = clean_gemini_schema(tool.get("parameters")) or {}
            param_count = len(cleaned.get("properties") or {})
            required = cleaned.get("required") or []
            logger.info(
                "ROOMKIT_GEMINI_DEBUG: | %-44s params=%d required=%d desc=%s",
                name,
                param_count,
                len(required),
                desc,
            )
            for prop_name, prop_schema in (cleaned.get("properties") or {}).items():
                if isinstance(prop_schema, dict) and "type" not in prop_schema:
                    properties_without_type.append(f"{name}.{prop_name}")
        logger.info("ROOMKIT_GEMINI_DEBUG: ===== /tools =====")

        if properties_without_type:
            logger.warning(
                "ROOMKIT_GEMINI_DEBUG: %d tool properties have NO type after cleaning "
                "(Gemini will silently reject these tools): %s",
                len(properties_without_type),
                properties_without_type[:20],
            )

        # ── Other config ─────────────────────────────────────────────
        speech = config.get("speech_config")
        voice_name = ""
        if speech is not None:
            vc = getattr(speech, "voice_config", None)
            if vc is not None:
                pre = getattr(vc, "prebuilt_voice_config", None)
                if pre is not None:
                    voice_name = getattr(pre, "voice_name", "") or ""
        logger.info(
            "ROOMKIT_GEMINI_DEBUG: voice=%r temperature=%s response_modalities=%s "
            "session_resumption=on context_window_compression=sliding",
            voice_name,
            config.get("temperature"),
            config.get("response_modalities"),
        )

        # ── First tool's full cleaned schema (for paranoid review) ──
        if tools:
            first = tools[0]
            cleaned_first = {
                "name": first.get("name"),
                "description": (first.get("description") or "")[:200],
                "parameters": clean_gemini_schema(first.get("parameters")) or {},
            }
            logger.info(
                "ROOMKIT_GEMINI_DEBUG: first_tool_full_schema=%s",
                _json.dumps(cleaned_first)[:2000],
            )

    def _log_event(self, session_id: str, label: str, **fields: Any) -> None:
        """Log a single server event from Gemini Live for diagnostics.

        Called from the receive loop and message handlers. ``label`` is
        a short tag (text_delta, tool_call, transcription, error, …) and
        ``fields`` are the salient attributes to log. Gated on the same
        ``ROOMKIT_GEMINI_DEBUG`` env var as the config dump.
        """
        import os as _os

        if _os.environ.get("ROOMKIT_GEMINI_DEBUG", "").lower() not in {"1", "true", "yes"}:
            return
        rendered = " ".join(f"{k}={v!r}" for k, v in fields.items())
        logger.info(
            "ROOMKIT_GEMINI_DEBUG: <<< %s session=%s %s",
            label,
            session_id[:8],
            rendered,
        )

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
            server_vad=server_vad,
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
            input_sample_rate=input_sample_rate,
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
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
        state = self._sessions.get(session.id)
        if state is None:
            return

        # Buffer audio while reconnecting instead of dropping it
        if state.live_session is None or session.state == VoiceSessionState.CONNECTING:
            if session.state == VoiceSessionState.CONNECTING:
                state.audio_buffer.append(audio)
            return

        # Skip if connection is already closed
        if session.state != VoiceSessionState.ACTIVE:
            return

        try:
            await state.live_session.send_realtime_input(
                audio=self._make_audio_blob(audio, state.input_sample_rate),
            )
            # Mark that realtime input has been used — send_client_content is
            # no longer safe for this session (interleaving causes 1007 disconnects).
            state.realtime_input_sent = True
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
                await self._fire(
                    self._error_callbacks, session, "send_audio_failed", str(exc), label="error"
                )
            return

    async def start_audio_stream(self, session: VoiceSession) -> None:
        """Open the realtime audio input path by sending 20 ms of silence.

        Gemini Live exposes two protocol paths on the same WebSocket —
        ``send_client_content`` for structured text turns and
        ``send_realtime_input`` for streaming audio.  Interleaving them
        after audio has started causes the server to close the socket
        with code 1008/1007 on some preview models.  Sending one frame
        of silence up-front commits the session to the realtime path so
        later :meth:`inject_text` calls stay interleave-safe.

        No-op if the session is not active or the stream is already open.
        """
        state = self._get_active_state(session)
        if state is None or state.realtime_input_sent:
            return
        if state.live_session is None:
            return
        silence = b"\x00" * (state.input_sample_rate // 50)  # 20 ms PCM-16
        await state.live_session.send_realtime_input(
            audio=self._make_audio_blob(silence, state.input_sample_rate)
        )
        state.realtime_input_sent = True

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> None:
        if (state := self._get_active_state(session)) is None:
            return

        # Sanitize to prevent 1007 disconnects from control chars / surrogates.
        text = _sanitize_gemini_text(text)
        if not text.strip():
            logger.debug(
                "inject_text: empty after sanitization, skipping (session %s)",
                session.id,
            )
            return

        # Queue when tool results are pending — Gemini rejects input while
        # waiting for function responses (same guard as inject_image).
        if state.pending_tool_calls > 0:
            logger.debug(
                "Queuing text injection for session %s (pending tool calls: %d)",
                session.id,
                state.pending_tool_calls,
            )
            state.queued_text_injections.append((text, role, silent))
            return

        await self._send_text(state, text, role, silent)

    async def _send_text(
        self,
        state: _GeminiSessionState,
        text: str,
        role: str,
        silent: bool,
    ) -> None:
        from google.genai import types

        effective_role = role if role in ("user", "model") else "user"
        if effective_role != role:
            logger.debug(
                "inject_text session %s: role %r not supported by Gemini, coerced to %r",
                state.session.id,
                role,
                effective_role,
            )
        logger.debug(
            "inject_text session %s: role=%s (original=%s), silent=%s, "
            "realtime=%s, len=%d, preview=%.200s",
            state.session.id,
            effective_role,
            role,
            silent,
            state.realtime_input_sent,
            len(text),
            text,
        )

        if not state.realtime_input_sent:
            # No audio sent yet — send_client_content is safe and gives full
            # control over role and turn_complete semantics.
            await state.live_session.send_client_content(
                turns=types.Content(
                    role=effective_role,
                    parts=[types.Part(text=text)],
                ),
                turn_complete=not silent,
            )
            return

        # Audio is flowing — must use send_realtime_input to avoid 1007
        # disconnects from interleaving send_client_content with realtime input.
        # Limitations: no role parameter, no turn_complete control.
        if effective_role == "model":
            logger.warning(
                "inject_text session %s: role='model' not supported via "
                "send_realtime_input — sending as user context instead",
                state.session.id,
            )
            text = f"[Assistant previously said] {text}"

        if silent:
            text = f"[Context update, do not respond to this] {text}"
            logger.debug(
                "inject_text session %s: silent mode is best-effort via "
                "send_realtime_input (model may still respond)",
                state.session.id,
            )

        await state.live_session.send_realtime_input(text=text)

    async def inject_image(
        self,
        session: VoiceSession,
        image_data: bytes,
        mime_type: str = "image/png",
        *,
        prompt: str = "",
        silent: bool = False,
    ) -> None:
        if (state := self._get_active_state(session)) is None:
            return

        # Gemini Live API does not accept client_content while a tool response
        # is pending.  Queue the injection and flush after submit_tool_result.
        if state.pending_tool_calls > 0:
            logger.debug(
                "Queuing image injection for session %s (pending tool calls: %d)",
                session.id,
                state.pending_tool_calls,
            )
            state.queued_injections.append((image_data, mime_type, prompt, silent))
            return

        await self._send_image(state, image_data, mime_type, prompt, silent)

    async def _send_image(
        self,
        state: _GeminiSessionState,
        image_data: bytes,
        mime_type: str,
        prompt: str,
        silent: bool,
    ) -> None:
        from google.genai import types

        # Sanitize once, before branching.
        if prompt:
            prompt = _sanitize_gemini_text(prompt)
            if not prompt.strip():
                prompt = ""

        if not state.realtime_input_sent:
            # No audio sent yet — send_client_content is safe.
            parts: list[types.Part] = []
            if prompt:
                parts.append(types.Part(text=prompt))
            parts.append(types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_data)))
            await state.live_session.send_client_content(
                turns=types.Content(role="user", parts=parts),
                turn_complete=not silent,
            )
            return

        # Audio is flowing — use send_realtime_input to avoid 1007
        # disconnects.  The SDK only accepts one argument per call,
        # so text prompt and media are sent as separate messages.
        if prompt:
            if silent:
                prompt = f"[Context update, do not respond to this] {prompt}"
            await state.live_session.send_realtime_input(text=prompt)
        elif silent:
            # No prompt but silent — send a standalone instruction so the
            # model doesn't react to the image (no turn_complete equivalent
            # on the realtime path).
            await state.live_session.send_realtime_input(
                text="[Context update, do not respond to this image]"
            )

        await state.live_session.send_realtime_input(
            media=types.Blob(mime_type=mime_type, data=image_data),
        )

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        import json

        from google.genai import types

        if (state := self._get_active_state(session)) is None:
            return

        # Track tool result bytes for debugging
        state.tool_result_bytes += len(result)

        # Diagnostic: log every tool result we send back to Gemini so
        # the request → response → result cycle is visible end-to-end.
        # Body is truncated to 800 chars in the log; the full thing is
        # still sent to Gemini.
        self._log_event(
            session.id,
            "submit_tool_result",
            call_id=call_id,
            len=len(result),
            preview=(result[:800] + ("…" if len(result) > 800 else "")),
        )

        if len(result) > 16384:
            logger.warning(
                "Large tool result (%d chars) for call %s may cause Gemini to "
                "disconnect or silently fail (session %s)",
                len(result),
                call_id,
                session.id,
            )

        try:
            parsed = json.loads(result)
            result_dict = parsed if isinstance(parsed, dict) else {"result": parsed}
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

        # Decrement pending counter and flush queued injections
        state.pending_tool_calls = max(0, state.pending_tool_calls - 1)
        if state.pending_tool_calls == 0:
            if state.queued_text_injections:
                text_injections = state.queued_text_injections[:]
                state.queued_text_injections.clear()
                for text, role, silent in text_injections:
                    logger.debug(
                        "Flushing queued text injection for session %s (len=%d)",
                        session.id,
                        len(text),
                    )
                    await self._send_text(state, text, role, silent)
            if state.queued_injections:
                injections = state.queued_injections[:]
                state.queued_injections.clear()
                for image_data, mime_type, prompt, silent in injections:
                    logger.debug(
                        "Flushing queued image injection for session %s (mime=%s, size=%d)",
                        session.id,
                        mime_type,
                        len(image_data),
                    )
                    await self._send_image(state, image_data, mime_type, prompt, silent)

    async def interrupt(self, session: VoiceSession) -> None:
        # Gemini doesn't have a direct cancel; send empty to reset
        if self._get_active_state(session) is None:
            return
        logger.debug("Interrupt requested for Gemini session %s (no-op)", session.id)

    async def send_activity_start(self, session: VoiceSession) -> None:
        """Send ActivityStart to Gemini (manual VAD mode)."""
        if (state := self._get_active_state(session)) is None:
            return
        from google.genai import types

        await state.live_session.send_realtime_input(
            activity_start=types.ActivityStart(),
        )
        logger.debug("Sent ActivityStart for session %s", session.id)

    async def send_activity_end(self, session: VoiceSession) -> None:
        """Send ActivityEnd to Gemini (manual VAD mode)."""
        if (state := self._get_active_state(session)) is None:
            return
        from google.genai import types

        await state.live_session.send_realtime_input(
            activity_end=types.ActivityEnd(),
        )
        logger.debug("Sent ActivityEnd for session %s", session.id)

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
            "Gemini session %s disconnected: received=%d audio chunks",
            session.id,
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

        Reconfigure semantics: parameters left at ``None`` are
        preserved from the session's most recent config. ``_build_config``
        treats ``None`` as "absent" (it omits the field from the
        resulting LiveConnectConfig), so without this preservation a
        partial update like ``reconfigure(system_prompt=new)`` would
        wipe the existing tools and voice. Passing an empty list /
        empty string explicitly does still clear the field.
        """
        import contextlib

        state = self._sessions.get(session.id)
        if state is None:
            return

        # Discard stale queued injections from the old configuration
        state.queued_text_injections.clear()
        state.queued_injections.clear()

        # Preserve unspecified fields from the previous config so a
        # partial reconfigure (e.g. system_prompt-only) doesn't wipe
        # tools/voice/temperature. ``_build_config`` treats ``None``
        # as "absent" and would otherwise produce a config with no
        # tools at all.
        effective_prompt = system_prompt if system_prompt is not None else state.system_prompt
        effective_voice = voice if voice is not None else state.voice
        effective_tools = tools if tools is not None else state.tools
        effective_temperature = temperature if temperature is not None else state.temperature

        new_config = self._build_config(
            system_prompt=effective_prompt,
            voice=effective_voice,
            tools=effective_tools,
            temperature=effective_temperature,
            provider_config=provider_config,
        )

        # Remember effective values so the next partial reconfigure
        # preserves them.
        state.system_prompt = effective_prompt
        state.voice = effective_voice
        state.tools = effective_tools
        state.temperature = effective_temperature
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

    # -- Hot-path helpers (avoid per-call imports and string formatting) --

    def _make_audio_blob(self, data: bytes, sample_rate: int) -> Any:
        """Create a Blob without per-call import or string formatting."""
        if self._blob_cls is None:
            from google.genai import types

            self._blob_cls = types.Blob
        mime = self._mime_cache.get(sample_rate)
        if mime is None:
            mime = f"audio/pcm;rate={sample_rate}"
            self._mime_cache[sample_rate] = mime
        return self._blob_cls(data=data, mime_type=mime)

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
                    await self._fire(
                        self._error_callbacks,
                        session,
                        "max_reconnects",
                        f"Connection lost after {self._MAX_RECONNECTS} reconnect attempts",
                        label="error",
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

                async for response in live_session.receive():
                    reconnect_count = 0
                    await self._handle_server_response(session, response)

            except asyncio.CancelledError:
                raise
            except _GoAwayError:
                # Server warned it's about to disconnect — proactive reconnect.
                # This does NOT count against the reconnect limit.
                logger.info("Proactive reconnect (GoAway) for session %s", session.id)
                state.live_session = None
                reconnect_count = 0
            except Exception as exc:
                if session.state == VoiceSessionState.ENDED:
                    return  # state may be mutated by close() during await

                uptime = time.monotonic() - state.started_at if state.started_at else 0.0
                close_code = getattr(exc, "code", None)
                # Extract detailed error info from Gemini APIError
                response_json = getattr(exc, "response_json", None)
                status_code = getattr(exc, "status_code", None)
                logger.warning(
                    "Gemini session %s disconnected — "
                    "uptime=%.1fs, turns=%d, tool_result_bytes=%d, "
                    "audio_chunks=%d, close_code=%s, error=%s: %s, "
                    "status_code=%s, response_json=%s, pending_tools=%d",
                    session.id,
                    uptime,
                    state.turn_count,
                    state.tool_result_bytes,
                    state.audio_chunk_count,
                    close_code,
                    type(exc).__name__,
                    exc,
                    status_code,
                    response_json,
                    state.pending_tool_calls,
                )
                state.live_session = None
                # Suppress duplicate send_audio_failed errors during reconnect
                state.error_suppressed = True

    def _clear_transcription_buffers(self, session_id: str) -> None:
        """Remove all transcription buffer entries for a session."""
        self._transcription_buffer.clear_session(session_id)

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
                mime = f"audio/pcm;rate={state.input_sample_rate}"
                for chunk in state.audio_buffer:
                    await live_session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type=mime),
                    )
                state.realtime_input_sent = True
            finally:
                state.audio_buffer.clear()

        # Now safe to accept new audio directly
        session.state = VoiceSessionState.ACTIVE

        # Re-enable error callbacks for the next reconnection cycle
        state.error_suppressed = False

        logger.info("Gemini Live session %s reconnected", session.id)

    # Ordered dispatch table for server response handling.  Each entry is
    # (response_attribute, handler_method).  Order matters: go_away is
    # processed LAST so all data in the message is handled first.
    _RESPONSE_HANDLERS: list[tuple[str, str]] = [
        ("session_resumption_update", "_on_session_resumption"),
        ("voice_activity", "_on_voice_activity"),
        ("server_content", "_on_server_content"),
        ("data", "_on_audio_data"),
        ("tool_call", "_on_tool_call"),
        ("usage_metadata", "_on_usage_metadata"),
        ("go_away", "_on_go_away"),
    ]

    async def _handle_server_response(self, session: VoiceSession, response: Any) -> None:
        """Map Gemini Live responses to callbacks."""
        state = self._sessions.get(session.id)
        if state is None:
            return

        self._log_server_message(session, response)

        for attr, method in self._RESPONSE_HANDLERS:
            value = getattr(response, attr, None)
            if value:
                await getattr(self, method)(session, state, value)

    def _log_server_message(self, session: VoiceSession, response: Any) -> None:
        """Build a compact debug log line summarising the server message."""
        parts: list[str] = []
        if getattr(response, "data", None):
            parts.append(f"audio={len(response.data)}B")
        sc = getattr(response, "server_content", None)
        if sc:
            if getattr(sc, "model_turn", None):
                parts.append("model_turn")
            if getattr(sc, "turn_complete", None):
                parts.append("turn_complete")
            if getattr(sc, "interrupted", None):
                parts.append("interrupted")
            if getattr(sc, "input_transcription", None):
                parts.append(f"input_tx={sc.input_transcription.text!r}")
            if getattr(sc, "output_transcription", None):
                parts.append(f"output_tx={sc.output_transcription.text!r}")
        if getattr(response, "tool_call", None):
            parts.append("tool_call")
        va = getattr(response, "voice_activity", None)
        if va:
            parts.append(f"vad={getattr(va, 'voice_activity_type', '?')}")
        if getattr(response, "go_away", None):
            parts.append("go_away")
        if getattr(response, "session_resumption_update", None):
            parts.append("resumption_update")
        um = getattr(response, "usage_metadata", None)
        if um:
            parts.append(
                f"usage(prompt={getattr(um, 'prompt_token_count', '?')}"
                f",response={getattr(um, 'response_token_count', '?')}"
                f",total={getattr(um, 'total_token_count', '?')})"
            )
        if not parts:
            parts.append(f"unknown_keys={[k for k in dir(response) if not k.startswith('_')]}")
        logger.debug("[Gemini] recv: %s (session %s)", ", ".join(parts), session.id)

    async def _on_session_resumption(
        self, session: VoiceSession, state: _GeminiSessionState, update: Any
    ) -> None:
        if update.resumable and update.new_handle:
            state.resumption_handle = update.new_handle
            logger.debug(
                "Session resumption handle updated for %s (resumable=%s)",
                session.id,
                update.resumable,
            )

    async def _on_voice_activity(
        self, session: VoiceSession, state: _GeminiSessionState, va: Any
    ) -> None:
        vtype = getattr(va, "voice_activity_type", None)
        if not vtype:
            return
        if vtype == "ACTIVITY_START":
            logger.info("[VAD] speech_start (session %s)", session.id)
            state.user_speech_active = True
            await self._fire(self._speech_start_callbacks, session, label="speech_start")
        elif vtype == "ACTIVITY_END":
            logger.info("[VAD] speech_end (session %s)", session.id)
            state.user_speech_active = False
            await self._flush_transcription_buffer(session, "user")
            await self._fire(self._speech_end_callbacks, session, label="speech_end")

    async def _on_server_content(
        self, session: VoiceSession, state: _GeminiSessionState, content: Any
    ) -> None:
        # Input transcription (user speech-to-text)
        tr = getattr(content, "input_transcription", None)
        if tr and tr.text:
            await self._handle_transcription_chunk(session, tr.text, "user", bool(tr.finished))

        # Output transcription (model speech-to-text)
        tr = getattr(content, "output_transcription", None)
        if tr and tr.text:
            await self._handle_transcription_chunk(
                session, tr.text, "assistant", bool(tr.finished)
            )

        # Model started generating
        if getattr(content, "model_turn", None) and not state.response_started:
            await self._flush_transcription_buffer(session, "user")
            state.response_started = True
            state.audio_chunk_count = 0
            logger.info("[Gemini] response_start (session %s)", session.id)
            self._log_event(session.id, "response_start", turn=state.turn_count)
            await self._fire(self._response_start_callbacks, session, label="response_start")

        # Interrupted — user barged in while model was speaking
        if getattr(content, "interrupted", None):
            logger.info("[Gemini] INTERRUPTED — AI cut off by barge-in (session %s)", session.id)
            await self._flush_transcription_buffer(session, "assistant")
            # Fire speech_start ONLY if ACTIVITY_START wasn't already
            # received — Gemini doesn't always send voice_activity before
            # interrupted, so this may be the only trigger.
            if not state.user_speech_active:
                state.user_speech_active = True
                await self._fire(self._speech_start_callbacks, session, label="speech_start")
            if state.response_started:
                state.response_started = False
                await self._fire(self._response_end_callbacks, session, label="response_end")

        # Turn complete
        if getattr(content, "turn_complete", None):
            state.turn_count += 1
            logger.info(
                "[Gemini] turn_complete (session %s, %d audio chunks)",
                session.id,
                state.audio_chunk_count,
            )
            self._log_event(
                session.id,
                "turn_complete",
                turn=state.turn_count,
                audio_chunks=state.audio_chunk_count,
                pending_tool_calls=state.pending_tool_calls,
            )
            await self._flush_transcription_buffer(session, "user")
            await self._flush_transcription_buffer(session, "assistant")
            state.response_started = False
            state.user_speech_active = False
            await self._fire(self._response_end_callbacks, session, label="response_end")

    async def _on_audio_data(
        self, session: VoiceSession, state: _GeminiSessionState, data: bytes
    ) -> None:
        state.audio_chunk_count += 1
        if state.audio_chunk_count % 50 == 1:
            logger.debug(
                "[Gemini] audio chunk #%d (%d bytes) for session %s",
                state.audio_chunk_count,
                len(data),
                session.id,
            )
        await self._fire(self._audio_callbacks, session, data, label="audio")

    async def _on_tool_call(
        self, session: VoiceSession, state: _GeminiSessionState, tool_call: Any
    ) -> None:
        for fc in tool_call.function_calls:
            state.pending_tool_calls += 1
            args_dict = dict(fc.args) if fc.args else {}
            self._log_event(
                session.id,
                "function_call",
                name=fc.name,
                id=fc.id,
                args=args_dict,
            )
            await self._fire(
                self._tool_call_callbacks,
                session,
                fc.id,
                fc.name,
                args_dict,
                label="tool_call",
            )

    async def _on_usage_metadata(
        self, session: VoiceSession, state: _GeminiSessionState, meta: Any
    ) -> None:
        prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
        response_tokens = getattr(meta, "response_token_count", 0) or 0
        total_tokens = getattr(meta, "total_token_count", 0) or 0
        logger.debug(
            "[Gemini] usage: prompt=%d response=%d total=%d (session %s)",
            prompt_tokens,
            response_tokens,
            total_tokens,
            session.id,
        )
        # Only log the "real" usage tick — when the model evaluates a
        # new prompt round (prompt_tokens > 0). Gemini emits one usage
        # event per audio chunk during a response, ALL with
        # prompt_tokens=0 — those would flood the diagnostic log
        # without telling us anything useful.
        if prompt_tokens:
            self._log_event(
                session.id,
                "usage",
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
            )
        self._record_usage(session, prompt_tokens, response_tokens)

    async def _on_go_away(
        self, session: VoiceSession, state: _GeminiSessionState, go_away: Any
    ) -> None:
        time_left = getattr(go_away, "time_left", "unknown")
        logger.warning(
            "Gemini GoAway received for session %s (time_left=%s)",
            session.id,
            time_left,
        )
        raise _GoAwayError()

    async def _handle_transcription_chunk(
        self, session: VoiceSession, text: str, role: str, finished: bool
    ) -> None:
        """Accumulate transcription chunks and fire callback when complete."""
        full_text = self._transcription_buffer.append(session.id, role, text, finished)
        if full_text:
            # Log the FINAL transcription so we can see what each side
            # actually said in the same stream as tool_call events.
            # Truncate to keep log lines readable; full text still goes
            # into the room transcript via the callback.
            self._log_event(
                session.id,
                "transcription",
                role=role,
                text=full_text[:600] + ("…" if len(full_text) > 600 else ""),
                len=len(full_text),
            )
            await self._fire(
                self._transcription_callbacks,
                session,
                full_text,
                role,
                True,
                label="transcription",
            )
        elif not finished:
            # Send non-final for real-time display in the voice modal
            await self._fire(
                self._transcription_callbacks,
                session,
                text,
                role,
                False,
                label="transcription",
            )

    async def _flush_transcription_buffer(self, session: VoiceSession, role: str) -> None:
        """Flush buffered transcription at lifecycle boundaries."""
        full_text = self._transcription_buffer.flush(session.id, role)
        if full_text:
            logger.debug(
                "Flushing %s transcription buffer (%d chars) for session %s",
                role,
                len(full_text),
                session.id,
            )
            await self._fire(
                self._transcription_callbacks,
                session,
                full_text,
                role,
                True,
                label="transcription",
            )
