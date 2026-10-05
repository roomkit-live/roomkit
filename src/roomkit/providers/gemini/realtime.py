"""Google Gemini Live API provider for speech-to-speech conversations."""

from __future__ import annotations

import contextlib
import logging
import time
from copy import deepcopy
from typing import Any

from pydantic import SecretStr

from roomkit.core.task_utils import _finish_cleanup, cancel_and_wait
from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.gemini.realtime_config import (
    blocking_tool_names,
    build_live_config,
    debug_enabled,
)
from roomkit.providers.gemini.realtime_connection import GeminiLiveConnectionMixin
from roomkit.providers.gemini.realtime_handlers import GeminiLiveEventHandlersMixin
from roomkit.providers.gemini.realtime_input import GeminiLiveInputMixin
from roomkit.providers.gemini.realtime_models import (
    MODELS,
)
from roomkit.providers.gemini.realtime_state import (  # noqa: F401 - tests read these
    _GeminiSessionState,
    _GoAwayError,
    _TranscriptionBuffer,
)
from roomkit.providers.gemini.realtime_tools import GeminiLiveToolsMixin
from roomkit.providers.gemini.realtime_transcription import GeminiLiveTranscriptionMixin
from roomkit.providers.gemini.voices import VOICES as _VOICES
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import RealtimeVoiceProvider, VoiceInfo

logger = logging.getLogger("roomkit.providers.gemini.realtime")


class GeminiLiveProvider(
    GeminiLiveConnectionMixin,
    GeminiLiveInputMixin,
    GeminiLiveToolsMixin,
    GeminiLiveEventHandlersMixin,
    GeminiLiveTranscriptionMixin,
    RealtimeVoiceProvider,
):
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
        model: str = "gemini-3.8-live",
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
    def model_name(self) -> str:
        """The Gemini Live model this provider connects to, end to end."""
        return self._model

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """Curated, offline catalog of Gemini Live native-audio voices (fixed set)."""
        return list(_VOICES)

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Gemini Live models."""
        return list(MODELS)

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        # gemini-3.x live models reject send_client_content with WS 1007
        # after the first model turn and offer no documented dynamic
        # system_instruction update. Their session_resumption is also
        # fragile with non-trivial system prompts. Disable mid-session
        # reconfigure for the whole 3.x family so callers route changes
        # through session-start delivery instead. 2.5-era models keep
        # the old behavior.
        return not (self._model.startswith("gemini-3.") or self._model.startswith("gemini-3-"))

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
        warned: set[str] | None = None,
    ) -> Any:
        """The LiveConnectConfig for this provider's model.

        Thin over :func:`~roomkit.providers.gemini.realtime_config.build_live_config`,
        kept as a method because ``connect``, ``reconfigure`` and the tests
        address it on the provider.
        """
        return build_live_config(
            self._model,
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            provider_config=provider_config,
            server_vad=server_vad,
            warned=warned,
        )

    def _blocking_tool_names(
        self, tools: list[dict[str, Any]] | None, warned: set[str] | None = None
    ) -> set[str]:
        """Names whose calls the API waits on, for this provider's model."""
        return blocking_tool_names(self._model, tools, warned)

    def _log_event(self, session_id: str, label: str, **fields: Any) -> None:
        """Log a single server event from Gemini Live for diagnostics.

        Called from the receive loop and message handlers. ``label`` is
        a short tag (text_delta, tool_call, transcription, error, …) and
        ``fields`` are the salient attributes to log. Gated on the same
        ``ROOMKIT_GEMINI_DEBUG`` env var as the config dump.
        """
        if not debug_enabled():
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
        warned: set[str] = set()
        live_config = self._build_config(
            system_prompt=system_prompt,
            voice=voice,
            tools=tools,
            temperature=temperature,
            provider_config=provider_config,
            server_vad=server_vad,
            warned=warned,
        )

        ctxmgr, live_session = await self._open_live_session(live_config)

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
            server_vad=server_vad,
            provider_config=deepcopy(provider_config or {}),
            blocking_tool_names=self._blocking_tool_names(tools, warned),
            warned_unsupported=warned,
        )
        await self._abandon_previous_connection(session)
        self._sessions[session.id] = state

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        self._start_receive_loop(state)

        logger.info("Gemini Live session connected: %s", session.id)

    async def disconnect(self, session: VoiceSession) -> None:
        state = self._sessions.pop(session.id, None)
        if state is None:
            session.state = VoiceSessionState.ENDED
            return

        await self._abandon_open_calls(state)
        session.state = VoiceSessionState.ENDED
        state.audio_buffer.clear()

        # Cancel receive task
        await cancel_and_wait(state.receive_task, log_errors_to=logger)

        # Clean up transcription buffers
        self._clear_transcription_buffers(session.id)

        # Record session metrics before cleanup
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

        ``gemini-3.8-live`` resumes a session under the instruction it
        started with, ignoring the new one: a session with no conversation
        reconnects fresh, and a session with one gets the new instructions
        with its next non-silent injection (a handoff greeting), which the
        model follows (RFC §12.4).
        """
        state = self._sessions.get(session.id)
        if state is None:
            return

        if state.provider_config.get("preserve_context"):
            raise ValueError("Reconfiguration is unavailable while preserving Gemini context")

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

        effective_provider_config = deepcopy(state.provider_config)
        if provider_config is not None:
            effective_provider_config.update(deepcopy(provider_config))

        new_config = self._build_config(
            system_prompt=effective_prompt,
            voice=effective_voice,
            tools=effective_tools,
            temperature=effective_temperature,
            provider_config=effective_provider_config,
            server_vad=state.server_vad,
            warned=state.warned_unsupported,
        )

        # Remember effective values so the next partial reconfigure
        # preserves them.
        state.system_prompt = effective_prompt
        state.voice = effective_voice
        state.tools = effective_tools
        state.temperature = effective_temperature
        # The new declarations decide what blocks from here on. The calls
        # outstanding on the old socket do not carry over: the reconnect
        # below releases and reports them, since the new connection never
        # issued their ids.
        state.blocking_tool_names = self._blocking_tool_names(
            effective_tools, state.warned_unsupported
        )
        state.live_config = new_config
        state.provider_config = effective_provider_config
        logger.info(
            "Reconfiguring Gemini session %s (voice=%s)",
            session.id,
            voice,
        )

        await _finish_cleanup(self._switch_connection(session, state))

    async def _switch_connection(self, session: VoiceSession, state: _GeminiSessionState) -> None:
        """Stop the old receive loop, reconnect under the new config, read again.

        Run to its end even when the caller is cancelled, whose cancellation
        is raised afterwards: stopped halfway, the session would hold its new
        config on the old socket, with nobody reading it.
        """
        # Cancel the old receive task BEFORE reconnecting to prevent it
        # from detecting the disconnection and triggering a second
        # auto-reconnect (double-reconnect bug).
        await cancel_and_wait(state.receive_task, log_errors_to=logger)
        state.receive_task = None
        self._carry_instructions_past_resumption(state)
        await self._reconnect(session)
        # Start a fresh receive loop for the new connection.
        self._start_receive_loop(state)

    def _carry_instructions_past_resumption(self, state: _GeminiSessionState) -> None:
        """Make the new instruction take effect where resuming would drop it.

        A session with nothing said in it reconnects fresh: nothing to keep.
        On a model that resumes under the original instruction, a session
        with a conversation keeps its context, and the new instruction rides
        its next non-silent injection instead.
        """
        state.pending_instructions = None
        if state.resumption_handle is None:
            return
        if not state.has_conversation:
            logger.info(
                "Gemini session %s has no conversation yet: reconnecting fresh",
                state.session.id,
            )
            state.resumption_handle = None
        elif self._resumption_keeps_instructions and state.system_prompt:
            state.pending_instructions = state.system_prompt

    @property
    def _resumption_keeps_instructions(self) -> bool:
        """Measured on gemini-3.8-live: a resumed session keeps the system
        instruction it started with (3.1 and 2.5 take the new one)."""
        return self._model.startswith("gemini-3.8")

    async def close(self) -> None:
        for session_id in list(self._sessions.keys()):
            state = self._sessions.get(session_id)
            if state:
                await self.disconnect(state.session)
