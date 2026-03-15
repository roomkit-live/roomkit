"""Anam AI Realtime provider — photorealistic avatar with audio+video output.

Connects to Anam's cloud pipeline via the ``anam`` SDK (WebRTC).
Anam handles STT → LLM → TTS → face animation and delivers
synchronized audio and video frames.

Requirements:
    pip install roomkit[anam]   # installs: anam, av, numpy
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any

from roomkit.providers.anam.config import AnamConfig
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.provider import (
    RealtimeAudioCallback,
    RealtimeAudioVideoProvider,
    RealtimeErrorCallback,
    RealtimeResponseEndCallback,
    RealtimeResponseStartCallback,
    RealtimeSpeechEndCallback,
    RealtimeSpeechStartCallback,
    RealtimeToolCallCallback,
    RealtimeTranscriptionCallback,
    RealtimeVideoCallback,
)

logger = logging.getLogger("roomkit.providers.anam.realtime")

# Lazy-loaded optional dependencies
_anam_mod: Any = None
_np: Any = None


def _ensure_deps() -> None:
    """Verify required optional dependencies are importable."""
    global _anam_mod, _np  # noqa: PLW0603
    if _anam_mod is None:
        try:
            import anam

            _anam_mod = anam
        except ImportError as exc:
            msg = "anam SDK is required for AnamRealtimeProvider. Install with: pip install anam"
            raise ImportError(msg) from exc
    if _np is None:
        try:
            import numpy as np

            _np = np
        except ImportError as exc:
            msg = "numpy is required for AnamRealtimeProvider. Install with: pip install numpy"
            raise ImportError(msg) from exc


@dataclass
class _SessionState:
    """Per-session state for an Anam connection."""

    session: VoiceSession
    client: Any = None
    # The context manager returned by client.connect()
    anam_ctx: Any = None
    # The session object from __aenter__
    anam_session: Any = None
    audio_task: asyncio.Task[None] | None = None
    video_task: asyncio.Task[None] | None = None
    responding: bool = False
    _closed: bool = False


class AnamRealtimeProvider(RealtimeAudioVideoProvider):
    """Realtime audio+video provider using Anam AI avatars.

    Connects to Anam's cloud pipeline via the ``anam`` Python SDK.
    The avatar renders a photorealistic talking head driven by the LLM
    response, delivering synchronized audio and video frames.

    Requires the ``anam``, ``av``, and ``numpy`` packages::

        pip install roomkit[anam]

    Example::

        from roomkit.providers.anam import AnamConfig, AnamRealtimeProvider

        provider = AnamRealtimeProvider(
            AnamConfig(api_key="ak-...", persona_id="my-persona")
        )
        provider.on_audio(handle_audio)
        provider.on_video(handle_video)

        await provider.connect(session, system_prompt="You are a helpful avatar.")

    Provider config keys (via ``provider_config`` dict):
        persona_id (str): Override persona for this session.
        enable_audio_passthrough (bool): Bypass Anam STT.
    """

    def __init__(self, config: AnamConfig) -> None:
        self._config = config
        self._states: dict[str, _SessionState] = {}
        self._input_sample_rate: int = 16000

        # Callbacks
        self._audio_cbs: list[RealtimeAudioCallback] = []
        self._video_cbs: list[RealtimeVideoCallback] = []
        self._transcription_cbs: list[RealtimeTranscriptionCallback] = []
        self._speech_start_cbs: list[RealtimeSpeechStartCallback] = []
        self._speech_end_cbs: list[RealtimeSpeechEndCallback] = []
        self._tool_call_cbs: list[RealtimeToolCallCallback] = []
        self._response_start_cbs: list[RealtimeResponseStartCallback] = []
        self._response_end_cbs: list[RealtimeResponseEndCallback] = []
        self._error_cbs: list[RealtimeErrorCallback] = []

    @property
    def name(self) -> str:
        return "AnamRealtimeProvider"

    # -- Connection lifecycle --------------------------------------------------

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
        _ensure_deps()
        self._input_sample_rate = input_sample_rate

        pc = provider_config or {}
        persona_id = pc.get("persona_id", self._config.persona_id)
        prompt = system_prompt or self._config.system_prompt
        enable_passthrough = pc.get(
            "enable_audio_passthrough",
            self._config.enable_audio_passthrough,
        )

        # Build AnamClient with PersonaConfig.
        # persona_id → pre-defined persona (from Anam Lab).
        # avatar_id + voice_id + llm_id → ephemeral persona.
        pcfg_kwargs: dict[str, Any] = {
            "enable_audio_passthrough": enable_passthrough,
        }
        if persona_id:
            pcfg_kwargs["persona_id"] = persona_id
        if self._config.avatar_id:
            pcfg_kwargs["avatar_id"] = self._config.avatar_id
        if self._config.avatar_model:
            pcfg_kwargs["avatar_model"] = self._config.avatar_model
        if voice or self._config.voice_id:
            pcfg_kwargs["voice_id"] = voice or self._config.voice_id
        if self._config.llm_id:
            pcfg_kwargs["llm_id"] = self._config.llm_id
        if prompt:
            pcfg_kwargs["system_prompt"] = prompt
        if self._config.language_code:
            pcfg_kwargs["language_code"] = self._config.language_code

        pcfg = _anam_mod.PersonaConfig(**pcfg_kwargs)
        client = _anam_mod.AnamClient(
            api_key=self._config.api_key,
            persona_config=pcfg,
        )

        # client.connect() returns an async context manager; enter it
        # to get the session, and store the ctx for cleanup on disconnect.
        anam_ctx = client.connect()
        try:
            anam_session = await asyncio.wait_for(
                anam_ctx.__aenter__(),
                timeout=self._config.timeout,
            )
        except Exception:
            logger.exception("Anam connect failed for session %s", session.id)
            raise

        state = _SessionState(
            session=session,
            client=client,
            anam_ctx=anam_ctx,
            anam_session=anam_session,
        )
        self._states[session.id] = state

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        # Start background consume loops for audio and video
        state.audio_task = asyncio.create_task(
            self._audio_consume_loop(session.id),
            name=f"anam_audio:{session.id}",
        )
        state.video_task = asyncio.create_task(
            self._video_consume_loop(session.id),
            name=f"anam_video:{session.id}",
        )

        logger.info(
            "Anam session connected: %s (persona=%s)",
            session.id,
            persona_id,
        )

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        state = self._states.get(session.id)
        if state is None or state.anam_session is None:
            return
        try:
            rate = self._input_sample_rate or 16000
            state.anam_session.send_user_audio(audio, rate, 1)
        except Exception:
            logger.debug("send_audio failed (session %s)", session.id, exc_info=True)

    async def inject_text(self, session: VoiceSession, text: str, *, role: str = "user") -> None:
        state = self._states.get(session.id)
        if state is None or state.anam_session is None:
            return
        try:
            # send_message goes through the LLM; talk() bypasses it
            state.anam_session.send_message(text)
        except Exception:
            logger.debug("inject_text failed (session %s)", session.id, exc_info=True)

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        logger.warning(
            "Anam does not support tool results; ignored (session %s)",
            session.id,
        )

    async def interrupt(self, session: VoiceSession) -> None:
        state = self._states.get(session.id)
        if state is None or state.anam_session is None:
            return
        try:
            state.anam_session.interrupt()
        except Exception:
            logger.debug("interrupt failed (session %s)", session.id, exc_info=True)

    async def disconnect(self, session: VoiceSession) -> None:
        state = self._states.pop(session.id, None)
        if state is None:
            return
        state._closed = True

        # Cancel consume tasks
        for task in (state.audio_task, state.video_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        # Flush pending response end
        if state.responding:
            await self._fire_callbacks(self._response_end_cbs, session)

        # Exit the async context manager (closes WebRTC session)
        if state.anam_ctx is not None:
            with contextlib.suppress(Exception):
                await state.anam_ctx.__aexit__(None, None, None)

        session.state = VoiceSessionState.ENDED
        logger.info("Anam session disconnected: %s", session.id)

    async def close(self) -> None:
        for sid in list(self._states):
            state = self._states.get(sid)
            if state:
                await self.disconnect(state.session)

    # -- Callback registration -------------------------------------------------

    def on_audio(self, cb: RealtimeAudioCallback) -> None:
        self._audio_cbs.append(cb)

    def on_video(self, cb: RealtimeVideoCallback) -> None:
        self._video_cbs.append(cb)

    def on_transcription(self, cb: RealtimeTranscriptionCallback) -> None:
        self._transcription_cbs.append(cb)

    def on_speech_start(self, cb: RealtimeSpeechStartCallback) -> None:
        self._speech_start_cbs.append(cb)

    def on_speech_end(self, cb: RealtimeSpeechEndCallback) -> None:
        self._speech_end_cbs.append(cb)

    def on_tool_call(self, cb: RealtimeToolCallCallback) -> None:
        self._tool_call_cbs.append(cb)

    def on_response_start(self, cb: RealtimeResponseStartCallback) -> None:
        self._response_start_cbs.append(cb)

    def on_response_end(self, cb: RealtimeResponseEndCallback) -> None:
        self._response_end_cbs.append(cb)

    def on_error(self, cb: RealtimeErrorCallback) -> None:
        self._error_cbs.append(cb)

    # -- Internal: consume loops -----------------------------------------------

    async def _audio_consume_loop(self, session_id: str) -> None:
        """Consume audio frames from Anam and fire audio callbacks."""
        state = self._states.get(session_id)
        if state is None or state.anam_session is None:
            return
        session = state.session

        audio_frame_num = 0
        try:
            async for av_frame in state.anam_session.audio_frames():
                if state._closed:
                    break
                audio_frame_num += 1

                # Debug: log raw PyAV frame properties on first few frames
                if audio_frame_num <= 3:
                    ndarray = av_frame.to_ndarray()
                    logger.info(
                        "RAW PyAV audio #%d: format=%s rate=%s "
                        "layout=%s samples=%s ndarray.shape=%s "
                        "dtype=%s min=%.4f max=%.4f",
                        audio_frame_num,
                        getattr(av_frame, "format", "?"),
                        getattr(av_frame, "sample_rate", "?"),
                        getattr(av_frame, "layout", "?"),
                        getattr(av_frame, "samples", "?"),
                        ndarray.shape,
                        ndarray.dtype,
                        float(ndarray.min()),
                        float(ndarray.max()),
                    )

                # Signal response start on first audio frame
                if not state.responding:
                    state.responding = True
                    await self._fire_callbacks(self._response_start_cbs, session)

                # Convert PyAV AudioFrame → PCM int16 bytes
                try:
                    pcm = self._av_audio_to_pcm(av_frame)
                except Exception:
                    logger.debug("Audio frame conversion failed", exc_info=True)
                    continue

                await self._fire_audio_cbs(session, pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            if session.state == VoiceSessionState.ACTIVE and not state._closed:
                logger.warning("Anam audio stream ended unexpectedly (session %s)", session_id)
                await self._fire_error_cbs(session, "audio_stream_closed", "Audio stream ended")

    async def _video_consume_loop(self, session_id: str) -> None:
        """Consume video frames from Anam and fire video callbacks."""
        state = self._states.get(session_id)
        if state is None or state.anam_session is None:
            return
        session = state.session
        frame_seq = 0

        try:
            async for av_frame in state.anam_session.video_frames():
                if state._closed:
                    break

                # Convert PyAV VideoFrame → RoomKit VideoFrame
                try:
                    video_frame = self._av_video_to_frame(av_frame, frame_seq)
                    frame_seq += 1
                except Exception:
                    logger.debug("Video frame conversion failed", exc_info=True)
                    continue

                await self._fire_video_cbs(session, video_frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            if session.state == VoiceSessionState.ACTIVE and not state._closed:
                logger.warning("Anam video stream ended unexpectedly (session %s)", session_id)

    # -- Format conversion (delegated to _convert module) -----------------------

    @staticmethod
    def _av_audio_to_pcm(av_frame: Any) -> bytes:
        """Convert a PyAV AudioFrame to PCM int16 bytes."""
        from roomkit.providers.anam._convert import av_audio_to_pcm

        return av_audio_to_pcm(av_frame, _np)

    @staticmethod
    def _av_video_to_frame(av_frame: Any, sequence: int = 0) -> Any:
        """Convert a PyAV VideoFrame to a RoomKit VideoFrame."""
        from roomkit.providers.anam._convert import av_video_to_frame

        return av_video_to_frame(av_frame, sequence)

    # -- Callback helpers ------------------------------------------------------

    @staticmethod
    async def _fire(
        callbacks: list[Any],
        *args: Any,
        label: str = "callback",
    ) -> None:
        """Fire a list of callbacks with the given arguments."""
        for cb in callbacks:
            try:
                result = cb(*args)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("%s error", label)

    async def _fire_callbacks(
        self,
        callbacks: list[Any],
        session: VoiceSession,
    ) -> None:
        await self._fire(callbacks, session, label="callback")

    async def _fire_audio_cbs(
        self,
        session: VoiceSession,
        audio: bytes,
    ) -> None:
        await self._fire(self._audio_cbs, session, audio, label="audio")

    async def _fire_video_cbs(
        self,
        session: VoiceSession,
        video_frame: Any,
    ) -> None:
        await self._fire(self._video_cbs, session, video_frame, label="video")

    async def _fire_transcription_cbs(
        self,
        session: VoiceSession,
        text: str,
        role: str,
        is_final: bool,
    ) -> None:
        await self._fire(
            self._transcription_cbs,
            session,
            text,
            role,
            is_final,
            label="transcription",
        )

    async def _fire_error_cbs(
        self,
        session: VoiceSession,
        code: str,
        message: str,
    ) -> None:
        await self._fire(
            self._error_cbs,
            session,
            code,
            message,
            label="error",
        )
