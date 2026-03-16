"""NVIDIA PersonaPlex Realtime provider for speech-to-speech conversations.

Connects to a self-hosted PersonaPlex server via WebSocket, streaming
bidirectional audio using the PersonaPlex binary protocol (Opus-encoded).

PersonaPlex handles VAD, turn-taking, interruptions, and backchannels
natively via its Moshi architecture.  No tool calling or text injection
support — this is a pure conversational speech-to-speech model.

Requirements:
    pip install websockets 'sphn>=0.1.4,<0.2' numpy
    A running PersonaPlex server (GPU: A100/H100 recommended)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode

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

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

logger = logging.getLogger("roomkit.providers.personaplex.realtime")

# PersonaPlex binary protocol message types
_MSG_HANDSHAKE = 0x00
_MSG_AUDIO = 0x01
_MSG_TEXT = 0x02
_MSG_CONTROL = 0x03
_MSG_METADATA = 0x04
_MSG_ERROR = 0x05
_MSG_PING = 0x06

# Control actions (client → server)
_CTRL_START = 0x00
_CTRL_END_TURN = 0x01
_CTRL_PAUSE = 0x02
_CTRL_RESTART = 0x03

# PersonaPlex native sample rate
NATIVE_SAMPLE_RATE = 24000


@dataclass
class _SessionState:
    """Per-session state for a PersonaPlex connection."""

    session: VoiceSession
    ws: Any = None
    receive_task: asyncio.Task[None] | None = None
    opus_writer: Any = None
    opus_reader: Any = None
    responding: bool = False
    response_end_task: asyncio.Task[None] | None = None
    text_buffer: list[str] = field(default_factory=list)
    response_end_timeout: float = 1.0
    _logged_first_frame: bool = False


class PersonaPlexRealtimeProvider(RealtimeVoiceProvider):
    """Realtime voice provider using NVIDIA PersonaPlex.

    Connects to a self-hosted PersonaPlex server via WebSocket,
    streaming audio bidirectionally with the binary protocol.

    PersonaPlex handles VAD, turn-taking, and response generation
    natively (Moshi architecture, 7B parameters).

    Requires ``websockets``, ``sphn``, and ``numpy`` packages.

    Available voice prompts (shipped with PersonaPlex):
        Natural: NATF0-3 (female), NATM0-3 (male)
        Varied:  VARF0-4 (female), VARM0-4 (male)

    Example::

        provider = PersonaPlexRealtimeProvider(
            server_url="wss://gpu-host:8998/api/chat",
        )
        provider.on_audio(handle_output_audio)
        provider.on_transcription(handle_transcription)

        await provider.connect(
            session,
            system_prompt="You are a helpful assistant.",
            voice="NATF2.pt",
        )

    Provider config keys (via ``provider_config`` dict):
        seed (int): Random seed for reproducibility (-1 = disabled).
        response_end_timeout (float): Seconds of silence before
            firing response_end (default 1.0).
    """

    def __init__(
        self,
        *,
        server_url: str = "wss://localhost:8998/api/chat",
        ssl_verify: bool = False,
        default_voice_prompt: str = "NATF2.pt",
        response_end_timeout: float = 1.0,
        seed: int = -1,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._ssl_verify = ssl_verify
        self._default_voice_prompt = default_voice_prompt
        self._default_response_end_timeout = response_end_timeout
        self._default_seed = seed

        self._states: dict[str, _SessionState] = {}

        # Callbacks
        self._audio_cbs: list[RealtimeAudioCallback] = []
        self._transcription_cbs: list[RealtimeTranscriptionCallback] = []
        self._speech_start_cbs: list[RealtimeSpeechStartCallback] = []
        self._speech_end_cbs: list[RealtimeSpeechEndCallback] = []
        self._tool_call_cbs: list[RealtimeToolCallCallback] = []
        self._response_start_cbs: list[RealtimeResponseStartCallback] = []
        self._response_end_cbs: list[RealtimeResponseEndCallback] = []
        self._error_cbs: list[RealtimeErrorCallback] = []

    @property
    def name(self) -> str:
        return "PersonaPlexRealtimeProvider"

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
        import websockets  # noqa: F811 — lazy import

        self._ensure_deps()

        if tools:
            logger.warning("PersonaPlex does not support tool calling; tools ignored")
        if temperature is not None:
            logger.warning("PersonaPlex does not support temperature; ignored")

        pc = provider_config or {}
        voice_prompt = voice or pc.get("voice_prompt", self._default_voice_prompt)
        text_prompt = system_prompt or ""
        seed = pc.get("seed", self._default_seed)
        timeout = pc.get("response_end_timeout", self._default_response_end_timeout)

        url = self._build_url(voice_prompt, text_prompt, seed)
        ssl_ctx = self._build_ssl_context(url)

        ws = await asyncio.wait_for(
            websockets.connect(url, ssl=ssl_ctx, max_size=2**20),
            timeout=30.0,
        )

        # Wait for server handshake (byte 0x00)
        handshake = await asyncio.wait_for(ws.recv(), timeout=60.0)
        if not isinstance(handshake, bytes) or not handshake or handshake[0] != _MSG_HANDSHAKE:
            await ws.close()
            msg = "PersonaPlex server did not send handshake"
            raise ConnectionError(msg)

        import sphn  # noqa: F811 — lazy import

        state = _SessionState(
            session=session,
            ws=ws,
            opus_writer=sphn.OpusStreamWriter(NATIVE_SAMPLE_RATE),
            opus_reader=sphn.OpusStreamReader(NATIVE_SAMPLE_RATE),
            response_end_timeout=timeout,
        )
        self._states[session.id] = state

        session.state = VoiceSessionState.ACTIVE
        session.provider_session_id = session.id

        # Signal conversation start
        await ws.send(bytes([_MSG_CONTROL, _CTRL_START]))

        state.receive_task = asyncio.create_task(
            self._receive_loop(session.id),
            name=f"personaplex_recv:{session.id}",
        )
        logger.info("PersonaPlex session connected: %s (voice=%s)", session.id, voice_prompt)

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        state = self._states.get(session.id)
        if state is None or state.ws is None:
            return
        # Convert int16 PCM to float32 for Opus encoder
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        # sphn.append_pcm returns encoded Ogg/Opus bytes directly;
        # also try read_bytes() for API compatibility across sphn versions.
        encoded = state.opus_writer.append_pcm(pcm)
        if not encoded:
            encoded = state.opus_writer.read_bytes()
        if encoded:
            await state.ws.send(bytes([_MSG_AUDIO]) + bytes(encoded))

    async def inject_text(
        self, session: VoiceSession, text: str, *, role: str = "user", silent: bool = False
    ) -> None:
        logger.warning(
            "PersonaPlex does not support text injection; ignored (session %s)", session.id
        )

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        logger.warning(
            "PersonaPlex does not support tool calling; tool result ignored (session %s)",
            session.id,
        )

    async def interrupt(self, session: VoiceSession) -> None:
        state = self._states.get(session.id)
        if state is None or state.ws is None:
            return
        logger.debug("Sending pause control (session %s)", session.id)
        await state.ws.send(bytes([_MSG_CONTROL, _CTRL_PAUSE]))

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        """Send a raw PersonaPlex protocol message.

        Supported event types:
            ``{"type": "control", "action": 0-3}`` — control message
            ``{"type": "ping"}`` — keepalive ping
        """
        state = self._states.get(session.id)
        if state is None or state.ws is None:
            return
        msg_type = event.get("type", "")
        if msg_type == "control":
            action = int(event.get("action", _CTRL_START))
            await state.ws.send(bytes([_MSG_CONTROL, action]))
        elif msg_type == "ping":
            await state.ws.send(bytes([_MSG_PING]))

    async def disconnect(self, session: VoiceSession) -> None:
        state = self._states.pop(session.id, None)
        if state is None:
            return
        if state.response_end_task is not None:
            state.response_end_task.cancel()
        if state.receive_task is not None:
            state.receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await state.receive_task
        if state.ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(state.ws.close(), timeout=2.0)
        # Flush pending state before marking ended
        if state.responding:
            await self._fire_callbacks(self._response_end_cbs, session)
        if state.text_buffer:
            text = "".join(state.text_buffer)
            await self._fire_transcription_cbs(session, text, "assistant", True)
        session.state = VoiceSessionState.ENDED
        logger.info("PersonaPlex session disconnected: %s", session.id)

    async def close(self) -> None:
        for sid in list(self._states):
            state = self._states.get(sid)
            if state:
                await self.disconnect(state.session)

    # -- Callback registration --

    def on_audio(self, cb: RealtimeAudioCallback) -> None:
        self._audio_cbs.append(cb)

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

    # -- Internal helpers --

    @staticmethod
    def _ensure_deps() -> None:
        """Verify required optional dependencies are importable."""
        if np is None:
            msg = (
                "numpy is required for PersonaPlexRealtimeProvider. "
                "Install with: pip install numpy"
            )
            raise ImportError(msg)
        try:
            import sphn  # noqa: F401
        except ImportError as exc:
            msg = (
                "sphn is required for PersonaPlexRealtimeProvider (Opus codec). "
                "Install with: pip install 'sphn>=0.1.4,<0.2'"
            )
            raise ImportError(msg) from exc

    def _build_url(self, voice_prompt: str, text_prompt: str, seed: int) -> str:
        params: dict[str, str] = {
            "voice_prompt": voice_prompt,
            "text_prompt": text_prompt,
        }
        # Only include seed when explicitly set; PersonaPlex server has a
        # bug where presence of "seed" in query triggers a read from
        # request._state instead of request.query.
        if seed >= 0:
            params["seed"] = str(seed)
        return f"{self._server_url}?{urlencode(params, quote_via=quote)}"

    def _build_ssl_context(self, url: str) -> Any:
        if not url.startswith("wss://"):
            return None
        import ssl

        if self._ssl_verify:
            return ssl.create_default_context()
        # PersonaPlex default: self-signed certs
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    # -- Receive loop --

    async def _receive_loop(self, session_id: str) -> None:
        state = self._states.get(session_id)
        if state is None or state.ws is None:
            return
        session = state.session
        try:
            async for raw in state.ws:
                if not isinstance(raw, bytes) or not raw:
                    continue
                try:
                    await self._handle_message(session_id, raw[0], raw[1:])
                except Exception:
                    logger.exception("Error handling message (session %s)", session_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            if session.state == VoiceSessionState.ACTIVE:
                logger.warning("WebSocket closed unexpectedly (session %s)", session_id)
                session.state = VoiceSessionState.ENDED
                await self._fire_error_cbs(session, "connection_closed", "WebSocket closed")

    async def _handle_message(self, session_id: str, msg_type: int, payload: bytes) -> None:
        state = self._states.get(session_id)
        if state is None:
            return
        dispatch = {
            _MSG_AUDIO: self._handle_audio,
            _MSG_TEXT: self._handle_text,
            _MSG_ERROR: self._handle_error,
            _MSG_METADATA: self._handle_metadata,
        }
        handler = dispatch.get(msg_type)
        if handler:
            await handler(state, payload)
        else:
            logger.debug("Unknown message type 0x%02x (session %s)", msg_type, session_id)

    async def _handle_audio(self, state: _SessionState, opus_data: bytes) -> None:
        if not opus_data:
            return
        if not state.responding:
            state.responding = True
            await self._fire_callbacks(self._response_start_cbs, state.session)
        # sphn.append_bytes returns decoded PCM directly in some versions;
        # in others it buffers and read_pcm() drains.  Try both.
        pcm_float = state.opus_reader.append_bytes(opus_data)
        if pcm_float is None or not hasattr(pcm_float, "shape") or pcm_float.shape[-1] == 0:
            pcm_float = state.opus_reader.read_pcm()
        if pcm_float is not None and hasattr(pcm_float, "shape") and pcm_float.shape[-1] > 0:
            if not state._logged_first_frame:
                state._logged_first_frame = True
                logger.info(
                    "First audio frame: %d samples, shape=%s (session %s)",
                    pcm_float.shape[-1],
                    pcm_float.shape,
                    state.session.id,
                )
            pcm_int16 = (pcm_float * 32768.0).clip(-32768, 32767).astype(np.int16)
            await self._fire_audio_cbs(state.session, pcm_int16.tobytes())
        self._schedule_response_end(state)

    async def _handle_text(self, state: _SessionState, payload: bytes) -> None:
        token = payload.decode("utf-8", errors="replace")
        state.text_buffer.append(token)
        text = "".join(state.text_buffer)
        await self._fire_transcription_cbs(state.session, text, "assistant", False)
        if not state.responding:
            state.responding = True
            await self._fire_callbacks(self._response_start_cbs, state.session)
        self._schedule_response_end(state)

    async def _handle_error(self, state: _SessionState, payload: bytes) -> None:
        error_msg = payload.decode("utf-8", errors="replace")
        logger.error("[PersonaPlex] error: %s (session %s)", error_msg, state.session.id)
        await self._fire_error_cbs(state.session, "server_error", error_msg)

    async def _handle_metadata(self, state: _SessionState, payload: bytes) -> None:
        logger.debug("[PersonaPlex] metadata: %s (session %s)", payload[:200], state.session.id)

    # -- Response end debounce --

    def _schedule_response_end(self, state: _SessionState) -> None:
        if state.response_end_task is not None:
            state.response_end_task.cancel()
        state.response_end_task = asyncio.create_task(
            self._delayed_response_end(state),
            name=f"personaplex_resp_end:{state.session.id}",
        )

    async def _delayed_response_end(self, state: _SessionState) -> None:
        await asyncio.sleep(state.response_end_timeout)
        if state.responding:
            state.responding = False
            if state.text_buffer:
                text = "".join(state.text_buffer)
                state.text_buffer.clear()
                await self._fire_transcription_cbs(state.session, text, "assistant", True)
            await self._fire_callbacks(self._response_end_cbs, state.session)

    # -- Callback helpers --

    async def _fire_callbacks(self, callbacks: list[Any], session: VoiceSession) -> None:
        for cb in callbacks:
            try:
                result = cb(session)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Callback error (session %s)", session.id)

    async def _fire_audio_cbs(self, session: VoiceSession, audio: bytes) -> None:
        for cb in self._audio_cbs:
            try:
                result = cb(session, audio)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Audio callback error (session %s)", session.id)

    async def _fire_transcription_cbs(
        self, session: VoiceSession, text: str, role: str, is_final: bool
    ) -> None:
        for cb in self._transcription_cbs:
            try:
                result = cb(session, text, role, is_final)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Transcription callback error (session %s)", session.id)

    async def _fire_error_cbs(self, session: VoiceSession, code: str, message: str) -> None:
        for cb in self._error_cbs:
            try:
                result = cb(session, code, message)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Error callback error (session %s)", session.id)
