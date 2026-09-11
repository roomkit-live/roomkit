"""In-session client events of the OpenAI GPT-Live provider.

What the provider sends once a session is started: audio appends, context
appends (instructions, spoken or silent), a reasoning backend's outputs, and
the raw escape hatch. Interruption is deliberately a no-op here: a full-duplex
model handles being talked over itself (RFC §12.4.1).
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from roomkit.providers.openai.live_config import _LOG_TAG, _LiveSession
from roomkit.providers.openai.live_events import (
    EVT_COMMENTARY_APPEND,
    EVT_INPUT_AUDIO_APPEND,
    EVT_INSTRUCTIONS_APPEND,
    EVT_THINKING_APPEND,
    chunk_text,
)
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.openai.live")


class OpenAILiveClientMixin(RealtimeVoiceProvider):
    """Outbound client API for the GPT-Live wire.

    Mixed into ``OpenAILiveProvider``, which owns the connection state
    (``_states``) and the input resampler.
    """

    # Owned by OpenAILiveProvider.__init__; declared for typing.
    _states: dict[str, _LiveSession]
    _resampler: LinearResamplerProvider

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        state = self._states.get(session.id)
        if state is None or not state.started.is_set():
            return
        if state.input_rate != state.session_rate:
            frame = AudioFrame(
                data=audio, sample_rate=state.input_rate, channels=1, sample_width=2
            )
            audio = self._resampler.resample(frame, state.session_rate, 1, 2, session.id).data
        if state.codec is not None:
            if len(audio) % 2:
                raise ValueError("PCM16 audio must contain complete two-byte samples")
            audio = state.codec.encode(audio)
        if not audio:
            return
        await state.ws.send(
            json.dumps(
                {"type": EVT_INPUT_AUDIO_APPEND, "audio": base64.b64encode(audio).decode("ascii")}
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
        state = self._states.get(session.id)
        if state is None:
            return
        if role == "system":
            event_type = EVT_INSTRUCTIONS_APPEND
        elif silent:
            event_type = EVT_THINKING_APPEND
        else:
            event_type = EVT_COMMENTARY_APPEND
        logger.debug(
            "[%s →] %s (role=%s, silent=%s, session %s)",
            _LOG_TAG,
            event_type,
            role,
            silent,
            session.id,
        )
        await self._send_append(state, event_type, None, text)

    async def submit_delegation_output(
        self,
        session: VoiceSession,
        delegation_id: str,
        text: str,
        *,
        spoken: bool,
    ) -> None:
        state = self._states.get(session.id)
        if state is None:
            return
        event_type = EVT_COMMENTARY_APPEND if spoken else EVT_THINKING_APPEND
        logger.debug(
            "[%s →] %s for delegation %s (session %s)",
            _LOG_TAG,
            event_type,
            delegation_id,
            session.id,
        )
        await self._send_append(state, event_type, delegation_id, text)

    async def _send_append(
        self, state: _LiveSession, event_type: str, delegation_id: str | None, text: str
    ) -> None:
        """Append context in as many bounded pieces as the API needs.

        ``delegation_id`` is always sent, ``None`` included: on these events
        the field is required, and ``None`` means general session context.
        """
        for chunk in chunk_text(text):
            await state.ws.send(
                json.dumps({"type": event_type, "delegation_id": delegation_id, "content": chunk})
            )

    async def interrupt(self, session: VoiceSession) -> None:
        """No-op: a full-duplex model handles being talked over itself (RFC §12.4.1)."""
        logger.debug("[%s] interrupt ignored — full-duplex (session %s)", _LOG_TAG, session.id)

    async def truncate_audio(self, session: VoiceSession, audio_end_ms: int) -> None:
        """No-op: the model's context is not truncated on user speech (RFC §12.4.1)."""
        logger.debug(
            "[%s] truncate_audio(%d) ignored — full-duplex (session %s)",
            _LOG_TAG,
            audio_end_ms,
            session.id,
        )

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        state = self._states.get(session.id)
        if state is None:
            return
        await state.ws.send(json.dumps(event))
