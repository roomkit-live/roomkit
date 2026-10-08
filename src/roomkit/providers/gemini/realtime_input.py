"""Outbound input for the Gemini Live provider.

Audio frames, text and image injections, manual activity markers: the side of
the socket the caller feeds. What comes back is handled by
:class:`~roomkit.providers.gemini.realtime_handlers.GeminiLiveEventHandlersMixin`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from roomkit._text import fence
from roomkit.providers.gemini.realtime_config import genai_types
from roomkit.providers.gemini.realtime_state import _GeminiSessionState
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.injection import VoiceInjectionResult, say_line_instruction
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


def _into_conversation(state: _GeminiSessionState, text: str, silent: bool) -> str:
    """The text as it enters the session's conversation.

    Sending it gives the session a conversation. A non-silent injection also
    carries the instruction a resumption left unapplied (RFC §12.4): the
    model follows an instruction it is handed in a turn. The instruction is
    set apart in a block of its own before the text, so the text never reads
    as more of it.
    """
    state.has_conversation = True
    if silent or state.pending_instructions is None:
        return text
    instructions, state.pending_instructions = state.pending_instructions, None
    return (
        "Your instructions have been replaced. From now on, follow these instead "
        f"of the ones you started with:\n{fence('instructions', instructions)}\n\n{text}"
    )


class GeminiLiveInputMixin(RealtimeVoiceProvider):
    """Everything that goes to Gemini other than a tool result.

    Mixed into ``GeminiLiveProvider``. Caller audio, the text and image
    injections with their sanitising and their queue behind a blocking call,
    the manual activity markers, and the audio blob cache.
    """

    # Owned by GeminiLiveProvider / its other mixins; declared for typing.
    _sessions: dict[str, _GeminiSessionState]
    _get_active_state: Callable[[VoiceSession], _GeminiSessionState | None]
    _blob_cls: Any
    _mime_cache: dict[int, str]

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        state = self._sessions.get(session.id)
        if state is None:
            return

        if session.state not in (VoiceSessionState.ACTIVE, VoiceSessionState.CONNECTING):
            return
        # Buffer audio while reconnecting instead of dropping it
        if state.live_session is None or session.state == VoiceSessionState.CONNECTING:
            session.state = VoiceSessionState.CONNECTING
            state.buffer_audio(audio)
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
    ) -> VoiceInjectionResult:
        """Inject text as a turn (RFC §12.4).

        Gemini Live takes ``user`` and ``model`` turns and has no system role
        in them — instructions are fixed at setup — so a ``system``
        instruction is delivered as a ``user`` turn, which the model follows
        and answers, and an ``assistant`` line as a ``user`` turn asking the
        model to say it. Before any audio is sent the turn goes through
        ``clientContent`` (``silent`` leaves it incomplete); once audio flows
        it goes through ``realtimeInput``, which carries no role, and
        ``silent`` becomes a best-effort "do not respond" marker.
        """
        if (state := self._get_active_state(session)) is None:
            return VoiceInjectionResult(
                status="not_sent", reason="voice_not_connected", retryable=True
            )

        # Sanitize to prevent 1007 disconnects from control chars / surrogates.
        text = _sanitize_gemini_text(text)
        if not text.strip():
            logger.debug(
                "inject_text: empty after sanitization, skipping (session %s)",
                session.id,
            )
            return VoiceInjectionResult(status="not_sent", reason="voice_empty_text")

        # Queue while a blocking tool call is outstanding: the API refuses
        # input until its function response comes back. A background call is
        # not one the API waits on, and queueing there would hold the
        # injection back for no reason. Queue too behind injections still
        # waiting to go out (the flush is draining them), so none overtakes.
        if state.blocking_call_ids or state.queued_text_injections:
            logger.debug(
                "Queuing text injection for session %s (blocking tool calls: %d)",
                session.id,
                len(state.blocking_call_ids),
            )
            state.queued_text_injections.append((text, role, silent))
            return VoiceInjectionResult(status="unknown", reason="voice_provider_queued")

        await self._send_text(state, text, role, silent)
        return VoiceInjectionResult(status="sent")

    async def _send_text(
        self,
        state: _GeminiSessionState,
        text: str,
        role: str,
        silent: bool,
    ) -> None:
        types = genai_types()

        if role == "assistant":
            # No turn makes the model speak a given text (a ``model`` turn
            # records it as already said), so the line becomes an instruction.
            role, text = "system", say_line_instruction(text)
        text = _into_conversation(state, text, silent)
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

        # Same guard as inject_text: only a blocking call makes the API refuse
        # client_content. Queue the injection and flush after
        # submit_tool_result, and behind images still waiting to go out.
        if state.blocking_call_ids or state.queued_injections:
            logger.debug(
                "Queuing image injection for session %s (blocking tool calls: %d)",
                session.id,
                len(state.blocking_call_ids),
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
        types = genai_types()

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

    async def interrupt(self, session: VoiceSession) -> None:
        # Gemini doesn't have a direct cancel; send empty to reset
        if self._get_active_state(session) is None:
            return
        logger.debug("Interrupt requested for Gemini session %s (no-op)", session.id)

    async def send_activity_start(self, session: VoiceSession) -> None:
        """Send ActivityStart to Gemini (manual VAD mode)."""
        if (state := self._get_active_state(session)) is None:
            return
        types = genai_types()

        await state.live_session.send_realtime_input(
            activity_start=types.ActivityStart(),
        )
        # The user speaks again: a call the model then fails to write is
        # worth telling it about (RFC §12.4).
        state.malformed_call_nudged = False
        logger.debug("Sent ActivityStart for session %s", session.id)

    async def send_activity_end(self, session: VoiceSession) -> None:
        """Send ActivityEnd to Gemini (manual VAD mode)."""
        if (state := self._get_active_state(session)) is None:
            return
        types = genai_types()

        await state.live_session.send_realtime_input(
            activity_end=types.ActivityEnd(),
        )
        logger.debug("Sent ActivityEnd for session %s", session.id)

    def _make_audio_blob(self, data: bytes, sample_rate: int) -> Any:
        """Create a Blob without per-call import or string formatting."""
        if self._blob_cls is None:
            types = genai_types()

            self._blob_cls = types.Blob
        mime = self._mime_cache.get(sample_rate)
        if mime is None:
            mime = f"audio/pcm;rate={sample_rate}"
            self._mime_cache[sample_rate] = mime
        return self._blob_cls(data=data, mime_type=mime)
