"""OpenAI GPT-Live speech-to-speech provider (Live API, full-duplex).

GPT-Live listens and speaks at the same time and hands reasoning and tool use
to a backend model while it keeps talking. The Live API shares nothing with
the Realtime API beyond JSON over a WebSocket — one immutable
``session.start``, continuous audio, transcript deltas without turns,
delegations, context appends — so this provider is not built on
``OpenAIRealtimeBase``. It implements the full-duplex contract of RFC
§12.4.1: response and speech boundaries are synthesized from the transcript,
``interrupt`` and ``truncate_audio`` are no-ops, and both delegation modes are
carried.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import SecretStr

from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.openai.live_events import (
    DELEGATION_TARGETS,
    EVT_COMMENTARY_APPEND,
    EVT_DELEGATION_CREATED,
    EVT_ERROR,
    EVT_INPUT_AUDIO_APPEND,
    EVT_INPUT_TRANSCRIPT_DELTA,
    EVT_INSTRUCTIONS_APPEND,
    EVT_OUTPUT_AUDIO_DELTA,
    EVT_OUTPUT_TRANSCRIPT_DELTA,
    EVT_RESPONSE_CREATE,
    EVT_RESPONSE_EVENT,
    EVT_RESPONSE_ITEM_CREATE,
    EVT_SESSION_CLOSE,
    EVT_SESSION_CLOSED,
    EVT_SESSION_START,
    EVT_SESSION_STARTED,
    EVT_SESSION_UPDATE,
    EVT_SESSION_UPDATED,
    EVT_SESSION_USAGE_UPDATED,
    EVT_THINKING_APPEND,
    NOISY_EVENTS,
    UNCORRELATED_DELEGATION,
    PendingResponse,
    TurnGrouper,
    build_audio_format,
    chunk_text,
    format_backend_tools,
    history_items,
)
from roomkit.providers.openai.live_models import MODELS
from roomkit.telemetry.base import Attr
from roomkit.voice._g711 import _G711Codec, _get_codec
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.realtime.provider import RealtimeVoiceProvider, VoiceInfo

logger = logging.getLogger("roomkit.providers.openai.live")

_DEFAULT_BASE_URL = "wss://api.openai.com/v1/live/sessions"
_DEFAULT_MODEL = "gpt-live-1"
_CONNECT_TIMEOUT = 30.0
_CLOSE_TIMEOUT = 2.0
_LOG_TAG = "GPT-Live"

_VOICES: list[VoiceInfo] = [
    VoiceInfo(id="marin", name="Marin", description="Default GPT-Live voice"),
    VoiceInfo(id="cedar", name="Cedar"),
]


@dataclass(frozen=True)
class HostedReasoning:
    """Hosted backend: OpenAI runs the Responses model the live model delegates to.

    The channel's tools become this model's tools; its function calls reach
    the channel through ``on_tool_call`` and are answered through
    ``submit_tool_result``, exactly as with any other provider (RFC §12.4.1,
    hosted backend).

    Attributes:
        model: Responses model id (e.g. ``"gpt-5.6-terra"``). Required.
        instructions: The backend's own instructions.
        reasoning_effort: ``"low"``/``"medium"``/``"high"`` when the backend
            model grades it; omitted otherwise.
        max_output_tokens: Output cap for one delegated response.
        service_tier: Responses service tier (``auto``, ``default``, ``flex``,
            ``priority``).
        parallel_tool_calls: Whether the backend may call several tools at once.
        extra: Other ``delegation.responses`` fields, merged last.
    """

    model: str
    instructions: str | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    service_tier: str | None = None
    parallel_tool_calls: bool | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_config(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Render the ``delegation`` block of ``session.start``."""
        responses: dict[str, Any] = {"model": self.model}
        if self.instructions:
            responses["instructions"] = self.instructions
        if self.reasoning_effort:
            responses["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_output_tokens is not None:
            responses["max_output_tokens"] = int(self.max_output_tokens)
        if self.service_tier:
            responses["service_tier"] = self.service_tier
        if self.parallel_tool_calls is not None:
            responses["parallel_tool_calls"] = bool(self.parallel_tool_calls)
        if tools:
            responses["tools"] = format_backend_tools(tools)
        responses.update(self.extra)
        return {"type": "responses", "responses": responses}


@dataclass(frozen=True)
class IntegratorReasoning:
    """Integrator backend: the channel's ReasoningBackend answers delegations.

    The model signals only that it is handing work over; the provider fires
    ``on_delegation(session, delegation_id, "integrator")`` and the answer
    returns through ``submit_delegation_output`` (RFC §12.4.1, integrator
    backend). Tools given to ``connect()`` stay off the wire: they are the
    backend's, and the channel keeps them as its declared catalogue.
    """

    def to_config(self, tools: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: ARG002
        """Render the ``delegation`` block of ``session.start``."""
        return {"type": "client"}


@dataclass
class _LiveSession:
    """Per-session connection state."""

    ws: Any
    session: VoiceSession
    session_rate: int
    input_rate: int
    output_rate: int
    codec: _G711Codec | None
    system_prompt: str | None
    voice: str | None
    tools: list[dict[str, Any]]
    provider_config: dict[str, Any]
    user_turn: TurnGrouper
    assistant_turn: TurnGrouper
    started: asyncio.Event = field(default_factory=asyncio.Event)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    start_error: str | None = None
    receive_task: asyncio.Task[None] | None = None
    responding: bool = False
    # Hosted delegation: Responses runs by delegation id, and the open
    # function calls each still owes an output.
    pending: dict[str, PendingResponse] = field(default_factory=dict)
    open_calls: dict[str, str] = field(default_factory=dict)
    live_seconds: float = 0.0


class OpenAILiveProvider(RealtimeVoiceProvider):
    """Speech-to-speech provider for OpenAI GPT-Live (``/v1/live/sessions``).

    **Full-duplex.** The model handles being talked over by itself, so
    :attr:`full_duplex` is ``True``: the channel never flushes playback or
    gates the model's audio on user speech, :meth:`interrupt` and
    :meth:`truncate_audio` do nothing, and a pipeline VAD stays in the
    observation role. The wire carries no response or speech boundaries;
    this provider synthesizes them from the transcript deltas with a quiet
    gap of ``turn_gap_ms`` per speaker (RFC §12.4.1).

    **Transcripts.** Partial transcriptions carry *deltas* for both roles and
    the final carries the whole turn, closed by the gap. A channel that keeps
    a transcript ledger reads the partials.

    **Reasoning delegation.** The model holds no tools. Pass
    :class:`HostedReasoning` to let OpenAI run the backend model — the
    channel's tools become its tools and its function calls flow through the
    usual ``on_tool_call`` / ``submit_tool_result`` path — or
    :class:`IntegratorReasoning` (the default) to serve delegations from a
    ``ReasoningBackend`` configured on the channel.

    **Fixed session.** Model, instructions, voice, audio format, delegation
    mode and seeded history are set once by ``session.start``.
    :meth:`reconfigure` appends a changed system prompt and, in hosted mode,
    updates the backend's tools without replacing the session; a voice change
    reconnects. :attr:`supports_mid_session_reconfigure` is therefore
    ``False``.

    **Text injection is paraphrased.** ``inject_text`` maps a ``system`` role
    to an instructions append and a ``user`` role to a spoken-context append
    (or a silent one with ``silent=True``); the model relays the text in its
    own words rather than reading it. Appends over the API's per-append bound
    are split on sentence boundaries. Image injection is not available on
    the Live endpoint.

    **Audio.** One wire format serves both directions, chosen from the
    channel's ``output_sample_rate``: PCM16 at 16 or 24 kHz, or G.711 at
    8 kHz with ``provider_config={"codec": "pcmu" | "pcma"}``. Input at a
    different ``input_sample_rate`` is resampled here.

    **Usage.** The live model bills session seconds, reported in
    ``session._last_usage["live_seconds"]``; a hosted backend's token usage
    is reported under ``session._last_usage["backend"]`` with its own model.

    ``provider_config`` keys: ``codec`` (``"pcm"``, ``"pcmu"``, ``"pcma"``)
    and ``history`` (a list of ``{"role", "text"}`` text messages seeding the
    session, at most 128).

    Requires the ``websockets`` package (``pip install 'roomkit[realtime-openai]'``).

    Example:
        provider = OpenAILiveProvider(
            api_key="sk-...",
            delegation=HostedReasoning(model="gpt-5.6-terra", instructions="..."),
        )
    """

    _EVENT_HANDLERS: dict[str, str] = {
        EVT_SESSION_STARTED: "_on_session_started",
        EVT_SESSION_UPDATED: "_on_session_updated",
        EVT_SESSION_CLOSED: "_on_session_closed",
        EVT_SESSION_USAGE_UPDATED: "_on_usage_updated",
        EVT_OUTPUT_AUDIO_DELTA: "_on_output_audio_delta",
        EVT_OUTPUT_TRANSCRIPT_DELTA: "_on_output_transcript_delta",
        EVT_INPUT_TRANSCRIPT_DELTA: "_on_input_transcript_delta",
        EVT_DELEGATION_CREATED: "_on_delegation_created",
        EVT_RESPONSE_EVENT: "_on_response_event",
        EVT_ERROR: "_on_error",
    }

    def __init__(
        self,
        *,
        api_key: str | SecretStr,
        model: str = _DEFAULT_MODEL,
        base_url: str | None = None,
        delegation: HostedReasoning | IntegratorReasoning | None = None,
        turn_gap_ms: int = 800,
        close_timeout_s: float = 5.0,
    ) -> None:
        super().__init__()
        if turn_gap_ms <= 0:
            raise ValueError("turn_gap_ms must be a positive number of milliseconds")
        if close_timeout_s < 0:
            raise ValueError("close_timeout_s must not be negative")
        self._api_key = SecretStr(api_key) if isinstance(api_key, str) else api_key
        self._model = model
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._delegation: HostedReasoning | IntegratorReasoning = (
            delegation if delegation is not None else IntegratorReasoning()
        )
        self._turn_gap_s = turn_gap_ms / 1000.0
        self._close_timeout_s = close_timeout_s
        self._states: dict[str, _LiveSession] = {}
        self._resampler = LinearResamplerProvider()

    # -- Identity and capabilities ------------------------------------------

    @property
    def name(self) -> str:
        return "OpenAILiveProvider"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def full_duplex(self) -> bool:
        return True

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        return False

    @property
    def delegation(self) -> HostedReasoning | IntegratorReasoning:
        """The delegation mode this provider opens its sessions with."""
        return self._delegation

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        """Curated, offline catalog of GPT-Live voices."""
        return list(_VOICES)

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of GPT-Live models."""
        return list(MODELS)

    def is_responding(self, session_id: str) -> bool:
        state = self._states.get(session_id)
        return state is not None and state.responding

    # -- Connection lifecycle -----------------------------------------------

    @staticmethod
    def _import_websockets() -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "websockets is required for OpenAILiveProvider. "
                "Install with: pip install 'roomkit[realtime-openai]'"
            ) from exc
        return websockets

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    def _build_session_config(
        self,
        *,
        system_prompt: str | None,
        voice: str | None,
        tools: list[dict[str, Any]],
        audio_format: dict[str, Any],
        pc: dict[str, Any],
    ) -> dict[str, Any]:
        config: dict[str, Any] = {"model": self._model}
        if system_prompt:
            config["instructions"] = system_prompt
        audio: dict[str, Any] = {"format": audio_format}
        if voice:
            audio["output"] = {"voice": voice}
        config["audio"] = audio
        config["delegation"] = self._delegation.to_config(tools)
        history = pc.get("history")
        if history:
            items = history_items(list(history))
            if items:
                config["input"] = items
        return config

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
        websockets = self._import_websockets()
        pc = dict(provider_config or {})
        tool_list = list(tools or [])

        # Built before opening the socket so validation errors fail fast.
        audio_format, law = build_audio_format(output_sample_rate, str(pc.get("codec", "pcm")))
        config = self._build_session_config(
            system_prompt=system_prompt,
            voice=voice,
            tools=tool_list,
            audio_format=audio_format,
            pc=pc,
        )
        if temperature is not None:
            logger.debug(
                "[%s] temperature ignored: the Live API has no sampling controls", _LOG_TAG
            )
        if not server_vad:
            logger.debug(
                "[%s] server_vad=False ignored: a full-duplex model takes no activity signals",
                _LOG_TAG,
            )
        codec = await asyncio.to_thread(_get_codec, law) if law is not None else None

        logger.info(
            "[%s →] session.start model=%s format=%s/%s delegation=%s (session %s)",
            _LOG_TAG,
            self._model,
            audio_format["type"],
            audio_format["rate"],
            config["delegation"]["type"],
            session.id,
        )
        ws = await asyncio.wait_for(
            websockets.connect(self._base_url, additional_headers=self._auth_headers()),
            timeout=_CONNECT_TIMEOUT,
        )
        state = _LiveSession(
            ws=ws,
            session=session,
            session_rate=int(audio_format["rate"]),
            input_rate=input_sample_rate,
            output_rate=output_sample_rate,
            codec=codec,
            system_prompt=system_prompt,
            voice=voice,
            tools=tool_list,
            provider_config=pc,
            user_turn=TurnGrouper(
                self._turn_gap_s,
                on_open=lambda: self._user_turn_opened(session),
                on_close=lambda text: self._user_turn_closed(session, text),
            ),
            assistant_turn=TurnGrouper(
                self._turn_gap_s,
                on_open=lambda: self._assistant_turn_opened(session),
                on_close=lambda text: self._assistant_turn_closed(session, text),
            ),
        )
        self._states[session.id] = state

        try:
            await ws.send(json.dumps({"type": EVT_SESSION_START, "session": config}))
        except BaseException:
            await self._discard(state)
            raise

        state.receive_task = asyncio.create_task(
            self._receive_loop(state), name=f"openai_live_recv:{session.id}"
        )
        try:
            await asyncio.wait_for(state.started.wait(), timeout=_CONNECT_TIMEOUT)
        except TimeoutError:
            await self._discard(state)
            raise TimeoutError(
                f"GPT-Live session did not start within {_CONNECT_TIMEOUT:.0f}s"
            ) from None
        if state.start_error is not None:
            await self._discard(state)
            raise ConnectionError(f"GPT-Live session failed to start: {state.start_error}")

        session.state = VoiceSessionState.ACTIVE
        logger.info("[%s] session connected: %s", _LOG_TAG, session.id)

    async def _receive_loop(self, state: _LiveSession) -> None:
        session_id = state.session.id
        try:
            async for raw in state.ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from %s for session %s", _LOG_TAG, session_id)
                    continue
                if not isinstance(event, dict):
                    continue
                try:
                    await self._handle_server_event(state, event)
                except Exception:
                    logger.exception(
                        "Error handling %s event for session %s", _LOG_TAG, session_id
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._retire_lost_connection(
                state, f"WebSocket closed unexpectedly for session {session_id}: {exc}"
            )
        else:
            await self._retire_lost_connection(state, f"WebSocket closed for session {session_id}")

    async def _retire_lost_connection(self, state: _LiveSession, message: str) -> None:
        """Release a socket whose receive loop stopped before disconnect()."""
        if self._states.get(state.session.id) is not state:
            return  # disconnect() already owns the teardown
        await self._discard(state, error_message=message)

    async def _discard(self, state: _LiveSession, *, error_message: str | None = None) -> None:
        """Remove one exact connection from the ownership map and close it."""
        session = state.session
        if self._states.get(session.id) is state:
            del self._states[session.id]
        state.user_turn.cancel()
        state.assistant_turn.cancel()
        was_active = session.state == VoiceSessionState.ACTIVE
        session.state = VoiceSessionState.ENDED
        if not state.started.is_set():
            state.start_error = state.start_error or (
                error_message or "connection closed before session.started"
            )
            state.started.set()
        state.closed.set()
        if error_message is not None and was_active:
            logger.warning("%s %s", _LOG_TAG, error_message)
            await self._fire(
                self._error_callbacks, session, "connection_closed", error_message, label="error"
            )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(state.ws.close(), timeout=_CLOSE_TIMEOUT)

    async def disconnect(self, session: VoiceSession) -> None:
        state = self._states.get(session.id)
        if state is None:
            session.state = VoiceSessionState.ENDED
            return

        # Deliver the finals of turns still open while the session can take
        # them; a session the channel already ended gets nothing more.
        if session.state == VoiceSessionState.ACTIVE:
            await state.assistant_turn.close()
            await state.user_turn.close()
        else:
            state.assistant_turn.cancel()
            state.user_turn.cancel()

        if state.started.is_set() and not state.closed.is_set():
            logger.debug("[%s →] session.close (session %s)", _LOG_TAG, session.id)
            with contextlib.suppress(Exception):
                await state.ws.send(json.dumps({"type": EVT_SESSION_CLOSE}))
            if self._close_timeout_s > 0:
                try:
                    await asyncio.wait_for(state.closed.wait(), timeout=self._close_timeout_s)
                except TimeoutError:
                    logger.warning(
                        "[%s] no session.closed within %.1fs (session %s)",
                        _LOG_TAG,
                        self._close_timeout_s,
                        session.id,
                    )

        if self._states.get(session.id) is state:
            del self._states[session.id]
        task = state.receive_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        with contextlib.suppress(Exception):
            await asyncio.wait_for(state.ws.close(), timeout=_CLOSE_TIMEOUT)
        state.user_turn.cancel()
        state.assistant_turn.cancel()
        session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for state in list(self._states.values()):
            await self.disconnect(state.session)

    # -- Inbound server events ----------------------------------------------

    async def _handle_server_event(self, state: _LiveSession, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type not in NOISY_EVENTS:
            logger.debug(
                "[%s ←] %s %s",
                _LOG_TAG,
                event_type,
                {k: v for k, v in event.items() if k not in ("type", "delta", "audio")},
            )
        handler_name = self._EVENT_HANDLERS.get(event_type)
        if handler_name is not None:
            await getattr(self, handler_name)(state, event)

    async def _on_session_started(self, state: _LiveSession, event: dict[str, Any]) -> None:
        live = event.get("session") or {}
        state.session.provider_session_id = str(live.get("id") or state.session.id)
        logger.info(
            "[%s] session.started id=%s expires_at=%s (session %s)",
            _LOG_TAG,
            live.get("id"),
            live.get("expires_at"),
            state.session.id,
        )
        state.started.set()

    async def _on_session_updated(self, state: _LiveSession, event: dict[str, Any]) -> None:
        logger.debug("[%s] session.updated (session %s)", _LOG_TAG, state.session.id)

    async def _on_session_closed(self, state: _LiveSession, event: dict[str, Any]) -> None:
        usage = event.get("usage") or {}
        seconds = usage.get("seconds")
        if isinstance(seconds, int | float):
            self._record_live_seconds(state, float(seconds))
        logger.info(
            "[%s] session.closed reason=%s (session %s)",
            _LOG_TAG,
            event.get("reason"),
            state.session.id,
        )
        state.closed.set()

    async def _on_usage_updated(self, state: _LiveSession, event: dict[str, Any]) -> None:
        seconds = (event.get("usage") or {}).get("seconds")
        if isinstance(seconds, int | float):
            self._record_live_seconds(state, float(seconds))

    async def _on_output_audio_delta(self, state: _LiveSession, event: dict[str, Any]) -> None:
        audio_b64 = event.get("delta", "")
        if not audio_b64:
            return
        audio = base64.b64decode(audio_b64)
        if state.codec is not None:
            audio = state.codec.decode(audio)
        await self._fire(self._audio_callbacks, state.session, audio, label="audio")

    async def _on_output_transcript_delta(
        self, state: _LiveSession, event: dict[str, Any]
    ) -> None:
        delta = event.get("delta", "")
        if not delta:
            return
        # Opening the turn fires response_start before the first partial.
        await state.assistant_turn.feed(delta)
        await self._fire(
            self._transcription_callbacks,
            state.session,
            delta,
            "assistant",
            False,
            label="transcription",
        )

    async def _on_input_transcript_delta(self, state: _LiveSession, event: dict[str, Any]) -> None:
        delta = event.get("delta", "")
        if not delta:
            return
        await state.user_turn.feed(delta)
        await self._fire(
            self._transcription_callbacks,
            state.session,
            delta,
            "user",
            False,
            label="transcription",
        )

    async def _assistant_turn_opened(self, session: VoiceSession) -> None:
        state = self._states.get(session.id)
        if state is not None:
            state.responding = True
        logger.info("[%s] response_start (synthesized, session %s)", _LOG_TAG, session.id)
        await self._fire(self._response_start_callbacks, session, label="response_start")

    async def _assistant_turn_closed(self, session: VoiceSession, text: str) -> None:
        if text.strip():
            await self._fire(
                self._transcription_callbacks,
                session,
                text.strip(),
                "assistant",
                True,
                label="transcription",
            )
        state = self._states.get(session.id)
        if state is not None:
            state.responding = False
        logger.info("[%s] response_end (synthesized, session %s)", _LOG_TAG, session.id)
        await self._fire(self._response_end_callbacks, session, label="response_end")

    async def _user_turn_opened(self, session: VoiceSession) -> None:
        logger.info("[%s] speech_start (synthesized, session %s)", _LOG_TAG, session.id)
        await self._fire(self._speech_start_callbacks, session, label="speech_start")

    async def _user_turn_closed(self, session: VoiceSession, text: str) -> None:
        if text.strip():
            await self._fire(
                self._transcription_callbacks,
                session,
                text.strip(),
                "user",
                True,
                label="transcription",
            )
        logger.info("[%s] speech_end (synthesized, session %s)", _LOG_TAG, session.id)
        await self._fire(self._speech_end_callbacks, session, label="speech_end")

    async def _on_delegation_created(self, state: _LiveSession, event: dict[str, Any]) -> None:
        delegation = event.get("delegation") or {}
        delegation_id = delegation.get("id")
        target = DELEGATION_TARGETS.get(str(delegation.get("target", "")))
        if not delegation_id or target is None:
            logger.warning(
                "[%s] delegation without id or known target ignored: %s (session %s)",
                _LOG_TAG,
                delegation,
                state.session.id,
            )
            return
        logger.info(
            "[%s] delegation %s → %s backend (session %s)",
            _LOG_TAG,
            delegation_id,
            target,
            state.session.id,
        )
        await self._fire(
            self._delegation_callbacks,
            state.session,
            str(delegation_id),
            target,
            label="delegation",
        )

    async def _on_response_event(self, state: _LiveSession, event: dict[str, Any]) -> None:
        """Dispatch a wrapped Responses lifecycle event (hosted backend)."""
        inner = event.get("event") or {}
        inner_type = str(inner.get("type", ""))
        key = str(event.get("delegation_id") or UNCORRELATED_DELEGATION)

        if inner_type == "response.created":
            state.pending.setdefault(key, PendingResponse())
        elif inner_type == "response.output_item.done":
            await self._on_backend_output_item(state, key, inner.get("item") or {})
        elif inner_type in ("response.completed", "response.incomplete", "response.failed"):
            response = inner.get("response") or {}
            if inner_type == "response.completed":
                self._record_backend_usage(state, response)
            else:
                detail = response.get("error") or response.get("incomplete_details") or {}
                message = (
                    (detail.get("message") or detail.get("reason"))
                    if isinstance(detail, dict)
                    else None
                )
                await self._fire(
                    self._error_callbacks,
                    state.session,
                    inner_type,
                    str(
                        message or response.get("status") or "delegated response did not complete"
                    ),
                    label="error",
                )
            pending = state.pending.get(key)
            if pending is not None:
                pending.finished = True
                await self._maybe_continue_response(state, key)
        else:
            logger.debug(
                "[%s] %s (session %s)", _LOG_TAG, inner_type or "response.event", state.session.id
            )

    async def _on_backend_output_item(
        self, state: _LiveSession, key: str, item: dict[str, Any]
    ) -> None:
        if item.get("type") != "function_call":
            return
        if item.get("status", "completed") != "completed":
            logger.debug("[%s] ignoring %s function call item", _LOG_TAG, item.get("status"))
            return
        call_id = item.get("call_id")
        name = item.get("name")
        if not call_id or not name:
            logger.warning("[%s] function call item without call_id or name: %s", _LOG_TAG, item)
            return
        if call_id in state.open_calls:
            logger.warning("[%s] function call %s already in progress", _LOG_TAG, call_id)
            return
        raw_args = item.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (json.JSONDecodeError, TypeError, ValueError):
            arguments = {"raw": raw_args}

        pending = state.pending.setdefault(key, PendingResponse())
        pending.call_ids.add(str(call_id))
        pending.had_calls = True
        state.open_calls[str(call_id)] = key
        await self._fire(
            self._tool_call_callbacks,
            state.session,
            str(call_id),
            str(name),
            arguments,
            label="tool_call",
        )

    async def _maybe_continue_response(self, state: _LiveSession, key: str) -> None:
        """Resume the backend once every call of its response has an output."""
        pending = state.pending.get(key)
        if pending is None or not pending.finished or pending.call_ids:
            return
        del state.pending[key]
        if not pending.had_calls:
            return  # a text-only response needs no continuation
        logger.debug("[%s →] response.create (delegation %s)", _LOG_TAG, key)
        await state.ws.send(json.dumps({"type": EVT_RESPONSE_CREATE}))

    async def _on_error(self, state: _LiveSession, event: dict[str, Any]) -> None:
        error = event.get("error") or {}
        code = str(error.get("code") or error.get("type") or "unknown")
        message = str(error.get("message") or "Unknown error")
        if error.get("param"):
            message = f"{message} (param: {error['param']})"
        logger.error("[%s] error [%s] %s (session %s)", _LOG_TAG, code, message, state.session.id)
        if not state.started.is_set():
            # No session has started on this connection, so this one will not.
            state.start_error = f"{code}: {message}"
            state.started.set()
            return
        await self._fire(self._error_callbacks, state.session, code, message, label="error")

    # -- Usage ---------------------------------------------------------------

    def _record_live_seconds(self, state: _LiveSession, seconds: float) -> None:
        """Record the live model's cumulative audio duration, never as tokens."""
        delta = max(0.0, seconds - state.live_seconds)
        state.live_seconds = seconds
        state.session._last_usage["live_seconds"] = seconds
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None and delta > 0:
            telemetry.record_metric(
                "roomkit.realtime.live_seconds",
                delta,
                unit="s",
                attributes={"session_id": state.session.id, Attr.MODEL: self._model},
            )

    def _record_backend_usage(self, state: _LiveSession, response: dict[str, Any]) -> None:
        """Attribute a hosted backend response's tokens to the backend model."""
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return
        backend_model = str(
            response.get("model")
            or (self._delegation.model if isinstance(self._delegation, HostedReasoning) else "")
        )
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        record = {
            "model": backend_model,
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cached_tokens": int(input_details.get("cached_tokens") or 0),
            "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
        }
        state.session._last_usage["backend"] = record
        logger.info(
            "[%s] backend usage model=%s input=%d output=%d (session %s)",
            _LOG_TAG,
            backend_model,
            record["input_tokens"],
            record["output_tokens"],
            state.session.id,
        )
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            attrs = {"session_id": state.session.id, Attr.MODEL: backend_model}
            telemetry.record_metric(
                "roomkit.realtime.input_tokens",
                float(record["input_tokens"]),
                unit="tokens",
                attributes=attrs,
            )
            telemetry.record_metric(
                "roomkit.realtime.output_tokens",
                float(record["output_tokens"]),
                unit="tokens",
                attributes=attrs,
            )

    # -- Outbound client API ------------------------------------------------

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

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        state = self._states.get(session.id)
        if state is None:
            return
        key = state.open_calls.pop(call_id, None)
        if key is None:
            logger.warning(
                "[%s] tool result for unknown call %s dropped (session %s)",
                _LOG_TAG,
                call_id,
                session.id,
            )
            return
        logger.debug(
            "[%s →] response.item.create call=%s (session %s)", _LOG_TAG, call_id, session.id
        )
        await state.ws.send(
            json.dumps(
                {
                    "type": EVT_RESPONSE_ITEM_CREATE,
                    "item": {"type": "function_call_output", "call_id": call_id, "output": result},
                }
            )
        )
        pending = state.pending.get(key)
        if pending is not None:
            pending.call_ids.discard(call_id)
        await self._maybe_continue_response(state, key)

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
        """Apply what the fixed session can take; reconnect for the rest.

        A changed system prompt is appended to the instructions and, in
        hosted mode, a changed tool list reaches the backend through
        ``session.update``. A voice change (or a new codec) needs a new
        session: the connection is replaced and reseeded from the remembered
        settings (RFC §12.4.1).
        """
        state = self._states.get(session.id)
        if state is None:
            await super().reconfigure(
                session,
                system_prompt=system_prompt,
                voice=voice,
                tools=tools,
                temperature=temperature,
                provider_config=provider_config,
            )
            return
        if temperature is not None:
            logger.debug("[%s] temperature ignored on reconfigure", _LOG_TAG)

        new_codec = (provider_config or {}).get("codec")
        needs_restart = (voice is not None and voice != state.voice) or (
            new_codec is not None and new_codec != state.provider_config.get("codec", "pcm")
        )
        if needs_restart:
            await self._restart(
                state,
                system_prompt=system_prompt,
                voice=voice,
                tools=tools,
                provider_config=provider_config,
            )
            return

        if system_prompt is not None and system_prompt != state.system_prompt:
            logger.info(
                "[%s →] instructions append (reconfigure, session %s)", _LOG_TAG, session.id
            )
            await self._send_append(state, EVT_INSTRUCTIONS_APPEND, None, system_prompt)
            state.system_prompt = system_prompt

        if tools is not None:
            new_tools = list(tools)
            if isinstance(self._delegation, HostedReasoning) and format_backend_tools(
                new_tools
            ) != format_backend_tools(state.tools):
                logger.info(
                    "[%s →] session.update tools=%d (session %s)",
                    _LOG_TAG,
                    len(new_tools),
                    session.id,
                )
                await state.ws.send(
                    json.dumps(
                        {
                            "type": EVT_SESSION_UPDATE,
                            "session": {
                                "delegation": {
                                    "type": "responses",
                                    "responses": {"tools": format_backend_tools(new_tools)},
                                }
                            },
                        }
                    )
                )
            state.tools = new_tools

    async def _restart(
        self,
        state: _LiveSession,
        *,
        system_prompt: str | None,
        voice: str | None,
        tools: list[dict[str, Any]] | None,
        provider_config: dict[str, Any] | None,
    ) -> None:
        session = state.session
        merged_pc = {**state.provider_config, **(provider_config or {})}
        logger.info("[%s] replacing session %s (voice or codec changed)", _LOG_TAG, session.id)
        await self.disconnect(session)
        # The participant's session did not end — only the upstream connection
        # did (RFC §12.1 forbids a transition out of ENDED otherwise).
        session.renegotiate()
        await self.connect(
            session,
            system_prompt=system_prompt if system_prompt is not None else state.system_prompt,
            voice=voice if voice is not None else state.voice,
            tools=tools if tools is not None else state.tools,
            input_sample_rate=state.input_rate,
            output_sample_rate=state.output_rate,
            provider_config=merged_pc,
        )
