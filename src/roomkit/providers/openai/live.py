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

The provider is assembled from four modules: this one owns construction,
the connection lifecycle and reconfiguration; ``live_client`` sends the
in-session client events; ``live_handlers`` translates server events into
callbacks; ``live_hosted`` carries the hosted delegation's Responses
envelope. ``live_config`` holds what a session is opened with.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from pydantic import SecretStr

from roomkit.core.task_utils import cancel_and_wait, log_task_exception
from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.ai.tool_declaration import ToolNameRule
from roomkit.providers.openai.ai import OPENAI_TOOL_NAMES
from roomkit.providers.openai.live_client import OpenAILiveClientMixin
from roomkit.providers.openai.live_config import (
    _ACKNOWLEDGED_CLOSE_TIMEOUT,
    _CLOSE_TIMEOUT,
    _CONNECT_TIMEOUT,
    _DEFAULT_BASE_URL,
    _DEFAULT_MODEL,
    _LOG_TAG,
    _VOICES,
    HostedReasoning,
    IntegratorReasoning,
    _LiveSession,
)
from roomkit.providers.openai.live_events import (
    EVT_INSTRUCTIONS_APPEND,
    EVT_SESSION_CLOSE,
    EVT_SESSION_START,
    EVT_SESSION_UPDATE,
    TurnGrouper,
    build_audio_format,
    format_backend_tools,
    history_items,
    tokenizer,
)
from roomkit.providers.openai.live_handlers import OpenAILiveEventHandlersMixin
from roomkit.providers.openai.live_hosted import OpenAILiveHostedDelegationMixin
from roomkit.providers.openai.live_models import MODELS
from roomkit.providers.vendor_endpoint import is_vendor_endpoint
from roomkit.voice._g711 import _get_codec
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.realtime.provider import VoiceInfo

__all__ = ["HostedReasoning", "IntegratorReasoning", "OpenAILiveProvider"]

logger = logging.getLogger("roomkit.providers.openai.live")


class OpenAILiveProvider(
    OpenAILiveClientMixin, OpenAILiveHostedDelegationMixin, OpenAILiveEventHandlersMixin
):
    """Speech-to-speech provider for OpenAI GPT-Live (``/v1/live/sessions``).

    **Full-duplex.** The model handles being talked over by itself, so
    :attr:`full_duplex` is ``True``: the channel never flushes playback or
    gates the model's audio on user speech, :meth:`interrupt` and
    :meth:`truncate_audio` do nothing, and a pipeline VAD stays in the
    observation role. The wire carries no response or speech boundaries;
    this provider synthesizes them from transcript deltas and assistant audio
    activity, with a quiet gap of ``turn_gap_ms`` per speaker (RFC §12.4.1).

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
    own words rather than reading it. The session takes no user text once
    started, so ``user`` is what the model will say, never a turn it answers:
    an instruction, the opening greeting included, is ``role="system"``
    (OpenAI's documented way to have the model speak first), and a line the
    application wrote (``role="assistant"``) becomes an instructions append
    asking the model to say it — as commentary, the model improvised its own. An append is
    measured with the model's
    tokenizer (``tiktoken``, installed with the extra; UTF-8 bytes bound it
    when that is missing) and split on sentence boundaries only where it
    exceeds the API's per-append bound — the model voices each piece of a
    split commentary. Image injection is not available on the Live endpoint.

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
        # Sockets whose protocol is already over, still finishing their TCP
        # close on their own. See :meth:`_release_socket`.
        self._deferred_closes: set[asyncio.Task[None]] = set()

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

    @property
    def _tool_name_rule(self) -> ToolNameRule | None:
        """The tool names the hosted backend accepts: an OpenAI Responses
        model on OpenAI's own endpoint takes OpenAI's rule, and the session
        accepts any name, so a refused one would only fail the delegation
        later, opaquely (measured on ``gpt-live-1``, RFC §6.7). The integrator
        backend's own provider checks its tools; behind ``base_url`` the
        server decides."""
        own = is_vendor_endpoint(self._base_url, _DEFAULT_BASE_URL)
        if isinstance(self._delegation, HostedReasoning) and own:
            return OPENAI_TOOL_NAMES
        return None

    def _check_tool_names(self, tools: list[dict[str, Any]]) -> None:
        """Raise before anything is sent for a tool name the backend refuses."""
        rule = self._tool_name_rule
        if rule is not None:
            rule.check(
                tool.get("name", "")
                for tool in tools
                if tool.get("type", "function") == "function"
            )

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
        self._check_tool_names(tool_list)

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
        # Appends are measured with the model's tokenizer, whose table is
        # fetched on first use: load it here so the session's first append
        # does not wait for it.
        await tokenizer()

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

        state.receive_task = self._session_task(
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
        await self._abandon_open_calls(state)
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
        await self._close_socket(state)

    async def _abandon_open_calls(self, state: _LiveSession) -> None:
        """Report the hosted calls this connection leaves unanswered: no other
        connection issued them, so their results will never be read (RFC §12.4)."""
        await self._abandon_open_tool_calls(state.session, where=lambda booked: booked[0] is state)

    async def _close_socket(self, state: _LiveSession, *, timeout: float | None = None) -> None:
        """Close one socket under the library's own bound.

        ``close_timeout`` is what websockets consults once the close frame is
        out: on expiry it aborts the transport, so the socket is released
        whether or not the peer answers. Cancelling ``close()`` from outside
        would bound the wait and leave the transport open. The outer wait only
        nets a send stalled on flow control, which that deadline does not
        cover, and aborts the transport itself.
        """
        if timeout is None:
            timeout = _CLOSE_TIMEOUT
        ws = state.ws
        with contextlib.suppress(Exception):
            ws.close_timeout = timeout
            try:
                await asyncio.wait_for(ws.close(), timeout=timeout + _CLOSE_TIMEOUT)
            except TimeoutError:
                ws.transport.abort()

    def _release_socket(self, state: _LiveSession) -> None:
        """Close a socket whose protocol is over without waiting for the peer.

        Once ``session.closed`` has landed there is nothing behind the close
        handshake: the billed seconds rode that event and the turns are
        already settled. The API answers no close frame after it and drops
        the TCP connection itself about two seconds later (measured with the
        bare ``websockets`` client, whether or not a close frame was sent),
        so ``ws.close()``, which waits for the handshake and the TCP close,
        would buy those two seconds and nothing else. The close frame is still
        sent; a ``close_timeout`` of zero then makes the library abort the
        transport at once. That bound applies while no close deadline exists
        yet, which is the case here: the peer has sent no close frame. It runs
        on its own task, off the path of a caller that is usually tearing a
        call down; :meth:`close` is where it is awaited again, and finds it
        done.
        """
        task = asyncio.create_task(
            self._close_socket(state, timeout=_ACKNOWLEDGED_CLOSE_TIMEOUT),
            name=f"roomkit-live-close-{state.session.id}",
        )
        self._deferred_closes.add(task)
        task.add_done_callback(self._deferred_closes.discard)
        task.add_done_callback(log_task_exception)

    async def disconnect(self, session: VoiceSession) -> None:
        state = self._states.get(session.id)
        if state is None:
            session.state = VoiceSessionState.ENDED
            return
        state.closing = True
        await self._abandon_open_calls(state)

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
        await cancel_and_wait(state.receive_task, log_errors_to=logger)
        # Either ``session.closed`` landed — the protocol is over — or
        # ``_discard`` won the race and already closed this socket. Both mean
        # the teardown has nothing left to deliver, so nobody waits on it.
        if state.closed.is_set():
            self._release_socket(state)
        else:
            await self._close_socket(state)
        state.user_turn.cancel()
        state.assistant_turn.cancel()
        session.state = VoiceSessionState.ENDED

    async def close(self) -> None:
        for state in list(self._states.values()):
            await self.disconnect(state.session)
        # ``disconnect`` hands an acknowledged socket to a task, closed without
        # waiting for the peer. The provider-wide teardown is where that task
        # is finally somebody's: ``close()`` releases every resource, so it is
        # awaited here, and costs what the abort cost, which is nothing.
        deferred = list(self._deferred_closes)
        if deferred:
            await asyncio.gather(*deferred, return_exceptions=True)

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
        # Before anything is sent or torn down: a refused name leaves the
        # session as it was.
        if tools is not None:
            self._check_tool_names(list(tools))

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
        # The disconnect reports the calls the old connection left open; one
        # whose own handler asked for this restart (a handoff) is the
        # channel's to spare.
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
