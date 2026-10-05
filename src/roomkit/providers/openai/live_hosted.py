"""Hosted reasoning delegation for the OpenAI GPT-Live provider.

In the hosted mode OpenAI runs the backend model and wraps its Responses
lifecycle in ``response.event`` envelopes. This mixin collects the function
calls each delegated response makes, hands them to ``on_tool_call``, and
resumes the backend with ``response.create`` only once every call of that
response has an output — the hosted service rejects a partial continuation
(RFC §12.4.1, hosted backend). Backend token usage is attributed to the
backend model, never to the live one.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from roomkit.providers.ai.tool_calls import realtime_call_arguments
from roomkit.providers.openai.live_config import (
    _LOG_TAG,
    HostedReasoning,
    IntegratorReasoning,
    _LiveSession,
)
from roomkit.providers.openai.live_events import (
    EVT_RESPONSE_CREATE,
    EVT_RESPONSE_ITEM_CREATE,
    UNCORRELATED_DELEGATION,
)
from roomkit.providers.openai.response_calls import PendingResponse
from roomkit.telemetry.base import Attr
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.providers.openai.live")


class OpenAILiveHostedDelegationMixin(RealtimeVoiceProvider):
    """Responses-envelope handling and tool-result submission for the hosted backend.

    Mixed into ``OpenAILiveProvider``, which owns ``_states`` and the
    delegation mode.
    """

    # Connection state owned by OpenAILiveProvider.__init__; declared for typing.
    _states: dict[str, _LiveSession]
    _delegation: HostedReasoning | IntegratorReasoning

    # Nested Responses event type → handler. The three terminal types share one.
    _RESPONSE_HANDLERS: dict[str, str] = {
        "response.created": "_on_backend_response_created",
        "response.output_item.done": "_on_backend_output_item_done",
        "response.completed": "_on_backend_response_finished",
        "response.incomplete": "_on_backend_response_finished",
        "response.failed": "_on_backend_response_finished",
    }

    async def _on_response_event(self, state: _LiveSession, event: dict[str, Any]) -> None:
        """Dispatch a wrapped Responses lifecycle event (hosted backend)."""
        inner = event.get("event") or {}
        inner_type = str(inner.get("type", ""))
        key = str(event.get("delegation_id") or UNCORRELATED_DELEGATION)
        handler_name = self._RESPONSE_HANDLERS.get(inner_type)
        if handler_name is None:
            logger.debug(
                "[%s] %s (session %s)", _LOG_TAG, inner_type or "response.event", state.session.id
            )
            return
        await getattr(self, handler_name)(state, key, inner)

    async def _on_backend_response_created(
        self, state: _LiveSession, key: str, inner: dict[str, Any]
    ) -> None:
        state.pending.setdefault(key, PendingResponse())

    async def _on_backend_output_item_done(
        self, state: _LiveSession, key: str, inner: dict[str, Any]
    ) -> None:
        await self._on_backend_output_item(state, key, inner.get("item") or {})

    async def _on_backend_response_finished(
        self, state: _LiveSession, key: str, inner: dict[str, Any]
    ) -> None:
        """The run emitted all of its items: account for it and resume if it owes nothing."""
        inner_type = str(inner.get("type", ""))
        response = inner.get("response") or {}
        if inner_type == "response.completed":
            self._record_backend_usage(state, response)
        else:
            await self._report_backend_failure(state, inner_type, response)
        pending = state.pending.get(key)
        if pending is not None:
            pending.finished = True
            await self._maybe_continue_response(state, key)

    async def _report_backend_failure(
        self, state: _LiveSession, inner_type: str, response: dict[str, Any]
    ) -> None:
        detail = response.get("error") or response.get("incomplete_details") or {}
        message = (
            (detail.get("message") or detail.get("reason")) if isinstance(detail, dict) else None
        )
        await self._fire(
            self._error_callbacks,
            state.session,
            inner_type,
            str(message or response.get("status") or "delegated response did not complete"),
            label="error",
        )

    async def _on_backend_output_item(
        self, state: _LiveSession, key: str, item: dict[str, Any]
    ) -> None:
        if item.get("type") != "function_call":
            return
        name = str(item.get("name") or "")
        if not name:
            # Handed on all the same: the channel refuses it and answers under
            # its id (RFC §12.4).
            logger.warning("[%s] function call item without a name: %s", _LOG_TAG, item)
        call_id = str(item.get("call_id") or "")
        # A call the output cap cut is handed on too, as OpenAI Realtime's: it
        # runs when its argument text reads, else the channel refuses it as
        # cut off and reports it (RFC §6.4, §12.4).
        cut = item.get("status") == "incomplete"
        arguments = realtime_call_arguments(item.get("arguments"), cut=cut)

        # Booked with the connection that issued it and the delegation it
        # answers: only that connection's end abandons it (RFC §12.4).
        if self._book_tool_call(state.session, call_id, (state, key)):
            # Only a call the channel may answer holds the response open: one
            # without an id, or under an id in flight, is refused and reported
            # with nothing sent (RFC §12.4).
            pending = state.pending.setdefault(key, PendingResponse())
            pending.call_ids.add(call_id)
            pending.had_calls = True
        await self._fire(
            self._tool_call_callbacks,
            state.session,
            call_id,
            name,
            arguments,
            label="tool_call",
        )

    async def _maybe_continue_response(self, state: _LiveSession, key: str) -> None:
        """Resume the backend once every call of its response has an output."""
        pending = state.pending.get(key)
        if pending is None or not pending.settled:
            return
        del state.pending[key]
        if not pending.ready_to_continue:
            return  # a text-only response needs no continuation
        logger.debug("[%s →] response.create (delegation %s)", _LOG_TAG, key)
        await state.ws.send(json.dumps({"type": EVT_RESPONSE_CREATE}))

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
        self._publish_usage(state.session)
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

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        state = self._states.get(session.id)
        if state is None:
            return
        held, booked = self._answerable_tool_call(session, call_id)
        if not held:
            return
        _, key = booked
        logger.debug(
            "[%s →] response.item.create call=%s (session %s)", _LOG_TAG, call_id, session.id
        )
        # Off the response's books before the send yields, as off the open
        # calls: a call issued under the id meanwhile is a new call, which
        # holds its response (RFC §12.4).
        pending = state.pending.get(key)
        if pending is not None:
            pending.call_ids.discard(call_id)
        await state.ws.send(
            json.dumps(
                {
                    "type": EVT_RESPONSE_ITEM_CREATE,
                    "item": {"type": "function_call_output", "call_id": call_id, "output": result},
                }
            )
        )
        await self._maybe_continue_response(state, key)
