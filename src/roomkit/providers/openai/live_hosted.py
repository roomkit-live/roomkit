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
    PendingResponse,
)
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
