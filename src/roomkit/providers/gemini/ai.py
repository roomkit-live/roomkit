"""Google Gemini AI provider — generates responses via the Google Generative AI API.

The provider owns the client, the call and the stream. Turning RoomKit
messages and context into the request lives in ``request.py``.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    AIToolCall,
    FirstToken,
    ModelInfo,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    answered_by,
    tool_call_of,
)
from roomkit.providers.ai.response_schema import checked_stream, schema_for_generate
from roomkit.providers.ai.tool_calls import CallIds
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.gemini.errors import (
    REFUSAL_FINISH_REASONS,
    prompt_block_reason,
    reason_name,
    wrap_gemini_error,
)
from roomkit.providers.gemini.models import MODELS, TAKES_LEVELS
from roomkit.providers.gemini.request import (
    build_gen_config,
    format_messages,
    reject_model_turn_tail,
)
from roomkit.providers.gemini.sdk import build_genai_client, close_genai_client
from roomkit.providers.utils import _aclose_stream

logger = logging.getLogger(__name__)


def _refusal(done: StreamDone) -> str | None:
    """A refusal stop reason, or the reason the prompt itself was blocked."""
    if done.finish_reason in REFUSAL_FINISH_REASONS:
        return done.finish_reason
    return done.metadata.get("prompt_block_reason")


def _parts_layout(parts: list[Any]) -> str:
    """Compact one-line summary of a streamed chunk's parts for diagnostics.

    Renders each part as ``kind(sig=yes|no)`` so a thought_signature that
    arrives on a different part (or a later chunk) than its function_call
    is visible. ``kind`` is fcall:<name> / thought / text / other.
    """
    out: list[str] = []
    for p in parts:
        has_sig = getattr(p, "thought_signature", None) is not None
        fc = getattr(p, "function_call", None)
        if fc is not None:
            kind = f"fcall:{getattr(fc, 'name', '?')}"
        elif getattr(p, "thought", False):
            kind = "thought"
        elif getattr(p, "text", None):
            kind = "text"
        else:
            kind = "other"
        out.append(f"{kind}(sig={'yes' if has_sig else 'no'})")
    return " ".join(out) or "(empty)"


def _usage_from_metadata(meta: Any) -> dict[str, int]:
    """Map Gemini's usage metadata to roomkit's canonical, disjoint counters.

    The SDK calls them prompt and candidates counts. The prompt count includes
    implicitly cached tokens, reported apart so the cached prefix is not
    charged twice and cache rates apply. Thinking is billed as output but
    counted outside candidates: it joins ``output_tokens``, and its share is
    also exposed as ``reasoning_tokens``, a detail never priced on its own
    (RFC §6, usage counters).
    """
    prompt = meta.prompt_token_count or 0
    cached = getattr(meta, "cached_content_token_count", None) or 0
    thoughts = getattr(meta, "thoughts_token_count", None) or 0
    usage = {
        "input_tokens": max(prompt - cached, 0),
        "output_tokens": (meta.candidates_token_count or 0) + thoughts,
    }
    if cached:
        usage["cache_read_input_tokens"] = cached
    if thoughts:
        usage["reasoning_tokens"] = thoughts
    return usage


def _call_key(
    fc: Any,
    name: str,
    args: dict[str, Any],
    in_chunk: dict[str, int],
    calls: dict[str, dict[str, Any]],
) -> str:
    """Which call of the response a function-call part is.

    Gemini can re-emit a call in a later chunk (the first carrying its
    thought_signature), and a re-emission must fold into it; but two
    identical calls in one chunk ("roll two dice") are two calls. A call's
    own id identifies it when an earlier part carried the same; otherwise its
    name and arguments do, counted within the chunk, so a later chunk's
    re-emission lands on the first occurrence whether either copy carries an
    id or not. Two copies carrying different ids are two calls (RFC §6.4).
    """
    call_id = getattr(fc, "id", None)
    if call_id:
        for key, call in calls.items():
            if call["server_id"] == call_id:
                return key
    try:
        fingerprint = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        fingerprint = repr(args)
    base = f"{name}::{fingerprint}"
    occurrence = in_chunk.get(base, 0)
    in_chunk[base] = occurrence + 1
    key = f"{base}#{occurrence}"
    held = calls.get(key)
    if call_id and held is not None and held["server_id"] not in (None, call_id):
        return f"{key}@{call_id}"
    return key


def _fold_function_call(
    part: Any,
    in_chunk: dict[str, int],
    calls: dict[str, dict[str, Any]],
    order: list[str],
) -> None:
    """Fold a function-call part into the response's calls: a new call, or a
    re-emission of one, which keeps what either copy carries (RFC §6.4)."""
    fc = part.function_call
    name = fc.name or ""
    args = dict(fc.args) if fc.args else {}
    # thought_signature lives on the Part (bytes), not the FunctionCall.
    # Encode to a portable str for metadata.
    raw_sig = getattr(part, "thought_signature", None)
    sig = base64.b64encode(raw_sig).decode("ascii") if isinstance(raw_sig, bytes) else raw_sig
    key = _call_key(fc, name, args, in_chunk, calls)
    held = calls.get(key)
    if held is None:
        calls[key] = {
            "server_id": getattr(fc, "id", None) or None,
            "name": name,
            "arguments": args,
            "signature": sig,
        }
        order.append(key)
        return
    held["signature"] = held["signature"] or sig
    held["server_id"] = held["server_id"] or getattr(fc, "id", None)


class GeminiAIProvider(AIProvider):
    """AI provider using the Google Gemini API."""

    def __init__(self, config: GeminiConfig) -> None:
        self._config = config
        # The client carries the connect/read split; see ``build_genai_client``
        # for why it cannot go on the request.
        self._client, self._http, self._types = build_genai_client(
            config, provider="GeminiAIProvider", api_key=config.api_key.get_secret_value()
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    @property
    def supports_vision(self) -> bool:
        """All Gemini models support vision."""
        return True

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Gemini models."""
        return list(MODELS)

    async def list_models(self) -> list[ModelInfo]:
        """List generate-content models the Gemini API currently exposes.

        Serves both surfaces this provider family speaks to, which name their
        models differently and describe them unequally:

        - Developer API (AI Studio): ``models/gemini-3.5-flash``, and each entry
          declares ``supported_actions``.
        - Vertex (:class:`~roomkit.providers.gemini.vertex.GeminiVertexProvider`):
          ``publishers/google/models/gemini-2.5-flash``, with no
          ``supported_actions`` and no metadata at all.

        So the id is the last path segment rather than one stripped prefix — a
        prefixed id matches nothing in the curated catalog, which silently
        emptied the metadata, and would be written to a caller's config as a
        model name the API then rejects. And where no action is declared, the
        listing mixes in models this call cannot serve, which the family and
        embedding checks drop without hiding a model too new to be curated.
        """
        live: list[ModelInfo] = []
        pager = await self._client.aio.models.list()  # ty: ignore[unresolved-attribute]
        curated_ids = {m.id for m in self.available_models()}
        async for m in pager:
            name = (m.name or "").rsplit("/", 1)[-1]
            actions = getattr(m, "supported_actions", None) or []
            if not name or (actions and "generateContent" not in actions):
                continue
            if not actions:
                # Vertex declares no actions, so the name is the only signal
                # left: keep the Gemini generative family (curated, or new
                # enough that the catalog has not caught up) and drop the
                # embedding line, which answers embedContent, not this call.
                generative = name in curated_ids or name.startswith("gemini-")
                if not generative or "embedding" in name:
                    continue
            live.append(
                ModelInfo(
                    id=name,
                    display_name=getattr(m, "display_name", None),
                    context_window=getattr(m, "input_token_limit", None),
                )
            )
        return self._listing(live)

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Wrap an SDK exception into a ProviderError."""
        return wrap_gemini_error(exc)

    def _build_gen_config(self, context: AIContext) -> Any:
        """The generation config for one turn, overridable by a subclass whose
        config carries a field this API refuses (Vertex's billing labels)."""
        entry = self.catalog_entry()
        capabilities = entry.capabilities if entry is not None else []
        return build_gen_config(self._types, self._config, context, capabilities)

    def _calls_need_signatures(self) -> bool:
        """Whether the model refuses a function call without its thought
        signature: a Gemini 3 model, the one family that takes thinking
        levels (measured 2026-10-03 on ``gemini-3.8-flash``), or a model the
        catalogue does not know, taken as recent; its own calls are signed
        either way, so only another vendor's round changes form."""
        entry = self.catalog_entry()
        if entry is None or self._config.thinking_level is not None:
            return True
        return TAKES_LEVELS in entry.capabilities

    @property
    def supports_response_schema(self) -> bool:
        """Controlled generation, through ``response_json_schema``."""
        return True

    @property
    def supports_response_schema_with_tools(self) -> bool:
        """Function calling and ``response_json_schema`` share a request
        (verified live on ``gemini-3.8-flash``, 2026-09-27)."""
        return True

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events from the Gemini streaming API.

        A response schema is checked before the done event (RFC §6.7).
        """
        schema_for_generate(
            context,
            supported=self.supports_response_schema,
            with_tools=self.supports_response_schema_with_tools,
            provider="gemini",
        )
        stream = checked_stream(
            self._events(context),
            context,
            provider="gemini",
            refusal=_refusal,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await _aclose_stream(stream)

    async def _events(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """The streamed call itself, shared by :meth:`generate`."""
        gen_config = self._build_gen_config(context)
        contents = format_messages(
            self._types, context.messages, signed_calls=self._calls_need_signatures()
        )
        # Before the try below, whose ``_wrap_error`` is for SDK exceptions and
        # would restate this one as an opaque provider failure.
        reject_model_turn_tail(contents)

        first_token = FirstToken(self)
        # Gemini can stream the same function call across several chunks — the
        # first carries its thought_signature, a later one re-emits the call
        # without it. Naively appending one tool call per part produced a
        # duplicate with no signature, which Gemini 3 then rejects with a 400
        # ("Function call is missing a thought_signature") on the next turn.
        # Accumulate by (name, args) and keep the signature from whichever
        # emission carried it, so each distinct call surfaces exactly once.
        fcalls: dict[str, dict[str, Any]] = {}
        fcall_order: list[str] = []
        usage: dict[str, int] = {}
        finish_reason: str | None = None
        block_reason: str | None = None
        answered: str | None = None

        try:
            response_stream = await self._client.aio.models.generate_content_stream(  # ty: ignore[unresolved-attribute]
                model=self._config.model,
                contents=contents,
                config=gen_config,
            )
            async for chunk in response_stream:
                # Extract usage from each chunk (last one has the totals)
                if chunk.usage_metadata:
                    usage = _usage_from_metadata(chunk.usage_metadata)
                answered = getattr(chunk, "model_version", None) or answered

                block_reason = prompt_block_reason(chunk) or block_reason
                # Read before the content guards below: the chunk that reports
                # MAX_TOKENS is often the one whose candidate carries no parts,
                # so capturing it after them would drop the very case the tool
                # loop needs — a round truncated with nothing to show for it.
                if chunk.candidates:
                    finish_reason = (
                        reason_name(getattr(chunk.candidates[0], "finish_reason", None))
                        or finish_reason
                    )

                if not chunk.candidates or not chunk.candidates[0].content:
                    continue

                parts = chunk.candidates[0].content.parts
                if not parts:
                    continue

                # Diagnostic: log the chunk's part layout whenever a function
                # call or a thought_signature is present, so WHERE Gemini puts
                # the signature in streaming (on the function_call part, on a
                # thought part, or on a later chunk) stays observable when a
                # round turns out to replay badly.
                if logger.isEnabledFor(logging.DEBUG) and any(
                    getattr(p, "function_call", None) is not None
                    or getattr(p, "thought_signature", None) is not None
                    for p in parts
                ):
                    logger.debug("Gemini stream chunk parts: %s", _parts_layout(parts))

                # A call re-emitted in a later chunk folds into the first; two
                # identical calls of one chunk are two calls (RFC §6.4).
                in_chunk: dict[str, int] = {}
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        first_token.seen()
                        # Thought-summary parts are flagged thought=True.
                        if getattr(part, "thought", False):
                            yield StreamThinkingDelta(thinking=part.text)
                        else:
                            yield StreamTextDelta(text=part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        _fold_function_call(part, in_chunk, fcalls, fcall_order)

            if fcall_order:
                logger.debug(
                    "Gemini tool calls finalized: %s",
                    [
                        f"{fcalls[k]['name']}({'sig' if fcalls[k]['signature'] else 'NOSIG'})"
                        for k in fcall_order
                    ],
                )
                if not any(fcalls[k]["signature"] for k in fcall_order):
                    # Gemini signs the first functionCall part of a round only,
                    # and ``format_messages`` lends that signature to the
                    # round's other calls — so a single signature anywhere in
                    # the round replays fine and is not worth a word. None at
                    # all is the case with nothing to lend: if the model was
                    # thinking, Gemini 3 rejects the whole round on the next
                    # turn ("Function call is missing a thought_signature").
                    logger.warning(
                        "Gemini round of %d function call(s) %s carries no "
                        "thought_signature (thinking config %s) — "
                        "nothing to replay them signed with; Gemini 3 rejects the next "
                        "turn when the model was thinking",
                        len(fcall_order),
                        [fcalls[k]["name"] for k in fcall_order],
                        getattr(gen_config, "thinking_config", None),
                    )
            # Each call takes the server's id when no earlier call of the
            # response took it, a minted one otherwise (RFC §6.4); a
            # re-emission already folded into its call above.
            call_ids = CallIds()
            for key in fcall_order:
                fc_data = fcalls[key]
                meta: dict[str, Any] = {}
                if fc_data["signature"] is not None:
                    meta["thought_signature"] = fc_data["signature"]
                yield StreamToolCall(
                    id=call_ids(fc_data["server_id"], fc_data["name"]),
                    name=fc_data["name"],
                    arguments=fc_data["arguments"],
                    metadata=meta,
                )

            yield StreamDone(
                usage=usage,
                finish_reason=finish_reason,
                metadata={
                    **answered_by(answered, self._config.model),
                    **({"prompt_block_reason": block_reason} if block_reason else {}),
                },
            )

        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def generate(self, context: AIContext) -> AIResponse:
        """Generate by consuming the structured stream."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[AIToolCall] = []
        done_event: StreamDone | None = None

        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamThinkingDelta):
                thinking_parts.append(event.thinking)
            elif isinstance(event, StreamTextDelta):
                text_parts.append(event.text)
            elif isinstance(event, StreamToolCall):
                tool_calls.append(tool_call_of(event))
            elif isinstance(event, StreamDone):
                done_event = event

        finish_reason = done_event.finish_reason if done_event else None
        return AIResponse(
            content="".join(text_parts),
            thinking="".join(thinking_parts) if thinking_parts else None,
            usage=done_event.usage if done_event else {},
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            metadata=done_event.metadata if done_event else {},
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text deltas as they arrive from the Gemini API."""
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    async def close(self) -> None:
        """Close the SDK and the httpx client it was given."""
        client, self._client = self._client, None
        http, self._http = self._http, None
        await close_genai_client(client, http)
