"""Google Gemini AI provider — generates responses via the Google Generative AI API.

The provider owns the client, the call and the stream. Turning RoomKit
messages and context into the request lives in ``request.py``.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    AIToolCall,
    ModelInfo,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.gemini.errors import wrap_gemini_error
from roomkit.providers.gemini.models import MODELS
from roomkit.providers.gemini.request import (
    build_gen_config,
    format_messages,
    reject_model_turn_tail,
)
from roomkit.providers.gemini.sdk import build_genai_client, close_genai_client

logger = logging.getLogger(__name__)


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
        return self._merge_curated(live)

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Wrap an SDK exception into a ProviderError."""
        return wrap_gemini_error(exc)

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events from the Gemini streaming API."""
        gen_config = build_gen_config(self._types, self._config, context)
        contents = format_messages(self._types, context.messages)
        # Before the try below, whose ``_wrap_error`` is for SDK exceptions and
        # would restate this one as an opaque provider failure.
        reject_model_turn_tail(contents)

        t0 = time.monotonic()
        first_token = True
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

        try:
            response_stream = await self._client.aio.models.generate_content_stream(  # ty: ignore[unresolved-attribute]
                model=self._config.model,
                contents=contents,
                config=gen_config,
            )
            async for chunk in response_stream:
                # Extract usage from each chunk (last one has the totals)
                if chunk.usage_metadata:
                    # Canonical key names (input_tokens / output_tokens) so the
                    # downstream usage tracker records Gemini like every other
                    # provider — the SDK calls them prompt/candidates counts.
                    # Gemini's prompt count INCLUDES implicitly-cached tokens;
                    # report them separately (Anthropic-style accounting, where
                    # input excludes cache reads) so cost math doesn't double-
                    # charge the cached prefix and cache rates can apply.
                    prompt = chunk.usage_metadata.prompt_token_count or 0
                    cached = getattr(chunk.usage_metadata, "cached_content_token_count", None) or 0
                    usage = {
                        "input_tokens": max(prompt - cached, 0),
                        "output_tokens": chunk.usage_metadata.candidates_token_count or 0,
                    }
                    if cached:
                        usage["cache_read_input_tokens"] = cached

                # Read before the content guards below: the chunk that reports
                # MAX_TOKENS is often the one whose candidate carries no parts,
                # so capturing it after them would drop the very case the tool
                # loop needs — a round truncated with nothing to show for it.
                # The SDK hands back a FinishReason enum; ``.name`` is its wire
                # spelling ("MAX_TOKENS"), and a plain string passes through.
                if chunk.candidates:
                    raw_reason = getattr(chunk.candidates[0], "finish_reason", None)
                    if raw_reason is not None:
                        finish_reason = getattr(raw_reason, "name", None) or str(raw_reason)

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

                for part in parts:
                    if hasattr(part, "text") and part.text:
                        if first_token:
                            self._record_ttfb(t0)
                            first_token = False
                        # Thought-summary parts are flagged thought=True.
                        if getattr(part, "thought", False):
                            yield StreamThinkingDelta(thinking=part.text)
                        else:
                            yield StreamTextDelta(text=part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        fc_name = fc.name or ""
                        fc_args = dict(fc.args) if fc.args else {}
                        # thought_signature lives on the Part (bytes), not the
                        # FunctionCall. Encode to a portable str for metadata.
                        raw_sig = getattr(part, "thought_signature", None)
                        sig = (
                            base64.b64encode(raw_sig).decode("ascii")
                            if isinstance(raw_sig, bytes)
                            else raw_sig
                        )
                        try:
                            fp = json.dumps(fc_args, sort_keys=True, default=str)
                        except (TypeError, ValueError):
                            fp = repr(fc_args)
                        key = f"{fc_name}::{fp}"
                        if key not in fcalls:
                            fcalls[key] = {
                                "id": f"call_{uuid4().hex[:12]}",
                                "name": fc_name,
                                "arguments": fc_args,
                                "signature": sig,
                            }
                            fcall_order.append(key)
                        elif fcalls[key]["signature"] is None and sig is not None:
                            # Re-emission carried the signature the first did not.
                            fcalls[key]["signature"] = sig

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
                        "thought_signature (thinking_level=%s, thinking_budget=%s) — "
                        "nothing to replay them signed with; Gemini 3 rejects the next "
                        "turn when the model was thinking",
                        len(fcall_order),
                        [fcalls[k]["name"] for k in fcall_order],
                        self._config.thinking_level,
                        context.thinking_budget,
                    )
            for key in fcall_order:
                fc_data = fcalls[key]
                meta: dict[str, Any] = {}
                if fc_data["signature"] is not None:
                    meta["thought_signature"] = fc_data["signature"]
                yield StreamToolCall(
                    id=fc_data["id"],
                    name=fc_data["name"],
                    arguments=fc_data["arguments"],
                    metadata=meta,
                )

            yield StreamDone(
                usage=usage,
                finish_reason=finish_reason,
                metadata={"model": self._config.model},
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
                tool_calls.append(
                    AIToolCall(
                        id=event.id,
                        name=event.name,
                        arguments=event.arguments,
                        metadata=event.metadata,
                    )
                )
            elif isinstance(event, StreamDone):
                done_event = event

        return AIResponse(
            content="".join(text_parts),
            thinking="".join(thinking_parts) if thinking_parts else None,
            usage=done_event.usage if done_event else {},
            tool_calls=tool_calls,
            finish_reason=done_event.finish_reason if done_event else None,
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
